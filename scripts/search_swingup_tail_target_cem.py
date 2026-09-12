#!/usr/bin/env python
"""Exact serial CEM search for a target-directed swing-up arrest tail.

The source prefix is replayed once in the final MuJoCo step loop.  CEM changes
only a declared tail, and every candidate is evaluated from the same physical
state at that boundary.  This is a discovery tool: the stitched result still
requires reset-free capture and held-out verification.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles, wrap_angle
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp


def interpolation_matrix(knot_count: int, step_count: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, knot_count, dtype=np.float64)
    target = np.linspace(0.0, 1.0, step_count, dtype=np.float64)
    right = np.searchsorted(source, target, side="right").clip(1, knot_count - 1)
    left = right - 1
    fraction = (target - source[left]) / (source[right] - source[left])
    matrix = np.zeros((step_count, knot_count), dtype=np.float64)
    rows = np.arange(step_count)
    matrix[rows, left] = 1.0 - fraction
    matrix[rows, right] += fraction
    return matrix


def load_source_controls(path: str, *, steps: int, dt: float) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    controller = payload.get("controller") if isinstance(payload, dict) else None
    if isinstance(controller, dict) and controller.get("controls") is not None:
        source = np.asarray(controller["controls"], dtype=np.float64)
        source_seconds = float(controller.get("horizon_seconds", source.size * dt))
    else:
        # The serial global CEM records the exact validated waveform as best.knots.
        # Accepting that schema avoids converting a measured force trace by hand.
        record = payload.get("best") if isinstance(payload, dict) else None
        if not isinstance(record, dict) or record.get("knots") is None:
            raise ValueError(f"{path} must contain controller.controls or best.knots")
        source = np.asarray(record["knots"], dtype=np.float64)
        source_seconds = float(payload.get("search", {}).get("seconds", source.size * dt))
    if source.ndim != 1 or source.size < 2:
        raise ValueError("source controls must be a one-dimensional sequence")
    source_t = np.linspace(0.0, max(source_seconds, dt), source.size, dtype=np.float64)
    target_t = np.arange(steps, dtype=np.float64) * dt
    return np.clip(np.interp(target_t, source_t, source, left=source[0], right=source[-1]), -1.0, 1.0)


def load_target(path: str, *, scale: float, n_links: int) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(states, list) or not states or not isinstance(states[0], dict):
        raise ValueError(f"{path} must contain a non-empty states list")
    state = states[0]
    qpos = np.asarray(state.get("qpos", []), dtype=np.float64)
    qvel = np.asarray(state.get("qvel", []), dtype=np.float64)
    expected = n_links + 1
    if qpos.shape != (expected,) or qvel.shape != (expected,):
        raise ValueError(f"target qpos/qvel must have shape {(expected,)}")
    if not 0.0 < scale <= 1.0:
        raise ValueError("target scale must be in (0, 1]")
    target_qpos = np.zeros(expected, dtype=np.float64)
    target_qpos[0] = scale * qpos[0]
    target_qpos[1:] = scale * wrap_angle(qpos[1:])
    target_qvel = scale * qvel
    target = np.r_[target_qpos, target_qvel]
    return target, {"source": file_metadata(path), "scale": float(scale), "qpos": target_qpos.tolist(), "qvel": target_qvel.tolist()}


def state_metrics(env: NLinkCartPoleEnv, target: np.ndarray) -> dict[str, float]:
    qpos = np.asarray(env.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    absolute = serial_absolute_angles(qpos[1:])
    target_absolute = serial_absolute_angles(target[1 : env.n + 1])
    angle_error = wrap_angle(absolute - target_absolute)
    absolute_rate = np.cumsum(qvel[1:])
    target_absolute_rate = np.cumsum(target[env.n + 2 :])
    rate_error = absolute_rate - target_absolute_rate
    return {
        "max_abs_angle": float(np.max(np.abs(absolute))),
        "hinge_velocity_rms": float(np.sqrt(np.mean(qvel[1:] ** 2))),
        "absolute_rate_rms": float(np.sqrt(np.mean(absolute_rate**2))),
        "target_angle_error_rms": float(np.sqrt(np.mean(angle_error**2))),
        "target_angle_error_max": float(np.max(np.abs(angle_error))),
        "target_absolute_rate_error_rms": float(np.sqrt(np.mean(rate_error**2))),
        "target_cart_error": float(qpos[0] - target[0]),
        "target_cart_velocity_error": float(qvel[0] - target[env.n + 1]),
        "cart_abs": abs(float(qpos[0])),
        "cart_velocity_abs": abs(float(qvel[0])),
    }


def replay_prefix(env: NLinkCartPoleEnv, controls: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    env.reset(seed=0)
    for step, action in enumerate(controls[:steps]):
        _, _, terminated, truncated, info = env.step([float(action)])
        if terminated or truncated:
            raise RuntimeError(f"source prefix terminated at step {step}: {info.get('termination_reason')}")
    return np.asarray(env.data.qpos, dtype=np.float64).copy(), np.asarray(env.data.qvel, dtype=np.float64).copy()


def evaluate_candidate(
    env: NLinkCartPoleEnv,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    actions: np.ndarray,
    target: np.ndarray,
    *,
    target_angle_weight: float,
    target_rate_weight: float,
    target_cart_weight: float,
    target_cart_velocity_weight: float,
    stage_weight: float,
    rail_soft_limit: float,
) -> tuple[float, dict[str, Any]]:
    env.reset(seed=0, options={"qpos": start_qpos, "qvel": start_qvel})
    rows: list[dict[str, float]] = []
    action_sq = 0.0
    action_slew = 0.0
    previous = 0.0
    max_cart = abs(float(env.data.qpos[0]))
    for action in actions:
        _, _, terminated, truncated, info = env.step([float(action)])
        row = state_metrics(env, target)
        rows.append(row)
        action_sq += float(action) ** 2
        action_slew += (float(action) - previous) ** 2
        previous = float(action)
        max_cart = max(max_cart, row["cart_abs"])
        if terminated or truncated:
            return float("inf"), {"invalid": True, "termination_reason": info.get("termination_reason"), "rows": rows}
    terminal = rows[-1]
    stage = np.asarray(
        [
            row["target_angle_error_rms"] ** 2
            + 0.25 * row["target_absolute_rate_error_rms"] ** 2
            + 0.05 * row["target_cart_error"] ** 2
            for row in rows
        ],
        dtype=np.float64,
    )
    rail_penalty = 200000.0 * max(0.0, max_cart / rail_soft_limit - 1.0) ** 2
    cost = (
        float(target_angle_weight) * terminal["target_angle_error_rms"] ** 2
        + float(target_rate_weight) * terminal["target_absolute_rate_error_rms"] ** 2
        + float(target_cart_weight) * terminal["target_cart_error"] ** 2
        + float(target_cart_velocity_weight) * terminal["target_cart_velocity_error"] ** 2
        + float(stage_weight) * float(np.mean(stage))
        + rail_penalty
        + 0.01 * action_sq
        + 0.04 * action_slew
    )
    return float(cost), {
        "invalid": False,
        "termination_reason": "time_limit",
        "steps": len(rows),
        "cost": float(cost),
        "terminal": terminal,
        "max_cart": float(max_cart),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact serial CEM search for a target-directed swing-up arrest tail")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--source-json", required=True)
    parser.add_argument("--tail-start-seconds", type=float, default=3.0)
    parser.add_argument("--tail-seconds", type=float, default=1.56)
    parser.add_argument(
        "--tail-center",
        choices=("source", "zero"),
        default="source",
        help="initialize the optimized tail from the source continuation or zero force",
    )
    parser.add_argument(
        "--initial-tail-json",
        default=None,
        help="optional prior target-tail artifact whose best knots seed CEM",
    )
    parser.add_argument("--target-state", required=True)
    parser.add_argument("--target-scale", type=float, default=0.01)
    parser.add_argument("--knot-count", type=int, default=24)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--action-sigma", type=float, default=0.06)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.002)
    parser.add_argument("--target-angle-weight", type=float, default=500000.0)
    parser.add_argument("--target-rate-weight", type=float, default=500000.0)
    parser.add_argument("--target-cart-weight", type=float, default=5000.0)
    parser.add_argument("--target-cart-velocity-weight", type=float, default=5000.0)
    parser.add_argument("--stage-weight", type=float, default=0.25)
    parser.add_argument("--rail-soft-limit", type=float, default=2.90)
    parser.add_argument("--seed", type=int, default=20791)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if min(args.tail_start_seconds, args.tail_seconds, args.knot_count, args.population, args.elites, args.iterations) <= 0.0:
        raise ValueError("tail duration and CEM dimensions must be positive")
    if args.elites > args.population or args.knot_count < 2:
        raise ValueError("invalid CEM dimensions")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "episode_seconds": float(args.tail_start_seconds + args.tail_seconds + 1.0),
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "terminate_abs_angle": None,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    dt = env.dt
    total_steps = max(2, int(round((args.tail_start_seconds + args.tail_seconds) / dt)))
    tail_steps = max(2, int(round(args.tail_seconds / dt)))
    prefix_steps = max(0, int(round(args.tail_start_seconds / dt)))
    source_controls = load_source_controls(args.source_json, steps=total_steps, dt=dt)
    start_qpos, start_qvel = replay_prefix(env, source_controls, prefix_steps)
    target, target_metadata = load_target(args.target_state, scale=args.target_scale, n_links=env.n)
    interpolation = interpolation_matrix(args.knot_count, tail_steps)
    if args.tail_center == "zero":
        center = np.zeros(args.knot_count, dtype=np.float64)
    else:
        center = np.interp(
            np.linspace(0.0, 1.0, args.knot_count),
            np.linspace(0.0, 1.0, tail_steps),
            source_controls[prefix_steps : prefix_steps + tail_steps],
        )
    if args.initial_tail_json is not None:
        prior = json.loads(Path(args.initial_tail_json).read_text(encoding="utf-8"))
        prior_best = prior.get("best") if isinstance(prior, dict) else None
        prior_knots = None if not isinstance(prior_best, dict) else prior_best.get("knots")
        if prior_knots is None:
            raise ValueError("initial-tail-json must contain best.knots")
        prior_knots = np.asarray(prior_knots, dtype=np.float64)
        if prior_knots.ndim != 1 or prior_knots.size < 2:
            raise ValueError("initial best knots must be a one-dimensional sequence")
        center = np.interp(
            np.linspace(0.0, 1.0, args.knot_count),
            np.linspace(0.0, 1.0, prior_knots.size),
            prior_knots,
        )
    rng = np.random.default_rng(args.seed)
    spread = np.full(args.knot_count, float(args.action_sigma), dtype=np.float64)
    best_record: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    for iteration in range(args.iterations):
        candidates = np.clip(center[None, :] + rng.normal(0.0, spread, size=(args.population, args.knot_count)), -1.0, 1.0)
        candidates[0] = center
        records: list[dict[str, Any]] = []
        for index, knots in enumerate(candidates):
            actions = np.clip(knots @ interpolation.T, -1.0, 1.0)
            cost, metrics = evaluate_candidate(
                env,
                start_qpos,
                start_qvel,
                actions,
                target,
                target_angle_weight=args.target_angle_weight,
                target_rate_weight=args.target_rate_weight,
                target_cart_weight=args.target_cart_weight,
                target_cart_velocity_weight=args.target_cart_velocity_weight,
                stage_weight=args.stage_weight,
                rail_soft_limit=args.rail_soft_limit,
            )
            records.append({"index": index, "cost": float(cost), "metrics": metrics, "knots": knots})
        records.sort(key=lambda record: float(record["cost"]))
        top = records[0]
        elites = np.asarray([record["knots"] for record in records[: args.elites]], dtype=np.float64)
        center = np.mean(elites, axis=0)
        spread = np.maximum(np.std(elites, axis=0) * args.sigma_decay, args.sigma_floor)
        metrics = top["metrics"]
        record = {"iteration": iteration + 1, "cost": float(top["cost"]), **metrics}
        history.append(record)
        if best_record is None or record["cost"] < best_record["cost"]:
            best_record = {**record, "knots": top["knots"].astype(float).tolist()}
        terminal = metrics.get("terminal", {})
        print(
            f"iter={iteration + 1:03d} cost={float(top['cost']):.3f} "
            f"angle_err={terminal.get('target_angle_error_max', np.nan):.6f} "
            f"rate_err={terminal.get('target_absolute_rate_error_rms', np.nan):.6f} "
            f"xerr={terminal.get('target_cart_error', np.nan):.6f} "
            f"maxx={metrics.get('max_cart', np.nan):.3f}",
            flush=True,
        )
    assert best_record is not None
    best_knots = np.asarray(best_record["knots"], dtype=np.float64)
    tail_actions = np.clip(best_knots @ interpolation.T, -1.0, 1.0)
    stitched = np.r_[source_controls[:prefix_steps], tail_actions]
    env.reset(seed=0)
    rows: list[dict[str, Any]] = []
    for step, action in enumerate(stitched):
        _, _, terminated, truncated, info = env.step([float(action)])
        row = state_metrics(env, target)
        row.update({"step": int(step + 1), "time_seconds": float((step + 1) * dt), "action": float(action), "is_upright": bool(info.get("is_upright", False)), "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0))})
        rows.append(row)
        if terminated or truncated:
            break
    terminal = rows[-1]
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact serial CEM tail search aimed at a measured downstream-capturable state; reset-free verification required.",
        "config_path": str(Path(args.config)),
        "config_sha256": data_sha256(cfg),
        "source_controller": file_metadata(args.source_json),
        "target": target_metadata,
        "tail": {
            "start_seconds": float(args.tail_start_seconds),
            "seconds": float(args.tail_seconds),
            "center": args.tail_center,
            "initial_tail_json": args.initial_tail_json,
            "prefix_steps": int(prefix_steps),
            "steps": int(tail_steps),
            "knot_count": int(args.knot_count),
            "best_knots": best_knots.astype(float).tolist(),
            "best_actions": tail_actions.astype(float).tolist(),
            "stitched_controls": stitched.astype(float).tolist(),
            "start_qpos": start_qpos.astype(float).tolist(),
            "start_qvel": start_qvel.astype(float).tolist(),
        },
        "search": {
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "target_angle_weight": float(args.target_angle_weight),
            "target_rate_weight": float(args.target_rate_weight),
            "target_cart_weight": float(args.target_cart_weight),
            "target_cart_velocity_weight": float(args.target_cart_velocity_weight),
            "stage_weight": float(args.stage_weight),
            "rail_soft_limit": float(args.rail_soft_limit),
            "seed": int(args.seed),
            "wall_time_seconds": float(time.time() - started),
        },
        "best": best_record,
        "terminal_metrics": terminal,
        "replay": {"steps": len(rows), "termination_reason": "time_limit" if len(rows) == len(stitched) else "terminated", "rows": rows},
        "history": history,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(payload, Path(args.out))
    print(f"terminal_angle_error={terminal['target_angle_error_max']:.6f} terminal_rate_error={terminal['target_absolute_rate_error_rms']:.6f} terminal_x={terminal['cart_abs']:.6f}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
