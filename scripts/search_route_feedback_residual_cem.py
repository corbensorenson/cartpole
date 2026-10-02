#!/usr/bin/env python
"""Search a phase-dependent feedback residual around a transferred route.

This is a narrow n+1 continuation probe.  The inherited route supplies the
feedforward swing timing while a low-dimensional, time-varying gain acts on
cart state and the distal mode.  Every candidate is evaluated with the exact
environment step loop; the result is diagnostic until it passes canonical
replay and the project's noisy gates.
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
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp


def load_route(path: str | Path) -> tuple[np.ndarray, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    controller = payload.get("controller", payload)
    if not isinstance(controller, dict):
        raise ValueError("route must contain a controller object")
    controls = np.asarray(controller.get("controls", []), dtype=np.float64)
    seconds = float(
        controller.get(
            "horizon_seconds",
            payload.get("search", {}).get("seconds", 0.0),
        )
    )
    if controls.ndim != 1 or controls.size < 2 or seconds <= 0.0:
        raise ValueError("route controls or horizon are invalid")
    return np.clip(controls, -1.0, 1.0), seconds


def feature_vector(env: NLinkCartPoleEnv) -> np.ndarray:
    qpos = np.asarray(env.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    relative = wrap_angle(qpos[1 : 1 + env.n])
    absolute = serial_absolute_angles(relative)
    relative_rate = qvel[1 : 1 + env.n]
    absolute_rate = np.cumsum(relative_rate)
    return np.asarray(
        [
            qpos[0] / 1.25,
            qvel[0] / 0.50,
            absolute[-2] / 0.15,
            absolute[-1] / 0.15,
            relative[-2] / 0.15,
            relative[-1] / 0.15,
            absolute_rate[-2] / 0.75,
            absolute_rate[-1] / 0.75,
            relative_rate[-2] / 0.75,
            relative_rate[-1] / 0.75,
        ],
        dtype=np.float64,
    )


def interpolated_gains(gains: np.ndarray, step: int, step_count: int) -> np.ndarray:
    phase = float(step) * (gains.shape[0] - 1) / max(1, step_count - 1)
    left = min(gains.shape[0] - 1, int(np.floor(phase)))
    right = min(gains.shape[0] - 1, left + 1)
    fraction = phase - float(left)
    return (1.0 - fraction) * gains[left] + fraction * gains[right]


def evaluate(
    cfg: dict[str, Any],
    base_controls: np.ndarray,
    gains: np.ndarray,
    *,
    seed: int,
    rail_soft_limit: float,
    rail_weight: float,
    potential_weight: float,
    late_start_fraction: float,
    keep_trace: bool = False,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
    env.reset(seed=seed)
    weights = np.asarray(env.morphology.lengths, dtype=np.float64) * (
        np.cumsum(np.asarray(env.morphology.masses, dtype=np.float64)[::-1])[::-1]
        - 0.5 * np.asarray(env.morphology.masses, dtype=np.float64)
    )
    rows: list[dict[str, Any]] = []
    previous_action = 0.0
    max_cart = 0.0
    for step, base in enumerate(base_controls):
        features = feature_vector(env)
        residual = float(interpolated_gains(gains, step, base_controls.size) @ features)
        action = float(np.clip(float(base) + residual, -1.0, 1.0))
        _, _, terminated, truncated, info = env.step([action])
        qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
        absolute = serial_absolute_angles(qpos[1:])
        absolute_rate = np.cumsum(qvel[1:])
        potential = float(
            np.sum(weights * (np.cos(absolute) + 1.0) * 0.5)
            / max(1.0e-12, float(np.sum(weights)))
        )
        state_cost = (
            35.0 * (float(np.max(np.abs(absolute))) / 0.15) ** 2
            + 16.0 * (float(np.sqrt(np.mean(qvel[1:] ** 2))) / 0.75) ** 2
            + 8.0 * (float(np.sqrt(np.mean(absolute_rate**2))) / 0.75) ** 2
            + 3.0 * (abs(float(qpos[0])) / 1.25) ** 2
            + 3.0 * (abs(float(qvel[0])) / 0.50) ** 2
        )
        max_cart = max(max_cart, abs(float(qpos[0])))
        rows.append(
            {
                "step": int(step + 1),
                "time_seconds": float((step + 1) * env.dt),
                "action": action,
                "residual": residual,
                "x": float(qpos[0]),
                "cart_velocity": float(qvel[0]),
                "max_abs_angle": float(np.max(np.abs(absolute))),
                "hinge_velocity_rms": float(np.sqrt(np.mean(qvel[1:] ** 2))),
                "absolute_rate_rms": float(np.sqrt(np.mean(absolute_rate**2))),
                "potential_fraction": potential,
                "state_cost": state_cost,
                "action_slew": float(action - previous_action),
                "qpos": qpos.astype(float).tolist() if keep_trace else None,
                "qvel": qvel.astype(float).tolist() if keep_trace else None,
                "is_upright": bool(info.get("is_upright", False)),
                "max_upright_streak_seconds": float(
                    info.get("max_upright_streak_seconds", 0.0)
                ),
            }
        )
        previous_action = action
        if terminated or truncated:
            break

    env.close()
    if not rows:
        return {
            "score": 1.0e15,
            "success": False,
            "termination_reason": "no_rows",
            "steps": 0,
            "rows": rows,
        }
    late_start = min(len(rows) - 1, max(1, int(len(base_controls) * late_start_fraction)))
    late = rows[late_start:]
    state_costs = np.asarray([row["state_cost"] for row in late], dtype=np.float64)
    potentials = np.asarray([row["potential_fraction"] for row in late], dtype=np.float64)
    window = min(12, len(state_costs))
    rolling = np.asarray(
        [
            np.mean(state_costs[index : index + window])
            for index in range(len(state_costs) - window + 1)
        ],
        dtype=np.float64,
    )
    best_index = late_start + int(np.argmin(rolling)) + window - 1
    terminal = rows[-1]
    max_streak = max(float(row["max_upright_streak_seconds"]) for row in rows)
    rail_penalty = rail_weight * max(0.0, max_cart / rail_soft_limit - 1.0) ** 2
    incomplete_penalty = 1.0e8 if len(rows) != len(base_controls) else 0.0
    score = (
        0.55 * float(np.min(rolling))
        + 0.25 * float(terminal["state_cost"])
        + 0.20 * float(np.mean(state_costs[-min(30, len(state_costs)) :]))
        - potential_weight * float(np.max(potentials))
        + rail_penalty
        + incomplete_penalty
        + 0.02 * float(np.mean(np.asarray([row["action"] for row in rows]) ** 2))
        + 0.08
        * float(
            np.mean(
                np.asarray([row["action_slew"] for row in rows], dtype=np.float64)
                ** 2
            )
        )
    )
    return {
        "score": float(score),
        "success": bool(terminal.get("is_upright", False) and max_streak >= 5.0),
        "termination_reason": None if len(rows) == len(base_controls) else "rail_violation",
        "steps": len(rows),
        "max_cart_abs": float(max_cart),
        "max_upright_streak_seconds": float(max_streak),
        "max_potential_fraction": float(np.max(potentials)),
        "best_time_seconds": float(rows[best_index]["time_seconds"]),
        "best_state_cost": float(rows[best_index]["state_cost"]),
        "best_angle": float(rows[best_index]["max_abs_angle"]),
        "best_cart": float(rows[best_index]["x"]),
        "best_cart_velocity": float(rows[best_index]["cart_velocity"]),
        "best_absolute_rate_rms": float(rows[best_index]["absolute_rate_rms"]),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--initial-controller", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--segments", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=24)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--gain-sigma", type=float, default=0.12)
    parser.add_argument("--gain-sigma-floor", type=float, default=0.005)
    parser.add_argument("--gain-limit", type=float, default=2.0)
    parser.add_argument("--rail-soft-limit", type=float, default=2.8)
    parser.add_argument("--rail-weight", type=float, default=1.0e6)
    parser.add_argument("--potential-weight", type=float, default=500.0)
    parser.add_argument("--late-start-fraction", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=21121)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.segments < 2 or args.iterations < 1 or args.population < 2:
        raise ValueError("segments, iterations, and population must be positive")
    if not 1 <= args.elites <= args.population:
        raise ValueError("elites must be in 1..population")
    if min(args.gain_sigma, args.gain_sigma_floor, args.gain_limit) <= 0.0:
        raise ValueError("gain scales must be positive")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][key] = 0.0
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    base_controls, route_seconds = load_route(args.initial_controller)
    feature_names = [
        "cart_position",
        "cart_velocity",
        "distal_absolute_angle",
        "tip_absolute_angle",
        "distal_relative_angle",
        "tip_relative_angle",
        "distal_absolute_rate",
        "tip_absolute_rate",
        "distal_relative_rate",
        "tip_relative_rate",
    ]
    gain_shape = (args.segments, len(feature_names))
    center = np.zeros(gain_shape, dtype=np.float64)
    sigma = np.full(gain_shape, args.gain_sigma, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    started = time.time()
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for iteration in range(args.iterations):
        candidates = np.clip(
            center[None, :, :]
            + rng.normal(0.0, sigma, size=(args.population, *gain_shape)),
            -args.gain_limit,
            args.gain_limit,
        )
        candidates[0] = center
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            metrics = evaluate(
                cfg,
                base_controls,
                candidate,
                seed=args.seed,
                rail_soft_limit=args.rail_soft_limit,
                rail_weight=args.rail_weight,
                potential_weight=args.potential_weight,
                late_start_fraction=args.late_start_fraction,
            )
            records.append({"gains": candidate, "metrics": metrics})
        records.sort(key=lambda row: float(row["metrics"]["score"]))
        elite = np.asarray([row["gains"] for row in records[: args.elites]])
        center = np.mean(elite, axis=0)
        sigma = np.maximum(np.std(elite, axis=0) * 0.90, args.gain_sigma_floor)
        top = records[0]
        if best is None or float(top["metrics"]["score"]) < float(best["metrics"]["score"]):
            best = {
                "gains": top["gains"].copy(),
                "metrics": top["metrics"],
                "iteration": iteration + 1,
            }
        tm = top["metrics"]
        history.append(
            {
                "iteration": int(iteration + 1),
                "score": float(tm["score"]),
                "max_potential_fraction": float(tm.get("max_potential_fraction", 0.0)),
                "best_angle": tm.get("best_angle"),
                "best_state_cost": tm.get("best_state_cost"),
                "max_cart_abs": tm.get("max_cart_abs"),
                "max_upright_streak_seconds": tm.get("max_upright_streak_seconds"),
                "sigma_max": float(np.max(sigma)),
            }
        )
        print(
            f"iter={iteration + 1:03d} score={tm['score']:.2f} "
            f"potential={tm.get('max_potential_fraction', 0.0):.3f} "
            f"angle={tm.get('best_angle')} x={tm.get('max_cart_abs')} "
            f"hold={tm.get('max_upright_streak_seconds', 0.0):.3f}s",
            flush=True,
        )

    assert best is not None
    traced = evaluate(
        cfg,
        base_controls,
        np.asarray(best["gains"], dtype=np.float64),
        seed=args.seed,
        rail_soft_limit=args.rail_soft_limit,
        rail_weight=args.rail_weight,
        potential_weight=args.potential_weight,
        late_start_fraction=args.late_start_fraction,
        keep_trace=True,
    )
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_route_feedback_residual_cem_not_solution",
        "not_solution": True,
        "summary": "Exact serial CEM for a phase-dependent distal-mode feedback residual around a transferred 11-link route.",
        "config_path": str(Path(args.config)),
        "initial_controller": {
            "path": str(Path(args.initial_controller)),
            "sha256": data_sha256(json.loads(Path(args.initial_controller).read_text(encoding="utf-8"))),
        },
        "route_seconds": float(route_seconds),
        "feature_names": feature_names,
        "best_gains": np.asarray(best["gains"], dtype=np.float64).astype(float).tolist(),
        "best": traced,
        "search": {
            "algorithm": "serial_exact_mujoco_phase_dependent_distal_feedback_cem",
            "segments": int(args.segments),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "gain_sigma": float(args.gain_sigma),
            "gain_sigma_floor": float(args.gain_sigma_floor),
            "gain_limit": float(args.gain_limit),
            "rail_soft_limit": float(args.rail_soft_limit),
            "rail_weight": float(args.rail_weight),
            "potential_weight": float(args.potential_weight),
            "late_start_fraction": float(args.late_start_fraction),
            "seed": int(args.seed),
            "history": history,
            "wall_time_seconds": float(time.time() - started),
        },
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(result, Path(args.out))
    print(
        f"best potential={traced.get('max_potential_fraction', 0.0):.3f} "
        f"angle={traced.get('best_angle')} cart={traced.get('max_cart_abs')} "
        f"hold={traced.get('max_upright_streak_seconds', 0.0):.3f}s"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
