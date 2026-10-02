#!/usr/bin/env python
"""Search a time-reversed down-fall seed for a serial n-link swing-up.

The probe searches one smooth force waveform on the exact target plant.  It
first applies that waveform from the upright equilibrium and penalizes the
terminal state unless it is close to the hanging equilibrium.  It then
reverses the same waveform and evaluates it from the exact hanging start.
This is a development seed only: the reversed open-loop route still needs
target-plant feedback refinement and all canonical evidence gates.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp


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


def state_config(
    cfg: dict[str, Any], *, qpos: list[float], qvel: list[float], seconds: float
) -> dict[str, Any]:
    env = {
        **cfg["env"],
        "init_mode": "fixed_state",
        "init_qpos": list(qpos),
        "init_qvel": list(qvel),
        "episode_seconds": float(seconds),
        "terminate_abs_angle": None,
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
    }
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        env[f"{key}_start"] = 0.0
        env[f"{key}_end"] = 0.0
    return {**cfg, "env": env}


def angle_error_to_hanging(qpos: np.ndarray) -> float:
    absolute = serial_absolute_angles(np.asarray(qpos[1:], dtype=np.float64))
    error = np.arctan2(np.sin(absolute - np.pi), np.cos(absolute - np.pi))
    return float(np.sqrt(np.mean(error**2)))


def row_metrics(env: NLinkCartPoleEnv, info: dict[str, Any]) -> dict[str, Any]:
    absolute = serial_absolute_angles(np.asarray(env.data.qpos[1:], dtype=np.float64))
    absolute_rate = np.cumsum(np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64))
    return {
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "max_abs_angle": float(np.max(np.abs(absolute))),
        "angle_rms": float(np.sqrt(np.mean(absolute**2))),
        "hinge_velocity_rms": float(
            np.sqrt(np.mean(np.asarray(env.data.qvel[1 : 1 + env.n]) ** 2))
        ),
        "absolute_angular_velocity_rms": float(np.sqrt(np.mean(absolute_rate**2))),
        "x": float(env.data.qpos[0]),
        "cart_velocity": float(env.data.qvel[0]),
        "is_upright": bool(info.get("is_upright", False)),
        "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
        "max_upright_streak_seconds": float(
            info.get("max_upright_streak_seconds", 0.0)
        ),
    }


def rollout(
    cfg: dict[str, Any],
    actions: np.ndarray,
    *,
    qpos: list[float],
    qvel: list[float],
    seconds: float,
    trace: bool = False,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(
        state_config(cfg, qpos=qpos, qvel=qvel, seconds=seconds),
        progress=1.0,
        seed=0,
    )
    env.reset(seed=0)
    all_rows: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    max_cart = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    for step, action in enumerate(np.asarray(actions, dtype=np.float64)):
        _, _, terminated, truncated, info = env.step([float(action)])
        row = row_metrics(env, info)
        row["step"] = int(step + 1)
        row["time_seconds"] = float((step + 1) * env.dt)
        row["action"] = float(action)
        max_cart = max(max_cart, abs(float(row["x"])))
        all_rows.append(row)
        final_info = dict(info)
        if trace and (step % 2 == 0 or row["is_upright"]):
            trace_rows.append(row)
        if terminated or truncated:
            break
    final_row = all_rows[-1] if all_rows else {}
    env.close()
    return {
        "rows": all_rows,
        "trace_rows": trace_rows,
        "final": final_row,
        "max_cart_abs": float(max_cart),
        "steps": int(len(all_rows)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def evaluate_candidate(
    cfg: dict[str, Any],
    actions: np.ndarray,
    *,
    seconds: float,
    handoff_min_time: float,
    rail_penalty_limit: float,
    down_weight: float,
    reverse_weight: float,
    terminal_down_weight: float,
    return_trace: bool = False,
) -> tuple[float, dict[str, Any]]:
    n = int(cfg["env"]["n_links"])
    upright_qpos = [0.0] * (n + 1)
    upright_qvel = [0.0] * (n + 1)
    hanging_qpos = [0.0, float(np.pi)] + [0.0] * (n - 1)
    hanging_qvel = [0.0] * (n + 1)

    down = rollout(
        cfg,
        actions,
        qpos=upright_qpos,
        qvel=upright_qvel,
        seconds=seconds,
        trace=return_trace,
    )
    reversed_actions = np.asarray(actions, dtype=np.float64)[::-1]
    reverse = rollout(
        cfg,
        reversed_actions,
        qpos=hanging_qpos,
        qvel=hanging_qvel,
        seconds=seconds,
        trace=return_trace,
    )

    down_final_qpos = np.asarray(down["final"].get("qpos", upright_qpos), dtype=np.float64)
    down_final_qvel = np.asarray(down["final"].get("qvel", upright_qvel), dtype=np.float64)
    down_angle_error = angle_error_to_hanging(down_final_qpos)
    down_terminal_cost = (
        (down_angle_error / 0.15) ** 2
        + 0.75 * (np.linalg.norm(down_final_qvel) / 0.75) ** 2
        + (abs(float(down_final_qpos[0])) / 1.25) ** 2
        + (abs(float(down_final_qvel[0])) / 0.50) ** 2
    )
    # The exact serial rollout keeps all rows for scoring; only the selected
    # candidate stores the compact trace in the final artifact.
    score_rows = reverse["rows"]
    late_rows = [row for row in score_rows if row["time_seconds"] >= handoff_min_time]
    if not late_rows:
        late_rows = score_rows
    reverse_scores = [
        28.0 * (row["max_abs_angle"] / 0.15) ** 2
        + 10.0 * (row["hinge_velocity_rms"] / 0.75) ** 2
        + 10.0 * (row["absolute_angular_velocity_rms"] / 0.75) ** 2
        + 3.0 * (abs(row["x"]) / 1.25) ** 2
        + 3.0 * (abs(row["cart_velocity"]) / 0.50) ** 2
        for row in late_rows
    ]
    best_index = int(np.argmin(reverse_scores)) if reverse_scores else 0
    best_reverse = late_rows[best_index] if late_rows else {}
    reverse_best_cost = float(min(reverse_scores)) if reverse_scores else 1.0e12
    max_streak = max(
        (float(row["max_upright_streak_seconds"]) for row in score_rows),
        default=0.0,
    )
    max_cart = max(float(down["max_cart_abs"]), float(reverse["max_cart_abs"]))
    rail_penalty = 1000.0 * max(0.0, max_cart / max(1e-9, rail_penalty_limit) - 0.95) ** 2
    smoothness = float(np.mean(np.diff(np.asarray(actions, dtype=np.float64)) ** 2))
    score = (
        float(down_weight) * down_terminal_cost
        + float(terminal_down_weight) * down_terminal_cost
        + float(reverse_weight) * reverse_best_cost
        - 2500.0 * max_streak
        + rail_penalty
        + 0.05 * smoothness
    )
    return float(score), {
        "down_terminal_cost": float(down_terminal_cost),
        "down_angle_error_to_hanging": float(down_angle_error),
        "down_final": down["final"],
        "reverse_best_cost": float(reverse_best_cost),
        "reverse_best": best_reverse,
        "reverse_max_upright_streak_seconds": float(max_streak),
        "max_cart_abs": float(max_cart),
        "rail_limit": float(cfg["env"]["rail_limit"]),
        "reverse_actions": reversed_actions.astype(float).tolist(),
        "down_trace": down["trace_rows"] if return_trace else None,
        "reverse_trace": reverse["trace_rows"] if return_trace else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup11_uniform.yaml")
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--knot-count", type=int, default=32)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--handoff-min-time", type=float, default=8.0)
    parser.add_argument("--action-sigma", type=float, default=0.02)
    parser.add_argument("--sigma-decay", type=float, default=0.92)
    parser.add_argument("--sigma-floor", type=float, default=0.0005)
    parser.add_argument("--rail-penalty-limit", type=float, default=10.0)
    parser.add_argument("--down-weight", type=float, default=1.0)
    parser.add_argument("--reverse-weight", type=float, default=0.35)
    parser.add_argument("--terminal-down-weight", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=11015)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.knot_count < 2 or args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")
    if min(args.seconds, args.handoff_min_time, args.action_sigma, args.sigma_decay, args.sigma_floor) <= 0.0:
        raise ValueError("seconds and CEM scales must be positive")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"], "init_mode": "fixed_state"}
    env_probe = NLinkCartPoleEnv(
        state_config(
            cfg,
            qpos=[0.0] * (int(cfg["env"]["n_links"]) + 1),
            qvel=[0.0] * (int(cfg["env"]["n_links"]) + 1),
            seconds=args.seconds,
        ),
        progress=1.0,
        seed=0,
    )
    step_count = max(2, round(args.seconds / env_probe.dt))
    interpolation = interpolation_matrix(args.knot_count, step_count)
    env_probe.close()

    rng = np.random.default_rng(args.seed)
    center = np.zeros(args.knot_count, dtype=np.float64)
    sigma = np.full(args.knot_count, args.action_sigma, dtype=np.float64)
    best_record: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    for iteration in range(args.iterations):
        knots = np.clip(
            center[None, :]
            + rng.normal(0.0, sigma, size=(args.population, args.knot_count)),
            -1.0,
            1.0,
        )
        knots[0] = center
        records: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(knots):
            actions = np.clip(candidate @ interpolation.T, -1.0, 1.0)
            cost, metrics = evaluate_candidate(
                cfg,
                actions,
                seconds=args.seconds,
                handoff_min_time=args.handoff_min_time,
                rail_penalty_limit=args.rail_penalty_limit,
                down_weight=args.down_weight,
                reverse_weight=args.reverse_weight,
                terminal_down_weight=args.terminal_down_weight,
            )
            records.append(
                {
                    "index": int(candidate_index),
                    "cost": float(cost),
                    "knots": candidate.astype(float).tolist(),
                    "actions": actions.astype(float).tolist(),
                    "metrics": metrics,
                }
            )
        records.sort(key=lambda row: float(row["cost"]))
        elite = np.asarray(
            [row["knots"] for row in records[: args.elites]], dtype=np.float64
        )
        center = np.mean(elite, axis=0)
        sigma = np.maximum(np.std(elite, axis=0) * args.sigma_decay, args.sigma_floor)
        top = records[0]
        metrics = top["metrics"]
        history.append(
            {
                "iteration": int(iteration + 1),
                "cost": float(top["cost"]),
                "down_terminal_cost": float(metrics["down_terminal_cost"]),
                "down_angle_error_to_hanging": float(metrics["down_angle_error_to_hanging"]),
                "reverse_best_cost": float(metrics["reverse_best_cost"]),
                "reverse_best_angle": float(metrics["reverse_best"].get("max_abs_angle", np.nan)),
                "reverse_best_hinge_rms": float(metrics["reverse_best"].get("hinge_velocity_rms", np.nan)),
                "max_cart_abs": float(metrics["max_cart_abs"]),
            }
        )
        if best_record is None or float(top["cost"]) < float(best_record["cost"]):
            best_record = dict(top)
        print(
            f"iter={iteration + 1:03d} cost={top['cost']:.3f} "
            f"down={metrics['down_angle_error_to_hanging']:.3f}rad "
            f"rev_angle={metrics['reverse_best'].get('max_abs_angle', np.nan):.3f} "
            f"rev_hinge={metrics['reverse_best'].get('hinge_velocity_rms', np.nan):.3f} "
            f"rail={metrics['max_cart_abs']:.3f}",
            flush=True,
        )

    assert best_record is not None
    best_actions = np.asarray(best_record["actions"], dtype=np.float64)
    best_cost, best_metrics = evaluate_candidate(
        cfg,
        best_actions,
        seconds=args.seconds,
        handoff_min_time=args.handoff_min_time,
        rail_penalty_limit=args.rail_penalty_limit,
        down_weight=args.down_weight,
        reverse_weight=args.reverse_weight,
        terminal_down_weight=args.terminal_down_weight,
        return_trace=True,
    )
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "claim_status": "time_reversed_downfall_seed_diagnostic",
        "summary": "Exact serial co-optimization of an upright-to-hanging down-fall and its reversed hanging-start swing-up; development only.",
        "config_sha256": data_sha256(cfg),
        "config_path": str(Path(args.config)),
        "search": {
            "seconds": float(args.seconds),
            "knot_count": int(args.knot_count),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "handoff_min_time": float(args.handoff_min_time),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "rail_penalty_limit": float(args.rail_penalty_limit),
            "down_weight": float(args.down_weight),
            "reverse_weight": float(args.reverse_weight),
            "terminal_down_weight": float(args.terminal_down_weight),
            "seed": int(args.seed),
            "wall_time_seconds": float(time.time() - started),
        },
        "controller": {
            "type": "time_reversed_downfall_force_knots",
            "knots": np.asarray(best_record["knots"], dtype=np.float64).astype(float).tolist(),
            "controls": best_actions.astype(float).tolist(),
            "reversed_controls": best_actions[::-1].astype(float).tolist(),
            "horizon_steps": int(best_actions.size),
            "horizon_seconds": float(best_actions.size * env_probe.dt),
        },
        "best": {
            "cost": float(best_cost),
            "metrics": best_metrics,
        },
        "history": history,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(result, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
