#!/usr/bin/env python
"""Search a shared nonlinear-squashed full-state capture feedback law.

This is a bounded diagnostic for the six-link capture envelope.  It searches
one stationary feedback law across several exact measured states, rather than
optimizing an open-loop action sequence for one state.  The result is useful
only if exact MuJoCo replay shows a shared basin; it is never benchmark
evidence by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp

try:
    from scripts.search_capture_sequence import fixed_state_cfg
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg


def parse_indices(value: str, count: int) -> list[int]:
    if value.strip().lower() == "first":
        return list(range(min(count, 8)))
    indices = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not indices:
        raise ValueError("--indices must contain at least one state index")
    if any(index < 0 or index >= count for index in indices):
        raise IndexError(f"state indices must be in [0, {count})")
    return indices


def features(env: NLinkCartPoleEnv, *, feature_set: str) -> np.ndarray:
    qpos = np.asarray(env.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    relative_angles = qpos[1 : 1 + env.n]
    relative_rates = qvel[1 : 1 + env.n]
    absolute_angles = serial_absolute_angles(relative_angles)
    absolute_rates = np.cumsum(relative_rates)
    if feature_set == "scaled_raw":
        return np.concatenate(
            [
                np.asarray([1.0, qpos[0] / 1.25, qvel[0] / 0.50]),
                absolute_angles / 0.15,
                absolute_rates / 3.0,
                relative_angles / 0.30,
                relative_rates / 0.75,
            ]
        )
    if feature_set == "trigonometric":
        return np.concatenate(
            [
                np.asarray([1.0, qpos[0] / 1.25, qvel[0] / 0.50]),
                np.sin(absolute_angles),
                np.cos(absolute_angles) - 1.0,
                np.sin(relative_angles),
                np.cos(relative_angles) - 1.0,
                absolute_rates / 0.75,
                relative_rates / 1.0,
            ]
        )
    raise ValueError("feature_set must be 'scaled_raw' or 'trigonometric'")


def evaluate(
    weights: np.ndarray,
    envs: list[NLinkCartPoleEnv],
    indices: list[int],
    *,
    seed: int,
    seconds: float,
    feature_set: str,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for slot, env in enumerate(envs):
        env.reset(seed=seed + indices[slot])
        stage_tail: list[float] = []
        max_cart = abs(float(env.data.qpos[0]))
        final_info: dict[str, Any] = {}
        max_steps = min(env.max_steps, int(round(seconds / env.dt)))
        for step in range(max_steps):
            action = float(np.tanh(float(weights @ features(env, feature_set=feature_set))))
            _, _, terminated, truncated, info = env.step([action])
            qpos = np.asarray(env.data.qpos, dtype=np.float64)
            qvel = np.asarray(env.data.qvel, dtype=np.float64)
            absolute_angles = serial_absolute_angles(qpos[1 : 1 + env.n])
            absolute_rates = np.cumsum(qvel[1 : 1 + env.n])
            stage = (
                12.0 * (np.max(np.abs(absolute_angles)) / 0.15) ** 2
                + (np.sqrt(np.mean(qvel[1 : 1 + env.n] ** 2)) / 0.75) ** 2
                + 3.0 * (np.sqrt(np.mean(absolute_rates**2)) / 3.0) ** 2
                + 2.0 * (qpos[0] / 1.25) ** 2
                + (qvel[0] / 0.50) ** 2
            )
            if step >= max(0, max_steps - 125):
                stage_tail.append(float(stage))
            max_cart = max(max_cart, abs(float(info["x"])))
            final_info = dict(info)
            if terminated or truncated:
                break
        rail_hit = final_info.get("termination_reason") == "rail_violation"
        rows.append(
            {
                "state_index": int(indices[slot]),
                "success": bool(final_info.get("success", False)),
                "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
                "max_cart_excursion": float(max_cart),
                "rail_hit": bool(rail_hit),
                "termination_reason": final_info.get("termination_reason"),
                "tail_stage_cost": float(np.mean(stage_tail) if stage_tail else 1.0e9),
            }
        )
    costs = np.asarray([row["tail_stage_cost"] for row in rows], dtype=np.float64)
    rail_count = sum(bool(row["rail_hit"]) for row in rows)
    hold = np.asarray([row["max_upright_streak_seconds"] for row in rows], dtype=np.float64)
    score = float(
        np.mean(costs)
        + 0.4 * np.percentile(costs, 75)
        + 10_000.0 * rail_count
        - 15.0 * np.mean(np.minimum(hold, 15.0))
    )
    return score, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/test.json")
    parser.add_argument("--indices", default="first")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--elites", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--feature-set", choices=["scaled_raw", "trigonometric"], default="trigonometric")
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.population < 2 or not 1 <= args.elites <= args.population:
        raise ValueError("population and elites are inconsistent")
    if args.iterations < 1 or args.seconds <= 0.0:
        raise ValueError("iterations and seconds must be positive")

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    states = dataset.get("states", dataset) if isinstance(dataset, dict) else dataset
    if not isinstance(states, list) or not states:
        raise ValueError("dataset must contain a non-empty states list")
    indices = parse_indices(args.indices, len(states))
    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"], "action_lqr_residual": {"enabled": False}, "action_lqr_switch": {"enabled": False}}
    envs = [
        NLinkCartPoleEnv(
            fixed_state_cfg(cfg, states[index], args.seconds),
            progress=args.progress,
            seed=args.seed + index,
        )
        for index in indices
    ]
    dimension = int(features(envs[0], feature_set=args.feature_set).size)
    rng = np.random.default_rng(args.seed)
    center = np.zeros(dimension, dtype=np.float64)
    sigma = np.full(dimension, args.sigma, dtype=np.float64)
    best_score = float("inf")
    best_weights = center.copy()
    best_rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
    for iteration in range(args.iterations + 1):
        candidates = np.empty((args.population, dimension), dtype=np.float64)
        candidates[0] = center
        candidates[1:] = center + rng.normal(0.0, sigma, size=(args.population - 1, dimension))
        records: list[tuple[float, np.ndarray, list[dict[str, Any]]]] = []
        for candidate in candidates:
            score, rows = evaluate(
                candidate,
                envs,
                indices,
                seed=args.seed,
                seconds=args.seconds,
                feature_set=args.feature_set,
            )
            records.append((score, candidate.copy(), rows))
        records.sort(key=lambda item: item[0])
        if records[0][0] < best_score:
            best_score, best_weights, best_rows = records[0]
        elite = np.asarray([row[1] for row in records[: args.elites]], dtype=np.float64)
        center = elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0) * args.sigma_decay, args.sigma_floor)
        history.append(
            {
                "iteration": int(iteration),
                "iteration_score": float(records[0][0]),
                "best_score": float(best_score),
                "successes": int(sum(row["success"] for row in best_rows)),
                "rail_hits": int(sum(row["rail_hit"] for row in best_rows)),
                "mean_hold_seconds": float(np.mean([row["max_upright_streak_seconds"] for row in best_rows])),
                "sigma_mean": float(np.mean(sigma)),
            }
        )
    for env in envs:
        env.close()
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "shared_linear_feedback_diagnostic",
        "not_solution": True,
        "summary": "Shared exact-MuJoCo full-state tanh feedback search on a small P1 endpoint slice.",
        "config": file_metadata(Path(args.config)),
        "config_sha256": data_sha256(cfg),
        "dataset": file_metadata(dataset_path),
        "progress": float(args.progress),
        "feature_set": args.feature_set,
        "indices": indices,
        "search": {
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "seconds": float(args.seconds),
            "sigma": float(args.sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "seed": int(args.seed),
        },
        "best_score": float(best_score),
        "feature_order": (
            ["bias", "cart_position/1.25", "cart_velocity/0.50",
             "absolute_angles/0.15", "absolute_rates/3.0",
             "relative_angles/0.30", "relative_rates/0.75"]
            if args.feature_set == "scaled_raw"
            else ["bias", "cart_position/1.25", "cart_velocity/0.50",
                  "sin_absolute_angles", "cos_absolute_angles_minus_one",
                  "sin_relative_angles", "cos_relative_angles_minus_one",
                  "absolute_rates/0.75", "relative_rates/1.0"]
        ),
        "weights": best_weights.astype(float).tolist(),
        "best_rows": best_rows,
        "history": history,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, args.out)
    print(
        f"best_score={best_score:.3f} successes={sum(row['success'] for row in best_rows)}/{len(best_rows)} "
        f"rails={sum(row['rail_hit'] for row in best_rows)}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
