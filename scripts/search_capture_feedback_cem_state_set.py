#!/usr/bin/env python
"""Search one reset-free nonlinear capture actor over a measured state set."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from search_capture_feedback_cem import capture_features, load_state, rollout


def load_states(path: str, state_index: int) -> list[dict[str, Any]]:
    if state_index != -1:
        return [load_state(path, state_index)]
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError(f"{path} does not contain a non-empty states list")
    return [dict(value) for value in values]


def score_state_set(
    cfg: dict[str, Any],
    states: list[dict[str, Any]],
    vector: np.ndarray,
    *,
    seconds: float,
    centered_weight: float,
    absolute_low_momentum_weight: float,
    max_cart_target: float,
    max_cart_weight: float,
    stable_time_weight: float,
    return_trace: bool = False,
) -> dict[str, Any]:
    metrics = [
        rollout(
            cfg,
            state=state,
            vector=vector,
            seconds=seconds,
            return_trace=return_trace,
            centered_weight=centered_weight,
            absolute_low_momentum_weight=absolute_low_momentum_weight,
            max_cart_target=max_cart_target,
            max_cart_weight=max_cart_weight,
            stable_time_weight=stable_time_weight,
        )
        for state in states
    ]
    scores = np.asarray([float(row["score"]) for row in metrics], dtype=np.float64)
    return {
        "score": float(0.75 * np.max(scores) + 0.25 * np.mean(scores)),
        "mean_score": float(np.mean(scores)),
        "worst_score": float(np.max(scores)),
        "success_count": int(sum(bool(row["success"]) for row in metrics)),
        "min_upright_streak_seconds": float(
            min(row["max_upright_streak_seconds"] for row in metrics)
        ),
        "min_centered_upright_streak_seconds": float(
            min(row["max_centered_upright_streak_seconds"] for row in metrics)
        ),
        "min_absolute_low_momentum_upright_streak_seconds": float(
            min(row["max_absolute_low_momentum_upright_streak_seconds"] for row in metrics)
        ),
        "max_cart_abs": float(max(row["max_cart_abs"] for row in metrics)),
        "states": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CEM search for one capture actor across measured handoff states"
    )
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument("--state-json", action="append", required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--centered-weight", type=float, default=15000.0)
    parser.add_argument("--absolute-low-momentum-weight", type=float, default=15000.0)
    parser.add_argument("--max-cart-target", type=float, default=1.25)
    parser.add_argument("--max-cart-weight", type=float, default=250.0)
    parser.add_argument("--stable-time-weight", type=float, default=0.0)
    parser.add_argument("--seed-actor", default=None)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")

    cfg = apply_overrides(load_config(args.config), args.override)
    states = [state for path in args.state_json for state in load_states(path, args.state_index)]
    probe_cfg = {
        **cfg,
        "env": {
            **cfg["env"],
            "init_mode": "fixed_state",
            "init_qpos": states[0]["qpos"],
            "init_qvel": states[0]["qvel"],
            "obs_include_capture_features": False,
            "action_lqr_residual": {"enabled": False},
            "action_lqr_switch": {"enabled": False},
        },
    }
    probe = NLinkCartPoleEnv(probe_cfg, progress=0.0, seed=0)
    feature_dim = int(capture_features(probe).size)
    probe.close()

    rng = np.random.default_rng(args.seed)
    center = np.zeros(feature_dim + 1, dtype=np.float64)
    if args.seed_actor:
        seed_payload = json.loads(Path(args.seed_actor).read_text(encoding="utf-8"))
        seed_vector = np.asarray(seed_payload["best_vector"], dtype=np.float64)
        if seed_vector.shape != center.shape:
            raise ValueError(f"seed actor has vector shape {seed_vector.shape}, expected {center.shape}")
        center = seed_vector.copy()
    sigma = np.full(center.shape, float(args.sigma), dtype=np.float64)
    sigma[-1] = 0.15
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()

    for iteration in range(args.iterations + 1):
        candidates = [center.copy()]
        if iteration > 0:
            candidates.extend(
                center + rng.normal(0.0, sigma, size=center.shape)
                for _ in range(args.population - 1)
            )
        records = []
        for vector in candidates:
            metrics = score_state_set(
                cfg,
                states,
                np.asarray(vector, dtype=np.float64),
                seconds=args.seconds,
                centered_weight=args.centered_weight,
                absolute_low_momentum_weight=args.absolute_low_momentum_weight,
                max_cart_target=args.max_cart_target,
                max_cart_weight=args.max_cart_weight,
                stable_time_weight=args.stable_time_weight,
            )
            records.append({"vector": np.asarray(vector), "metrics": metrics})
        records.sort(key=lambda row: float(row["metrics"]["score"]))
        top = records[0]
        if best is None or float(top["metrics"]["score"]) < float(best["score"]):
            best = {
                "score": float(top["metrics"]["score"]),
                "vector": top["vector"].astype(float).tolist(),
                "metrics": top["metrics"],
            }
        elite = np.asarray([row["vector"] for row in records[: args.elites]], dtype=np.float64)
        center = np.mean(elite, axis=0)
        sigma = np.maximum(np.std(elite, axis=0) * args.sigma_decay, args.sigma_floor)
        tm = top["metrics"]
        history.append(
            {
                "iteration": int(iteration),
                "score": float(tm["score"]),
                "best_score": float(best["score"]),
                "success_count": int(tm["success_count"]),
                "min_upright_streak_seconds": float(tm["min_upright_streak_seconds"]),
                "min_centered_upright_streak_seconds": float(
                    tm["min_centered_upright_streak_seconds"]
                ),
                "min_absolute_low_momentum_upright_streak_seconds": float(
                    tm["min_absolute_low_momentum_upright_streak_seconds"]
                ),
            }
        )
        print(
            f"iter={iteration:03d} score={tm['score']:.2f} "
            f"successes={tm['success_count']}/{len(states)} "
            f"min_streak={tm['min_upright_streak_seconds']:.3f}s "
            f"min_centered={tm['min_centered_upright_streak_seconds']:.3f}s "
            f"min_absolute_low={tm['min_absolute_low_momentum_upright_streak_seconds']:.3f}s",
            flush=True,
        )

    assert best is not None
    best_vector = np.asarray(best["vector"], dtype=np.float64)
    final = score_state_set(
        cfg,
        states,
        best_vector,
        seconds=args.seconds,
        centered_weight=args.centered_weight,
        absolute_low_momentum_weight=args.absolute_low_momentum_weight,
        max_cart_target=args.max_cart_target,
        max_cart_weight=args.max_cart_weight,
        stable_time_weight=args.stable_time_weight,
        return_trace=False,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "One reset-free nonlinear capture actor optimized over measured handoff states; component diagnostic only.",
        "config_path": str(Path(args.config)),
        "state_jsons": [str(Path(path)) for path in args.state_json],
        "state_index": int(args.state_index),
        "seconds": float(args.seconds),
        "feature_names": (
            ["cart_position"]
            + [f"sin_absolute_angle_{i}" for i in range(int(cfg["env"]["n_links"]))]
            + [f"cos_minus_one_absolute_angle_{i}" for i in range(int(cfg["env"]["n_links"]))]
            + ["cart_velocity"]
            + [
                f"absolute_angular_velocity_{i}"
                for i in range(int(cfg["env"]["n_links"]))
            ]
        ),
        "best_vector": best_vector.astype(float).tolist(),
        "best_actor": {
            "weight": best_vector[:-1].astype(float).tolist(),
            "bias": float(best_vector[-1]),
        },
        "best_search": best,
        "final_eval": final,
        "search": {
            "seed": int(args.seed),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "sigma": float(args.sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "centered_weight": float(args.centered_weight),
            "absolute_low_momentum_weight": float(args.absolute_low_momentum_weight),
            "max_cart_target": float(args.max_cart_target),
            "max_cart_weight": float(args.max_cart_weight),
            "stable_time_weight": float(args.stable_time_weight),
            "seed_actor": args.seed_actor,
            "wall_time_seconds": float(time.time() - started),
        },
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
