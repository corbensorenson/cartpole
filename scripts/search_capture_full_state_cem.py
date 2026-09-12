#!/usr/bin/env python
"""Search a full-state nonlinear capture actor from a real handoff."""

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
from search_capture_feedback_cem import load_state, rollout


def full_state_features(env: NLinkCartPoleEnv) -> np.ndarray:
    qpos = np.asarray(env.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    relative_angles = wrap_angle(qpos[1 : 1 + env.n])
    absolute_angles = serial_absolute_angles(qpos[1 : 1 + env.n])
    relative_rates = qvel[1 : 1 + env.n]
    absolute_rates = np.cumsum(relative_rates)
    return np.r_[
        qpos[0] / 1.25,
        np.sin(absolute_angles),
        np.cos(absolute_angles) - 1.0,
        np.sin(relative_angles),
        np.cos(relative_angles) - 1.0,
        qvel[0] / 0.50,
        absolute_rates / 0.75,
        relative_rates / 1.0,
    ].astype(np.float64)


def full_state_action(env: NLinkCartPoleEnv, vector: np.ndarray) -> float:
    features = full_state_features(env)
    expected = features.size + 1
    if vector.shape != (expected,):
        raise ValueError(f"expected full-state actor vector {(expected,)}, got {vector.shape}")
    return float(np.tanh(features @ vector[:-1] + vector[-1]))


def score_state(
    cfg: dict[str, Any],
    state: dict[str, Any],
    vector: np.ndarray,
    *,
    progress: float,
    seconds: float,
    centered_weight: float,
    absolute_low_momentum_weight: float,
    max_cart_target: float,
    max_cart_weight: float,
    stable_time_weight: float,
) -> dict[str, Any]:
    return rollout(
        cfg,
        state=state,
        vector=None,
        progress=progress,
        seconds=seconds,
        centered_weight=centered_weight,
        absolute_low_momentum_weight=absolute_low_momentum_weight,
        max_cart_target=max_cart_target,
        max_cart_weight=max_cart_weight,
        stable_time_weight=stable_time_weight,
        action_fn=lambda env, actor=vector: full_state_action(env, actor),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="CEM search for full relative-plus-absolute capture feedback")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.005)
    parser.add_argument("--centered-weight", type=float, default=5000.0)
    parser.add_argument("--absolute-low-momentum-weight", type=float, default=5000.0)
    parser.add_argument("--max-cart-target", type=float, default=1.25)
    parser.add_argument("--max-cart-weight", type=float, default=200.0)
    parser.add_argument("--stable-time-weight", type=float, default=10000.0)
    parser.add_argument("--seed", type=int, default=20780)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")
    if min(args.seconds, args.iterations, args.sigma, args.sigma_floor) <= 0.0:
        raise ValueError("duration, iterations, and sigma values must be positive")

    cfg = apply_overrides(load_config(args.config), args.override)
    state = load_state(args.state_json, args.state_index)
    probe_cfg = {
        **cfg,
        "env": {
            **cfg["env"],
            "init_mode": "fixed_state",
            "init_qpos": state["qpos"],
            "init_qvel": state["qvel"],
            "obs_include_capture_features": False,
            "action_lqr_residual": {"enabled": False},
            "action_lqr_switch": {"enabled": False},
        },
    }
    probe = NLinkCartPoleEnv(probe_cfg, progress=args.progress, seed=0)
    feature_dim = int(full_state_features(probe).size)
    probe.close()
    center = np.zeros(feature_dim + 1, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    sigma = np.full(feature_dim + 1, float(args.sigma), dtype=np.float64)
    sigma[-1] = min(0.15, float(args.sigma))
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
            metrics = score_state(
                cfg,
                state,
                np.asarray(vector, dtype=np.float64),
                progress=args.progress,
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
                "max_upright_streak_seconds": float(tm["max_upright_streak_seconds"]),
                "max_centered_upright_streak_seconds": float(tm["max_centered_upright_streak_seconds"]),
                "max_absolute_low_momentum_upright_streak_seconds": float(
                    tm["max_absolute_low_momentum_upright_streak_seconds"]
                ),
                "max_cart_abs": float(tm["max_cart_abs"]),
            }
        )
        print(
            f"iter={iteration:03d} score={tm['score']:.2f} "
            f"hold={tm['max_upright_streak_seconds']:.3f}s "
            f"centered={tm['max_centered_upright_streak_seconds']:.3f}s "
            f"low={tm['max_absolute_low_momentum_upright_streak_seconds']:.3f}s",
            flush=True,
        )

    assert best is not None
    best_vector = np.asarray(best["vector"], dtype=np.float64)
    final = score_state(
        cfg,
        state,
        best_vector,
        progress=args.progress,
        seconds=args.seconds,
        centered_weight=args.centered_weight,
        absolute_low_momentum_weight=args.absolute_low_momentum_weight,
        max_cart_target=args.max_cart_target,
        max_cart_weight=args.max_cart_weight,
        stable_time_weight=args.stable_time_weight,
    )
    feature_names = (
        ["cart_position"]
        + [f"sin_absolute_angle_{i}" for i in range(int(cfg["env"]["n_links"]))]
        + [f"cos_minus_one_absolute_angle_{i}" for i in range(int(cfg["env"]["n_links"]))]
        + [f"sin_relative_angle_{i}" for i in range(int(cfg["env"]["n_links"]))]
        + [f"cos_minus_one_relative_angle_{i}" for i in range(int(cfg["env"]["n_links"]))]
        + ["cart_velocity"]
        + [f"absolute_angular_velocity_{i}" for i in range(int(cfg["env"]["n_links"]))]
        + [f"relative_angular_velocity_{i}" for i in range(int(cfg["env"]["n_links"]))]
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Full relative-plus-absolute nonlinear capture CEM from an exact seven-link handoff; proposal only.",
        "config_path": str(Path(args.config)),
        "state_json": str(Path(args.state_json)),
        "state_index": int(args.state_index),
        "seconds": float(args.seconds),
        "feature_names": feature_names,
        "best_vector": best_vector.astype(float).tolist(),
        "best_actor": {"weight": best_vector[:-1].astype(float).tolist(), "bias": float(best_vector[-1])},
        "best_search": best,
        "eval": final,
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
            "wall_time_seconds": float(time.time() - started),
        },
        "handoff_state": state,
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
