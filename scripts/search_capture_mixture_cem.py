#!/usr/bin/env python
"""Search a two-mode reset-free capture controller over measured handoffs.

The low-momentum mode maintains the upright equilibrium. The high-momentum
mode brakes incoming cart/link motion. A smooth gate based on measured
whole-chain momentum selects between them, so the search can test a
capture-then-stabilize decomposition without a phase reset.
"""

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


def load_seed_vector(path: str, expected_size: int) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    vector = payload.get("best_vector")
    if vector is None:
        actor = payload.get("best_actor", {})
        vector = list(actor.get("weight", [])) + [float(actor.get("bias", 0.0))]
    result = np.asarray(vector, dtype=np.float64)
    if result.shape != (expected_size,):
        raise ValueError(f"{path} has vector shape {result.shape}, expected {(expected_size,)}")
    return result


def decode_actor(vector: np.ndarray, feature_dim: int) -> tuple[np.ndarray, float, np.ndarray, float, float, float]:
    block = feature_dim + 1
    expected = 2 * block + 2
    if vector.shape != (expected,):
        raise ValueError(f"expected mixture vector {(expected,)}, got {vector.shape}")
    low_weight = vector[:feature_dim]
    low_bias = float(vector[feature_dim])
    high_start = block
    high_weight = vector[high_start : high_start + feature_dim]
    high_bias = float(vector[high_start + feature_dim])
    threshold = float(0.20 + 0.20 * np.tanh(vector[2 * block]))
    gain = float(np.clip(np.exp(np.clip(vector[2 * block + 1], -2.0, 3.4)), 1.0, 30.0))
    return low_weight, low_bias, high_weight, high_bias, threshold, gain


def mixture_action(env: NLinkCartPoleEnv, vector: np.ndarray) -> float:
    features = capture_features(env)
    low_weight, low_bias, high_weight, high_bias, threshold, gain = decode_actor(
        vector, int(features.size)
    )
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    absolute_omega = np.cumsum(qvel[1 : 1 + env.n])
    momentum = float(
        np.sqrt(np.mean((absolute_omega / 0.75) ** 2)) + abs(float(qvel[0])) / 0.50
    )
    gate = float(1.0 / (1.0 + np.exp(-gain * (momentum - threshold))))
    low_action = float(np.tanh(features @ low_weight + low_bias))
    high_action = float(np.tanh(features @ high_weight + high_bias))
    return float((1.0 - gate) * low_action + gate * high_action)


def score_states(
    cfg: dict[str, Any],
    states: list[dict[str, Any]],
    vector: np.ndarray,
    *,
    progress: float,
    seconds: float,
    centered_weight: float,
    absolute_low_momentum_weight: float,
    max_cart_target: float,
    max_cart_weight: float,
) -> dict[str, Any]:
    metrics = [
        rollout(
            cfg,
            state=state,
            vector=None,
            progress=progress,
            seconds=seconds,
            centered_weight=centered_weight,
            absolute_low_momentum_weight=absolute_low_momentum_weight,
            max_cart_target=max_cart_target,
            max_cart_weight=max_cart_weight,
            action_fn=lambda env, actor=vector: mixture_action(env, actor),
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
        description="CEM search for a smooth capture-then-stabilize actor over measured states"
    )
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument("--state-json", action="append", required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--centered-weight", type=float, default=15000.0)
    parser.add_argument("--absolute-low-momentum-weight", type=float, default=15000.0)
    parser.add_argument("--max-cart-target", type=float, default=1.25)
    parser.add_argument("--max-cart-weight", type=float, default=250.0)
    parser.add_argument("--seed-low-actor", default=None)
    parser.add_argument("--seed-high-actor", default=None)
    parser.add_argument("--seed", type=int, default=20260928)
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
    probe = NLinkCartPoleEnv(probe_cfg, progress=args.progress, seed=0)
    feature_dim = int(capture_features(probe).size)
    probe.close()
    block = feature_dim + 1
    vector_size = 2 * block + 2

    center = np.zeros(vector_size, dtype=np.float64)
    center[2 * block + 1] = np.log(5.0)
    if args.seed_low_actor:
        center[:block] = load_seed_vector(args.seed_low_actor, block)
    if args.seed_high_actor:
        center[block : 2 * block] = load_seed_vector(args.seed_high_actor, block)
    rng = np.random.default_rng(args.seed)
    sigma = np.full(vector_size, float(args.sigma), dtype=np.float64)
    sigma[2 * block :] = 0.20
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
            metrics = score_states(
                cfg,
                states,
                np.asarray(vector, dtype=np.float64),
                progress=args.progress,
                seconds=args.seconds,
                centered_weight=args.centered_weight,
                absolute_low_momentum_weight=args.absolute_low_momentum_weight,
                max_cart_target=args.max_cart_target,
                max_cart_weight=args.max_cart_weight,
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
            f"min_centered={tm['min_centered_upright_streak_seconds']:.3f}s "
            f"min_absolute_low={tm['min_absolute_low_momentum_upright_streak_seconds']:.3f}s",
            flush=True,
        )

    assert best is not None
    best_vector = np.asarray(best["vector"], dtype=np.float64)
    final = score_states(
        cfg,
        states,
        best_vector,
        progress=args.progress,
        seconds=args.seconds,
        centered_weight=args.centered_weight,
        absolute_low_momentum_weight=args.absolute_low_momentum_weight,
        max_cart_target=args.max_cart_target,
        max_cart_weight=args.max_cart_weight,
    )
    low_weight, low_bias, high_weight, high_bias, threshold, gain = decode_actor(
        best_vector, feature_dim
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Two-mode reset-free nonlinear capture-then-stabilize CEM; component diagnostic only.",
        "config_path": str(Path(args.config)),
        "state_jsons": [str(Path(path)) for path in args.state_json],
        "state_index": int(args.state_index),
        "seconds": float(args.seconds),
        "feature_names": (
            ["cart_position"]
            + [f"sin_absolute_angle_{i}" for i in range(int(cfg["env"]["n_links"]))]
            + [f"cos_minus_one_absolute_angle_{i}" for i in range(int(cfg["env"]["n_links"]))]
            + ["cart_velocity"]
            + [f"absolute_angular_velocity_{i}" for i in range(int(cfg["env"]["n_links"]))]
        ),
        "gate": {
            "momentum": "sqrt(mean((absolute_angular_velocity/0.75)^2)) + abs(cart_velocity)/0.50",
            "threshold": threshold,
            "gain": gain,
        },
        "best_vector": best_vector.astype(float).tolist(),
        "low_momentum_actor": {"weight": low_weight.astype(float).tolist(), "bias": low_bias},
        "high_momentum_actor": {"weight": high_weight.astype(float).tolist(), "bias": high_bias},
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
            "seed_low_actor": args.seed_low_actor,
            "seed_high_actor": args.seed_high_actor,
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
