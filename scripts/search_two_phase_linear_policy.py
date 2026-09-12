#!/usr/bin/env python
"""Search a two-expert linear/tanh chain on an exact MuJoCo plant.

Expert one is held fixed while a second state-feedback actor is optimized from
its actual post-switch states. This is a discovery probe for the proposed
swing-up plus capture/stabilize architecture, not final evidence.
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


def load_actor(path: str, obs_dim: int) -> tuple[np.ndarray, float]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    actor = payload.get("best_actor", payload.get("capture_actor", {}))
    weight = np.asarray(actor.get("weight", []), dtype=np.float64)
    if weight.shape != (obs_dim,):
        raise ValueError(f"{path} actor has shape {weight.shape}; expected {(obs_dim,)}")
    return weight, float(actor.get("bias", 0.0))


def rollout(
    cfg: dict[str, Any],
    *,
    seed: int,
    swing_weight: np.ndarray,
    swing_bias: float,
    capture_weight: np.ndarray,
    capture_bias: float,
    switch_time: float,
    seconds: float,
) -> dict[str, Any]:
    local_cfg = {**cfg, "env": {**cfg["env"], "episode_seconds": float(seconds)}}
    env = NLinkCartPoleEnv(local_cfg, progress=1.0, seed=seed)
    obs, _ = env.reset(seed=seed)
    terminated = False
    truncated = False
    ep_return = 0.0
    action_sq = 0.0
    action_smooth = 0.0
    prev_action = 0.0
    max_cart = 0.0
    max_capture_quality = 0.0
    best_handoff_score = float("inf")
    best_handoff: dict[str, float] = {}
    final_info: dict[str, Any] = {}
    switch_step = int(round(switch_time / env.dt))
    step = 0
    while not (terminated or truncated):
        actor_weight, actor_bias = (
            (swing_weight, swing_bias) if step < switch_step else (capture_weight, capture_bias)
        )
        action = float(np.tanh(float(obs @ actor_weight + actor_bias)))
        obs, reward, terminated, truncated, info = env.step([action])
        ep_return += float(reward)
        action_sq += action * action
        action_smooth += (action - prev_action) ** 2
        prev_action = action
        angle = float(info.get("max_abs_angle", np.inf))
        hinge = float(info.get("hinge_velocity_rms", np.inf))
        cart = abs(float(info.get("x", 0.0)))
        cart_velocity = abs(float(env.data.qvel[0]))
        handoff_score = (
            30.0 * (angle / 0.15) ** 2
            + 10.0 * (hinge / 0.75) ** 2
            + 3.0 * (cart / 1.25) ** 2
            + 3.0 * (cart_velocity / 0.50) ** 2
        )
        if handoff_score < best_handoff_score:
            best_handoff_score = handoff_score
            best_handoff = {
                "time_seconds": float((step + 1) * env.dt),
                "max_abs_angle": angle,
                "hinge_velocity_rms": hinge,
                "cart_abs": cart,
                "cart_velocity_abs": cart_velocity,
            }
        max_cart = max(max_cart, cart)
        max_capture_quality = max(max_capture_quality, float(info.get("capture_quality", 0.0)))
        final_info = dict(info)
        step += 1
    success = bool(final_info.get("success", False))
    ever_upright = final_info.get("time_to_first_upright") is not None
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    max_low_streak = float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0))
    rail_over = max(0.0, max_cart - float(local_cfg["env"]["rail_limit"]))
    score = (
        -30000.0 * float(success)
        -7000.0 * max_streak
        -6500.0 * max_low_streak
        -1500.0 * float(ever_upright)
        -1200.0 * max_capture_quality
        +5.0 * best_handoff_score
        +100.0 * float(terminated and not success)
        +140.0 * rail_over * rail_over
        +0.01 * action_sq
        +0.04 * action_smooth
    )
    result = {
        "score": float(score),
        "return": float(ep_return),
        "success": success,
        "ever_upright": bool(ever_upright),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "max_upright_streak_seconds": max_streak,
        "max_low_momentum_upright_streak_seconds": max_low_streak,
        "best_handoff_score": float(best_handoff_score),
        "best_handoff": best_handoff,
        "max_cart_abs": float(max_cart),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "final_info": final_info,
    }
    env.close()
    return result


def evaluate(
    cfg: dict[str, Any],
    *,
    seed: int,
    episodes: int,
    swing_weight: np.ndarray,
    swing_bias: float,
    capture_weight: np.ndarray,
    capture_bias: float,
    switch_time: float,
    seconds: float,
) -> dict[str, Any]:
    rows = [
        rollout(
            cfg,
            seed=seed + index,
            swing_weight=swing_weight,
            swing_bias=swing_bias,
            capture_weight=capture_weight,
            capture_bias=capture_bias,
            switch_time=switch_time,
            seconds=seconds,
        )
        for index in range(episodes)
    ]
    return {
        "score": float(np.mean([row["score"] for row in rows])),
        "return_mean": float(np.mean([row["return"] for row in rows])),
        "success_rate": float(np.mean([float(row["success"]) for row in rows])),
        "ever_upright_rate": float(np.mean([float(row["ever_upright"]) for row in rows])),
        "max_upright_streak_mean": float(np.mean([row["max_upright_streak_seconds"] for row in rows])),
        "max_upright_streak_max": float(np.max([row["max_upright_streak_seconds"] for row in rows])),
        "max_low_momentum_upright_streak_mean": float(
            np.mean([row["max_low_momentum_upright_streak_seconds"] for row in rows])
        ),
        "best_handoff_score_min": float(np.min([row["best_handoff_score"] for row in rows])),
        "best_handoff_hinge_velocity_rms_min": float(
            np.min([row["best_handoff"].get("hinge_velocity_rms", np.inf) for row in rows])
        ),
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--swing-json", required=True)
    parser.add_argument("--capture-init-json", required=True)
    parser.add_argument("--switch-time", type=float, default=9.8)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=20260936)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.002)
    parser.add_argument("--capture-sigma", type=float, default=0.25)
    parser.add_argument("--capture-bias-sigma", type=float, default=0.20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")
    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"], "episode_seconds": float(args.seconds)}
    probe = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    obs_dim = int(probe.observation_space.shape[0])
    probe.close()
    swing_weight, swing_bias = load_actor(args.swing_json, obs_dim)
    capture_weight, capture_bias = load_actor(args.capture_init_json, obs_dim)
    center = np.r_[capture_weight, capture_bias]
    sigma = np.full(obs_dim + 1, float(args.capture_sigma), dtype=np.float64)
    sigma[-1] = float(args.capture_bias_sigma)
    rng = np.random.default_rng(args.seed)
    best: dict[str, Any] | None = None
    best_vector = center.copy()
    history: list[dict[str, Any]] = []
    started = time.time()

    for iteration in range(args.iterations + 1):
        candidates = [center.copy()]
        if iteration > 0:
            candidates = [best_vector.copy()]
            candidates.extend(
                center + rng.normal(0.0, sigma, size=center.shape)
                for _ in range(args.population - 1)
            )
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            metrics = evaluate(
                cfg,
                seed=args.seed,
                episodes=args.episodes,
                swing_weight=swing_weight,
                swing_bias=swing_bias,
                capture_weight=np.asarray(candidate[:-1]),
                capture_bias=float(candidate[-1]),
                switch_time=args.switch_time,
                seconds=args.seconds,
            )
            records.append({"vector": np.asarray(candidate), "metrics": metrics})
        records.sort(key=lambda row: float(row["metrics"]["score"]))
        top = records[0]
        if best is None or top["metrics"]["score"] < best["score"]:
            best = {
                "score": float(top["metrics"]["score"]),
                "metrics": top["metrics"],
                "vector": top["vector"].astype(float).tolist(),
            }
            best_vector = top["vector"].copy()
        m = top["metrics"]
        history.append(
            {
                "iteration": iteration,
                "score": float(m["score"]),
                "success_rate": float(m["success_rate"]),
                "ever_upright_rate": float(m["ever_upright_rate"]),
                "max_upright_streak_max": float(m["max_upright_streak_max"]),
                "best_handoff_score_min": float(m["best_handoff_score_min"]),
                "best_handoff_hinge_velocity_rms_min": float(m["best_handoff_hinge_velocity_rms_min"]),
            }
        )
        print(
            f"iter={iteration:03d} score={m['score']:.3f} succ={m['success_rate']:.2f} "
            f"ever={m['ever_upright_rate']:.2f} streak={m['max_upright_streak_max']:.3f}s "
            f"handoff={m['best_handoff_score_min']:.2f} hhinge={m['best_handoff_hinge_velocity_rms_min']:.3f}",
            flush=True,
        )
        if iteration > 0:
            elite = np.asarray([row["vector"] for row in records[: args.elites]])
            center = best_vector.copy()
            sigma = np.maximum(elite.std(axis=0), args.sigma_floor) * args.sigma_decay

    assert best is not None
    out = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Two-expert linear/tanh chain search with fixed swing actor and optimized capture actor.",
        "swing_json": str(args.swing_json),
        "capture_init_json": str(args.capture_init_json),
        "switch_time": float(args.switch_time),
        "seconds": float(args.seconds),
        "obs_dim": int(obs_dim),
        "search": {
            "seed": int(args.seed),
            "episodes": int(args.episodes),
            "eval_episodes": int(args.eval_episodes),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "wall_time_seconds": float(time.time() - started),
        },
        "best": best,
        "capture_actor": {
            "weight": best_vector[:-1].astype(float).tolist(),
            "bias": float(best_vector[-1]),
        },
        "eval": evaluate(
            cfg,
            seed=args.seed + 4444,
            episodes=args.eval_episodes,
            swing_weight=swing_weight,
            swing_bias=swing_bias,
            capture_weight=best_vector[:-1],
            capture_bias=float(best_vector[-1]),
            switch_time=args.switch_time,
            seconds=args.seconds,
        ),
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(out, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
