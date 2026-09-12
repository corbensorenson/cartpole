#!/usr/bin/env python
"""CPU-only CEM search for a closed-loop linear tanh swing controller.

This is a discovery branch that uses the environment's state features as
feedback rather than a fixed action waveform.  It intentionally saves a
portable JSON policy proposal instead of pretending to be final evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config, save_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp


def time_feature_count(cfg: dict[str, Any]) -> int:
    env_cfg = cfg.get("env", {})
    if not bool(env_cfg.get("obs_include_time", False)):
        return 0
    return 1 + 2 * len(list(env_cfg.get("obs_time_frequencies", [])))


def build_sigma(
    *,
    n_links: int,
    obs_dim: int,
    time_features: int,
    cart: float,
    cart_velocity: float,
    angle: float,
    hinge_velocity: float,
    morphology: float,
    time: float,
    bias: float,
) -> np.ndarray:
    sigma = np.zeros(obs_dim + 1, dtype=np.float64)
    sigma[0] = cart
    sigma[1] = cart_velocity
    sigma[2 : 2 + 2 * n_links] = angle
    rel_start = 2 + 2 * n_links
    vel_start = 2 + 3 * n_links
    sigma[rel_start : rel_start + n_links] = angle
    sigma[vel_start : vel_start + n_links] = hinge_velocity
    morphology_start = 2 + 4 * n_links
    sigma[morphology_start : obs_dim] = morphology
    if time_features:
        sigma[obs_dim - time_features : obs_dim] = time
    sigma[-1] = bias
    return sigma


def rollout(
    cfg: dict[str, Any],
    *,
    seed: int,
    weight: np.ndarray,
    bias: float,
    seconds: float,
    handoff_weight: float,
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
    best_angle = float("inf")
    best_hinge = float("inf")
    best_cart = float("inf")
    best_cart_velocity = float("inf")
    best_handoff_score = float("inf")
    best_handoff_angle = float("inf")
    best_handoff_hinge = float("inf")
    best_handoff_cart = float("inf")
    best_handoff_cart_velocity = float("inf")
    max_cart = 0.0
    max_capture_quality = 0.0
    final_info: dict[str, Any] = {}
    while not (terminated or truncated):
        action = float(np.tanh(float(obs @ weight + bias)))
        obs, reward, terminated, truncated, info = env.step([action])
        ep_return += float(reward)
        action_sq += action * action
        action_smooth += (action - prev_action) ** 2
        prev_action = action
        angle = float(info.get("max_abs_angle", np.inf))
        hinge = float(info.get("hinge_velocity_rms", np.inf))
        cart = abs(float(info.get("x", 0.0)))
        cart_velocity = abs(float(env.data.qvel[0]))
        best_angle = min(best_angle, angle)
        best_hinge = min(best_hinge, hinge)
        best_cart = min(best_cart, cart)
        best_cart_velocity = min(best_cart_velocity, cart_velocity)
        handoff_score = (
            30.0 * (angle / 0.15) ** 2
            + 10.0 * (hinge / 0.75) ** 2
            + 3.0 * (cart / 1.25) ** 2
            + 3.0 * (cart_velocity / 0.50) ** 2
        )
        if handoff_score < best_handoff_score:
            best_handoff_score = handoff_score
            best_handoff_angle = angle
            best_handoff_hinge = hinge
            best_handoff_cart = cart
            best_handoff_cart_velocity = cart_velocity
        max_cart = max(max_cart, cart)
        max_capture_quality = max(
            max_capture_quality,
            float(info.get("capture_quality", 0.0)),
        )
        final_info = dict(info)
    success = bool(final_info.get("success", False))
    ever_upright = final_info.get("time_to_first_upright") is not None
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    max_centered_streak = float(final_info.get("max_centered_upright_streak_seconds", 0.0))
    max_low_momentum_streak = float(
        final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
    )
    rail_over = max(0.0, max_cart - float(local_cfg["env"]["rail_limit"]))
    score = (
        -30000.0 * float(success)
        - 7000.0 * max_streak
        - 4500.0 * max_centered_streak
        - 6500.0 * max_low_momentum_streak
        - 1500.0 * float(ever_upright)
        - 1200.0 * max_capture_quality
        + 22.0 * best_angle
        + 6.0 * best_hinge
        + 2.0 * best_cart
        + 1.0 * best_cart_velocity
        + float(handoff_weight) * best_handoff_score
        + 100.0 * float(terminated and not success)
        + 140.0 * rail_over * rail_over
        + 0.01 * action_sq
        + 0.04 * action_smooth
    )
    result = {
        "score": float(score),
        "return": float(ep_return),
        "success": success,
        "ever_upright": bool(ever_upright),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "best_max_abs_angle": float(best_angle),
        "best_hinge_velocity_rms": float(best_hinge),
        "best_cart_abs": float(best_cart),
        "best_cart_velocity_abs": float(best_cart_velocity),
        "best_handoff_score": float(best_handoff_score),
        "best_handoff_max_abs_angle": float(best_handoff_angle),
        "best_handoff_hinge_velocity_rms": float(best_handoff_hinge),
        "best_handoff_cart_abs": float(best_handoff_cart),
        "best_handoff_cart_velocity_abs": float(best_handoff_cart_velocity),
        "max_cart_abs": float(max_cart),
        "max_capture_quality": float(max_capture_quality),
        "max_upright_streak_seconds": max_streak,
        "max_centered_upright_streak_seconds": max_centered_streak,
        "max_low_momentum_upright_streak_seconds": max_low_momentum_streak,
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
    vector: np.ndarray,
    obs_dim: int,
    seconds: float,
    handoff_weight: float,
) -> dict[str, Any]:
    weight = vector[:obs_dim]
    bias = float(vector[obs_dim])
    rows = [
        rollout(
            cfg,
            seed=seed + episode,
            weight=weight,
            bias=bias,
            seconds=seconds,
            handoff_weight=handoff_weight,
        )
        for episode in range(episodes)
    ]
    return {
        "score": float(np.mean([row["score"] for row in rows])),
        "return_mean": float(np.mean([row["return"] for row in rows])),
        "success_rate": float(np.mean([float(row["success"]) for row in rows])),
        "ever_upright_rate": float(np.mean([float(row["ever_upright"]) for row in rows])),
        "max_upright_streak_mean": float(
            np.mean([row["max_upright_streak_seconds"] for row in rows])
        ),
        "max_upright_streak_max": float(
            np.max([row["max_upright_streak_seconds"] for row in rows])
        ),
        "max_low_momentum_upright_streak_mean": float(
            np.mean([row["max_low_momentum_upright_streak_seconds"] for row in rows])
        ),
                "best_max_abs_angle_min": float(
                    np.min([row["best_max_abs_angle"] for row in rows])
                ),
        "best_handoff_score_min": float(
            np.min([row["best_handoff_score"] for row in rows])
        ),
        "best_handoff_hinge_velocity_rms_min": float(
            np.min([row["best_handoff_hinge_velocity_rms"] for row in rows])
        ),
        "max_capture_quality_max": float(
            np.max([row["max_capture_quality"] for row in rows])
        ),
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--init-json",
        default=None,
        help="warm-start from best_actor in a prior JSON search result",
    )
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260931)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--sigma-decay", type=float, default=0.88)
    parser.add_argument("--sigma-floor", type=float, default=0.003)
    parser.add_argument("--cart-sigma", type=float, default=0.50)
    parser.add_argument("--cart-vel-sigma", type=float, default=0.40)
    parser.add_argument("--angle-sigma", type=float, default=0.50)
    parser.add_argument("--hinge-vel-sigma", type=float, default=0.25)
    parser.add_argument("--morphology-sigma", type=float, default=0.0)
    parser.add_argument("--time-sigma", type=float, default=0.30)
    parser.add_argument("--bias-sigma", type=float, default=0.25)
    parser.add_argument("--handoff-weight", type=float, default=5.0)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg.setdefault("ppo", {})["hidden_sizes"] = []
    cfg["env"] = {**cfg["env"], "episode_seconds": float(args.seconds)}
    probe = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    obs_dim = int(probe.observation_space.shape[0])
    n_links = int(probe.n)
    probe.close()
    sigma = build_sigma(
        n_links=n_links,
        obs_dim=obs_dim,
        time_features=time_feature_count(cfg),
        cart=args.cart_sigma,
        cart_velocity=args.cart_vel_sigma,
        angle=args.angle_sigma,
        hinge_velocity=args.hinge_vel_sigma,
        morphology=args.morphology_sigma,
        time=args.time_sigma,
        bias=args.bias_sigma,
    )
    rng = np.random.default_rng(args.seed)
    center = np.zeros(obs_dim + 1, dtype=np.float64)
    if args.init_json:
        prior = json.loads(Path(args.init_json).read_text(encoding="utf-8"))
        actor = prior.get("best_actor", {})
        prior_weight = np.asarray(actor.get("weight", []), dtype=np.float64)
        if prior_weight.shape != (obs_dim,):
            raise ValueError(
                f"{args.init_json} best_actor weight has shape {prior_weight.shape}; "
                f"expected {(obs_dim,)}"
            )
        center[:obs_dim] = prior_weight
        center[obs_dim] = float(actor.get("bias", 0.0))
    best_vector = center.copy()
    best: dict[str, Any] | None = None
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
        for vector in candidates:
            metrics = evaluate(
                cfg,
                seed=args.seed,
                episodes=args.episodes,
                vector=np.asarray(vector, dtype=np.float64),
                obs_dim=obs_dim,
                seconds=args.seconds,
                handoff_weight=args.handoff_weight,
            )
            records.append({"vector": np.asarray(vector, dtype=np.float64), "metrics": metrics})
        records.sort(key=lambda row: float(row["metrics"]["score"]))
        top = records[0]
        if best is None or float(top["metrics"]["score"]) < float(best["score"]):
            best = {
                "score": float(top["metrics"]["score"]),
                "metrics": top["metrics"],
                "vector": top["vector"].astype(float).tolist(),
            }
            best_vector = top["vector"].copy()
        top_metrics = top["metrics"]
        history.append(
            {
                "iteration": iteration,
                "score": float(top_metrics["score"]),
                "best_score": float(best["score"]),
                "return_mean": float(top_metrics["return_mean"]),
                "success_rate": float(top_metrics["success_rate"]),
                "ever_upright_rate": float(top_metrics["ever_upright_rate"]),
                "max_upright_streak_mean": float(top_metrics["max_upright_streak_mean"]),
                "max_upright_streak_max": float(top_metrics["max_upright_streak_max"]),
                "best_max_abs_angle_min": float(top_metrics["best_max_abs_angle_min"]),
                "best_handoff_score_min": float(top_metrics["best_handoff_score_min"]),
                "best_handoff_hinge_velocity_rms_min": float(
                    top_metrics["best_handoff_hinge_velocity_rms_min"]
                ),
            }
        )
        print(
            f"iter={iteration:03d} score={top_metrics['score']:.3f} "
            f"ret={top_metrics['return_mean']:.1f} succ={top_metrics['success_rate']:.2f} "
            f"ever={top_metrics['ever_upright_rate']:.2f} "
            f"streak={top_metrics['max_upright_streak_max']:.3f}s "
            f"angle={top_metrics['best_max_abs_angle_min']:.3f} "
            f"handoff={top_metrics['best_handoff_score_min']:.2f} "
            f"hhinge={top_metrics['best_handoff_hinge_velocity_rms_min']:.3f}",
            flush=True,
        )
        if iteration > 0:
            elite = np.asarray([row["vector"] for row in records[: args.elites]])
            center = best_vector.copy()
            sigma = np.maximum(elite.std(axis=0), args.sigma_floor) * args.sigma_decay

    assert best is not None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_path.with_name(out_path.stem + ".config.resolved.yaml"))
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "CPU NumPy CEM search for a closed-loop linear tanh seven-link swing policy; proposal only.",
        "config_path": str(Path(args.config)),
        "init_json": args.init_json,
        "progress": float(args.progress),
        "seconds": float(args.seconds),
        "obs_dim": int(obs_dim),
        "search": {
            "seed": int(args.seed),
            "episodes": int(args.episodes),
            "eval_episodes": int(args.eval_episodes),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "handoff_weight": float(args.handoff_weight),
            "wall_time_seconds": float(time.time() - started),
        },
        "best_search": best,
        "best_actor": {
            "weight": best_vector[:obs_dim].astype(float).tolist(),
            "bias": float(best_vector[obs_dim]),
        },
        "eval": evaluate(
            cfg,
            seed=args.seed + 4444,
            episodes=args.eval_episodes,
            vector=best_vector,
            obs_dim=obs_dim,
            seconds=args.seconds,
            handoff_weight=args.handoff_weight,
        ),
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
