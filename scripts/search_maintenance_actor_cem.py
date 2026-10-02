#!/usr/bin/env python
"""Search one shared nonlinear maintenance actor over upright perturbations.

This is a component diagnostic, not a swing-up result.  The purpose is to
answer a narrow n10 question: does a static full-state feedback law have a
usable upright basin when it is optimized against several real MuJoCo
initial states instead of a single hand-picked reset?
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


def make_initial_states(
    cfg: dict[str, Any],
    *,
    count: int,
    seed: int,
    angle_noise: float,
    velocity_noise: float,
    cart_noise: float,
    cart_velocity_noise: float,
) -> list[dict[str, list[float]]]:
    probe_cfg = {
        **cfg,
        "env": {
            **cfg["env"],
            "init_mode": "upright",
            "init_angle_noise": float(angle_noise),
            "init_vel_noise": float(velocity_noise),
            "init_cart_noise": float(cart_noise),
            "init_cart_vel_noise": float(cart_velocity_noise),
        },
    }
    env = NLinkCartPoleEnv(probe_cfg, progress=1.0, seed=seed)
    states: list[dict[str, list[float]]] = []
    for index in range(int(count)):
        env.reset(seed=seed + index)
        states.append(
            {
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
            }
        )
    env.close()
    return states


def load_state_bank(path: str) -> list[dict[str, list[float]]]:
    """Load exact qpos/qvel states from a materialized handoff bank."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    raw_states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_states, list) or not raw_states:
        raise ValueError(f"{source} must contain a non-empty states list")
    states: list[dict[str, list[float]]] = []
    for index, state in enumerate(raw_states):
        if not isinstance(state, dict):
            raise ValueError(f"{source} state {index} is not an object")
        qpos = np.asarray(state.get("qpos", []), dtype=np.float64)
        qvel = np.asarray(state.get("qvel", []), dtype=np.float64)
        if qpos.ndim != 1 or qvel.ndim != 1 or qpos.shape != qvel.shape or qpos.size < 2:
            raise ValueError(f"{source} state {index} has invalid qpos/qvel arrays")
        states.append({"qpos": qpos.astype(float).tolist(), "qvel": qvel.astype(float).tolist()})
    return states


def actor_action(env: NLinkCartPoleEnv, vector: np.ndarray) -> float:
    features = full_state_features(env)
    expected = features.size + 1
    if vector.shape != (expected,):
        raise ValueError(f"expected actor vector {(expected,)}, got {vector.shape}")
    return float(np.tanh(features @ vector[:-1] + vector[-1]))


def rollout(
    cfg: dict[str, Any],
    state: dict[str, list[float]],
    vector: np.ndarray,
    *,
    seconds: float,
) -> dict[str, Any]:
    env_cfg = {
        **cfg["env"],
        "init_mode": "fixed_state",
        "init_qpos": state["qpos"],
        "init_qvel": state["qvel"],
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "episode_seconds": float(seconds),
        "terminate_abs_angle": None,
    }
    env = NLinkCartPoleEnv({**cfg, "env": env_cfg}, progress=1.0, seed=0)
    env.reset(seed=0)
    integrated_cost = 0.0
    max_cart = abs(float(env.data.qpos[0]))
    max_angle = 0.0
    max_abs_rate = 0.0
    max_streak = 0.0
    max_low_momentum = 0.0
    previous_action = 0.0
    action_sq = 0.0
    action_delta_sq = 0.0
    final_info: dict[str, Any] = {}
    steps = min(env.max_steps, int(seconds / env.dt))
    for _ in range(steps):
        action = actor_action(env, vector)
        _, _, terminated, truncated, info = env.step([action])
        angle = float(info["max_abs_angle"])
        abs_rate = float(info["absolute_angular_velocity_rms"])
        hinge_rate = float(info["hinge_velocity_rms"])
        cart = abs(float(info["x"]))
        cart_velocity = abs(float(env.data.qvel[0]))
        # Keep the optimizer sensitive to the entire recovery trajectory.  A
        # terminal-only score lets candidates leave the upright basin and
        # return to it by the last step, which is not a maintenance solution.
        integrated_cost += (
            (angle / 0.15) ** 2
            + 0.75 * (abs_rate / 0.75) ** 2
            + 0.25 * (hinge_rate / 0.75) ** 2
            + 0.20 * (cart / 1.25) ** 2
            + 0.20 * (cart_velocity / 0.50) ** 2
            + 0.01 * action * action
        )
        max_cart = max(max_cart, cart)
        max_angle = max(max_angle, angle)
        max_abs_rate = max(max_abs_rate, float(info["max_absolute_angular_velocity"]))
        max_streak = max(max_streak, float(info["max_upright_streak_seconds"]))
        max_low_momentum = max(
            max_low_momentum,
            float(info["max_low_momentum_upright_streak_seconds"]),
        )
        action_sq += action * action
        action_delta_sq += (action - previous_action) ** 2
        previous_action = action
        final_info = dict(info)
        if terminated or truncated:
            break
    mean_cost = integrated_cost / max(1, steps)
    rail_over = max(0.0, max_cart - float(env.rail_limit))
    # Reward a continuous upright basin, but keep the smooth trajectory cost
    # dominant until a candidate can actually stay there.
    score = (
        mean_cost
        + 0.50 * max(0.0, float(final_info.get("max_abs_angle", np.inf)) / 0.15) ** 2
        + 0.25 * max(0.0, abs(float(final_info.get("x", np.inf))) / 1.25) ** 2
        + 0.02 * action_sq / max(1, steps)
        + 0.04 * action_delta_sq / max(1, steps)
        - 18.0 * max_streak
        - 10.0 * max_low_momentum
        + 300.0 * rail_over * rail_over
    )
    result = {
        "score": float(score),
        "success": bool(final_info.get("success", False)),
        "mean_cost": float(mean_cost),
        "max_cart_abs": float(max_cart),
        "max_abs_angle": float(max_angle),
        "max_absolute_angular_velocity": float(max_abs_rate),
        "max_upright_streak_seconds": float(max_streak),
        "max_low_momentum_upright_streak_seconds": float(max_low_momentum),
        "termination_reason": final_info.get("termination_reason"),
        "final_info": final_info,
    }
    env.close()
    return result


def score_vector(
    cfg: dict[str, Any],
    states: list[dict[str, list[float]]],
    vector: np.ndarray,
    *,
    seconds: float,
) -> dict[str, Any]:
    episodes = [rollout(cfg, state, vector, seconds=seconds) for state in states]
    scores = np.asarray([float(row["score"]) for row in episodes], dtype=np.float64)
    return {
        "score": float(np.mean(scores) + 0.50 * np.max(scores)),
        "mean_episode_score": float(np.mean(scores)),
        "worst_episode_score": float(np.max(scores)),
        "episodes": episodes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CEM search for an n10 upright maintenance actor")
    parser.add_argument("--config", default="configs/swingup10_uniform.yaml")
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.005)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument(
        "--states-json",
        default=None,
        help="materialized exact handoff bank; overrides synthetic upright perturbations",
    )
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--angle-noise", type=float, default=0.01)
    parser.add_argument("--velocity-noise", type=float, default=0.01)
    parser.add_argument("--cart-noise", type=float, default=0.0)
    parser.add_argument("--cart-velocity-noise", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260913)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")

    cfg = apply_overrides(load_config(args.config), args.override)
    states = (
        load_state_bank(args.states_json)
        if args.states_json is not None
        else make_initial_states(
            cfg,
            count=args.episodes,
            seed=args.seed,
            angle_noise=args.angle_noise,
            velocity_noise=args.velocity_noise,
            cart_noise=args.cart_noise,
            cart_velocity_noise=args.cart_velocity_noise,
        )
    )
    probe = NLinkCartPoleEnv(
        {**cfg, "env": {**cfg["env"], "init_mode": "fixed_state", "init_qpos": states[0]["qpos"], "init_qvel": states[0]["qvel"]}},
        progress=1.0,
        seed=0,
    )
    feature_dim = int(full_state_features(probe).size)
    probe.close()

    center = np.zeros(feature_dim + 1, dtype=np.float64)
    rng = np.random.default_rng(args.seed + 1)
    sigma = np.full(feature_dim + 1, float(args.sigma), dtype=np.float64)
    sigma[-1] = min(0.10, float(args.sigma))
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
            metrics = score_vector(cfg, states, np.asarray(vector, dtype=np.float64), seconds=args.seconds)
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
        episode_metrics = top["metrics"]["episodes"]
        history.append(
            {
                "iteration": int(iteration),
                "score": float(top["metrics"]["score"]),
                "best_score": float(best["score"]),
                "max_upright_streak_seconds": float(max(row["max_upright_streak_seconds"] for row in episode_metrics)),
                "mean_max_upright_streak_seconds": float(np.mean([row["max_upright_streak_seconds"] for row in episode_metrics])),
                "max_cart_abs": float(max(row["max_cart_abs"] for row in episode_metrics)),
                "sigma_mean": float(np.mean(sigma)),
            }
        )
        print(
            f"iter={iteration:03d} score={top['metrics']['score']:.2f} "
            f"hold_mean={history[-1]['mean_max_upright_streak_seconds']:.3f}s "
            f"hold_max={history[-1]['max_upright_streak_seconds']:.3f}s "
            f"cart={history[-1]['max_cart_abs']:.3f}",
            flush=True,
        )

    assert best is not None
    best_vector = np.asarray(best["vector"], dtype=np.float64)
    final = score_vector(cfg, states, best_vector, seconds=args.seconds)
    n = int(cfg["env"]["n_links"])
    feature_names = (
        ["cart_position"]
        + [f"sin_absolute_angle_{i}" for i in range(n)]
        + [f"cos_minus_one_absolute_angle_{i}" for i in range(n)]
        + [f"sin_relative_angle_{i}" for i in range(n)]
        + [f"cos_minus_one_relative_angle_{i}" for i in range(n)]
        + ["cart_velocity"]
        + [f"absolute_angular_velocity_{i}" for i in range(n)]
        + [f"relative_angular_velocity_{i}" for i in range(n)]
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Robust upright-maintenance actor CEM diagnostic over a fixed bank of n10 MuJoCo perturbations.",
        "config_path": str(Path(args.config)),
        "states_json": None if args.states_json is None else str(Path(args.states_json)),
        "feature_names": feature_names,
        "best_vector": best_vector.astype(float).tolist(),
        "best_actor": {"weight": best_vector[:-1].astype(float).tolist(), "bias": float(best_vector[-1])},
        "best_search": best,
        "eval": final,
        "initial_states": states,
        "search": {
            "seed": int(args.seed),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "sigma": float(args.sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "episodes": int(len(states)),
            "seconds": float(args.seconds),
            "angle_noise": float(args.angle_noise),
            "velocity_noise": float(args.velocity_noise),
            "cart_noise": float(args.cart_noise),
            "cart_velocity_noise": float(args.cart_velocity_noise),
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
