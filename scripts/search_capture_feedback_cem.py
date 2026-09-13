#!/usr/bin/env python
"""Search a bounded local nonlinear capture feedback law from a real state.

The actor is a tanh of normalized cart, absolute-angle, cart-speed, and
absolute-angular-speed features.  CEM is used only for the local capture
problem; every rollout begins from the supplied physical handoff state and is
reset-free after that point.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import (
    data_sha256,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)


def load_state(path: str, index: int) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(states, list) or not states:
        raise ValueError(f"{path} does not contain a non-empty states list")
    if index < 0 or index >= len(states):
        raise IndexError(f"state index {index} outside 0..{len(states) - 1}")
    return dict(states[index])


def capture_features(env: NLinkCartPoleEnv) -> np.ndarray:
    qpos = np.asarray(env.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    absolute = serial_absolute_angles(qpos[1 : 1 + env.n])
    relative_omega = qvel[1 : 1 + env.n]
    absolute_omega = np.cumsum(relative_omega)
    return np.r_[
        qpos[0] / 1.25,
        np.sin(absolute),
        np.cos(absolute) - 1.0,
        qvel[0] / 0.50,
        absolute_omega / 0.75,
    ].astype(np.float64)


def rollout(
    cfg: dict[str, Any],
    *,
    state: dict[str, Any],
    vector: np.ndarray | None,
    progress: float = 0.0,
    seconds: float,
    return_trace: bool = False,
    centered_weight: float = 5000.0,
    absolute_low_momentum_weight: float = 0.0,
    max_cart_target: float = 1.25,
    max_cart_weight: float = 120.0,
    stable_time_weight: float = 0.0,
    action_fn: Callable[[NLinkCartPoleEnv], float] | None = None,
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
        "obs_include_capture_features": False,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        env_cfg[f"{key}_start"] = 0.0
        env_cfg[f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv({**cfg, "env": env_cfg}, progress=progress, seed=0)
    obs, _ = env.reset(seed=0)
    feature_dim = int(capture_features(env).size)
    if action_fn is None and (vector is None or vector.shape != (feature_dim + 1,)):
        actual_shape = None if vector is None else vector.shape
        raise ValueError(
            f"expected actor vector {(feature_dim + 1,)}, got {actual_shape}"
        )
    weight = None if vector is None else vector[:feature_dim]
    bias = None if vector is None else float(vector[-1])
    best_cost = float("inf")
    best_row: dict[str, Any] | None = None
    trace: list[dict[str, Any]] = []
    action_sq = 0.0
    action_smooth = 0.0
    previous_action = 0.0
    max_cart = abs(float(env.data.qpos[0]))
    absolute_low_momentum_streak_steps = 0
    max_absolute_low_momentum_streak_steps = 0
    centered_upright_time_steps = 0
    absolute_low_momentum_time_steps = 0
    final_info: dict[str, Any] = {}
    steps = min(env.max_steps, int(seconds / env.dt))
    del obs

    for step in range(steps):
        features = capture_features(env)
        if action_fn is None:
            assert weight is not None and bias is not None
            action = float(np.tanh(features @ weight + bias))
        else:
            action = float(action_fn(env))
        _, _, terminated, truncated, info = env.step([action])
        cart_velocity = float(env.data.qvel[0])
        angle = float(info["max_abs_angle"])
        hinge = float(info["hinge_velocity_rms"])
        absolute_angular_velocity = np.cumsum(
            np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64)
        )
        absolute_angular_velocity_rms = float(
            np.sqrt(np.mean(absolute_angular_velocity**2))
        )
        cart = abs(float(info["x"]))
        absolute_low_momentum = bool(
            bool(info["is_upright"])
            and cart <= float(max_cart_target)
            and absolute_angular_velocity_rms <= 0.75
            and abs(cart_velocity) <= 0.50
        )
        centered_upright = bool(
            bool(info["is_upright"]) and cart <= float(max_cart_target)
        )
        centered_upright_time_steps += int(centered_upright)
        absolute_low_momentum_time_steps += int(absolute_low_momentum)
        if absolute_low_momentum:
            absolute_low_momentum_streak_steps += 1
        else:
            absolute_low_momentum_streak_steps = 0
        max_absolute_low_momentum_streak_steps = max(
            max_absolute_low_momentum_streak_steps,
            absolute_low_momentum_streak_steps,
        )
        action_sq += action * action
        action_smooth += (action - previous_action) ** 2
        previous_action = action
        max_cart = max(max_cart, cart)
        row = {
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "action": float(action),
            "max_abs_angle": angle,
            "hinge_velocity_rms": hinge,
            "absolute_angular_velocity_rms": absolute_angular_velocity_rms,
            "max_absolute_angular_velocity": float(
                np.max(np.abs(absolute_angular_velocity))
            ),
            "x": float(info["x"]),
            "cart_velocity": cart_velocity,
            "is_upright": bool(info["is_upright"]),
            "upright_streak_seconds": float(info["upright_streak_seconds"]),
            "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
            "max_low_momentum_upright_streak_seconds": float(
                info["max_low_momentum_upright_streak_seconds"]
            ),
            "absolute_low_momentum_upright_streak_seconds": float(
                absolute_low_momentum_streak_steps * env.dt
            ),
            "max_absolute_low_momentum_upright_streak_seconds": float(
                max_absolute_low_momentum_streak_steps * env.dt
            ),
            "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        }
        local_cost = (
            30.0 * (angle / 0.15) ** 2
            + 8.0 * (hinge / 0.75) ** 2
            + 12.0 * (absolute_angular_velocity_rms / 0.75) ** 2
            + 3.0 * (cart / 1.25) ** 2
            + 3.0 * (abs(cart_velocity) / 0.50) ** 2
        )
        if local_cost < best_cost:
            best_cost = local_cost
            best_row = row
        if return_trace and (step % 2 == 0 or info["is_upright"]):
            trace.append(row)
        final_info = dict(info)
        if terminated or truncated:
            break

    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    max_centered = float(final_info.get("max_centered_upright_streak_seconds", 0.0))
    max_low_momentum = float(
        final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
    )
    terminal_absolute_angular_velocity = np.cumsum(
        np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64)
    )
    terminal_absolute_angular_velocity_rms = float(
        np.sqrt(np.mean(terminal_absolute_angular_velocity**2))
    )
    terminal_cost = (
        30.0 * (float(final_info.get("max_abs_angle", np.inf)) / 0.15) ** 2
        + 8.0 * (float(final_info.get("hinge_velocity_rms", np.inf)) / 0.75) ** 2
        + 12.0 * (terminal_absolute_angular_velocity_rms / 0.75) ** 2
        + 3.0 * (abs(float(final_info.get("x", np.inf))) / 1.25) ** 2
        + 3.0 * (abs(float(env.data.qvel[0])) / 0.50) ** 2
    )
    rail_over = max(0.0, max_cart - float(env.rail_limit))
    score = (
        0.30 * best_cost
        + 0.35 * terminal_cost
        + 0.35 * (best_cost + terminal_cost) / 2.0
        - 12000.0 * max_streak
        - float(centered_weight) * max_centered
        - 8000.0 * max_low_momentum
        - float(absolute_low_momentum_weight)
        * float(max_absolute_low_momentum_streak_steps * env.dt)
        - float(stable_time_weight) * float(absolute_low_momentum_time_steps * env.dt)
        + 120.0 * rail_over * rail_over
        + float(max_cart_weight) * max(0.0, max_cart - float(max_cart_target)) ** 2
        + 0.01 * action_sq
        + 0.04 * action_smooth
    )
    result = {
        "score": float(score),
        "success": bool(final_info.get("success", False)),
        "best_cost": float(best_cost),
        "terminal_cost": float(terminal_cost),
        "best_row": best_row,
        "max_cart_abs": float(max_cart),
        "max_upright_streak_seconds": max_streak,
        "max_centered_upright_streak_seconds": max_centered,
        "max_low_momentum_upright_streak_seconds": max_low_momentum,
        "max_absolute_low_momentum_upright_streak_seconds": float(
            max_absolute_low_momentum_streak_steps * env.dt
        ),
        "centered_upright_time_seconds": float(centered_upright_time_steps * env.dt),
        "absolute_low_momentum_time_seconds": float(
            absolute_low_momentum_time_steps * env.dt
        ),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "termination_reason": final_info.get("termination_reason"),
        "final_info": final_info,
        "trace": trace,
    }
    env.close()
    return result


def evaluate(
    cfg: dict[str, Any],
    state: dict[str, Any],
    vector: np.ndarray,
    *,
    progress: float,
    seed: int,
    episodes: int,
    seconds: float,
    centered_weight: float,
    absolute_low_momentum_weight: float,
    max_cart_target: float,
    max_cart_weight: float,
    stable_time_weight: float,
) -> dict[str, Any]:
    rows = [
        rollout(
            cfg,
            state=state,
            vector=vector,
            progress=progress,
            seconds=seconds,
            centered_weight=centered_weight,
            absolute_low_momentum_weight=absolute_low_momentum_weight,
            max_cart_target=max_cart_target,
            max_cart_weight=max_cart_weight,
            stable_time_weight=stable_time_weight,
        )
        for _ in range(episodes)
    ]
    return {
        "score": float(np.mean([row["score"] for row in rows])),
        "success_rate": float(np.mean([float(row["success"]) for row in rows])),
        "max_upright_streak_mean": float(
            np.mean([row["max_upright_streak_seconds"] for row in rows])
        ),
        "max_upright_streak_max": float(
            np.max([row["max_upright_streak_seconds"] for row in rows])
        ),
        "max_centered_upright_streak_mean": float(
            np.mean([row["max_centered_upright_streak_seconds"] for row in rows])
        ),
        "max_absolute_low_momentum_upright_streak_mean": float(
            np.mean(
                [
                    row["max_absolute_low_momentum_upright_streak_seconds"]
                    for row in rows
                ]
            )
        ),
        "best_cost_min": float(np.min([row["best_cost"] for row in rows])),
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CEM search for nonlinear capture feedback from a real seven-link handoff"
    )
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument(
        "--state-json", default="runs/swingup7_cart_feasible_handoff_state.json"
    )
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--eval-episodes", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.40)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--centered-weight", type=float, default=5000.0)
    parser.add_argument("--absolute-low-momentum-weight", type=float, default=0.0)
    parser.add_argument("--max-cart-target", type=float, default=1.25)
    parser.add_argument("--max-cart-weight", type=float, default=120.0)
    parser.add_argument("--stable-time-weight", type=float, default=0.0)
    parser.add_argument("--seed-actor", default=None)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")
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
    feature_dim = int(capture_features(probe).size)
    probe.close()
    rng = np.random.default_rng(args.seed)
    center = np.zeros(feature_dim + 1, dtype=np.float64)
    if args.seed_actor:
        seed_payload = json.loads(Path(args.seed_actor).read_text(encoding="utf-8"))
        seed_vector = np.asarray(seed_payload["best_vector"], dtype=np.float64)
        if seed_vector.shape != center.shape:
            raise ValueError(
                f"seed actor has vector shape {seed_vector.shape}, expected {center.shape}"
            )
        center = seed_vector.copy()
    sigma = np.full(feature_dim + 1, float(args.sigma), dtype=np.float64)
    sigma[-1] = 0.15
    best_vector = center.copy()
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()

    for iteration in range(args.iterations + 1):
        candidates = [best_vector.copy()]
        if iteration > 0:
            candidates.extend(
                center + rng.normal(0.0, sigma, size=center.shape)
                for _ in range(args.population - 1)
            )
        records = []
        for vector in candidates:
            metrics = rollout(
                cfg,
                state=state,
                vector=np.asarray(vector, dtype=np.float64),
                progress=args.progress,
                seconds=args.seconds,
                centered_weight=args.centered_weight,
                absolute_low_momentum_weight=args.absolute_low_momentum_weight,
                max_cart_target=args.max_cart_target,
                max_cart_weight=args.max_cart_weight,
                stable_time_weight=args.stable_time_weight,
            )
            records.append(
                {"vector": np.asarray(vector, dtype=np.float64), "metrics": metrics}
            )
        records.sort(key=lambda row: float(row["metrics"]["score"]))
        top = records[0]
        if best is None or float(top["metrics"]["score"]) < float(best["score"]):
            best = {
                "score": float(top["metrics"]["score"]),
                "metrics": top["metrics"],
                "vector": top["vector"].astype(float).tolist(),
            }
            best_vector = top["vector"].copy()
        tm = top["metrics"]
        history.append(
            {
                "iteration": iteration,
                "score": float(tm["score"]),
                "best_score": float(best["score"]),
                "streak": float(tm["max_upright_streak_seconds"]),
                "centered_streak": float(tm["max_centered_upright_streak_seconds"]),
                "best_cost": float(tm["best_cost"]),
                "terminal_cost": float(tm["terminal_cost"]),
            }
        )
        print(
            f"iter={iteration:03d} score={tm['score']:.2f} "
            f"streak={tm['max_upright_streak_seconds']:.3f}s "
            f"centered={tm['max_centered_upright_streak_seconds']:.3f}s "
            f"best={tm['best_cost']:.1f} terminal={tm['terminal_cost']:.1f}",
            flush=True,
        )
        if iteration > 0:
            elite = np.asarray([row["vector"] for row in records[: args.elites]])
            center = best_vector.copy()
            sigma = np.maximum(elite.std(axis=0) * args.sigma_decay, args.sigma_floor)

    assert best is not None
    link_count = int(cfg["env"]["n_links"])
    final_eval = evaluate(
        cfg,
        state,
        best_vector,
        progress=args.progress,
        seed=args.seed + 4444,
        episodes=args.eval_episodes,
        seconds=args.seconds,
        centered_weight=args.centered_weight,
        absolute_low_momentum_weight=args.absolute_low_momentum_weight,
        max_cart_target=args.max_cart_target,
        max_cart_weight=args.max_cart_weight,
        stable_time_weight=args.stable_time_weight,
    )
    trace = rollout(
        cfg,
        state=state,
        vector=best_vector,
        progress=args.progress,
        seconds=args.seconds,
        return_trace=True,
        centered_weight=args.centered_weight,
        absolute_low_momentum_weight=args.absolute_low_momentum_weight,
        max_cart_target=args.max_cart_target,
        max_cart_weight=args.max_cart_weight,
        stable_time_weight=args.stable_time_weight,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Local nonlinear capture feedback CEM from an actual seven-link planner handoff; proposal only.",
        "config_path": str(Path(args.config)),
        "state_json": str(Path(args.state_json)),
        "state_index": int(args.state_index),
        "seconds": float(args.seconds),
        "feature_names": (
            ["cart_position"]
            + [f"sin_absolute_angle_{i}" for i in range(link_count)]
            + [f"cos_minus_one_absolute_angle_{i}" for i in range(link_count)]
            + ["cart_velocity"]
            + [f"absolute_angular_velocity_{i}" for i in range(link_count)]
        ),
        "best_vector": best_vector.astype(float).tolist(),
        "best_actor": {
            "weight": best_vector[:-1].astype(float).tolist(),
            "bias": float(best_vector[-1]),
        },
        "best_search": best,
        "eval": final_eval,
        "trace_rollout": trace,
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
