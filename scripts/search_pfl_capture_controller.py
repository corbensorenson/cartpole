#!/usr/bin/env python
"""Search a two-expert PFL swing-up plus upright-capture controller.

The first expert shapes passive-chain energy through cart acceleration.  The
second expert is the exact-model local LQR used for upright capture.  This is
discovery evidence only: a promising result still needs reset-free replay and
the canonical seven-link gates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv

try:
    from scripts.search_energy_shaping_feedback import (
        force_for_cart_acceleration,
        state_features,
    )
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from search_energy_shaping_feedback import force_for_cart_acceleration, state_features
    from search_swingup_capture import lqr_action, lqr_gain


PARAMETER_NAMES = [
    "energy_gain",
    "phase_gain",
    "phase_velocity_gain",
    "cart_kp",
    "cart_kd",
    "kick_amp",
    "kick_frequency",
    "kick_phase",
    "lqr_scale",
    "angle_limit",
    "hinge_rate_limit",
    "energy_error_limit",
    "gate_width",
]


def sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(np.clip(value, -60.0, 60.0)))))


def swing_acceleration(params: np.ndarray, features: dict[str, float], t: float) -> float:
    energy_gain, phase_gain, phase_velocity_gain, cart_kp, cart_kd = params[:5]
    kick_amp, kick_frequency, kick_phase = params[5:8]
    kick = kick_amp * np.sin(2.0 * np.pi * kick_frequency * t + kick_phase)
    return float(
        energy_gain * features["energy_error"] * features["horizontal_momentum"]
        + phase_gain * features["phase"]
        + phase_velocity_gain * features["phase_velocity"]
        - cart_kp * features["cart_position"]
        - cart_kd * features["cart_velocity"]
        + kick
    )


def capture_blend(params: np.ndarray, features: dict[str, float], env: NLinkCartPoleEnv) -> float:
    _, _, _, _, _, _, _, _, _, angle_limit, rate_limit, energy_limit, width = params
    _, absolute = env._angles()
    max_angle = float(np.max(np.abs(absolute)))
    hinge_rate = float(np.sqrt(np.mean(np.asarray(env.data.qvel[1 : 1 + env.n]) ** 2)))
    cart_abs = abs(float(features["cart_position"]))
    cart_velocity_abs = abs(float(features["cart_velocity"]))
    width = max(1.0e-3, float(width))
    angle_gate = sigmoid((float(angle_limit) - max_angle) / width)
    rate_gate = sigmoid((float(rate_limit) - hinge_rate) / width)
    energy_gate = sigmoid((float(energy_limit) - abs(float(features["energy_error"]))) / width)
    cart_gate = sigmoid((1.25 - cart_abs) / width)
    cart_velocity_gate = sigmoid((0.50 - cart_velocity_abs) / width)
    return float(np.clip(angle_gate * rate_gate * energy_gate * cart_gate * cart_velocity_gate, 0.0, 1.0))


def rollout(
    cfg: dict[str, Any],
    params: np.ndarray,
    *,
    gain: np.ndarray,
    progress: float,
    seconds: float,
    rail_limit: float | None = None,
    return_trace: bool = False,
) -> dict[str, Any]:
    env_cfg = {
        **cfg["env"],
        "init_mode": "hanging",
        "episode_seconds": float(seconds),
        "terminate_abs_angle": None,
        "obs_include_capture_features": False,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
    }
    if rail_limit is not None:
        env_cfg["rail_limit"] = float(rail_limit)
        env_cfg["rail_limit_start"] = float(rail_limit)
        env_cfg["rail_limit_end"] = float(rail_limit)
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        env_cfg[f"{key}_start"] = 0.0
        env_cfg[f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv({**cfg, "env": env_cfg}, progress=progress, seed=0)
    env.reset(seed=0)
    trace: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    max_cart = 0.0
    best_cost = float("inf")
    best_row: dict[str, Any] | None = None
    action_energy = 0.0
    action_slew = 0.0
    previous_action = 0.0
    steps = min(env.max_steps, int(seconds / env.dt))

    for step in range(steps):
        t = step * env.dt
        features = state_features(env, t)
        acceleration = swing_acceleration(params, features, t)
        force = force_for_cart_acceleration(env, acceleration)
        swing_action = float(np.clip(force / env.force_limit, -1.0, 1.0))
        blend = capture_blend(params, features, env)
        capture_action = lqr_action(env, gain, scale=float(params[8]), cart_target=0.0)
        action = float(np.clip((1.0 - blend) * swing_action + blend * capture_action, -1.0, 1.0))
        _, _, terminated, truncated, info = env.step([action])
        _, absolute = env._angles()
        hinge_rms = float(np.sqrt(np.mean(np.asarray(env.data.qvel[1 : 1 + env.n]) ** 2)))
        row = {
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "action": float(action),
            "swing_action": swing_action,
            "capture_action": float(capture_action),
            "capture_blend": blend,
            "desired_cart_acceleration": float(acceleration),
            "force": float(force),
            **features,
            "max_abs_angle": float(np.max(np.abs(absolute))),
            "hinge_velocity_rms": hinge_rms,
            "x": float(env.data.qpos[0]),
            "cart_velocity": float(env.data.qvel[0]),
            "absolute_angles": absolute.astype(float).tolist(),
            "is_upright": bool(info.get("is_upright", False)),
            "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
            "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
        }
        cost = (
            30.0 * (row["max_abs_angle"] / 0.15) ** 2
            + 8.0 * (row["hinge_velocity_rms"] / 0.75) ** 2
            + 3.0 * (abs(row["x"]) / 1.25) ** 2
            + 3.0 * (abs(row["cart_velocity"]) / 0.50) ** 2
        )
        if cost < best_cost:
            best_cost = cost
            best_row = dict(row)
        if return_trace and (step % 4 == 0 or row["is_upright"] or blend > 0.5):
            trace.append(row)
        max_cart = max(max_cart, abs(row["x"]))
        action_energy += action * action
        action_slew += (action - previous_action) ** 2
        previous_action = action
        final_info = dict(info)
        if terminated or truncated:
            break

    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    centered_streak = float(final_info.get("max_centered_upright_streak_seconds", 0.0))
    low_momentum_streak = float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0))
    rail_over = max(0.0, max_cart - float(env.rail_limit))
    score = (
        0.35 * best_cost
        - 12000.0 * max_streak
        - 5000.0 * centered_streak
        - 8000.0 * low_momentum_streak
        + 100.0 * rail_over * rail_over
        + 0.01 * action_energy
        + 0.02 * action_slew
    )
    result = {
        "score": float(score),
        "success": bool(final_info.get("success", False)),
        "ever_upright": final_info.get("time_to_first_upright") is not None,
        "max_upright_streak_seconds": max_streak,
        "max_centered_upright_streak_seconds": centered_streak,
        "max_low_momentum_upright_streak_seconds": low_momentum_streak,
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "max_cart_abs": float(max_cart),
        "termination_reason": final_info.get("termination_reason"),
        "best_row": best_row,
        "best_cost": float(best_cost),
        "final_info": final_info,
        "trace": trace,
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Search two-expert PFL swing-up plus LQR capture")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--rail-limit", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--init-json", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("--elites must be between one and population")
    cfg = apply_overrides(load_config(args.config), args.override)
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    rng = np.random.default_rng(args.seed)
    center = np.asarray([12.0, -4.0, -1.0, 1.0, 1.0, 2.0, 0.20, 0.0, 1.0, 0.35, 1.5, 0.40, 0.12], dtype=np.float64)
    if args.init_json:
        prior = json.loads(Path(args.init_json).read_text(encoding="utf-8"))
        prior_vector = np.asarray(prior.get("best", {}).get("vector", []), dtype=np.float64)
        if prior_vector.shape != center.shape:
            raise ValueError(f"{args.init_json} best vector has shape {prior_vector.shape}; expected {center.shape}")
        center = prior_vector.copy()
    sigma = np.full(center.shape, float(args.sigma), dtype=np.float64)
    sigma[6] = 0.08
    sigma[7] = 0.50
    sigma[8:] = 0.15
    lower = np.asarray([-40.0, -40.0, -20.0, 0.0, 0.0, -30.0, 0.02, -np.pi, 0.05, 0.12, 0.10, 0.03, 0.02], dtype=np.float64)
    upper = np.asarray([40.0, 40.0, 20.0, 20.0, 20.0, 30.0, 2.00, np.pi, 6.0, 0.80, 5.0, 1.50, 0.60], dtype=np.float64)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    for iteration in range(args.iterations + 1):
        candidates = [center.copy()] if iteration == 0 else [center + rng.normal(0.0, sigma) for _ in range(args.population)]
        records: list[dict[str, Any]] = []
        for vector in candidates:
            vector = np.clip(np.asarray(vector, dtype=np.float64), lower, upper)
            result = rollout(
                cfg,
                vector,
                gain=gain,
                progress=args.progress,
                seconds=args.seconds,
                rail_limit=args.rail_limit,
            )
            records.append({"vector": vector, "result": result})
        records.sort(key=lambda row: float(row["result"]["score"]))
        top = records[0]
        if best is None or float(top["result"]["score"]) < float(best["score"]):
            best = {"score": float(top["result"]["score"]), "vector": top["vector"].astype(float).tolist(), "result": top["result"]}
        elite = np.asarray([row["vector"] for row in records[: args.elites]], dtype=np.float64)
        center = elite.mean(axis=0)
        sigma = np.maximum(np.std(elite, axis=0) * args.sigma_decay, args.sigma_floor)
        top_result = top["result"]
        history.append({
            "iteration": int(iteration),
            "score": float(top_result["score"]),
            "best_score": float(best["score"]),
            "success": bool(top_result["success"]),
            "ever_upright": bool(top_result["ever_upright"]),
            "streak": float(top_result["max_upright_streak_seconds"]),
            "centered_streak": float(top_result["max_centered_upright_streak_seconds"]),
            "low_momentum_streak": float(top_result["max_low_momentum_upright_streak_seconds"]),
            "best_angle": float((top_result.get("best_row") or {}).get("max_abs_angle", np.inf)),
            "max_cart_abs": float(top_result["max_cart_abs"]),
        })
        print(
            f"iter={iteration:03d} score={top_result['score']:.2f} success={int(top_result['success'])} "
            f"upright={int(top_result['ever_upright'])} streak={top_result['max_upright_streak_seconds']:.3f}s "
            f"angle={(top_result.get('best_row') or {}).get('max_abs_angle', np.nan):.4f} "
            f"max_cart={top_result['max_cart_abs']:.2f}",
            flush=True,
        )

    assert best is not None
    final = rollout(
        cfg,
        np.asarray(best["vector"], dtype=np.float64),
        gain=gain,
        progress=args.progress,
        seconds=args.seconds,
        rail_limit=args.rail_limit,
        return_trace=True,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Two-expert PFL energy shaper plus exact-model LQR capture; discovery only.",
        "config_path": str(Path(args.config)),
        "progress": float(args.progress),
        "seconds": float(args.seconds),
        "rail_limit": args.rail_limit,
        "parameter_names": PARAMETER_NAMES,
        "search": {
            "seed": int(args.seed),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "sigma": float(args.sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "wall_time_seconds": float(time.time() - started),
        },
        "best": best,
        "final_eval": final,
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
