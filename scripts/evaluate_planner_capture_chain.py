#!/usr/bin/env python
"""Evaluate a saved hanging-start planner handoff followed by a capture actor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from probe_swingup_trajectory import trajectory_action
from search_capture_feedback_cem import capture_features
from search_swingup_tail_action_cem import load_controller


def load_actor(path: str) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    vector = payload.get("best_vector")
    if vector is None:
        raise ValueError(f"{path} does not contain best_vector")
    return np.asarray(vector, dtype=np.float64)


def reset_free_prefix(
    env: NLinkCartPoleEnv,
    source: dict[str, Any],
    tail_start_seconds: float,
    prefix_controls: np.ndarray,
) -> None:
    prefix_steps = int(round(tail_start_seconds / env.dt))
    for step in range(prefix_steps):
        action = trajectory_action(
            env,
            step * env.dt,
            np.asarray(source["knots"], dtype=np.float64),
            float(source["trajectory_seconds"]),
            float(source.get("kp", 0.0)),
            float(source.get("kd", 0.0)),
        )
        _, _, terminated, truncated, info = env.step([action])
        if terminated or truncated:
            raise RuntimeError(f"source ended at {step}: {info.get('termination_reason')}")
    for step, action in enumerate(prefix_controls):
        _, _, terminated, truncated, info = env.step([float(action)])
        if terminated or truncated:
            raise RuntimeError(f"tail prefix ended at {step}: {info.get('termination_reason')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved planner route and switch to a learned capture actor")
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument("--tail-json", required=True)
    parser.add_argument("--capture-json", required=True)
    parser.add_argument("--hold-seconds", type=float, default=8.0)
    parser.add_argument("--absolute-rate-threshold", type=float, default=0.75)
    parser.add_argument("--handoff-state-json", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    tail_payload = json.loads(Path(args.tail_json).read_text(encoding="utf-8"))
    controller = tail_payload["controller"]
    source_path = Path(tail_payload["swing_controller_json"])
    source = load_controller(str(source_path), None)
    prefix_controls = np.asarray(controller["prefix_controls"], dtype=np.float64)
    suffix_controls = np.asarray(controller["suffix_controls"], dtype=np.float64)
    capture_vector = load_actor(args.capture_json)
    capture_payload = json.loads(Path(args.capture_json).read_text(encoding="utf-8"))
    handoff_payload = None
    if args.handoff_state_json:
        handoff_payload = json.loads(Path(args.handoff_state_json).read_text(encoding="utf-8"))
        handoff_state = handoff_payload.get("states", [])[0]
    else:
        handoff_state = capture_payload.get("handoff_state")
    if not handoff_state:
        raise ValueError("a handoff state is required for endpoint replay comparison")

    base = apply_overrides(load_config(args.config), args.override)
    tail_start_seconds = float(tail_payload["tail_start_seconds"])
    prefix_seconds = float(tail_payload["prefix_seconds"])
    suffix_seconds = float(tail_payload["suffix_seconds"])
    total_seconds = tail_start_seconds + prefix_seconds + suffix_seconds + float(args.hold_seconds) + 1.0
    env_cfg = {
        **base["env"],
        "init_mode": "hanging",
        "episode_seconds": total_seconds,
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "terminate_abs_angle": None,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        env_cfg[f"{key}_start"] = 0.0
        env_cfg[f"{key}_end"] = 0.0
    cfg = {**base, "env": env_cfg}
    env = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
    env.reset(seed=0)
    reset_free_prefix(env, source, tail_start_seconds, prefix_controls)
    for action in suffix_controls:
        _, _, terminated, truncated, info = env.step([float(action)])
        if terminated or truncated:
            raise RuntimeError(f"saved suffix ended before handoff: {info.get('termination_reason')}")

    endpoint_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    endpoint_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
    expected_qpos = np.asarray(handoff_state["qpos"], dtype=np.float64)
    expected_qvel = np.asarray(handoff_state["qvel"], dtype=np.float64)
    endpoint_error = {
        "max_qpos_abs_error": float(np.max(np.abs(endpoint_qpos - expected_qpos))),
        "max_qvel_abs_error": float(np.max(np.abs(endpoint_qvel - expected_qvel))),
        "qpos_l2_error": float(np.linalg.norm(endpoint_qpos - expected_qpos)),
        "qvel_l2_error": float(np.linalg.norm(endpoint_qvel - expected_qvel)),
    }

    trace: list[dict[str, Any]] = []
    max_cart_abs = abs(float(env.data.qpos[0]))
    low_streak = 0
    max_low_streak = 0
    centered_streak = 0
    max_centered_streak = 0
    hold_steps = int(round(float(args.hold_seconds) / env.dt))
    completed_steps = 0
    feature_dim = int(capture_features(env).size)
    if capture_vector.shape != (feature_dim + 1,):
        raise ValueError(f"capture vector shape {capture_vector.shape} does not match {(feature_dim + 1,)}")
    for step in range(hold_steps):
        features = capture_features(env)
        action = float(np.tanh(features @ capture_vector[:-1] + capture_vector[-1]))
        _, _, terminated, truncated, info = env.step([action])
        absolute_omega = np.cumsum(np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64))
        absolute_rms = float(np.sqrt(np.mean(absolute_omega**2)))
        cart_abs = abs(float(info["x"]))
        centered = bool(info["is_upright"] and cart_abs <= 1.25)
        low_momentum = bool(
            centered
            and absolute_rms <= float(args.absolute_rate_threshold)
            and abs(float(env.data.qvel[0])) <= 0.50
        )
        centered_streak = centered_streak + 1 if centered else 0
        low_streak = low_streak + 1 if low_momentum else 0
        max_centered_streak = max(max_centered_streak, centered_streak)
        max_low_streak = max(max_low_streak, low_streak)
        max_cart_abs = max(max_cart_abs, cart_abs)
        completed_steps = step + 1
        if step % 5 == 0 or info["is_upright"]:
            trace.append(
                {
                    "step": int(step + 1),
                    "time_seconds": float(env.step_count * env.dt),
                    "action": action,
                    "max_abs_angle": float(info["max_abs_angle"]),
                    "absolute_angular_velocity_rms": absolute_rms,
                    "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                    "x": float(info["x"]),
                    "cart_velocity": float(env.data.qvel[0]),
                    "is_upright": bool(info["is_upright"]),
                    "centered": centered,
                    "low_momentum": low_momentum,
                    "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
                }
            )
        if terminated or truncated:
            break

    final_info = dict(info)
    required_hold_seconds = float(cfg["env"].get("success_sustain_seconds", 0.0))
    chain_success = bool(
        max_low_streak * env.dt >= required_hold_seconds
        and final_info.get("termination_reason") is None
    )
    env.close()
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Reset-free seven-link discovery planner followed by a learned capture actor; component evidence only.",
        "config_path": str(Path(args.config)),
        "tail_json": str(Path(args.tail_json)),
        "capture_json": str(Path(args.capture_json)),
        "handoff_state_json": args.handoff_state_json,
        "reset_free": True,
        "planner": {
            "source_controller": str(source_path),
            "tail_start_seconds": tail_start_seconds,
            "prefix_seconds": prefix_seconds,
            "suffix_seconds": suffix_seconds,
            "prefix_steps": int(prefix_controls.size),
            "suffix_steps": int(suffix_controls.size),
        },
        "endpoint_error": endpoint_error,
        "capture": {
            "hold_seconds_requested": float(args.hold_seconds),
            "hold_steps_completed": int(completed_steps),
            "max_centered_streak_seconds": float(max_centered_streak * env.dt),
            "max_absolute_low_momentum_streak_seconds": float(max_low_streak * env.dt),
            "max_cart_abs": float(max_cart_abs),
            "absolute_rate_threshold": float(args.absolute_rate_threshold),
            "success": chain_success,
            "required_hold_seconds": required_hold_seconds,
            "environment_success_flag": bool(final_info.get("success", False)),
            "termination_reason": final_info.get("termination_reason"),
            "final_info": final_info,
        },
        "trace": trace,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(result, Path(args.out))
    print(
        f"endpoint_qpos_err={endpoint_error['max_qpos_abs_error']:.3e} "
        f"endpoint_qvel_err={endpoint_error['max_qvel_abs_error']:.3e} "
        f"centered={max_centered_streak * env.dt:.3f}s "
        f"absolute_low={max_low_streak * env.dt:.3f}s "
        f"max_cart={max_cart_abs:.3f} success={chain_success}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
