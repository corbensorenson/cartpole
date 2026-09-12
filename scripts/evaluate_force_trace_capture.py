#!/usr/bin/env python
"""Evaluate a force-trace swing, optimized tail, and learned capture policy.

This is a discovery evaluator for the two-expert handoff.  It performs one
environment reset, applies the saved hanging-start force trace, applies the
optimized force tail, and then lets the learned capture policy run without a
reset.  The complete trace is retained so a promising result can be audited
before it is promoted to benchmark evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from torch_runtime import prepare_runtime

prepare_runtime()

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, text_sha256, utc_timestamp
from gcartpole.ppo_torch import ActorCritic, load_model, sample_action


def load_json(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def load_knots(payload: dict[str, Any], key: str) -> np.ndarray:
    record = payload.get(key)
    if not isinstance(record, dict) or not isinstance(record.get("knots"), list):
        raise ValueError(f"controller payload has no knot record at {key!r}")
    knots = np.asarray(record["knots"], dtype=np.float64)
    if knots.ndim != 1 or len(knots) < 2:
        raise ValueError("controller knots must be a one-dimensional list with at least two values")
    return knots


def force_from_knots(knots: np.ndarray, step: int, action_count: int) -> float:
    phase = float(np.clip(step, 0, action_count - 1)) / float(max(1, action_count - 1))
    return float(
        np.clip(
            np.interp(phase, np.linspace(0.0, 1.0, len(knots), dtype=np.float64), knots),
            -1.0,
            1.0,
        )
    )


def row_from_env(
    env: NLinkCartPoleEnv,
    *,
    step: int,
    stage: str,
    policy_action: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    relative_angles, absolute_angles = env._angles()
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "stage": stage,
        "policy_action": float(policy_action),
        "applied_action": float(info.get("applied_action_norm", policy_action)),
        "action_bias": float(info.get("action_bias_norm", 0.0)),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "x": float(info["x"]),
        "cart_velocity": float(env.data.qvel[0]),
        "relative_angles": relative_angles.astype(float).tolist(),
        "absolute_angles": absolute_angles.astype(float).tolist(),
        "max_abs_angle": float(info["max_abs_angle"]),
        "mean_abs_angle": float(info["mean_abs_angle"]),
        "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
        "absolute_angular_velocity_rms": float(info["absolute_angular_velocity_rms"]),
        "is_upright": bool(info["is_upright"]),
        "upright_streak_seconds": float(info["upright_streak_seconds"]),
        "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
        "centered_upright_streak_seconds": float(info["centered_upright_streak_seconds"]),
        "max_centered_upright_streak_seconds": float(info["max_centered_upright_streak_seconds"]),
        "low_momentum_upright_streak_seconds": float(info["low_momentum_upright_streak_seconds"]),
        "max_low_momentum_upright_streak_seconds": float(info["max_low_momentum_upright_streak_seconds"]),
        "capture_quality": float(info.get("capture_quality", 0.0)),
        "time_to_first_upright": info.get("time_to_first_upright"),
        "time_to_capture": info.get("time_to_capture"),
        "termination_reason": info.get("termination_reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an uninterrupted force-prefix/tail/capture chain")
    parser.add_argument("--config", required=True, help="capture-policy config with matching observation features")
    parser.add_argument("--swing-controller-json", required=True)
    parser.add_argument("--swing-record-key", default="best")
    parser.add_argument("--tail-json", required=True)
    parser.add_argument("--tail-record-key", default="best")
    parser.add_argument("--capture-checkpoint", required=True)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--tail-start-seconds", type=float, default=None)
    parser.add_argument("--tail-seconds", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    swing_payload = load_json(args.swing_controller_json)
    tail_payload = load_json(args.tail_json)
    swing_knots = load_knots(swing_payload, args.swing_record_key)
    tail_knots = load_knots(tail_payload, args.tail_record_key)
    swing_seconds = float(swing_payload.get("search", {}).get("seconds", 0.0))
    if swing_seconds <= 0.0:
        raise ValueError("swing controller payload has no positive search duration")
    tail_start = float(
        args.tail_start_seconds
        if args.tail_start_seconds is not None
        else tail_payload.get("tail_start_seconds", 0.0)
    )
    tail_seconds = float(
        args.tail_seconds
        if args.tail_seconds is not None
        else tail_payload.get("tail_horizon_seconds", 0.0)
    )
    if tail_start <= 0.0 or tail_seconds <= 0.0 or tail_start + tail_seconds >= args.seconds:
        raise ValueError("tail start/duration must be positive and fit inside the evaluation horizon")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "episode_seconds": max(float(cfg["env"].get("episode_seconds", 0.0)), float(args.seconds)),
    }
    for noise_key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        cfg["env"][noise_key] = 0.0
        cfg["env"][f"{noise_key}_start"] = 0.0
        cfg["env"][f"{noise_key}_end"] = 0.0

    residual_cfg = cfg["env"].get("action_lqr_residual")
    residual_enabled = bool(isinstance(residual_cfg, dict) and residual_cfg.get("enabled", False))
    if residual_enabled:
        residual_cfg["enabled"] = False

    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    observation, reset_info = env.reset(seed=args.seed)
    probe_obs_dim = int(observation.shape[0])
    hidden_sizes = [int(value) for value in cfg.get("ppo", {}).get("hidden_sizes", [256, 256])]
    model = ActorCritic(
        obs_dim=probe_obs_dim,
        act_dim=int(env.action_space.shape[0]),
        hidden_sizes=hidden_sizes,
        action_std_init=float(cfg.get("ppo", {}).get("action_std_init", 0.02)),
    )
    load_model(model, args.capture_checkpoint)
    model.eval()

    swing_steps = max(2, int(round(swing_seconds / env.dt)))
    tail_start_step = int(round(tail_start / env.dt))
    tail_steps = max(2, int(round(tail_seconds / env.dt)))
    total_steps = min(env.max_steps, int(round(args.seconds / env.dt)))
    if tail_start_step + tail_steps >= total_steps:
        raise ValueError("tail does not leave room for capture-policy evaluation")

    rows: list[dict[str, Any]] = []
    stage_events: list[dict[str, Any]] = [{"time_seconds": 0.0, "stage": "swing", "reason": "single_reset"}]
    done_events: list[dict[str, Any]] = []
    total_return = 0.0
    max_cart_abs = abs(float(reset_info["x"]))
    max_action_abs = 0.0
    capture_started = False
    final_info: dict[str, Any] = dict(reset_info)
    for step in range(total_steps):
        if step < tail_start_step:
            stage = "swing_force"
            policy_action = force_from_knots(swing_knots, step, swing_steps)
        elif step < tail_start_step + tail_steps:
            stage = "tail_force"
            policy_action = force_from_knots(tail_knots, step - tail_start_step, tail_steps)
        else:
            stage = "capture_policy"
            if residual_enabled and not capture_started:
                residual_cfg["enabled"] = True
                capture_started = True
                observation = env._get_obs()
                stage_events.append(
                    {
                        "time_seconds": float(step * env.dt),
                        "stage": stage,
                        "reason": "tail_complete",
                    }
                )
            policy_action_array, _, _ = sample_action(
                model,
                observation[None, :],
                deterministic=True,
            )
            policy_action = float(policy_action_array[0, 0])

        observation, reward, terminated, truncated, info = env.step([policy_action])
        total_return += float(reward)
        final_info = dict(info)
        row = row_from_env(
            env,
            step=step + 1,
            stage=stage,
            policy_action=policy_action,
            info=info,
        )
        rows.append(row)
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        max_action_abs = max(max_action_abs, abs(float(info.get("applied_action_norm", policy_action))))
        if terminated or truncated:
            done_events.append(
                {
                    "step": int(step + 1),
                    "time_seconds": float((step + 1) * env.dt),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "termination_reason": info.get("termination_reason"),
                    "success": bool(info.get("success", False)),
                }
            )
            break

    success = bool(final_info.get("success", False))
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "One-reset exact serial replay of force swing, optimized tail, and learned capture policy; discovery evidence pending held-out validation.",
        "reset_count": 1,
        "success": success,
        "failure_category": None if success else ("rail_violation" if final_info.get("termination_reason") == "rail_violation" else "capture_loss"),
        "source_controllers": {
            "swing": str(Path(args.swing_controller_json)),
            "swing_record_key": args.swing_record_key,
            "tail": str(Path(args.tail_json)),
            "tail_record_key": args.tail_record_key,
            "capture_checkpoint": str(Path(args.capture_checkpoint)),
        },
        "resolved_config_sha256": data_sha256(cfg),
        "generated_xml_sha256": text_sha256(env.xml),
        "replay": {
            "seed": int(args.seed),
            "progress": float(args.progress),
            "seconds_requested": float(args.seconds),
            "simulated_seconds": float(len(rows) * env.dt),
            "total_steps": int(total_steps),
            "swing_seconds": swing_seconds,
            "tail_start_seconds": tail_start,
            "tail_seconds": tail_seconds,
            "capture_start_seconds": float((tail_start_step + tail_steps) * env.dt),
            "dt_seconds": float(env.dt),
            "frame_skip": int(env.frame_skip),
            "rail_limit": float(env.rail_limit),
            "force_limit": float(env.force_limit),
            "hidden_sizes": hidden_sizes,
        },
        "stage_events": stage_events,
        "episode_return": float(total_return),
        "max_cart_abs": float(max_cart_abs),
        "max_action_abs": float(max_action_abs),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "max_centered_upright_streak_seconds": float(final_info.get("max_centered_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "time_to_capture": final_info.get("time_to_capture"),
        "termination_reason": final_info.get("termination_reason"),
        "done_events": done_events,
        "final_info": final_info,
        "trace": rows,
        "trajectory_sha256": data_sha256(rows),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(result, Path(args.out))
    print(
        f"Wrote {args.out} success={success} reset_count=1 steps={len(rows)} "
        f"max_streak={result['max_upright_streak_seconds']:.3f}s max_rail={max_cart_abs:.3f} "
        f"termination={result['termination_reason']}"
    )


if __name__ == "__main__":
    main()
