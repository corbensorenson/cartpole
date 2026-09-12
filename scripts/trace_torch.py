#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from torch_runtime import prepare_runtime

prepare_runtime()

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.ppo_torch import ActorCritic, load_model, sample_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Trace a deterministic CPU PyTorch policy and retain handoff states")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--min-time", type=float, default=0.5)
    parser.add_argument("--max-angle", type=float, default=0.35)
    parser.add_argument("--max-hinge-rms", type=float, default=2.5)
    parser.add_argument("--max-cart-abs", type=float, default=2.75)
    parser.add_argument("--max-cart-velocity", type=float, default=2.0)
    parser.add_argument("--keep-states", type=int, default=256)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    if args.zero_noise:
        cfg["env"] = {
            **cfg["env"],
            "init_angle_noise": 0.0,
            "init_vel_noise": 0.0,
            "init_cart_noise": 0.0,
            "init_cart_vel_noise": 0.0,
        }
    probe = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    obs, _ = probe.reset()
    model = ActorCritic(
        obs_dim=int(obs.shape[0]),
        act_dim=int(probe.action_space.shape[0]),
        hidden_sizes=list(cfg["ppo"].get("hidden_sizes", [256, 256])),
        action_std_init=float(cfg["ppo"].get("action_std_init", 0.7)),
    )
    probe.close()
    load_model(model, args.checkpoint)

    episodes: list[dict[str, object]] = []
    states: list[dict[str, object]] = []
    for episode in range(int(args.episodes)):
        env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed + 10_000 + episode)
        obs, _ = env.reset()
        trace: list[dict[str, object]] = []
        done = False
        while not done:
            action, _, _ = sample_action(model, obs[None, :], deterministic=True)
            action_scalar = float(np.asarray(action, dtype=np.float64).reshape(-1)[0])
            obs, reward, terminated, truncated, info = env.step([action_scalar])
            time_seconds = float(env.step_count * env.dt)
            qpos = np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist()
            qvel = np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist()
            row = {
                "step": int(env.step_count),
                "time_seconds": time_seconds,
                "action": action_scalar,
                "reward": float(reward),
                "qpos": qpos,
                "qvel": qvel,
                "x": float(info["x"]),
                "cart_velocity": float(env.data.qvel[0]),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "capture_quality": float(info.get("capture_quality", 0.0)),
                "is_upright": bool(info.get("is_upright", False)),
                "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
                "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
                "energy_fraction": float(info.get("energy_fraction", 0.0)),
                "policy_action_norm": float(info.get("policy_action_norm", action_scalar)),
                "applied_action_norm": float(info.get("applied_action_norm", action_scalar)),
                "action_bias_norm": float(info.get("action_bias_norm", 0.0)),
                "residual_scale": float(info.get("residual_scale", 1.0)),
                "controller_mode": info.get("controller_mode"),
            }
            trace.append(row)
            if (
                time_seconds >= float(args.min_time)
                and float(row["max_abs_angle"]) <= float(args.max_angle)
                and float(row["hinge_velocity_rms"]) <= float(args.max_hinge_rms)
                and abs(float(row["x"])) <= float(args.max_cart_abs)
                and abs(float(row["cart_velocity"])) <= float(args.max_cart_velocity)
            ):
                states.append(
                    {
                        "episode": int(episode),
                        "step": int(env.step_count),
                        "time_seconds": time_seconds,
                        "qpos": qpos,
                        "qvel": qvel,
                        "max_abs_angle": float(row["max_abs_angle"]),
                        "hinge_velocity_rms": float(row["hinge_velocity_rms"]),
                        "x": float(row["x"]),
                        "cart_velocity": float(row["cart_velocity"]),
                        "capture_quality": float(row["capture_quality"]),
                        "energy_fraction": float(row["energy_fraction"]),
                    }
                )
            done = bool(terminated or truncated)
        episodes.append(
            {
                "episode": int(episode),
                "seed": int(args.seed + 10_000 + episode),
                "success": bool(info.get("success", False)),
                "termination_reason": info.get("termination_reason"),
                "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
                "max_cart_excursion": float(info.get("max_cart_excursion", 0.0)),
                "trace": trace,
            }
        )
        env.close()

    states.sort(
        key=lambda state: (
            float(state["max_abs_angle"]),
            float(state["hinge_velocity_rms"]),
            abs(float(state["cart_velocity"])),
            abs(float(state["x"])),
        )
    )
    payload = {
        "generated_at": utc_timestamp(),
        "source": "trace_torch_policy",
        "not_solution": not any(bool(row["success"]) for row in episodes),
        "config_sha256": data_sha256(cfg),
        "checkpoint": str(Path(args.checkpoint)),
        "progress": float(args.progress),
        "seed": int(args.seed),
        "episodes": episodes,
        "states": states[: max(0, int(args.keep_states))],
        "state_filters": {
            "min_time": float(args.min_time),
            "max_angle": float(args.max_angle),
            "max_hinge_rms": float(args.max_hinge_rms),
            "max_cart_abs": float(args.max_cart_abs),
            "max_cart_velocity": float(args.max_cart_velocity),
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(f"Wrote {args.out} episodes={len(episodes)} states={len(payload['states'])}")


if __name__ == "__main__":
    main()
