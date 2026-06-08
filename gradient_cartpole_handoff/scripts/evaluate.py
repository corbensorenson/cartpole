#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, evaluate_policy, load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained MLX PPO checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--progress", type=float, default=1.0, help="1.0 = final uniform morphology")
    parser.add_argument("--out", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    probe = NLinkCartPoleEnv(cfg, progress=args.progress, seed=int(cfg["experiment"].get("seed", 0)))
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    probe.close()

    ppo = cfg["ppo"]
    model = ActorCritic(obs_dim, act_dim, list(ppo.get("hidden_sizes", [256, 256])), float(ppo.get("action_std_init", 0.7)))
    mx.eval(model.parameters())
    load_model(model, args.checkpoint)
    eval_seed = int(cfg["experiment"].get("seed", 0)) + 4444
    metrics = evaluate_policy(
        cfg,
        model,
        episodes=args.episodes,
        seed=eval_seed,
        progress=args.progress,
        return_episodes=True,
    )
    metrics["evidence"] = {
        "generated_at": utc_timestamp(),
        "deterministic_policy": True,
        "progress": float(args.progress),
        "eval_seed": eval_seed,
        "config": {
            "path": str(Path(args.config)),
            "resolved_sha256": data_sha256(cfg),
            "overrides": list(args.override),
        },
        "checkpoint": file_metadata(args.checkpoint),
        "environment": {
            "n_links": int(cfg["env"]["n_links"]),
            "init_mode": str(cfg["env"].get("init_mode", "upright")),
            "init_angle_noise": float(cfg["env"].get("init_angle_noise", 0.02)),
            "init_vel_noise": float(cfg["env"].get("init_vel_noise", 0.01)),
            "episode_seconds": float(cfg["env"]["episode_seconds"]),
            "max_steps": int(probe.max_steps),
            "force_limit": float(cfg["env"]["force_limit"]),
            "rail_limit": float(probe.rail_limit),
            "configured_rail_limit": float(cfg["env"]["rail_limit"]),
            "rail_limit_start": float(cfg["env"].get("rail_limit_start", cfg["env"]["rail_limit"])),
            "rail_limit_end": float(cfg["env"].get("rail_limit_end", cfg["env"]["rail_limit"])),
            "terminate_abs_angle": cfg["env"].get("terminate_abs_angle", 1.25),
            "success_upright_threshold": float(cfg["env"].get("success_upright_threshold", cfg["env"].get("reward", {}).get("upright_threshold", 0.10))),
            "success_sustain_seconds": float(cfg["env"].get("success_sustain_seconds", 0.0)),
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    print(metrics)
    if args.out:
        dump_json(metrics, Path(args.out))


if __name__ == "__main__":
    main()
