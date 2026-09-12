#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from torch_runtime import prepare_runtime

prepare_runtime()

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_torch import (
    ActorCritic,
    evaluate_policy,
    load_model,
    select_evaluation_state_indices,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a CPU PyTorch cart-pole policy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument(
        "--states-path",
        default=None,
        help="Evaluate from a saved state-list handoff instead of ordinary resets",
    )
    parser.add_argument(
        "--state-indices",
        default=None,
        help="Comma-separated state-list indices; otherwise choose a seeded no-replacement sample",
    )
    parser.add_argument("--out", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    reset_state_indices = None
    if args.states_path:
        cfg["env"]["init_mode"] = "state_list"
        cfg["env"]["init_states_path"] = str(args.states_path)
        if args.state_indices:
            reset_state_indices = [int(value.strip()) for value in args.state_indices.split(",") if value.strip()]
            if len(reset_state_indices) != args.episodes:
                raise ValueError("--state-indices must contain exactly --episodes values")
        else:
            reset_state_indices = select_evaluation_state_indices(args.states_path, args.episodes, args.seed)
    probe = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    obs, _ = probe.reset()
    obs_dim = int(obs.shape[0])
    act_dim = int(probe.action_space.shape[0])
    probe.close()
    model = ActorCritic(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_sizes=list(cfg["ppo"].get("hidden_sizes", [256, 256])),
        action_std_init=float(cfg["ppo"].get("action_std_init", 0.7)),
    )
    load_model(model, args.checkpoint)
    metrics = evaluate_policy(
        cfg,
        model,
        episodes=args.episodes,
        seed=args.seed,
        progress=args.progress,
        return_episodes=True,
        reset_state_indices=reset_state_indices,
    )
    if args.states_path:
        metrics["init_states_path"] = str(args.states_path)
        metrics["reset_state_indices"] = reset_state_indices
    print(metrics)
    if args.out:
        dump_json(metrics, Path(args.out))
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
