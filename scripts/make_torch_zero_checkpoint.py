#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from torch_runtime import prepare_runtime

prepare_runtime()

from gcartpole.config import apply_overrides, load_config, save_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_torch import ActorCritic, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a zero-mean CPU PyTorch actor checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    probe = NLinkCartPoleEnv(cfg, progress=0.0, seed=int(cfg["experiment"].get("seed", 0)))
    obs, _ = probe.reset()
    model = ActorCritic(
        obs_dim=int(obs.shape[0]),
        act_dim=int(probe.action_space.shape[0]),
        hidden_sizes=list(cfg["ppo"].get("hidden_sizes", [256, 256])),
        action_std_init=float(cfg["ppo"].get("action_std_init", 0.08)),
    )
    probe.close()
    with torch.no_grad():
        model.actor_out.weight.zero_()
        model.actor_out.bias.zero_()
    checkpoint = Path(args.checkpoint)
    save_model(model, checkpoint)
    save_config(cfg, checkpoint.parent.parent / "config.resolved.yaml")
    print(f"Wrote {checkpoint}")
    print(f"actor_output={tuple(model.actor_out.weight.shape)} zero")


if __name__ == "__main__":
    main()
