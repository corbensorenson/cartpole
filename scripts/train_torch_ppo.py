#!/usr/bin/env python
from __future__ import annotations

import argparse

from torch_runtime import prepare_runtime

prepare_runtime()

from gcartpole.config import apply_overrides, load_config
from gcartpole.ppo_torch import train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train n-link gradient cart-pole with CPU PyTorch PPO")
    parser.add_argument("--config", required=True, help="YAML config path")
    parser.add_argument("--init-checkpoint", default=None, help="Optional PyTorch .pt checkpoint to initialize from")
    parser.add_argument("--override", action="append", default=[], help="Override config values, e.g. ppo.num_envs=16")
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    train(cfg, init_checkpoint=args.init_checkpoint)


if __name__ == "__main__":
    main()
