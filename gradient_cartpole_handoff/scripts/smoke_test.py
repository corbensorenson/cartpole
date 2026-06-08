#!/usr/bin/env python
from __future__ import annotations

import platform
import sys

import numpy as np

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.morphology import build_morphology


def main() -> None:
    print(f"Python: {sys.version.split()[0]}  arch={platform.processor()}  machine={platform.machine()}")
    cfg = load_config("configs/debug6_fast.yaml")
    morph = build_morphology(cfg["env"], cfg["morphology"], progress=0.0)
    print("lengths:", np.round(morph.lengths, 4))
    print("masses: ", np.round(morph.masses, 4))
    env = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
    obs, info = env.reset()
    print("obs_dim:", obs.shape[0], "initial max_abs_angle:", info["max_abs_angle"])
    for _ in range(5):
        obs, reward, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
    print("step reward:", reward, "done:", term or trunc, "x:", info["x"])
    frame = env.render_rgb(width=320, height=180)
    print("render frame:", frame.shape, frame.dtype)
    env.close()

    try:
        import mlx.core as mx
        print("MLX OK; default device:", mx.default_device())
    except Exception as exc:
        print("MLX import failed:", exc)
        raise


if __name__ == "__main__":
    main()
