"""Experimental PufferLib adapter placeholder.

This file intentionally avoids importing PufferLib at module import time because the
4.0 API should be checked on the target Mac after installation.

The environment itself is Gymnasium-compatible:

    from gcartpole.env import NLinkCartPoleEnv
    env = NLinkCartPoleEnv(cfg)

Most PufferLib integrations start from wrapping a Gym/Gymnasium env through its
emulation/vectorization layer. After installing PufferLib 4.0 locally, inspect the
available API and wire it here.
"""

from __future__ import annotations

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv


def make_env(config_path: str = "configs/gradient6_curriculum.yaml", progress: float = 0.0, seed: int = 0):
    cfg = load_config(config_path)
    return NLinkCartPoleEnv(cfg, progress=progress, seed=seed)


if __name__ == "__main__":
    env = make_env()
    obs, info = env.reset()
    print("Gymnasium-compatible env ready")
    print("obs shape:", obs.shape)
    print("action space:", env.action_space)
    env.close()
