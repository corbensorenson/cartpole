#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from torch_runtime import prepare_runtime

prepare_runtime()

from gcartpole.config import apply_overrides, load_config, save_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_torch import ActorCritic, save_model
from make_lqr_checkpoint import (
    absolute_angle_cost,
    cart_target_bias,
    finite_difference_dynamics,
    state_gain_to_obs_weight,
)
from scipy.linalg import solve_discrete_are


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic CPU PyTorch checkpoint from a MuJoCo LQR linearization"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--control-cost", type=float, default=1000.0)
    parser.add_argument("--policy-scale", type=float, default=1.0)
    parser.add_argument("--cart-target", type=float, default=0.0)
    parser.add_argument("--cart-position-cost", type=float, default=0.1)
    parser.add_argument("--absolute-angle-cost", type=float, default=100.0)
    parser.add_argument("--cart-velocity-cost", type=float, default=0.1)
    parser.add_argument("--absolute-angular-velocity-cost", type=float, default=1.0)
    parser.add_argument("--relative-angle-cost", type=float, default=1.0)
    parser.add_argument("--relative-angular-velocity-cost", type=float, default=0.01)
    parser.add_argument("--cart-position-gain-add", type=float, default=0.0)
    parser.add_argument("--cart-velocity-gain-add", type=float, default=0.0)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    if list(cfg["ppo"].get("hidden_sizes", [])):
        raise ValueError("LQR checkpoint generation requires ppo.hidden_sizes: []")

    out_dir = Path(cfg["experiment"]["out_dir"])
    checkpoint_path = Path(args.checkpoint)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.resolved.yaml")

    a, b = finite_difference_dynamics(cfg, args.progress, args.fd_eps)
    n = int(cfg["env"]["n_links"])
    q_weights = {
        "cart_position": float(args.cart_position_cost),
        "absolute_angle": float(args.absolute_angle_cost),
        "cart_velocity": float(args.cart_velocity_cost),
        "absolute_angular_velocity": float(args.absolute_angular_velocity_cost),
        "relative_angle": float(args.relative_angle_cost),
        "relative_angular_velocity": float(args.relative_angular_velocity_cost),
    }
    q = absolute_angle_cost(n, q_weights)
    r = np.array([[float(args.control_cost)]], dtype=np.float64)
    p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(b.T @ p @ b + r, b.T @ p @ a).reshape(-1)
    gain[0] += float(args.cart_position_gain_add)
    gain[n + 1] += float(args.cart_velocity_gain_add)

    probe = NLinkCartPoleEnv(cfg, progress=args.progress, seed=int(cfg["experiment"].get("seed", 0)))
    obs, _ = probe.reset()
    obs_dim = int(obs.shape[0])
    act_dim = int(probe.action_space.shape[0])
    probe.close()

    action_std_init = float(cfg["ppo"].get("action_std_init", 0.01))
    model = ActorCritic(obs_dim, act_dim, [], action_std_init).to(torch.device("cpu"))
    obs_weight = state_gain_to_obs_weight(cfg, gain, obs_dim, float(args.policy_scale))
    with torch.no_grad():
        model.actor_out.weight.copy_(torch.as_tensor(obs_weight.reshape(1, -1), dtype=torch.float32))
        model.actor_out.bias.copy_(torch.as_tensor(
            [cart_target_bias(gain, float(args.policy_scale), float(args.cart_target))],
            dtype=torch.float32,
        ))
        model.critic_out.weight.zero_()
        model.critic_out.bias.zero_()
        model.log_std.fill_(float(np.log(action_std_init)))
    save_model(model, checkpoint_path)

    closed_loop_eigs = np.linalg.eigvals(a - b @ gain.reshape(1, -1))
    metadata = {
        "generated_at": utc_timestamp(),
        "method": "finite_difference_discrete_lqr",
        "backend": "torch_cpu",
        "scope": "uniform n-link near-upright stabilization; not swing-up",
        "checkpoint": str(checkpoint_path),
        "config_path": str(Path(args.config)),
        "config_resolved_sha256": data_sha256(cfg),
        "progress": float(args.progress),
        "fd_eps": float(args.fd_eps),
        "control_cost": float(args.control_cost),
        "policy_scale": float(args.policy_scale),
        "cart_target": float(args.cart_target),
        "q_weights": q_weights,
        "cart_position_gain_add": float(args.cart_position_gain_add),
        "cart_velocity_gain_add": float(args.cart_velocity_gain_add),
        "state_gain": gain.astype(float).tolist(),
        "nonzero_observation_weights": {
            str(i): float(v) for i, v in enumerate(obs_weight) if abs(float(v)) > 0.0
        },
        "open_loop_max_abs_eigenvalue": float(np.max(np.abs(np.linalg.eigvals(a)))),
        "closed_loop_max_abs_eigenvalue": float(np.max(np.abs(closed_loop_eigs))),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    meta_path = checkpoint_path.with_name(checkpoint_path.stem + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {meta_path}")
    print(f"closed_loop_max_abs_eigenvalue={metadata['closed_loop_max_abs_eigenvalue']:.9f}")
    print(json.dumps({"state_gain": metadata["state_gain"]}, indent=2))


if __name__ == "__main__":
    main()
