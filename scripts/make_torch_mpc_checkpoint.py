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
from evaluate_linear_mpc_upright import LinearMPC
from make_lqr_checkpoint import (
    absolute_angle_cost,
    finite_difference_dynamics,
    state_gain_to_obs_weight,
)

from scipy.linalg import solve_discrete_are


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a deterministic CPU PyTorch checkpoint from condensed upright linear MPC"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--progress", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--control-cost", type=float, default=1000.0)
    parser.add_argument("--terminal-q-factor", type=float, default=100.0)
    parser.add_argument("--policy-scale", type=float, default=1.0)
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    if list(cfg["ppo"].get("hidden_sizes", [])):
        raise ValueError("MPC checkpoint generation requires ppo.hidden_sizes: []")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")

    checkpoint_path = Path(args.checkpoint)
    out_dir = Path(cfg["experiment"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.resolved.yaml")

    a, b = finite_difference_dynamics(cfg, args.progress, args.fd_eps)
    n = int(cfg["env"]["n_links"])
    q = absolute_angle_cost(
        n,
        {
            "cart_position": 0.1,
            "absolute_angle": 100.0,
            "cart_velocity": 0.1,
            "absolute_angular_velocity": 1.0,
            "relative_angle": 1.0,
            "relative_angular_velocity": 0.01,
        },
    )
    r = np.asarray([[float(args.control_cost)]], dtype=np.float64)
    riccati = solve_discrete_are(a, b, q, r)
    cost_scale = float(np.max(np.diag(riccati)))
    q_scaled = q / cost_scale
    r_scaled = r / cost_scale
    terminal = float(args.terminal_q_factor) * q / cost_scale
    controller = LinearMPC(
        a,
        b,
        q_scaled,
        r_scaled,
        terminal,
        horizon=args.horizon,
        rail_constraint=float(cfg["env"]["rail_limit"]),
        fallback_gain=np.zeros(a.shape[0], dtype=np.float64),
        fallback_scale=0.0,
    )
    # The condensed MPC law is linear before action clipping. Its first action
    # is -Kx, and tanh in ActorCritic supplies the same bounded action shape.
    state_gain = np.linalg.solve(controller.hessian, controller.linear_cost_map)[0]

    probe = NLinkCartPoleEnv(cfg, progress=args.progress, seed=int(cfg["experiment"].get("seed", 0)))
    obs, _ = probe.reset()
    obs_dim = int(obs.shape[0])
    act_dim = int(probe.action_space.shape[0])
    probe.close()

    action_std_init = float(cfg["ppo"].get("action_std_init", 0.01))
    model = ActorCritic(obs_dim, act_dim, [], action_std_init).to(torch.device("cpu"))
    obs_weight = state_gain_to_obs_weight(
        cfg,
        state_gain,
        obs_dim,
        float(args.policy_scale),
    )
    with torch.no_grad():
        model.actor_out.weight.copy_(torch.as_tensor(obs_weight.reshape(1, -1), dtype=torch.float32))
        model.actor_out.bias.zero_()
        model.critic_out.weight.zero_()
        model.critic_out.bias.zero_()
        model.log_std.fill_(float(np.log(action_std_init)))
    save_model(model, checkpoint_path)

    metadata = {
        "generated_at": utc_timestamp(),
        "method": "condensed_linear_mpc_first_action",
        "backend": "torch_cpu",
        "scope": "upright maintenance warm start; not swing-up and not canonical evidence",
        "checkpoint": str(checkpoint_path),
        "config_path": str(Path(args.config)),
        "config_resolved_sha256": data_sha256(cfg),
        "progress": float(args.progress),
        "horizon_steps": int(args.horizon),
        "horizon_seconds": float(args.horizon * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]),
        "control_cost": float(args.control_cost),
        "terminal_q_factor": float(args.terminal_q_factor),
        "policy_scale": float(args.policy_scale),
        "cost_normalization": cost_scale,
        "state_gain": state_gain.astype(float).tolist(),
        "nonzero_observation_weights": {
            str(i): float(v) for i, v in enumerate(obs_weight) if abs(float(v)) > 0.0
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    meta_path = checkpoint_path.with_name(checkpoint_path.stem + ".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {checkpoint_path}")
    print(f"Wrote {meta_path}")
    print(json.dumps({"state_gain": metadata["state_gain"]}, indent=2))


if __name__ == "__main__":
    main()
