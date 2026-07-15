#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mujoco
import mlx.core as mx
import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.config import apply_overrides, dump_json, load_config, save_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, save_model


def finite_difference_dynamics(cfg: dict[str, Any], progress: float, eps: float) -> tuple[np.ndarray, np.ndarray]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=int(cfg["experiment"].get("seed", 0)))
    n = env.n
    d = n + 1
    state_dim = 2 * d

    def set_state(x: np.ndarray) -> None:
        env.data.qpos[:] = x[:d]
        env.data.qvel[:] = x[d:]
        env.data.ctrl[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

    def get_state() -> np.ndarray:
        return np.r_[env.data.qpos.copy(), env.data.qvel.copy()]

    def step_map(x: np.ndarray, action_norm: float) -> np.ndarray:
        set_state(x)
        env.data.ctrl[0] = float(np.clip(action_norm, -1.0, 1.0)) * env.force_limit
        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, env.data)
        return get_state()

    x0 = np.zeros(state_dim, dtype=np.float64)
    a = np.zeros((state_dim, state_dim), dtype=np.float64)
    b = np.zeros((state_dim, 1), dtype=np.float64)
    for i in range(state_dim):
        dx = np.zeros(state_dim, dtype=np.float64)
        dx[i] = eps
        a[:, i] = (step_map(x0 + dx, 0.0) - step_map(x0 - dx, 0.0)) / (2.0 * eps)
    b[:, 0] = (step_map(x0, eps) - step_map(x0, -eps)) / (2.0 * eps)
    env.close()
    return a, b


def absolute_angle_cost(n: int, weights: dict[str, float]) -> np.ndarray:
    d = n + 1
    state_dim = 2 * d
    c = np.zeros((2 * d, state_dim), dtype=np.float64)
    c[0, 0] = 1.0
    c[n + 1, d] = 1.0
    for i in range(n):
        c[1 + i, 1 : 2 + i] = 1.0
        c[n + 2 + i, d + 1 : d + 2 + i] = 1.0

    w = np.diag(
        [weights["cart_position"]]
        + [weights["absolute_angle"]] * n
        + [weights["cart_velocity"]]
        + [weights["absolute_angular_velocity"]] * n
    )
    q = c.T @ w @ c
    q += np.diag(
        [0.0]
        + [weights["relative_angle"]] * n
        + [0.0]
        + [weights["relative_angular_velocity"]] * n
    )
    return q


def state_gain_to_obs_weight(cfg: dict[str, Any], gain: np.ndarray, obs_dim: int, scale: float) -> np.ndarray:
    n = int(cfg["env"]["n_links"])
    d = n + 1
    rail_limit = float(cfg["env"]["rail_limit"])
    w = np.zeros(obs_dim, dtype=np.float32)
    w[0] = -gain[0] * rail_limit * scale
    w[1] = -gain[d] * scale
    rel_start = 2 + 2 * n
    vel_start = 2 + 3 * n
    w[rel_start : rel_start + n] = -gain[1 : 1 + n] * scale
    w[vel_start : vel_start + n] = -gain[d + 1 : d + 1 + n] * scale
    return w


def cart_target_bias(gain: np.ndarray, scale: float, cart_target: float) -> float:
    # action = -scale * K @ (state - [cart_target, 0, ...])
    return float(gain[0] * scale * cart_target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a deterministic MLX checkpoint from a MuJoCo LQR linearization")
    parser.add_argument("--config", default="configs/uniform6_near_upright_lqr.yaml")
    parser.add_argument("--checkpoint", default=None)
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
    hidden_sizes = list(cfg["ppo"].get("hidden_sizes", []))
    if hidden_sizes:
        raise ValueError("LQR checkpoint generation requires ppo.hidden_sizes: []")

    out_dir = Path(cfg["experiment"]["out_dir"])
    ckpt_path = Path(args.checkpoint) if args.checkpoint else out_dir / "checkpoints" / "best.safetensors"
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
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    probe.close()

    model = ActorCritic(obs_dim, act_dim, hidden_sizes, float(cfg["ppo"].get("action_std_init", 0.01)))
    obs_weight = state_gain_to_obs_weight(cfg, gain, obs_dim, float(args.policy_scale))
    params = {
        "actor_out": {
            "weight": mx.array(obs_weight.reshape(1, -1)),
            "bias": mx.array([cart_target_bias(gain, float(args.policy_scale), float(args.cart_target))], dtype=mx.float32),
        },
        "critic_out": {
            "weight": mx.zeros((1, obs_dim), dtype=mx.float32),
            "bias": mx.zeros((1,), dtype=mx.float32),
        },
        "log_std": mx.array([float(np.log(float(cfg["ppo"].get("action_std_init", 0.01))))], dtype=mx.float32),
    }
    model.update(params)
    mx.eval(model.parameters())
    save_model(model, ckpt_path)

    closed_loop_eigs = np.linalg.eigvals(a - b @ gain.reshape(1, -1))
    meta = {
        "generated_at": utc_timestamp(),
        "method": "finite_difference_discrete_lqr",
        "scope": "uniform 6-link near-upright stabilization; not swing-up and not the default PPO curriculum",
        "checkpoint": str(ckpt_path),
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
    dump_json(meta, ckpt_path.with_name("best.meta.json"))
    print(f"Wrote {ckpt_path}")
    print(f"closed_loop_max_abs_eigenvalue={meta['closed_loop_max_abs_eigenvalue']:.9f}")


if __name__ == "__main__":
    main()
