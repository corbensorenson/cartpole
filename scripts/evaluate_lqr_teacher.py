#!/usr/bin/env python
"""Audit finite-difference LQR teachers on the nonlinear cart-pole plant."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.generalized_modes import chain_normal_modes
from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics


def coordinate_transform(
    cfg: dict[str, Any], *, progress: float, state_coordinates: str
) -> np.ndarray:
    n = int(cfg["env"]["n_links"])
    d = n + 1
    transform = np.eye(2 * d, dtype=np.float64)
    if state_coordinates == "relative":
        return transform
    if state_coordinates == "absolute":
        serial = np.eye(d, dtype=np.float64)
        serial[1:, 1:] = np.tril(np.ones((n, n), dtype=np.float64))
        transform[:d, :d] = serial
        transform[d:, d:] = serial
        return transform
    if state_coordinates == "modal":
        env = NLinkCartPoleEnv(cfg, progress=progress, seed=0)
        modes = chain_normal_modes(env, equilibrium="upright")
        env.close()
        modal = np.eye(d, dtype=np.float64)
        modal[1:, 1:] = modes.relative_shapes.T @ modes.joint_mass_matrix
        transform[:d, :d] = modal
        transform[d:, d:] = modal
        return transform
    raise ValueError("--state-coordinates must be relative, absolute, or modal")


def build_gain(
    cfg: dict[str, Any],
    *,
    progress: float,
    fd_eps: float,
    control_cost: float,
    cart_position_cost: float,
    absolute_angle_cost_value: float,
    cart_velocity_cost: float,
    absolute_angular_velocity_cost: float,
    relative_angle_cost: float,
    relative_angular_velocity_cost: float,
    state_coordinates: str,
) -> tuple[np.ndarray, float, float, np.ndarray]:
    a, b = finite_difference_dynamics(cfg, progress, fd_eps)
    n = int(cfg["env"]["n_links"])
    q_relative = absolute_angle_cost(
        n,
        {
            "cart_position": cart_position_cost,
            "absolute_angle": absolute_angle_cost_value,
            "cart_velocity": cart_velocity_cost,
            "absolute_angular_velocity": absolute_angular_velocity_cost,
            "relative_angle": relative_angle_cost,
            "relative_angular_velocity": relative_angular_velocity_cost,
        },
    )
    transform = coordinate_transform(
        cfg, progress=progress, state_coordinates=state_coordinates
    )
    inverse = np.linalg.inv(transform)
    a_coordinates = transform @ a @ inverse
    b_coordinates = transform @ b
    q = inverse.T @ q_relative @ inverse
    r = np.array([[float(control_cost)]], dtype=np.float64)
    p = solve_discrete_are(a_coordinates, b_coordinates, q, r)
    gain = np.linalg.solve(b_coordinates.T @ p @ b_coordinates + r, b_coordinates.T @ p @ a_coordinates).reshape(-1)
    eigs = np.linalg.eigvals(a_coordinates - b_coordinates @ gain.reshape(1, -1))
    return (
        gain,
        float(np.max(np.abs(eigs))),
        float(np.max(np.abs(np.linalg.eigvals(a_coordinates)))),
        transform,
    )


def evaluate_gain(
    cfg: dict[str, Any],
    *,
    progress: float,
    gain: np.ndarray,
    gain_scale: float,
    episodes: int,
    seed: int,
    state_coordinates: str,
    state_transform: np.ndarray,
) -> dict[str, Any]:
    residual_cfg = cfg["env"].setdefault("action_lqr_residual", {})
    residual_cfg["enabled"] = True
    residual_cfg["state_gain"] = (gain * float(gain_scale)).astype(float).tolist()
    residual_cfg["state_coordinates"] = state_coordinates
    residual_cfg["state_transform"] = state_transform.astype(float).tolist()
    # The deterministic teacher is the residual bias; a zero policy isolates it.
    residual_cfg["residual_scale"] = 0.0

    episode_results: list[dict[str, Any]] = []
    saturation_steps = 0
    total_steps = 0
    for episode in range(int(episodes)):
        env = NLinkCartPoleEnv(cfg, progress=progress, seed=int(seed) + episode)
        obs, _ = env.reset()
        del obs
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(np.zeros(1, dtype=np.float32))
            total_steps += 1
            action = float(env.last_action_bias_norm)
            saturation_steps += int(abs(action) >= 0.999999)
            done = bool(terminated or truncated)
        episode_results.append(
            {
                "episode": episode,
                "success": bool(info.get("success", False)),
                "termination_reason": info.get("termination_reason"),
                "max_cart_excursion": float(info.get("max_cart_excursion", 0.0)),
                "max_abs_angle": float(info.get("max_abs_angle", 0.0)),
                "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
                "max_low_momentum_upright_streak_seconds": float(
                    info.get("max_low_momentum_upright_streak_seconds", 0.0)
                ),
                "max_capture_quality": float(info.get("max_capture_quality", 0.0)),
            }
        )
        env.close()

    successes = [bool(item["success"]) for item in episode_results]
    max_cart = [float(item["max_cart_excursion"]) for item in episode_results]
    max_streak = [float(item["max_upright_streak_seconds"]) for item in episode_results]
    return {
        "episodes": int(episodes),
        "successes": int(sum(successes)),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "mean_max_cart_excursion": float(np.mean(max_cart)) if max_cart else 0.0,
        "max_max_cart_excursion": float(np.max(max_cart)) if max_cart else 0.0,
        "mean_max_upright_streak_seconds": float(np.mean(max_streak)) if max_streak else 0.0,
        "saturation_fraction": float(saturation_steps / max(1, total_steps)),
        "episode_results": episode_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--progress", type=float, default=0.0025)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--control-cost", type=float, action="append", required=True)
    parser.add_argument("--gain-scale", type=float, action="append", default=[1.0])
    parser.add_argument("--cart-position-cost", type=float, default=0.1)
    parser.add_argument("--absolute-angle-cost", type=float, default=100.0)
    parser.add_argument("--cart-velocity-cost", type=float, default=0.1)
    parser.add_argument("--absolute-angular-velocity-cost", type=float, default=1.0)
    parser.add_argument("--relative-angle-cost", type=float, default=1.0)
    parser.add_argument("--relative-angular-velocity-cost", type=float, default=0.01)
    parser.add_argument(
        "--state-coordinates",
        choices=["relative", "absolute", "modal"],
        default="relative",
    )
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_cfg = apply_overrides(load_config(args.config), args.override)
    results: list[dict[str, Any]] = []
    gain_cache: dict[float, tuple[np.ndarray, float, float, np.ndarray]] = {}
    for control_cost in args.control_cost:
        if float(control_cost) not in gain_cache:
            gain_cache[float(control_cost)] = build_gain(
                base_cfg,
                progress=args.progress,
                fd_eps=args.fd_eps,
                control_cost=control_cost,
                cart_position_cost=args.cart_position_cost,
                absolute_angle_cost_value=args.absolute_angle_cost,
                cart_velocity_cost=args.cart_velocity_cost,
                absolute_angular_velocity_cost=args.absolute_angular_velocity_cost,
                relative_angle_cost=args.relative_angle_cost,
                relative_angular_velocity_cost=args.relative_angular_velocity_cost,
                state_coordinates=args.state_coordinates,
            )
        gain, closed_loop_radius, open_loop_radius, state_transform = gain_cache[float(control_cost)]
        for gain_scale in args.gain_scale:
            metrics = evaluate_gain(
                base_cfg,
                progress=args.progress,
                gain=gain,
                gain_scale=gain_scale,
                episodes=args.episodes,
                seed=args.seed,
                state_coordinates=args.state_coordinates,
                state_transform=state_transform,
            )
            results.append(
                {
                    "control_cost": float(control_cost),
                    "gain_scale": float(gain_scale),
                    "closed_loop_max_abs_eigenvalue": closed_loop_radius,
                    "open_loop_max_abs_eigenvalue": open_loop_radius,
                    "gain_max_abs": float(np.max(np.abs(gain * gain_scale))),
                    "gain_l2": float(np.linalg.norm(gain * gain_scale)),
                    **metrics,
                }
            )

    payload = {
        "method": "nonlinear_lqr_teacher_audit",
        "config": str(Path(args.config)),
        "progress": float(args.progress),
        "state_coordinates": args.state_coordinates,
        "state_transform": state_transform.astype(float).tolist(),
        "seed": int(args.seed),
        "teacher_parameters": {
            "fd_eps": float(args.fd_eps),
            "cart_position_cost": float(args.cart_position_cost),
            "absolute_angle_cost": float(args.absolute_angle_cost),
            "cart_velocity_cost": float(args.cart_velocity_cost),
            "absolute_angular_velocity_cost": float(args.absolute_angular_velocity_cost),
            "relative_angle_cost": float(args.relative_angle_cost),
            "relative_angular_velocity_cost": float(args.relative_angular_velocity_cost),
        },
        "results": results,
    }
    dump_json(payload, Path(args.out))
    print(f"Wrote {args.out}")
    for result in results:
        print(
            "cost={control_cost:g} scale={gain_scale:g} success={successes}/{episodes} "
            "streak={mean_max_upright_streak_seconds:.3f}s rail={max_max_cart_excursion:.3f} "
            "sat={saturation_fraction:.3f}".format(**result)
        )


if __name__ == "__main__":
    main()
