#!/usr/bin/env python
"""Build a finite-horizon LTV feedback route around an exact trajectory.

The route is intended as a reproducible warm-start artifact for the exact
MuJoCo FDDP tools. It does not change the feedforward trajectory and is not a
claim of swing-up success by itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform


def load_exact_route(
    path: Path,
    transition: MujocoTransition,
    n_links: int,
) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, list) or len(trajectory) < 2:
        raise ValueError("source artifact must contain at least two trajectory rows")
    controls = np.asarray([row["action"] for row in trajectory], dtype=np.float64)
    if controls.ndim != 1 or not np.all(np.isfinite(controls)):
        raise ValueError("source actions must be finite and one-dimensional")
    states = np.asarray(
        [
            transition.to_coordinates(data_state(transition.data))
        ]
        + [
            transition.to_coordinates(
                np.r_[
                    np.asarray(row["qpos"], dtype=np.float64),
                    np.asarray(row["qvel"], dtype=np.float64),
                ]
            )
            for row in trajectory
        ],
        dtype=np.float64,
    )
    expected = 2 * (int(n_links) + 1)
    if states.shape != (controls.size + 1, expected):
        raise ValueError("source trajectory state dimensions do not match config")
    if not np.all(np.isfinite(states)):
        raise ValueError("source trajectory states must be finite")
    return np.clip(controls, -1.0, 1.0), states


def ltv_feedback_gains(
    transition: MujocoTransition,
    states: np.ndarray,
    controls: np.ndarray,
    *,
    state_epsilon: float,
    action_epsilon: float,
    control_cost: float,
    position_cost: float,
    angle_cost: float,
    cart_velocity_cost: float,
    hinge_velocity_cost: float,
    terminal_scale: float,
) -> tuple[np.ndarray, dict[str, float]]:
    state_dim = states.shape[1]
    half = state_dim // 2
    q = np.zeros(state_dim, dtype=np.float64)
    q[0] = float(position_cost)
    q[1:half] = float(angle_cost)
    q[half] = float(cart_velocity_cost)
    q[half + 1 :] = float(hinge_velocity_cost)
    running_root = np.diag(np.sqrt(q))
    gains = np.zeros((controls.size, state_dim), dtype=np.float64)
    root = np.sqrt(float(terminal_scale)) * running_root
    spectral_max = 0.0
    gain_norm_max = 0.0
    for step in range(controls.size - 1, -1, -1):
        a, b = transition.linearize(
            states[step],
            float(controls[step]),
            state_epsilon=float(state_epsilon),
            action_epsilon=float(action_epsilon),
        )
        b = b[:, 0]
        root_b = root @ b
        root_a = root @ a
        s = float(control_cost) + float(root_b @ root_b)
        if not np.isfinite(s) or s <= 1.0e-12:
            raise ValueError(f"non-positive LTV control Hessian at step {step}: {s}")
        k = (root_b @ root_a) / s
        gains[step] = -k
        closed_loop = a - np.outer(b, k)
        stacked = np.vstack([running_root, np.sqrt(float(control_cost)) * k,
                             root @ closed_loop])
        # P = root.T @ root. QR propagates the Joseph-form cost factors
        # without forming an ill-conditioned P or subtracting large terms.
        _, root = np.linalg.qr(stacked, mode="reduced")
        if not np.all(np.isfinite(root)):
            raise ValueError(f"non-finite LTV Riccati factor at step {step}")
        spectral_max = max(spectral_max, float(np.max(np.abs(np.linalg.eigvals(closed_loop)))))
        gain_norm_max = max(gain_norm_max, float(np.linalg.norm(gains[step])))
    return gains, {
        "max_closed_loop_spectral_radius": float(spectral_max),
        "max_feedback_gain_norm": float(gain_norm_max),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--source", required=True)
    parser.add_argument("--optimization-route", action="store_true",
                        help="Use the source's fixed start, transform, and optimizer nodes; measure their dynamics gaps.")
    parser.add_argument("--continuous-angles", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--state-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--action-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--control-cost", type=float, default=0.5)
    parser.add_argument("--position-cost", type=float, default=1.0)
    parser.add_argument("--angle-cost", type=float, default=1.0)
    parser.add_argument("--cart-velocity-cost", type=float, default=1.0)
    parser.add_argument("--hinge-velocity-cost", type=float, default=1.0)
    parser.add_argument("--terminal-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    if min(
        args.state_epsilon,
        args.action_epsilon,
        args.control_cost,
        args.position_cost,
        args.angle_cost,
        args.cart_velocity_cost,
        args.hinge_velocity_cost,
        args.terminal_scale,
    ) <= 0.0:
        raise ValueError("linearization and LTV cost parameters must be positive")

    cfg = apply_overrides(load_config(args.config), args.override)
    source_path = Path(args.source).expanduser().resolve()
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "init_cart_noise": 0.0,
        "init_cart_noise_start": 0.0,
        "init_cart_noise_end": 0.0,
        "init_cart_vel_noise": 0.0,
        "init_cart_vel_noise_start": 0.0,
        "init_cart_vel_noise_end": 0.0,
        "init_angle_noise": 0.0,
        "init_angle_noise_start": 0.0,
        "init_angle_noise_end": 0.0,
        "init_vel_noise": 0.0,
        "init_vel_noise_start": 0.0,
        "init_vel_noise_end": 0.0,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    if args.optimization_route:
        try:
            from scripts.search_capture_sequence import fixed_state_cfg
        except ModuleNotFoundError:
            from search_capture_sequence import fixed_state_cfg
        cfg = fixed_state_cfg(cfg, source_payload["selected_state"], cfg["env"]["episode_seconds"])
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    env.reset(seed=args.seed)
    spec = load_config(args.spec)
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        int(cfg["env"]["n_links"]),
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    continuous_angles = args.continuous_angles or (
        args.optimization_route and source_payload["controller"].get("continuous_angles", False)
    )
    if args.optimization_route:
        # Legacy released routes used the benchmark transform implicitly.
        transform = np.asarray(source_payload["controller"].get("coordinate_transform", transform), dtype=np.float64)
    transition = MujocoTransition(env, coordinate_transform=transform, continuous_angles=continuous_angles)
    trajectory_diagnostics = None
    if args.optimization_route:
        try:
            from scripts.refine_ilqr_capture_chain import source_trajectory
            from scripts.search_fddp_capture import lift_nominal_trajectory, warm_start_diagnostics
        except ModuleNotFoundError:
            from refine_ilqr_capture_chain import source_trajectory
            from search_fddp_capture import lift_nominal_trajectory, warm_start_diagnostics
        controls, states, _ = source_trajectory(source_payload)
        start = transition.to_coordinates(data_state(env.data))
        if continuous_angles:
            states = lift_nominal_trajectory(transition, states, start)
        trajectory_diagnostics = warm_start_diagnostics(transition, start, controls, states)
    else:
        controls, states = load_exact_route(source_path, transition, env.n)
    started = time.monotonic()
    gains, ltv_metrics = ltv_feedback_gains(
        transition,
        states,
        controls,
        state_epsilon=args.state_epsilon,
        action_epsilon=args.action_epsilon,
        control_cost=args.control_cost,
        position_cost=args.position_cost,
        angle_cost=args.angle_cost,
        cart_velocity_cost=args.cart_velocity_cost,
        hinge_velocity_cost=args.hinge_velocity_cost,
        terminal_scale=args.terminal_scale,
    )
    elapsed = time.monotonic() - started
    env.close()

    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Finite-horizon LTV feedback diagnostic; source feasibility recorded separately; not P1 evidence.",
        "controller": {
            **(source_payload["controller"] if args.optimization_route else {}),
            "type": "finite_horizon_ltv_feedback_route",
            "controls": controls.astype(float).tolist(),
            "feedback_gains": gains.astype(float).tolist(),
            "periodic_coordinate_errors": False,
            "continuous_angles": bool(continuous_angles),
            "coordinate_transform": transform.tolist(),
            "state_epsilon": float(args.state_epsilon),
            "action_epsilon": float(args.action_epsilon),
            "control_cost": float(args.control_cost),
            "position_cost": float(args.position_cost),
            "angle_cost": float(args.angle_cost),
            "cart_velocity_cost": float(args.cart_velocity_cost),
            "hinge_velocity_cost": float(args.hinge_velocity_cost),
            "terminal_scale": float(args.terminal_scale),
            "ltv_metrics": ltv_metrics,
            "riccati_implementation": "square_root_qr_joseph_form",
            "source_trajectory_diagnostics": trajectory_diagnostics,
            "transform_source": "artifact" if "coordinate_transform" in source_payload.get("controller", {}) else "benchmark_spec",
        },
        "search": {
            "nominal_coordinate_states": states.astype(float).tolist(),
            "seconds": float(controls.size * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]),
            "wall_time_seconds": elapsed,
            "transition_evaluations": transition.evaluations,
        },
        "source": file_metadata(source_path),
        "evidence": {
            "config": {
                "path": str(Path(args.config)),
                "resolved_sha256": data_sha256(cfg),
                "progress": float(args.progress),
            },
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    if args.optimization_route:
        payload["selected_state"] = source_payload["selected_state"]
        payload["controller"].pop("solver_feedback_gains", None)
    dump_json(payload, args.out)
    print(
        f"steps={controls.size} seconds={payload['search']['seconds']:.3f} "
        f"max_gain_norm={ltv_metrics['max_feedback_gain_norm']:.3f} "
        f"max_rho={ltv_metrics['max_closed_loop_spectral_radius']:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
