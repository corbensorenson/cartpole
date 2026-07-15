#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.constrained_shooting import optimize_constrained_shooting
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import (
    MujocoTransition,
    QuadraticTrajectoryCost,
    add_terminal_cart_weights,
    data_state,
    optimize_ilqr,
)
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.refine_ilqr_capture_chain import source_trajectory
    from scripts.search_capture_sequence import fixed_state_cfg, load_state
    from scripts.search_ilqr_capture import execute_controller
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from refine_ilqr_capture_chain import source_trajectory
    from search_capture_sequence import fixed_state_cfg, load_state
    from search_ilqr_capture import execute_controller
    from search_swingup_capture import lqr_gain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search an exact-MuJoCo capture trajectory with hard terminal constraints"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--state-json", default="runs/p1_capture_envelope/validation.json"
    )
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--initial-controller", required=True)
    parser.add_argument("--seed", type=int, default=68001)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--control-weight", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--terminal-state-weight", type=float, default=1.0)
    parser.add_argument("--terminal-cart-weight", type=float, default=10.0)
    parser.add_argument("--terminal-cart-velocity-weight", type=float, default=10.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.25)
    parser.add_argument("--handoff-angle-abs", type=float, default=0.15)
    parser.add_argument("--handoff-cart-velocity-abs", type=float, default=0.5)
    parser.add_argument("--handoff-hinge-velocity-rms", type=float, default=0.75)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if (
        min(
            args.iterations,
            args.lqr_scale,
            args.control_weight,
            args.terminal_weight,
            args.handoff_lyapunov,
            args.handoff_cart_abs,
            args.handoff_angle_abs,
            args.handoff_cart_velocity_abs,
            args.handoff_hinge_velocity_rms,
        )
        <= 0.0
        or min(
            args.terminal_state_weight,
            args.terminal_cart_weight,
            args.terminal_cart_velocity_weight,
        )
        < 0.0
    ):
        raise ValueError("weights, thresholds, and iterations must be positive")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    selected_state, state_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(
        base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"])
    )
    spec = load_config(args.spec)
    distribution = spec["distribution"]
    n_links = int(cfg["env"]["n_links"])
    scales = StateScales(
        float(distribution["cart_position_abs_max"]),
        float(distribution["absolute_link_angle_abs_max"]),
        float(distribution["cart_velocity_abs_max"]),
        float(distribution["hinge_velocity_rms_max"]),
    )
    transform = dimensionless_absolute_transform(n_links, scales)
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix, input_matrix, gain, transform, feedback_scale=args.lqr_scale
    )

    initial_path = Path(args.initial_controller)
    initial_payload = json.loads(initial_path.read_text(encoding="utf-8"))
    initial_controls, _, _ = source_trajectory(initial_payload)
    terminal_metric = (
        args.terminal_weight * lyapunov / args.handoff_lyapunov
        + args.terminal_state_weight * np.eye(transform.shape[0], dtype=np.float64)
    )
    terminal_metric = add_terminal_cart_weights(
        terminal_metric,
        n_links,
        cart_weight=args.terminal_cart_weight,
        cart_velocity_weight=args.terminal_cart_velocity_weight,
    )

    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    env.reset(seed=args.seed)
    transition = MujocoTransition(env, coordinate_transform=transform)
    initial_state = transition.to_coordinates(data_state(env.data))
    started = time.time()
    search = optimize_constrained_shooting(
        transition,
        initial_state,
        initial_controls,
        terminal_metric,
        lyapunov,
        control_weight=args.control_weight,
        rail_limit=float(env.rail_limit * transform[0, 0]),
        handoff_lyapunov=args.handoff_lyapunov,
        handoff_cart_ratio=args.handoff_cart_abs / scales.cart_position,
        handoff_angle_ratio=args.handoff_angle_abs / scales.absolute_angle,
        handoff_cart_velocity_ratio=(
            args.handoff_cart_velocity_abs / scales.cart_velocity
        ),
        handoff_hinge_velocity_ratio=(
            args.handoff_hinge_velocity_rms / scales.hinge_velocity
        ),
        max_iterations=args.iterations,
    )
    search_seconds = time.time() - started
    tracking = optimize_ilqr(
        transition,
        initial_state,
        search.controls,
        QuadraticTrajectoryCost(
            stage_state=0.1 * np.eye(transform.shape[0], dtype=np.float64),
            terminal_state=terminal_metric,
            control=args.control_weight,
            rail_soft_limit=float(2.4 * transform[0, 0]),
            rail_limit=float(env.rail_limit * transform[0, 0]),
            rail_weight=100_000_000.0,
            wrap_angles=False,
        ),
        max_iterations=0,
    )
    env.close()
    nominal_values = np.maximum(
        0.0,
        np.einsum("ij,jk,ik->i", search.states, lyapunov, search.states),
    )
    result = execute_controller(
        cfg,
        seed=args.seed,
        controls=search.controls,
        nominal_states=search.states,
        feedback_gains=tracking.feedback_gains,
        gain=gain,
        lqr_scale=args.lqr_scale,
        transform=transform,
        lyapunov=lyapunov,
        handoff_lyapunov=args.handoff_lyapunov,
        handoff_cart_abs=args.handoff_cart_abs,
        handoff_angle_abs=args.handoff_angle_abs,
        handoff_cart_velocity_abs=args.handoff_cart_velocity_abs,
        handoff_hinge_velocity_rms=args.handoff_hinge_velocity_rms,
        tracking_mode="constrained_single_shooting_tracking",
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Single-state hard-constrained exact-MuJoCo shooting diagnostic; not P1 evidence.",
        "state_index": int(state_index),
        "selected_state": selected_state,
        "seed": int(args.seed),
        "controller": {
            "type": "exact_mujoco_constrained_single_shooting_then_lqr",
            "initial_controller": file_metadata(initial_path),
            "horizon_steps": int(search.controls.size),
            "horizon_seconds": float(
                search.controls.size * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]
            ),
            "iterations": int(args.iterations),
            "lqr_scale": float(args.lqr_scale),
            "control_weight": float(args.control_weight),
            "terminal_weight": float(args.terminal_weight),
            "terminal_state_weight": float(args.terminal_state_weight),
            "terminal_cart_weight": float(args.terminal_cart_weight),
            "terminal_cart_velocity_weight": float(args.terminal_cart_velocity_weight),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "handoff_angle_abs": float(args.handoff_angle_abs),
            "handoff_cart_velocity_abs": float(args.handoff_cart_velocity_abs),
            "handoff_hinge_velocity_rms": float(args.handoff_hinge_velocity_rms),
            "controls": search.controls.astype(float).tolist(),
            "feedback_gains": tracking.feedback_gains.astype(float).tolist(),
        },
        "search": {
            "success": bool(search.success),
            "status": int(search.status),
            "message": search.message,
            "cost": float(search.cost),
            "minimum_constraint_margin": float(search.minimum_constraint_margin),
            "maximum_constraint_jacobian_abs": float(
                search.maximum_constraint_jacobian_abs
            ),
            "maximum_objective_gradient_abs": float(
                search.maximum_objective_gradient_abs
            ),
            "iterations": int(search.iterations),
            "evaluations": int(search.evaluations),
            "wall_time_seconds": float(search_seconds),
            "minimum_lyapunov": float(np.min(nominal_values)),
            "terminal_lyapunov": float(nominal_values[-1]),
            "nominal_coordinate_states": search.states.astype(float).tolist(),
        },
        "result": result,
        "lyapunov": {
            "coordinate_source": file_metadata(args.spec),
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "evidence": {
            "config": {
                "path": str(Path(args.config)),
                "resolved_sha256": data_sha256(cfg),
            },
            "state_source": file_metadata(args.state_json),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"solver_success={search.success} margin={search.minimum_constraint_margin:.3e} "
        f"jacobian_max={search.maximum_constraint_jacobian_abs:.3e} "
        f"terminal_v={nominal_values[-1]:.2f} iterations={search.iterations} "
        f"wall={search_seconds:.1f}s"
    )
    print(
        f"success={result['success']} latched={result['latched']} "
        f"live_min_v={result['minimum_lyapunov']:.2f} "
        f"hold={result['max_upright_streak_seconds']:.3f}s "
        f"cart={result['max_cart_excursion']:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
