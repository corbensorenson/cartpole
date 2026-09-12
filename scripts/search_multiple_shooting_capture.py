#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import (
    MujocoTransition,
    QuadraticTrajectoryCost,
    data_state,
    optimize_ilqr,
)
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
)
from gcartpole.multiple_shooting import optimize_direct_collocation, optimize_multiple_shooting

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_capture_sequence import fixed_state_cfg, load_state
    from scripts.search_ilqr_capture import execute_controller, load_initial_controls
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_capture_sequence import fixed_state_cfg, load_state
    from search_ilqr_capture import execute_controller, load_initial_controls
    from search_swingup_capture import lqr_gain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search a sparse exact-MuJoCo multiple-shooting capture trajectory"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--state-json", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--seed", type=int, default=63001)
    parser.add_argument("--out", required=True)
    parser.add_argument("--initial-controller", default=None)
    parser.add_argument("--horizon-seconds", type=float, default=3.0)
    parser.add_argument("--segment-steps", type=int, default=5)
    parser.add_argument("--max-evaluations", type=int, default=100)
    parser.add_argument("--method", choices=["equality", "penalty"], default="equality")
    parser.add_argument(
        "--equality-solver",
        choices=["slsqp", "trust-constr"],
        default="trust-constr",
    )
    parser.add_argument("--defect-weight", type=float, default=1_000_000.0)
    parser.add_argument("--terminal-weight", type=float, default=100.0)
    parser.add_argument("--control-weight", type=float, default=0.1)
    parser.add_argument("--rail-weight", type=float, default=1_000_000.0)
    parser.add_argument("--rail-soft-limit", type=float, default=2.4)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if min(
        args.horizon_seconds,
        args.segment_steps,
        args.max_evaluations,
        args.defect_weight,
        args.terminal_weight,
        args.control_weight,
        args.rail_weight,
        args.rail_soft_limit,
        args.handoff_lyapunov,
        args.handoff_cart_abs,
    ) <= 0.0:
        raise ValueError("durations, counts, weights, and thresholds must be positive")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    selected_state, selected_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"]))
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
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
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix,
        input_matrix,
        gain,
        transform,
        feedback_scale=args.lqr_scale,
    )
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    env.reset(seed=args.seed)
    transition = MujocoTransition(env, coordinate_transform=transform)
    initial_state = transition.to_coordinates(data_state(env.data))
    policy_dt = float(env.dt)
    horizon_steps = max(2, int(round(args.horizon_seconds / policy_dt)))
    if horizon_steps % args.segment_steps != 0:
        raise ValueError("horizon must contain an integer number of shooting segments")
    initial_nodes = None
    if args.initial_controller is None:
        controls = np.zeros(horizon_steps, dtype=np.float64)
    else:
        controls = load_initial_controls(args.initial_controller, horizon_steps, policy_dt)
        initial_payload = json.loads(Path(args.initial_controller).read_text(encoding="utf-8"))
        saved_nodes = initial_payload.get("search", {}).get("node_states")
        if saved_nodes is not None:
            candidate_nodes = np.asarray(saved_nodes, dtype=np.float64)
            expected_shape = (horizon_steps // args.segment_steps, transform.shape[0])
            if candidate_nodes.shape == expected_shape:
                initial_nodes = candidate_nodes
    started = time.time()
    if args.method == "equality":
        search = optimize_direct_collocation(
            transition,
            initial_state,
            controls,
            segment_steps=args.segment_steps,
            terminal_weight=args.terminal_weight,
            control_weight=args.control_weight,
            rail_limit=env.rail_limit * transform[0, 0],
            max_iterations=args.max_evaluations,
            initial_node_states=initial_nodes,
            solver=args.equality_solver,
        )
    else:
        search = optimize_multiple_shooting(
            transition,
            initial_state,
            controls,
            segment_steps=args.segment_steps,
            defect_weight=args.defect_weight,
            terminal_weight=args.terminal_weight,
            control_weight=args.control_weight,
            rail_weight=args.rail_weight,
            rail_soft_limit=args.rail_soft_limit * transform[0, 0],
            rail_limit=env.rail_limit * transform[0, 0],
            max_evaluations=args.max_evaluations,
            initial_node_states=initial_nodes,
        )
    wall_time = time.time() - started
    exact_values = np.asarray(
        [max(0.0, state @ lyapunov @ state) for state in search.exact_states],
        dtype=np.float64,
    )
    exact_physical = np.asarray(
        [transition.to_physical(state) for state in search.exact_states], dtype=np.float64
    )
    node_terminal_value = float(max(0.0, search.node_states[-1] @ lyapunov @ search.node_states[-1]))
    tracking = optimize_ilqr(
        transition,
        initial_state,
        search.controls,
        QuadraticTrajectoryCost(
            stage_state=0.1 * np.eye(transform.shape[0], dtype=np.float64),
            terminal_state=args.terminal_weight
            * np.eye(transform.shape[0], dtype=np.float64),
            control=float(args.control_weight),
            rail_soft_limit=float(args.rail_soft_limit * transform[0, 0]),
            rail_limit=float(env.rail_limit * transform[0, 0]),
            rail_weight=float(args.rail_weight),
            wrap_angles=False,
        ),
        max_iterations=0,
    )
    env.close()
    result = execute_controller(
        cfg,
        seed=args.seed,
        controls=search.controls,
        nominal_states=search.exact_states,
        feedback_gains=tracking.feedback_gains,
        gain=gain,
        lqr_scale=args.lqr_scale,
        transform=transform,
        lyapunov=lyapunov,
        handoff_lyapunov=args.handoff_lyapunov,
        handoff_cart_abs=args.handoff_cart_abs,
        tracking_mode="multiple_shooting_ddp_tracking",
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Single-state sparse multiple-shooting capture diagnostic; not P1 evidence.",
        "state_index": selected_index,
        "selected_state": selected_state,
        "seed": int(args.seed),
        "controller": {
            "type": "exact_mujoco_sparse_multiple_shooting_then_lqr",
            "method": args.method,
            "equality_solver": args.equality_solver if args.method == "equality" else None,
            "horizon_steps": horizon_steps,
            "horizon_seconds": horizon_steps * policy_dt,
            "segment_steps": int(args.segment_steps),
            "segment_seconds": args.segment_steps * policy_dt,
            "initial_controller": (
                None
                if args.initial_controller is None
                else file_metadata(args.initial_controller)
            ),
            "defect_weight": float(args.defect_weight),
            "terminal_weight": float(args.terminal_weight),
            "control_weight": float(args.control_weight),
            "rail_weight": float(args.rail_weight),
            "rail_soft_limit": float(args.rail_soft_limit),
            "lqr_scale": float(args.lqr_scale),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "controls": search.controls.astype(float).tolist(),
            "feedback_gains": tracking.feedback_gains.astype(float).tolist(),
            "tracking_active_control_steps": int(tracking.active_control_steps),
        },
        "search": {
            "success": bool(search.success),
            "status": int(search.status),
            "message": search.message,
            "cost": float(search.cost),
            "optimality": float(search.optimality),
            "evaluations": int(search.evaluations),
            "wall_time_seconds": float(wall_time),
            "segment_defect_rms": float(np.sqrt(np.mean(search.segment_defects**2))),
            "segment_defect_abs_max": float(np.max(np.abs(search.segment_defects))),
            "node_terminal_state_norm": float(np.linalg.norm(search.node_states[-1])),
            "node_terminal_lyapunov": node_terminal_value,
            "exact_terminal_state_norm": float(np.linalg.norm(search.exact_states[-1])),
            "exact_initial_lyapunov": float(exact_values[0]),
            "exact_minimum_lyapunov": float(np.min(exact_values)),
            "exact_terminal_lyapunov": float(exact_values[-1]),
            "exact_max_cart_excursion": float(np.max(np.abs(exact_physical[:, 0]))),
            "node_states": search.node_states.astype(float).tolist(),
            "segment_defects": search.segment_defects.astype(float).tolist(),
            "exact_coordinate_states": search.exact_states.astype(float).tolist(),
            "exact_physical_states": exact_physical.astype(float).tolist(),
        },
        "result": result,
        "lyapunov": {
            "coordinate_source": file_metadata(args.spec),
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "evidence": {
            "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(cfg)},
            "state_source": file_metadata(args.state_json),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"node_norm={payload['search']['node_terminal_state_norm']:.3f} "
        f"defect_max={payload['search']['segment_defect_abs_max']:.3e} "
        f"exact_min_v={payload['search']['exact_minimum_lyapunov']:.2f} "
        f"exact_terminal_v={payload['search']['exact_terminal_lyapunov']:.2f} "
        f"cart={payload['search']['exact_max_cart_excursion']:.3f} "
        f"evals={search.evaluations} wall={wall_time:.1f}s"
    )
    print(
        f"success={result['success']} latched={result['latched']} "
        f"live_min_v={result['minimum_lyapunov']:.2f} "
        f"hold={result['max_upright_streak_seconds']:.3f}s"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
