#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import crocoddyl
except ImportError as error:
    raise SystemExit(
        "Crocoddyl is required; run `.venv/bin/pip install -r requirements-fddp.txt`"
    ) from error

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.fddp import MujocoActionModel, rollout_controls
from gcartpole.ilqr import MujocoTransition, QuadraticTrajectoryCost, data_state
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.refine_ilqr_capture_chain import source_trajectory
    from scripts.search_capture_sequence import fixed_state_cfg, load_state
    from scripts.search_ilqr_capture import (
        execute_controller,
        interpolate_initial_state,
    )
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from refine_ilqr_capture_chain import source_trajectory
    from search_capture_sequence import fixed_state_cfg, load_state
    from search_ilqr_capture import execute_controller, interpolate_initial_state
    from search_swingup_capture import lqr_gain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search one exact-MuJoCo capture trajectory with Box-FDDP"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--state-json", default="runs/p1_capture_envelope/validation.json"
    )
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--interpolate-from-state-index", default=None)
    parser.add_argument("--interpolation-alpha", type=float, default=1.0)
    parser.add_argument("--initial-controller", default=None)
    parser.add_argument("--horizon-seconds", type=float, default=4.5)
    parser.add_argument("--initial-feasible", action="store_true")
    parser.add_argument("--rebuild-initial-states", action="store_true")
    parser.add_argument("--seed", type=int, default=67001)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--initial-regularization", type=float, default=1e-6)
    parser.add_argument("--tracking-gain-scale", type=float, default=0.0)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--control-cost", type=float, default=0.1)
    parser.add_argument("--stage-weight", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=10_000.0)
    parser.add_argument("--terminal-state-weight", type=float, default=100_000.0)
    parser.add_argument("--rail-soft-limit", type=float, default=2.4)
    parser.add_argument("--rail-weight", type=float, default=100_000_000.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--switch-lyapunov", type=float, default=None)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--handoff-angle-abs", type=float, default=0.15)
    parser.add_argument("--handoff-cart-velocity-abs", type=float, default=0.5)
    parser.add_argument("--handoff-hinge-velocity-rms", type=float, default=0.75)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if (
        min(
            args.iterations,
            args.horizon_seconds,
            args.initial_regularization,
            args.lqr_scale,
            args.control_cost,
            args.stage_weight,
            args.terminal_weight,
            args.terminal_state_weight,
            args.rail_soft_limit,
            args.rail_weight,
            args.handoff_lyapunov,
            args.handoff_cart_abs,
            args.handoff_angle_abs,
            args.handoff_cart_velocity_abs,
            args.handoff_hinge_velocity_rms,
        )
        <= 0.0
    ):
        raise ValueError("counts, weights, and thresholds must be positive")
    if args.switch_lyapunov is not None and args.switch_lyapunov <= 0.0:
        raise ValueError("switch Lyapunov threshold must be positive")
    if args.tracking_gain_scale < 0.0:
        raise ValueError("tracking gain scale must be nonnegative")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    state, state_index = load_state(args.state_json, args.state_index)
    interpolation = None
    if args.interpolate_from_state_index is not None:
        source_state, source_index = load_state(
            args.state_json, args.interpolate_from_state_index
        )
        state = interpolate_initial_state(source_state, state, args.interpolation_alpha)
        interpolation = {
            "source_index": int(source_index),
            "target_index": int(state_index),
            "alpha": float(args.interpolation_alpha),
        }
    cfg = fixed_state_cfg(base_cfg, state, float(base_cfg["env"]["episode_seconds"]))

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
        state_matrix, input_matrix, gain, transform, feedback_scale=args.lqr_scale
    )

    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    env.reset(seed=args.seed)
    transition = MujocoTransition(env, coordinate_transform=transform)
    start_state = transition.to_coordinates(data_state(env.data))
    initial_path = (
        None if args.initial_controller is None else Path(args.initial_controller)
    )
    if initial_path is None:
        horizon_steps = max(2, int(round(args.horizon_seconds / env.dt)))
        initial_controls = np.zeros(horizon_steps, dtype=np.float64)
        initial_states = rollout_controls(transition, start_state, initial_controls)
    else:
        initial_payload = json.loads(initial_path.read_text(encoding="utf-8"))
        initial_controls, initial_states, _ = source_trajectory(initial_payload)
    if initial_states.shape != (initial_controls.size + 1, transform.shape[0]):
        raise ValueError("initial controller state/control horizon is inconsistent")
    initial_states = initial_states.copy()
    initial_states[0] = start_state
    if args.rebuild_initial_states:
        initial_states = rollout_controls(transition, start_state, initial_controls)
    trajectory_cost = QuadraticTrajectoryCost(
        stage_state=args.stage_weight * np.eye(transform.shape[0], dtype=np.float64),
        terminal_state=(
            args.terminal_weight * lyapunov / args.handoff_lyapunov
            + args.terminal_state_weight * np.eye(transform.shape[0], dtype=np.float64)
        ),
        control=float(args.control_cost),
        rail_soft_limit=float(args.rail_soft_limit * transform[0, 0]),
        rail_limit=float(env.rail_limit * transform[0, 0]),
        rail_weight=float(args.rail_weight),
        wrap_angles=False,
    )
    running_model = MujocoActionModel(transition, trajectory_cost)
    terminal_model = MujocoActionModel(transition, trajectory_cost, terminal=True)
    problem = crocoddyl.ShootingProblem(
        start_state,
        [running_model] * int(initial_controls.size),
        terminal_model,
    )
    solver = crocoddyl.SolverBoxFDDP(problem)
    initial_xs = [row.copy() for row in initial_states]
    initial_us = [np.asarray([action], dtype=np.float64) for action in initial_controls]
    started = time.time()
    converged = bool(
        solver.solve(
            initial_xs,
            initial_us,
            args.iterations,
            args.initial_feasible
            or args.rebuild_initial_states
            or initial_path is None,
            args.initial_regularization,
        )
    )
    search_seconds = time.time() - started
    controls = np.asarray([float(row[0]) for row in solver.us], dtype=np.float64)
    nominal_states = np.asarray(solver.xs, dtype=np.float64)
    solver_feedback_gains = -np.asarray(solver.K, dtype=np.float64).reshape(
        controls.size, transform.shape[0]
    )
    feedback_gains = args.tracking_gain_scale * solver_feedback_gains
    env.close()

    nominal_values = np.maximum(
        0.0,
        np.einsum("ij,jk,ik->i", nominal_states, lyapunov, nominal_states),
    )
    result = execute_controller(
        cfg,
        seed=args.seed,
        controls=controls,
        nominal_states=nominal_states,
        feedback_gains=feedback_gains,
        gain=gain,
        lqr_scale=args.lqr_scale,
        transform=transform,
        lyapunov=lyapunov,
        handoff_lyapunov=(
            args.handoff_lyapunov
            if args.switch_lyapunov is None
            else args.switch_lyapunov
        ),
        handoff_cart_abs=args.handoff_cart_abs,
        handoff_angle_abs=args.handoff_angle_abs,
        handoff_cart_velocity_abs=args.handoff_cart_velocity_abs,
        handoff_hinge_velocity_rms=args.handoff_hinge_velocity_rms,
        tracking_mode="box_fddp_tracking",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Single-state exact-MuJoCo Box-FDDP diagnostic; not P1 evidence.",
        "state_index": int(state_index),
        "selected_state": state,
        "state_interpolation": interpolation,
        "seed": int(args.seed),
        "controller": {
            "type": "crocoddyl_box_fddp_exact_mujoco_then_lqr",
            "initial_controller": (
                None if initial_path is None else file_metadata(initial_path)
            ),
            "initial_feasible": bool(
                args.initial_feasible
                or args.rebuild_initial_states
                or initial_path is None
            ),
            "rebuilt_initial_states": bool(args.rebuild_initial_states),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(
                controls.size * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]
            ),
            "iterations": int(args.iterations),
            "initial_regularization": float(args.initial_regularization),
            "tracking_gain_scale": float(args.tracking_gain_scale),
            "lqr_scale": float(args.lqr_scale),
            "control_cost": float(args.control_cost),
            "stage_weight": float(args.stage_weight),
            "terminal_weight": float(args.terminal_weight),
            "terminal_state_weight": float(args.terminal_state_weight),
            "rail_soft_limit": float(args.rail_soft_limit),
            "rail_weight": float(args.rail_weight),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "switch_lyapunov": (
                float(args.handoff_lyapunov)
                if args.switch_lyapunov is None
                else float(args.switch_lyapunov)
            ),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "handoff_angle_abs": float(args.handoff_angle_abs),
            "handoff_cart_velocity_abs": float(args.handoff_cart_velocity_abs),
            "handoff_hinge_velocity_rms": float(args.handoff_hinge_velocity_rms),
            "controls": controls.astype(float).tolist(),
            "feedback_gains": feedback_gains.astype(float).tolist(),
            "solver_feedback_gains": solver_feedback_gains.astype(float).tolist(),
        },
        "search": {
            "converged": converged,
            "iterations": int(solver.iter),
            "cost": float(solver.cost),
            "stopping_criterion": float(solver.stop),
            "is_feasible": bool(solver.isFeasible),
            "wall_time_seconds": float(search_seconds),
            "initial_lyapunov": float(nominal_values[0]),
            "minimum_lyapunov": float(np.min(nominal_values)),
            "terminal_lyapunov": float(nominal_values[-1]),
            "nominal_coordinate_states": nominal_states.astype(float).tolist(),
        },
        "lyapunov": {
            "coordinate_source": file_metadata(args.spec),
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "result": result,
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
        f"converged={converged} feasible={solver.isFeasible} "
        f"iterations={solver.iter} min_v={np.min(nominal_values):.2f} "
        f"terminal_v={nominal_values[-1]:.2f} wall={search_seconds:.1f}s"
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
