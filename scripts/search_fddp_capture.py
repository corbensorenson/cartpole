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
from gcartpole.ilqr import (
    MujocoTransition,
    QuadraticTrajectoryCost,
    add_terminal_cart_weights,
    data_state,
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


def rebuild_feedback_warm_start(
    transition: Any,
    start_state: np.ndarray,
    controls: np.ndarray,
    nominal_states: np.ndarray,
    feedback_gains: np.ndarray,
    *,
    feedback_scale: float = 1.0,
) -> np.ndarray:
    """Replay an inherited feedback trajectory on a new exact plant."""
    controls = np.asarray(controls, dtype=np.float64)
    nominal_states = np.asarray(nominal_states, dtype=np.float64)
    feedback_gains = np.asarray(feedback_gains, dtype=np.float64)
    start_state = np.asarray(start_state, dtype=np.float64)
    if controls.ndim != 1:
        raise ValueError("warm-start controls must be one-dimensional")
    if nominal_states.shape != (controls.size + 1, start_state.size):
        raise ValueError("warm-start nominal states have inconsistent dimensions")
    if feedback_gains.shape != (controls.size, start_state.size):
        raise ValueError("warm-start feedback gains have inconsistent dimensions")
    if feedback_scale < 0.0:
        raise ValueError("warm-start feedback scale must be nonnegative")

    states = [start_state.copy()]
    for step, control in enumerate(controls):
        error = states[-1] - nominal_states[step]
        action = float(
            np.clip(
                control + feedback_scale * feedback_gains[step] @ error,
                -1.0,
                1.0,
            )
        )
        states.append(transition(states[-1], action))
    return np.asarray(states, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search one exact-MuJoCo capture trajectory with Box-FDDP"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument(
        "--progress",
        type=float,
        default=1.0,
        help="morphology schedule progress for this exact replay; 1.0 is the endpoint plant",
    )
    parser.add_argument(
        "--lqr-progress",
        type=float,
        default=None,
        help="morphology progress used to linearize the post-handoff LQR; defaults to --progress",
    )
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--state-json", default="runs/p1_capture_envelope/validation.json"
    )
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--interpolate-from-state-index", default=None)
    parser.add_argument("--interpolation-alpha", type=float, default=1.0)
    parser.add_argument("--initial-controller", default=None)
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Skip Box-FDDP and replay the inherited controller directly on the target plant.",
    )
    parser.add_argument("--horizon-seconds", type=float, default=4.5)
    parser.add_argument(
        "--append-tail-seconds",
        type=float,
        default=0.0,
        help="append a zero-control settling tail to an inherited controller warm start",
    )
    parser.add_argument("--initial-feasible", action="store_true")
    parser.add_argument("--rebuild-initial-states", action="store_true")
    parser.add_argument(
        "--rebuild-initial-feedback",
        action="store_true",
        help=(
            "rebuild an inherited warm start by applying its saved feedback "
            "gains while rolling it through the target plant"
        ),
    )
    parser.add_argument(
        "--initial-feedback-scale",
        type=float,
        default=1.0,
        help="scale inherited feedback only while rebuilding a target-plant warm start",
    )
    parser.add_argument("--seed", type=int, default=67001)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--initial-regularization", type=float, default=1e-6)
    parser.add_argument("--tracking-gain-scale", type=float, default=0.0)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument(
        "--lqr-control-cost",
        type=float,
        default=1000.0,
        help="R term used to compute the post-handoff LQR gain",
    )
    parser.add_argument("--lqr-cart-position-cost", type=float, default=0.1)
    parser.add_argument("--lqr-absolute-angle-cost", type=float, default=100.0)
    parser.add_argument("--lqr-cart-velocity-cost", type=float, default=0.1)
    parser.add_argument("--lqr-absolute-angular-velocity-cost", type=float, default=1.0)
    parser.add_argument("--lqr-relative-angle-cost", type=float, default=1.0)
    parser.add_argument("--lqr-relative-angular-velocity-cost", type=float, default=0.01)
    parser.add_argument("--control-cost", type=float, default=0.1)
    parser.add_argument("--stage-weight", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=10_000.0)
    parser.add_argument("--terminal-state-weight", type=float, default=100_000.0)
    parser.add_argument("--terminal-cart-weight", type=float, default=0.0)
    parser.add_argument("--terminal-cart-velocity-weight", type=float, default=0.0)
    parser.add_argument(
        "--terminal-cart-velocity-factor",
        type=float,
        default=1.0,
        help="multiply the terminal cart-velocity block in the full-state objective",
    )
    parser.add_argument(
        "--terminal-hinge-velocity-factor",
        type=float,
        default=1.0,
        help="multiply the terminal hinge-velocity block in the full-state objective",
    )
    parser.add_argument(
        "--terminal-angle-factor",
        type=float,
        default=1.0,
        help="multiply the terminal absolute-angle block in the full-state objective",
    )
    parser.add_argument("--rail-soft-limit", type=float, default=2.4)
    parser.add_argument("--rail-weight", type=float, default=100_000_000.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--switch-lyapunov", type=float, default=None)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--handoff-angle-abs", type=float, default=0.15)
    parser.add_argument("--handoff-cart-velocity-abs", type=float, default=0.5)
    parser.add_argument("--handoff-hinge-velocity-rms", type=float, default=0.75)
    parser.add_argument(
        "--defer-handoff-until-horizon",
        action="store_true",
        help="Keep replaying the optimized feedback trajectory until its horizon before switching to LQR.",
    )
    parser.add_argument(
        "--phase-adaptive",
        action="store_true",
        help="Select the nearest forward nominal route phase within a bounded window during replay.",
    )
    parser.add_argument(
        "--phase-window",
        type=int,
        default=12,
        help="Maximum forward route steps considered by phase-adaptive replay.",
    )
    parser.add_argument(
        "--prefix-control-scale",
        type=float,
        default=1.0,
        help="multiply inherited feedforward controls during an initial prefix screen",
    )
    parser.add_argument(
        "--prefix-control-seconds",
        type=float,
        default=0.0,
        help="duration of the inherited-control prefix to scale; zero disables the screen",
    )
    parser.add_argument(
        "--allow-unstable-lyapunov",
        action="store_true",
        help="Use an identity terminal metric when the seven-link linear LQR is not asymptotically stable.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if (
        min(
            args.iterations,
            args.horizon_seconds,
            args.initial_regularization,
            args.lqr_scale,
            args.lqr_control_cost,
            args.lqr_cart_position_cost,
            args.lqr_absolute_angle_cost,
            args.lqr_cart_velocity_cost,
            args.lqr_absolute_angular_velocity_cost,
            args.lqr_relative_angle_cost,
            args.lqr_relative_angular_velocity_cost,
            args.control_cost,
            args.stage_weight,
            args.terminal_weight,
            args.rail_soft_limit,
            args.rail_weight,
            args.handoff_lyapunov,
            args.handoff_cart_abs,
            args.handoff_angle_abs,
            args.handoff_cart_velocity_abs,
            args.handoff_hinge_velocity_rms,
            args.terminal_cart_velocity_factor,
            args.terminal_hinge_velocity_factor,
            args.terminal_angle_factor,
        )
        <= 0.0
        or min(
            args.terminal_state_weight,
            args.terminal_cart_weight,
            args.terminal_cart_velocity_weight,
        )
        < 0.0
    ):
        raise ValueError("counts, weights, and thresholds must be positive")
    if args.switch_lyapunov is not None and args.switch_lyapunov <= 0.0:
        raise ValueError("switch Lyapunov threshold must be positive")
    if args.tracking_gain_scale < 0.0:
        raise ValueError("tracking gain scale must be nonnegative")
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    lqr_progress = args.progress if args.lqr_progress is None else args.lqr_progress
    if not 0.0 <= lqr_progress <= 1.0:
        raise ValueError("lqr-progress must be in [0, 1]")
    if args.append_tail_seconds < 0.0:
        raise ValueError("append-tail-seconds must be nonnegative")
    if args.prefix_control_seconds < 0.0:
        raise ValueError("prefix-control-seconds must be nonnegative")
    if args.prefix_control_seconds > 0.0 and args.initial_controller is None:
        raise ValueError("prefix-control-seconds requires an initial controller")
    if args.append_tail_seconds > 0.0 and args.initial_controller is None:
        raise ValueError("append-tail-seconds requires an initial controller")
    if args.rebuild_initial_feedback and args.initial_controller is None:
        raise ValueError("rebuild-initial-feedback requires an initial controller")
    if args.replay_only and args.initial_controller is None:
        raise ValueError("replay-only requires an initial controller")
    if args.initial_feedback_scale < 0.0:
        raise ValueError("initial feedback scale must be nonnegative")
    if args.phase_window < 0:
        raise ValueError("phase window must be nonnegative")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"].setdefault("action_lqr_residual", {})["enabled"] = False
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

    lqr_weights = {
        "cart_position": args.lqr_cart_position_cost,
        "absolute_angle": args.lqr_absolute_angle_cost,
        "cart_velocity": args.lqr_cart_velocity_cost,
        "absolute_angular_velocity": args.lqr_absolute_angular_velocity_cost,
        "relative_angle": args.lqr_relative_angle_cost,
        "relative_angular_velocity": args.lqr_relative_angular_velocity_cost,
    }
    gain = lqr_gain(
        cfg,
        progress=lqr_progress,
        fd_eps=1e-7,
        control_cost=args.lqr_control_cost,
        q_weights=lqr_weights,
    )
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
    state_matrix, input_matrix = finite_difference_dynamics(cfg, args.progress, 1e-7)
    try:
        lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
            state_matrix, input_matrix, gain, transform, feedback_scale=args.lqr_scale
        )
        lyapunov_source = "lqr_discrete_lyapunov"
    except ValueError:
        if not args.allow_unstable_lyapunov:
            raise
        # The uniform seven-link linearization has nearly uncontrollable
        # internal modes, so its LQR closed loop is not a valid Lyapunov
        # certificate.  Keep the exact nonlinear FDDP search available with a
        # positive-definite target metric, while recording the missing
        # certificate explicitly in the artifact.
        lyapunov = np.eye(transform.shape[0], dtype=np.float64)
        spectral_radius = float(
            np.max(
                np.abs(
                    np.linalg.eigvals(
                        state_matrix
                        - input_matrix
                        @ (args.lqr_scale * gain).reshape(1, -1)
                    )
                )
            )
        )
        lyapunov_source = "identity_fallback_unstable_lqr"

    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    env.reset(seed=args.seed)
    transition = MujocoTransition(env, coordinate_transform=transform)
    start_state = transition.to_coordinates(data_state(env.data))
    initial_path = (
        None if args.initial_controller is None else Path(args.initial_controller)
    )
    source_feedback_gains: np.ndarray | None = None
    if initial_path is None:
        horizon_steps = max(2, int(round(args.horizon_seconds / env.dt)))
        initial_controls = np.zeros(horizon_steps, dtype=np.float64)
        initial_states = rollout_controls(transition, start_state, initial_controls)
    else:
        initial_payload = json.loads(initial_path.read_text(encoding="utf-8"))
        initial_controls, initial_states, source_feedback_gains = source_trajectory(
            initial_payload
        )
        if args.append_tail_seconds > 0.0:
            tail_steps = max(1, int(round(args.append_tail_seconds / env.dt)))
            tail_controls = np.zeros(tail_steps, dtype=np.float64)
            tail_states = rollout_controls(
                transition, initial_states[-1], tail_controls
            )
            initial_controls = np.concatenate([initial_controls, tail_controls])
            initial_states = np.vstack([initial_states, tail_states[1:]])
            if source_feedback_gains is not None:
                source_feedback_gains = np.vstack(
                    [
                        source_feedback_gains,
                        np.zeros((tail_steps, transform.shape[0]), dtype=np.float64),
                    ]
                )
        if args.prefix_control_seconds > 0.0:
            prefix_steps = min(
                initial_controls.size,
                max(1, int(round(args.prefix_control_seconds / env.dt))),
            )
            initial_controls[:prefix_steps] = np.clip(
                initial_controls[:prefix_steps] * args.prefix_control_scale,
                -1.0,
                1.0,
            )
    if initial_states.shape != (initial_controls.size + 1, transform.shape[0]):
        raise ValueError("initial controller state/control horizon is inconsistent")
    initial_states = initial_states.copy()
    initial_states[0] = start_state
    replay_nominal_states = initial_states.copy()
    if args.rebuild_initial_feedback:
        assert source_feedback_gains is not None
        initial_states = rebuild_feedback_warm_start(
            transition,
            start_state,
            initial_controls,
            initial_states,
            source_feedback_gains,
            feedback_scale=args.initial_feedback_scale,
        )
    elif args.rebuild_initial_states:
        initial_states = rollout_controls(transition, start_state, initial_controls)
    terminal_identity = np.eye(transform.shape[0], dtype=np.float64)
    state_half = transform.shape[0] // 2
    terminal_identity[1:state_half, 1:state_half] *= args.terminal_angle_factor
    terminal_identity[state_half, state_half] *= args.terminal_cart_velocity_factor
    terminal_identity[state_half + 1 :, state_half + 1 :] *= args.terminal_hinge_velocity_factor
    terminal_metric = (
        args.terminal_weight * lyapunov / args.handoff_lyapunov
        + args.terminal_state_weight * terminal_identity
    )
    terminal_metric = add_terminal_cart_weights(
        terminal_metric,
        int(cfg["env"]["n_links"]),
        cart_weight=args.terminal_cart_weight,
        cart_velocity_weight=args.terminal_cart_velocity_weight,
    )
    trajectory_cost = QuadraticTrajectoryCost(
        stage_state=args.stage_weight * np.eye(transform.shape[0], dtype=np.float64),
        terminal_state=terminal_metric,
        control=float(args.control_cost),
        rail_soft_limit=float(args.rail_soft_limit * transform[0, 0]),
        rail_limit=float(env.rail_limit * transform[0, 0]),
        rail_weight=float(args.rail_weight),
        wrap_angles=False,
    )
    if args.replay_only:
        assert source_feedback_gains is not None
        controls = initial_controls.copy()
        nominal_states = replay_nominal_states
        solver_feedback_gains = source_feedback_gains.copy()
        feedback_gains = args.tracking_gain_scale * source_feedback_gains.copy()
        converged = False
        search_seconds = 0.0
        search_iterations = 0
        search_cost = 0.0
        search_stop = 0.0
        search_is_feasible = True
    else:
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
                or args.rebuild_initial_feedback
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
        search_iterations = int(solver.iter)
        search_cost = float(solver.cost)
        search_stop = float(solver.stop)
        search_is_feasible = bool(solver.isFeasible)
    env.close()

    nominal_values = np.maximum(
        0.0,
        np.einsum("ij,jk,ik->i", nominal_states, lyapunov, nominal_states),
    )
    result = execute_controller(
        cfg,
        progress=args.progress,
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
        defer_handoff_until_horizon=args.defer_handoff_until_horizon,
        phase_adaptive=args.phase_adaptive,
        phase_window=args.phase_window,
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
            "type": (
                "exact_mujoco_feedback_replay_then_lqr"
                if args.replay_only
                else "crocoddyl_box_fddp_exact_mujoco_then_lqr"
            ),
            "initial_controller": (
                None if initial_path is None else file_metadata(initial_path)
            ),
            "initial_feasible": bool(
                args.initial_feasible
                or args.rebuild_initial_states
                or args.replay_only
                or initial_path is None
            ),
            "replay_only": bool(args.replay_only),
            "rebuilt_initial_states": bool(args.rebuild_initial_states),
            "rebuilt_initial_feedback": bool(args.rebuild_initial_feedback),
            "initial_feedback_scale": float(args.initial_feedback_scale),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(
                controls.size * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]
            ),
            "iterations": int(args.iterations),
            "append_tail_seconds": float(args.append_tail_seconds),
            "initial_regularization": float(args.initial_regularization),
            "tracking_gain_scale": float(args.tracking_gain_scale),
            "lqr_scale": float(args.lqr_scale),
            "lqr_progress": float(lqr_progress),
            "lqr_control_cost": float(args.lqr_control_cost),
            "lqr_weights": {key: float(value) for key, value in lqr_weights.items()},
            "control_cost": float(args.control_cost),
            "stage_weight": float(args.stage_weight),
            "terminal_weight": float(args.terminal_weight),
            "terminal_state_weight": float(args.terminal_state_weight),
            "terminal_cart_weight": float(args.terminal_cart_weight),
            "terminal_cart_velocity_weight": float(args.terminal_cart_velocity_weight),
            "terminal_cart_velocity_factor": float(args.terminal_cart_velocity_factor),
            "terminal_hinge_velocity_factor": float(args.terminal_hinge_velocity_factor),
            "terminal_angle_factor": float(args.terminal_angle_factor),
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
            "defer_handoff_until_horizon": bool(args.defer_handoff_until_horizon),
            "phase_adaptive": bool(args.phase_adaptive),
            "phase_window": int(args.phase_window),
            "prefix_control_scale": float(args.prefix_control_scale),
            "prefix_control_seconds": float(args.prefix_control_seconds),
            "allow_unstable_lyapunov": bool(args.allow_unstable_lyapunov),
            "progress": float(args.progress),
            "controls": controls.astype(float).tolist(),
            "feedback_gains": feedback_gains.astype(float).tolist(),
            "solver_feedback_gains": solver_feedback_gains.astype(float).tolist(),
        },
        "search": {
            "converged": converged,
            "progress": float(args.progress),
            "iterations": search_iterations,
            "cost": search_cost,
            "stopping_criterion": search_stop,
            "is_feasible": search_is_feasible,
            "wall_time_seconds": float(search_seconds),
            "initial_lyapunov": float(nominal_values[0]),
            "minimum_lyapunov": float(np.min(nominal_values)),
            "terminal_lyapunov": float(nominal_values[-1]),
            "nominal_coordinate_states": nominal_states.astype(float).tolist(),
        },
        "lyapunov": {
            "coordinate_source": file_metadata(args.spec),
            "source": lyapunov_source,
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "result": result,
        "evidence": {
            "config": {
                "path": str(Path(args.config)),
                "resolved_sha256": data_sha256(cfg),
                "progress": float(args.progress),
            },
            "state_source": file_metadata(args.state_json),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"converged={converged} feasible={search_is_feasible} "
        f"iterations={search_iterations} min_v={np.min(nominal_values):.2f} "
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
