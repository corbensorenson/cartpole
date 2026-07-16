#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import lsq_linear

from gcartpole.capture_constraints import (
    exact_constraint_margins,
    normalized_constraint_score,
    terminal_bounds,
)
from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.fddp import rollout_controls
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

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.refine_ilqr_capture_chain import source_trajectory
    from scripts.search_capture_sequence import fixed_state_cfg, load_state
    from scripts.search_ilqr_capture import (
        execute_controller,
        initial_controls as lqr_initial_controls,
    )
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from refine_ilqr_capture_chain import source_trajectory
    from search_capture_sequence import fixed_state_cfg, load_state
    from search_ilqr_capture import (
        execute_controller,
        initial_controls as lqr_initial_controls,
    )
    from search_swingup_capture import lqr_gain


def endpoint_sensitivity(
    transition: MujocoTransition,
    states: np.ndarray,
    controls: np.ndarray,
    *,
    window_steps: int,
    window_offset: int,
    state_epsilon: float,
    action_epsilon: float,
) -> np.ndarray:
    window_offset = min(window_offset, controls.size - 1)
    window_steps = min(window_steps, controls.size - window_offset)
    sensitivity = np.zeros((states.shape[1], window_steps), dtype=np.float64)
    end = controls.size - window_offset
    start = end - window_steps
    for step in range(start, controls.size):
        state_matrix, input_matrix = transition.linearize(
            states[step],
            float(controls[step]),
            state_epsilon=state_epsilon,
            action_epsilon=action_epsilon,
        )
        sensitivity = state_matrix @ sensitivity
        if step < end:
            sensitivity[:, step - start] += input_matrix[:, 0]
    return sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Locally refine an exact-MuJoCo capture with sequential convex steps"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--state-json", default="runs/p1_capture_envelope/validation.json"
    )
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--initial-controller", required=True)
    parser.add_argument("--seed", type=int, default=69001)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--append-lqr-seconds", type=float, default=0.0)
    parser.add_argument("--window-steps", type=int, default=30)
    parser.add_argument("--window-offset", type=int, default=0)
    parser.add_argument(
        "--window-offsets",
        help="Comma-separated control-window offsets to cycle through",
    )
    parser.add_argument("--trust-region", type=float, default=0.005)
    parser.add_argument("--minimum-step-scale", type=float, default=1.0 / 128.0)
    parser.add_argument("--cart-penalty", type=float, default=100.0)
    parser.add_argument("--box-penalty", type=float, default=100.0)
    parser.add_argument("--control-regularization", type=float, default=0.01)
    parser.add_argument("--target-cart-margin", type=float, default=0.01)
    parser.add_argument("--state-epsilon", type=float, default=1e-5)
    parser.add_argument("--action-epsilon", type=float, default=1e-4)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.25)
    parser.add_argument("--handoff-angle-abs", type=float, default=0.15)
    parser.add_argument("--handoff-cart-velocity-abs", type=float, default=0.5)
    parser.add_argument("--handoff-hinge-velocity-abs", type=float, default=0.75)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if (
        min(
            args.iterations,
            args.window_steps,
            args.trust_region,
            args.minimum_step_scale,
            args.cart_penalty,
            args.box_penalty,
            args.control_regularization,
            args.state_epsilon,
            args.action_epsilon,
            args.lqr_scale,
            args.handoff_lyapunov,
            args.handoff_cart_abs,
            args.handoff_angle_abs,
            args.handoff_cart_velocity_abs,
            args.handoff_hinge_velocity_abs,
        )
        <= 0.0
        or args.target_cart_margin < 0.0
        or args.window_offset < 0
        or args.append_lqr_seconds < 0.0
    ):
        raise ValueError("solver and handoff settings must be positive")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    selected_state, state_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(
        base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"])
    )
    n_links = int(cfg["env"]["n_links"])
    distribution = load_config(args.spec)["distribution"]
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
    eigenvalues, eigenvectors = np.linalg.eigh(lyapunov / args.handoff_lyapunov)
    lyapunov_root = np.diag(np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    lower, upper = terminal_bounds(
        n_links,
        scales,
        cart_abs=args.handoff_cart_abs,
        angle_abs=args.handoff_angle_abs,
        cart_velocity_abs=args.handoff_cart_velocity_abs,
        hinge_velocity_abs=args.handoff_hinge_velocity_abs,
    )

    initial_path = Path(args.initial_controller)
    initial_payload = json.loads(initial_path.read_text(encoding="utf-8"))
    initial_controls, _, _ = source_trajectory(initial_payload)
    window_offsets = (
        [args.window_offset]
        if args.window_offsets is None
        else [int(value) for value in args.window_offsets.split(",")]
    )
    if (
        not window_offsets
        or min(window_offsets) < 0
        or max(window_offsets) >= initial_controls.size
    ):
        raise ValueError("window offset must leave at least one control to optimize")
    controls = initial_controls.copy()
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    env.reset(seed=args.seed)
    transition = MujocoTransition(env, coordinate_transform=transform)
    start_state = transition.to_coordinates(data_state(env.data))
    states = rollout_controls(transition, start_state, controls)
    appended_steps = int(round(args.append_lqr_seconds / env.dt))
    if appended_steps:
        controls = np.concatenate(
            (
                controls,
                lqr_initial_controls(
                    transition,
                    states[-1],
                    gain,
                    horizon_steps=appended_steps,
                    lqr_scale=args.lqr_scale,
                ),
            )
        )
        states = rollout_controls(transition, start_state, controls)
    warm_controls = controls.copy()
    rail_limit = float(env.rail_limit * transform[0, 0])
    initial_margins = exact_constraint_margins(
        states,
        lyapunov,
        lower,
        upper,
        rail_limit=rail_limit,
        handoff_lyapunov=args.handoff_lyapunov,
    )
    current_score = normalized_constraint_score(
        initial_margins,
        handoff_lyapunov=args.handoff_lyapunov,
        rail_limit=rail_limit,
    )
    history: list[dict[str, Any]] = []
    started = time.time()
    converged = current_score <= 1e-9
    consecutive_failures = 0
    for iteration in range(args.iterations):
        if converged:
            break
        active_offset = window_offsets[iteration % len(window_offsets)]
        sensitivity = endpoint_sensitivity(
            transition,
            states,
            controls,
            window_steps=args.window_steps,
            window_offset=active_offset,
            state_epsilon=args.state_epsilon,
            action_epsilon=args.action_epsilon,
        )
        window_steps = sensitivity.shape[1]
        terminal = states[-1]
        target_cart = np.clip(
            terminal[0],
            lower[0] + args.target_cart_margin,
            upper[0] - args.target_cart_margin,
        )
        target_terminal = np.clip(
            terminal,
            lower + args.target_cart_margin,
            upper - args.target_cart_margin,
        )
        matrix = np.vstack(
            (
                lyapunov_root @ sensitivity,
                np.sqrt(args.box_penalty) * sensitivity,
                np.sqrt(args.cart_penalty) * sensitivity[0:1],
                args.control_regularization * np.eye(window_steps),
            )
        )
        right_hand_side = np.r_[
            -lyapunov_root @ terminal,
            np.sqrt(args.box_penalty) * (target_terminal - terminal),
            np.sqrt(args.cart_penalty) * (target_cart - terminal[0]),
            np.zeros(window_steps),
        ]
        window_end = controls.size - active_offset
        window_start = window_end - window_steps
        control_window = controls[window_start:window_end]
        update_lower = np.maximum(-args.trust_region, -1.0 - control_window)
        update_upper = np.minimum(args.trust_region, 1.0 - control_window)
        subproblem = lsq_linear(
            matrix,
            right_hand_side,
            bounds=(update_lower, update_upper),
            max_iter=500,
            tol=1e-12,
            lsmr_tol=1e-12,
        )
        accepted = False
        step_scale = 1.0
        best_record: dict[str, Any] | None = None
        while step_scale >= args.minimum_step_scale:
            candidate_controls = controls.copy()
            candidate_controls[window_start:window_end] += step_scale * subproblem.x
            candidate_states = rollout_controls(
                transition, start_state, candidate_controls
            )
            candidate_margins = exact_constraint_margins(
                candidate_states,
                lyapunov,
                lower,
                upper,
                rail_limit=rail_limit,
                handoff_lyapunov=args.handoff_lyapunov,
            )
            candidate_score = normalized_constraint_score(
                candidate_margins,
                handoff_lyapunov=args.handoff_lyapunov,
                rail_limit=rail_limit,
            )
            best_record = {
                "iteration": iteration,
                "window_offset": active_offset,
                "step_scale": step_scale,
                "score_before": current_score,
                "score_after": candidate_score,
                "terminal_cart": float(candidate_states[-1, 0]),
                "terminal_lyapunov": float(
                    candidate_states[-1] @ lyapunov @ candidate_states[-1]
                ),
                "maximum_control_update": float(
                    np.max(np.abs(step_scale * subproblem.x))
                ),
                "subproblem_cost": float(subproblem.cost),
                "subproblem_optimality": float(subproblem.optimality),
                "accepted": candidate_score < current_score - 1e-12,
            }
            if best_record["accepted"]:
                controls = candidate_controls
                states = candidate_states
                current_score = candidate_score
                accepted = True
                break
            step_scale *= 0.5
        if best_record is not None:
            history.append(best_record)
        if not accepted:
            consecutive_failures += 1
            if consecutive_failures >= len(window_offsets):
                break
            continue
        consecutive_failures = 0
        converged = current_score <= 1e-9
    search_seconds = time.time() - started

    margins = exact_constraint_margins(
        states,
        lyapunov,
        lower,
        upper,
        rail_limit=rail_limit,
        handoff_lyapunov=args.handoff_lyapunov,
    )
    trajectory_cost = QuadraticTrajectoryCost(
        stage_state=np.zeros_like(lyapunov),
        terminal_state=lyapunov / args.handoff_lyapunov,
        control=0.1,
        rail_soft_limit=float(2.4 * transform[0, 0]),
        rail_limit=rail_limit,
        rail_weight=1e8,
        wrap_angles=False,
    )
    tracking = optimize_ilqr(
        transition, start_state, controls, trajectory_cost, max_iterations=0
    )
    env.close()
    result = execute_controller(
        cfg,
        seed=args.seed,
        controls=controls,
        nominal_states=states,
        feedback_gains=tracking.feedback_gains,
        gain=gain,
        lqr_scale=args.lqr_scale,
        transform=transform,
        lyapunov=lyapunov,
        handoff_lyapunov=args.handoff_lyapunov,
        handoff_cart_abs=args.handoff_cart_abs,
        handoff_angle_abs=args.handoff_angle_abs,
        handoff_cart_velocity_abs=args.handoff_cart_velocity_abs,
        handoff_hinge_velocity_rms=args.handoff_hinge_velocity_abs,
        tracking_mode="local_scp_tracking",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Single-state local exact-MuJoCo SCP diagnostic; not P1 evidence.",
        "state_index": int(state_index),
        "selected_state": selected_state,
        "seed": int(args.seed),
        "controller": {
            "type": "local_scp_exact_mujoco_then_lqr",
            "initial_controller": file_metadata(initial_path),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(controls.size * env.dt),
            "iterations": int(args.iterations),
            "append_lqr_seconds": float(args.append_lqr_seconds),
            "appended_steps": int(appended_steps),
            "window_steps": int(args.window_steps),
            "window_offset": int(args.window_offset),
            "window_offsets": window_offsets,
            "trust_region": float(args.trust_region),
            "minimum_step_scale": float(args.minimum_step_scale),
            "cart_penalty": float(args.cart_penalty),
            "box_penalty": float(args.box_penalty),
            "control_regularization": float(args.control_regularization),
            "target_cart_margin": float(args.target_cart_margin),
            "state_epsilon": float(args.state_epsilon),
            "action_epsilon": float(args.action_epsilon),
            "lqr_scale": float(args.lqr_scale),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "handoff_angle_abs": float(args.handoff_angle_abs),
            "handoff_cart_velocity_abs": float(args.handoff_cart_velocity_abs),
            "handoff_hinge_velocity_abs": float(args.handoff_hinge_velocity_abs),
            "controls": controls.astype(float).tolist(),
            "feedback_gains": tracking.feedback_gains.astype(float).tolist(),
        },
        "search": {
            "converged": bool(converged),
            "iterations": len(history),
            "wall_time_seconds": float(search_seconds),
            "initial_constraint_score": float(
                normalized_constraint_score(
                    initial_margins,
                    handoff_lyapunov=args.handoff_lyapunov,
                    rail_limit=rail_limit,
                )
            ),
            "final_constraint_score": float(current_score),
            "initial_constraint_margins": initial_margins,
            "exact_constraint_margins": margins,
            "history": history,
            "maximum_control_change": float(np.max(np.abs(controls - warm_controls))),
            "nominal_coordinate_states": states.astype(float).tolist(),
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
        f"converged={converged} iterations={len(history)} "
        f"score={current_score:.3e} hard_margin={margins['minimum_hard']:.3e} "
        f"terminal_v={-margins['lyapunov'] + args.handoff_lyapunov:.2f} "
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
