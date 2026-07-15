#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
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
    dimensionless_wrapped_state,
)
from gcartpole.predictive_sampling import (
    PredictiveSamplingConfig,
    PredictiveSamplingPlanner,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.refine_ilqr_capture_chain import source_trajectory
    from scripts.search_capture_sequence import (
        fixed_state_cfg,
        load_state,
        row_from_env,
    )
    from scripts.search_ilqr_capture import (
        initial_controls as lqr_initial_controls,
        lyapunov_value,
    )
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from refine_ilqr_capture_chain import source_trajectory
    from search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from search_ilqr_capture import (
        initial_controls as lqr_initial_controls,
        lyapunov_value,
    )
    from search_swingup_capture import lqr_action, lqr_gain


def shift_controls(controls: np.ndarray, elapsed_steps: int) -> np.ndarray:
    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim != 1 or controls.size < 2:
        raise ValueError("controls must be a one-dimensional horizon")
    if not 0 <= elapsed_steps <= controls.size:
        raise ValueError("elapsed_steps must lie inside the control horizon")
    return controls[elapsed_steps:].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate live receding iLQR after a saved exact-MuJoCo approach"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--state-json", default="runs/p1_capture_envelope/validation.json"
    )
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--initial-controller", required=True)
    parser.add_argument("--seed", type=int, default=65001)
    parser.add_argument("--out", required=True)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--mpc-seconds", type=float, default=2.0)
    parser.add_argument("--inter-segment-lqr-steps", type=int, default=0)
    parser.add_argument("--segment-predictive-sampling", action="store_true")
    parser.add_argument("--segment-knot-count", type=int, default=24)
    parser.add_argument("--segment-cem-iterations", type=int, default=4)
    parser.add_argument("--segment-cem-population", type=int, default=1024)
    parser.add_argument("--segment-cem-elites", type=int, default=64)
    parser.add_argument("--predictive-threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--control-cost", type=float, default=0.1)
    parser.add_argument("--stage-weight", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=10_000.0)
    parser.add_argument("--terminal-state-weight", type=float, default=100_000.0)
    parser.add_argument("--rail-soft-limit", type=float, default=2.4)
    parser.add_argument("--rail-weight", type=float, default=100_000_000.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if (
        min(
            args.replan_steps,
            args.mpc_seconds,
            args.iterations,
            args.lqr_scale,
            args.control_cost,
            args.stage_weight,
            args.terminal_weight,
            args.rail_soft_limit,
            args.rail_weight,
            args.handoff_lyapunov,
            args.handoff_cart_abs,
        )
        <= 0.0
        or args.terminal_state_weight < 0.0
        or args.inter_segment_lqr_steps < 0
        or min(
            args.segment_knot_count,
            args.segment_cem_iterations,
            args.segment_cem_population,
            args.segment_cem_elites,
            args.predictive_threads,
        )
        < 1
        or args.segment_cem_elites > args.segment_cem_population
    ):
        raise ValueError("durations, counts, scales, and weights must be valid")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    selected_state, selected_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(
        base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"])
    )
    controller_path = Path(args.initial_controller)
    controller_payload = json.loads(controller_path.read_text(encoding="utf-8"))
    controller_index = controller_payload.get("state_index")
    if controller_index is not None and int(controller_index) != int(selected_index):
        raise ValueError("initial controller state index does not match --state-index")
    initial_controls, initial_states, initial_gains = source_trajectory(
        controller_payload
    )
    replan_start_step = int(
        controller_payload.get("controller", {}).get("source_horizon_steps", 0)
    )
    if not 0 < replan_start_step < initial_controls.size:
        raise ValueError(
            "initial controller must identify a nonempty source approach and tail"
        )
    initial_tail_controls = initial_controls[replan_start_step:].copy()

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
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix, input_matrix, gain, transform, feedback_scale=args.lqr_scale
    )

    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    env.reset(seed=args.seed)
    transition = MujocoTransition(env, coordinate_transform=transform)
    policy_dt = float(env.dt)
    requested_mpc_steps = max(1, int(round(args.mpc_seconds / policy_dt)))
    mpc_stop_step = replan_start_step + requested_mpc_steps
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
    predictive_planner = None
    predictive_rng = np.random.default_rng(args.seed + 1)
    if args.segment_predictive_sampling:
        predictive_planner = PredictiveSamplingPlanner(
            env.model,
            frame_skip=env.frame_skip,
            force_limit=env.force_limit,
            coordinate_transform=transform,
            lyapunov=lyapunov,
            config=PredictiveSamplingConfig(
                horizon_steps=int(initial_tail_controls.size),
                replan_steps=args.replan_steps,
                knot_count=args.segment_knot_count,
                iterations=args.segment_cem_iterations,
                population=args.segment_cem_population,
                elites=args.segment_cem_elites,
                handoff_lyapunov=args.handoff_lyapunov,
                handoff_cart_abs=args.handoff_cart_abs,
            ),
            rail_limit=env.rail_limit,
            threads=args.predictive_threads,
        )

    plan = None
    plan_step = 0
    steps_since_replan = 0
    inter_segment_steps_remaining: int | None = None
    plan_events: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    latched = False
    first_handoff_step: int | None = None
    minimum_value = lyapunov_value(data_state(env.data), transform, lyapunov)
    initial_value = minimum_value
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    started = time.time()
    try:
        while not (terminated or truncated):
            state = data_state(env.data)
            value = lyapunov_value(state, transform, lyapunov)
            minimum_value = min(minimum_value, value)
            if (
                not latched
                and value <= args.handoff_lyapunov
                and abs(float(state[0])) <= args.handoff_cart_abs
            ):
                latched = True
                first_handoff_step = int(env.step_count)

            use_source = not latched and env.step_count < replan_start_step
            use_mpc = (
                not latched and replan_start_step <= env.step_count < mpc_stop_step
            )
            plan_exhausted = plan is not None and plan_step >= plan.controls.size
            if plan_exhausted and inter_segment_steps_remaining is None:
                inter_segment_steps_remaining = args.inter_segment_lqr_steps
            ready_after_coast = not plan_exhausted or inter_segment_steps_remaining == 0
            if use_mpc and (
                plan is None
                or (plan_exhausted and ready_after_coast)
                or (not plan_exhausted and steps_since_replan >= args.replan_steps)
            ):
                new_segment = plan is None or plan_exhausted
                if plan is None:
                    candidate_controls = initial_tail_controls
                elif plan_exhausted:
                    live_state = transition.to_coordinates(data_state(env.data))
                    if predictive_planner is None:
                        candidate_controls = lqr_initial_controls(
                            transition,
                            live_state,
                            gain,
                            horizon_steps=initial_tail_controls.size,
                            lqr_scale=args.lqr_scale,
                        )
                        predictive_search = None
                    else:
                        predictive_result = predictive_planner.search(
                            env.data, rng=predictive_rng
                        )
                        candidate_controls = np.asarray(
                            predictive_result["actions"], dtype=np.float64
                        )
                        predictive_search = {
                            "score": float(predictive_result["score"]),
                            "metrics": predictive_result["metrics"],
                            "history": predictive_result["history"],
                            "candidate_rollout_count": int(
                                predictive_result["candidate_rollout_count"]
                            ),
                        }
                else:
                    candidate_controls = shift_controls(plan.controls, plan_step)
                    predictive_search = None
                if plan is None:
                    predictive_search = None
                baseline_terminal = np.inf
                if plan is not None and not plan_exhausted:
                    baseline_terminal = float(
                        max(0.0, plan.states[-1] @ lyapunov @ plan.states[-1])
                    )
                live_state = transition.to_coordinates(data_state(env.data))
                search_started = time.time()
                candidate_plan = optimize_ilqr(
                    transition,
                    live_state,
                    candidate_controls,
                    trajectory_cost,
                    max_iterations=args.iterations,
                )
                plan_values = np.maximum(
                    0.0,
                    np.einsum(
                        "ij,jk,ik->i",
                        candidate_plan.states,
                        lyapunov,
                        candidate_plan.states,
                    ),
                )
                accepted = bool(plan_values[-1] <= baseline_terminal * (1.0 + 1e-9))
                if accepted:
                    plan = candidate_plan
                    plan_step = 0
                    inter_segment_steps_remaining = None
                steps_since_replan = 0
                plan_events.append(
                    {
                        "step": int(env.step_count),
                        "time_seconds": float(env.step_count * policy_dt),
                        "wall_time_seconds": float(time.time() - search_started),
                        "accepted": accepted,
                        "new_segment": bool(new_segment),
                        "predictive_search": predictive_search,
                        "baseline_terminal_lyapunov": (
                            None
                            if not np.isfinite(baseline_terminal)
                            else baseline_terminal
                        ),
                        "cost": float(candidate_plan.cost),
                        "iterations": int(candidate_plan.iterations),
                        "converged": bool(candidate_plan.converged),
                        "active_control_steps": int(
                            candidate_plan.active_control_steps
                        ),
                        "minimum_lyapunov": float(np.min(plan_values)),
                        "terminal_lyapunov": float(plan_values[-1]),
                    }
                )
                print(
                    f"plan={len(plan_events)} step={env.step_count} "
                    f"accepted={accepted} cost={candidate_plan.cost:.1f} "
                    f"min_v={np.min(plan_values):.1f} "
                    f"terminal_v={plan_values[-1]:.1f}",
                    flush=True,
                )

            if use_source:
                step = int(env.step_count)
                coordinate_state = dimensionless_wrapped_state(
                    env.data.qpos, env.data.qvel, transform
                )
                error = coordinate_state - initial_states[step]
                action = float(
                    np.clip(
                        initial_controls[step] + initial_gains[step] @ error,
                        -1.0,
                        1.0,
                    )
                )
                mode = "source_ilqr_tracking"
            elif (
                use_mpc
                and plan_exhausted
                and inter_segment_steps_remaining is not None
                and inter_segment_steps_remaining > 0
            ):
                action = lqr_action(env, gain, scale=args.lqr_scale, cart_target=0.0)
                inter_segment_steps_remaining -= 1
                mode = "inter_segment_lqr"
            elif use_mpc and plan is not None:
                coordinate_state = dimensionless_wrapped_state(
                    env.data.qpos, env.data.qvel, transform
                )
                error = coordinate_state - plan.states[plan_step]
                action = float(
                    np.clip(
                        plan.controls[plan_step]
                        + plan.feedback_gains[plan_step] @ error,
                        -1.0,
                        1.0,
                    )
                )
                plan_step += 1
                steps_since_replan += 1
                mode = "receding_ilqr_tracking"
            else:
                action = lqr_action(env, gain, scale=args.lqr_scale, cart_target=0.0)
                mode = "lqr"
            _, reward, terminated, truncated, info = env.step([action])
            row = row_from_env(
                env, step=env.step_count, action=action, reward=reward, info=info
            )
            row["controller_mode"] = mode
            row["dimensionless_lyapunov_value"] = lyapunov_value(
                data_state(env.data), transform, lyapunov
            )
            trajectory.append(row)
            final_info = dict(info)
    finally:
        env.close()

    result = {
        "success": bool(final_info.get("success", False)),
        "length": len(trajectory),
        "termination_reason": final_info.get("termination_reason"),
        "max_upright_streak_seconds": float(
            final_info.get("max_upright_streak_seconds", 0.0)
        ),
        "max_low_momentum_upright_streak_seconds": float(
            final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(final_info.get("max_cart_excursion", 0.0)),
        "initial_lyapunov": float(initial_value),
        "minimum_lyapunov": float(minimum_value),
        "latched": bool(latched),
        "first_handoff_step": first_handoff_step,
        "first_handoff_time": (
            None
            if first_handoff_step is None
            else float(first_handoff_step * policy_dt)
        ),
        "planning_wall_time_seconds": float(time.time() - started),
        "plan_event_count": len(plan_events),
        "plan_events": plan_events,
        "trajectory": trajectory,
        "final_info": final_info,
    }
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": (
            "Single-state reset-free saved approach plus live receding-iLQR "
            "diagnostic; not P1 evidence."
        ),
        "state_index": selected_index,
        "selected_state": selected_state,
        "seed": int(args.seed),
        "controller": {
            "type": "saved_ilqr_approach_then_receding_ilqr_then_lqr",
            "initial_controller": file_metadata(controller_path),
            "source_horizon_steps": int(replan_start_step),
            "tail_horizon_steps": int(initial_tail_controls.size),
            "replan_steps": int(args.replan_steps),
            "inter_segment_lqr_steps": int(args.inter_segment_lqr_steps),
            "segment_predictive_sampling": bool(args.segment_predictive_sampling),
            "segment_knot_count": int(args.segment_knot_count),
            "segment_cem_iterations": int(args.segment_cem_iterations),
            "segment_cem_population": int(args.segment_cem_population),
            "segment_cem_elites": int(args.segment_cem_elites),
            "predictive_threads": int(args.predictive_threads),
            "mpc_seconds": float(args.mpc_seconds),
            "effective_mpc_seconds": float(requested_mpc_steps * policy_dt),
            "iterations": int(args.iterations),
            "lqr_scale": float(args.lqr_scale),
            "control_cost": float(args.control_cost),
            "stage_weight": float(args.stage_weight),
            "terminal_weight": float(args.terminal_weight),
            "terminal_state_weight": float(args.terminal_state_weight),
            "rail_soft_limit": float(args.rail_soft_limit),
            "rail_weight": float(args.rail_weight),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
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
        f"success={result['success']} latched={result['latched']} "
        f"initial_v={initial_value:.2f} min_v={minimum_value:.2f} "
        f"hold={result['max_upright_streak_seconds']:.3f}s "
        f"cart={result['max_cart_excursion']:.3f} plans={len(plan_events)} "
        f"wall={result['planning_wall_time_seconds']:.1f}s"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
