#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
)
from gcartpole.predictive_sampling import (
    PredictiveSamplingConfig,
    PredictiveSamplingPlanner,
    shift_action_knots,
)

try:
    from scripts.evaluate_feedback_mpc_capture import state_lyapunov
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_capture_sequence import (
        fixed_state_cfg,
        load_state,
        row_from_env,
    )
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from evaluate_feedback_mpc_capture import state_lyapunov
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from search_swingup_capture import lqr_action, lqr_gain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate multithreaded direct-action predictive sampling on one capture state"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--state-json", default="runs/p1_capture_envelope/validation.json"
    )
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=64001)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon-seconds", type=float, default=3.0)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--mpc-seconds", type=float, default=6.0)
    parser.add_argument("--knot-count", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--population", type=int, default=1024)
    parser.add_argument("--elites", type=int, default=64)
    parser.add_argument("--action-sigma", type=float, default=0.7)
    parser.add_argument("--sigma-decay", type=float, default=0.8)
    parser.add_argument("--sigma-floor", type=float, default=0.025)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--handoff-angle-abs", type=float, default=0.15)
    parser.add_argument("--handoff-cart-velocity-abs", type=float, default=0.5)
    parser.add_argument("--handoff-hinge-velocity-rms", type=float, default=0.75)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    selected_state, selected_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(
        base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"])
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
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix, input_matrix, gain, transform, feedback_scale=args.lqr_scale
    )
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    env.reset(seed=args.seed)
    policy_dt = float(env.dt)
    planner_config = PredictiveSamplingConfig(
        horizon_steps=max(2, int(round(args.horizon_seconds / policy_dt))),
        replan_steps=args.replan_steps,
        knot_count=args.knot_count,
        iterations=args.iterations,
        population=args.population,
        elites=args.elites,
        action_sigma=args.action_sigma,
        sigma_decay=args.sigma_decay,
        sigma_floor=args.sigma_floor,
        handoff_lyapunov=args.handoff_lyapunov,
        handoff_cart_abs=args.handoff_cart_abs,
        handoff_angle_ratio=args.handoff_angle_abs
        / float(distribution["absolute_link_angle_abs_max"]),
        handoff_cart_velocity_ratio=args.handoff_cart_velocity_abs
        / float(distribution["cart_velocity_abs_max"]),
        handoff_hinge_velocity_ratio=args.handoff_hinge_velocity_rms
        / float(distribution["hinge_velocity_rms_max"]),
    )
    planner = PredictiveSamplingPlanner(
        env.model,
        frame_skip=env.frame_skip,
        force_limit=env.force_limit,
        coordinate_transform=transform,
        lyapunov=lyapunov,
        config=planner_config,
        rail_limit=env.rail_limit,
        threads=args.threads,
    )
    rng = np.random.default_rng(args.seed)
    mpc_steps = max(1, int(round(args.mpc_seconds / policy_dt)))
    latched = False
    first_handoff_step: int | None = None
    knots: np.ndarray | None = None
    actions: np.ndarray | None = None
    controller_step = 0
    plan_events: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    minimum_value = state_lyapunov(env.data.qpos, env.data.qvel, transform, lyapunov)
    initial_value = minimum_value
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    started = time.time()

    try:
        while not (terminated or truncated):
            handoff, value = planner.handoff_state(env.data.qpos, env.data.qvel)
            minimum_value = min(minimum_value, value)
            if not latched and handoff:
                latched = True
                first_handoff_step = int(env.step_count)
            use_mpc = not latched and env.step_count < mpc_steps
            if use_mpc and (actions is None or controller_step >= args.replan_steps):
                warm = None
                if knots is not None:
                    warm = shift_action_knots(
                        knots,
                        elapsed_steps=controller_step,
                        horizon_steps=planner_config.horizon_steps,
                    )
                search = planner.search(env.data, rng=rng, initial_knots=warm)
                knots = np.asarray(search["knots"], dtype=np.float64)
                actions = np.asarray(search["actions"], dtype=np.float64)
                controller_step = 0
                plan_events.append(
                    {
                        "step": int(env.step_count),
                        "time_seconds": float(env.step_count * policy_dt),
                        "score": float(search["score"]),
                        "metrics": search["metrics"],
                        "history": search["history"],
                        "knots": knots.astype(float).tolist(),
                    }
                )
                print(
                    f"plan={len(plan_events)} step={env.step_count} "
                    f"score={search['score']:.1f} "
                    f"min_v={search['metrics']['minimum_lyapunov']:.1f} "
                    f"latched={search['metrics']['latched']} "
                    f"cart={search['metrics']['max_cart_abs']:.3f}",
                    flush=True,
                )
            if use_mpc and actions is not None:
                action = float(actions[controller_step])
                controller_step += 1
                mode = "predictive_sampling"
            else:
                action = lqr_action(env, gain, scale=args.lqr_scale, cart_target=0.0)
                mode = "lqr"
            _, reward, terminated, truncated, info = env.step([action])
            row = row_from_env(
                env, step=env.step_count, action=action, reward=reward, info=info
            )
            row["controller_mode"] = mode
            row["dimensionless_lyapunov_value"] = state_lyapunov(
                env.data.qpos, env.data.qvel, transform, lyapunov
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
        "summary": "Single-state exact-model direct-action predictive-sampling diagnostic; not P1 evidence.",
        "state_index": selected_index,
        "selected_state": selected_state,
        "progress": float(args.progress),
        "seed": int(args.seed),
        "controller": {
            "type": "multithreaded_direct_action_predictive_sampling_then_lqr",
            **vars(planner_config),
            "horizon_seconds": float(planner_config.horizon_steps * policy_dt),
            "mpc_seconds": float(args.mpc_seconds),
            "threads": int(args.threads),
            "lqr_scale": float(args.lqr_scale),
            "lqr_gain": gain.astype(float).tolist(),
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
