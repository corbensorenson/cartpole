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

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_capture_sequence import (
        fixed_state_cfg,
        load_state,
        row_from_env,
    )
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from search_swingup_capture import lqr_action, lqr_gain


def lyapunov_value(
    state: np.ndarray, transform: np.ndarray, lyapunov: np.ndarray
) -> float:
    dimensionless = dimensionless_wrapped_state(
        state[: transform.shape[0] // 2],
        state[transform.shape[0] // 2 :],
        transform,
    )
    return float(max(0.0, dimensionless @ lyapunov @ dimensionless))


def initial_controls(
    transition: MujocoTransition,
    state: np.ndarray,
    gain: np.ndarray,
    *,
    horizon_steps: int,
    lqr_scale: float,
) -> np.ndarray:
    controls = np.empty(horizon_steps, dtype=np.float64)
    current = state.copy()
    for step in range(horizon_steps):
        wrapped = transition.to_physical(current)
        wrapped[1 : transition.env.n + 1] = (
            wrapped[1 : transition.env.n + 1] + np.pi
        ) % (2.0 * np.pi) - np.pi
        action = -float(lqr_scale) * float(gain @ wrapped)
        controls[step] = np.clip(action, -1.0, 1.0)
        current = transition(current, float(controls[step]))
    return controls


def load_initial_controls(
    path: str, horizon_steps: int, policy_dt: float
) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    saved_controls = payload.get("controller", {}).get("controls")
    trajectory_controls = saved_controls is None
    if trajectory_controls:
        saved_controls = [
            row["action"] for row in payload.get("result", {}).get("trajectory", [])
        ]
    controls = np.asarray(saved_controls, dtype=np.float64)
    if controls.ndim != 1 or len(controls) < 2:
        raise ValueError("initial controller must contain at least two scalar controls")
    if trajectory_controls and len(controls) >= horizon_steps:
        return np.clip(controls[:horizon_steps], -1.0, 1.0)
    if len(controls) == horizon_steps:
        return np.clip(controls, -1.0, 1.0)
    source_seconds = (
        len(controls) * policy_dt
        if trajectory_controls
        else float(payload["controller"]["horizon_seconds"])
    )
    source_dt = source_seconds / len(controls)
    source_time = np.arange(len(controls), dtype=np.float64) * source_dt
    target_time = np.arange(horizon_steps, dtype=np.float64) * policy_dt
    return np.clip(np.interp(target_time, source_time, controls, right=0.0), -1.0, 1.0)


def interpolate_initial_state(
    source: dict[str, Any], target: dict[str, Any], alpha: float
) -> dict[str, Any]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("interpolation alpha must lie in [0, 1]")
    source_qpos = np.asarray(source["qpos"], dtype=np.float64)
    target_qpos = np.asarray(target["qpos"], dtype=np.float64)
    source_qvel = np.asarray(source["qvel"], dtype=np.float64)
    target_qvel = np.asarray(target["qvel"], dtype=np.float64)
    if (
        source_qpos.shape != target_qpos.shape
        or source_qvel.shape != target_qvel.shape
        or source_qpos.size != source_qvel.size
    ):
        raise ValueError("source and target states must have matching qpos/qvel")
    qpos = (1.0 - alpha) * source_qpos + alpha * target_qpos
    qvel = (1.0 - alpha) * source_qvel + alpha * target_qvel
    absolute_angles = np.cumsum(qpos[1:])
    absolute_angles = (absolute_angles + np.pi) % (2.0 * np.pi) - np.pi
    return {
        "state_id": (
            f"interpolation-{source.get('state_id', 'source')}-to-"
            f"{target.get('state_id', 'target')}-alpha-{alpha:.8f}"
        ),
        "source": "state_space_continuation",
        "qpos": qpos.astype(float).tolist(),
        "qvel": qvel.astype(float).tolist(),
        "absolute_angles": absolute_angles.astype(float).tolist(),
        "cart_velocity": float(qvel[0]),
        "hinge_velocity_rms": float(np.sqrt(np.mean(qvel[1:] ** 2))),
    }


def handoff_bounds_satisfied(
    state: np.ndarray,
    n_links: int,
    *,
    angle_abs: float | None,
    cart_velocity_abs: float | None,
    hinge_velocity_rms: float | None,
) -> bool:
    state = np.asarray(state, dtype=np.float64)
    nq = n_links + 1
    relative_angles = (state[1:nq] + np.pi) % (2.0 * np.pi) - np.pi
    absolute_angles = np.cumsum(relative_angles)
    return bool(
        (angle_abs is None or np.max(np.abs(absolute_angles)) <= angle_abs)
        and (cart_velocity_abs is None or abs(float(state[nq])) <= cart_velocity_abs)
        and (
            hinge_velocity_rms is None
            or float(np.sqrt(np.mean(state[nq + 1 :] ** 2))) <= hinge_velocity_rms
        )
    )


def execute_controller(
    cfg: dict[str, Any],
    *,
    seed: int,
    controls: np.ndarray,
    nominal_states: np.ndarray,
    feedback_gains: np.ndarray,
    gain: np.ndarray,
    lqr_scale: float,
    transform: np.ndarray,
    lyapunov: np.ndarray,
    handoff_lyapunov: float,
    handoff_cart_abs: float,
    handoff_angle_abs: float | None = None,
    handoff_cart_velocity_abs: float | None = None,
    handoff_hinge_velocity_rms: float | None = None,
    tracking_mode: str = "ilqr_tracking",
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
    env.reset(seed=seed)
    trajectory: list[dict[str, Any]] = []
    latched = False
    first_handoff_step: int | None = None
    minimum_value = lyapunov_value(data_state(env.data), transform, lyapunov)
    episode_return = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    try:
        while not (terminated or truncated):
            state = data_state(env.data)
            value = lyapunov_value(state, transform, lyapunov)
            minimum_value = min(minimum_value, value)
            if (
                not latched
                and value <= handoff_lyapunov
                and abs(float(state[0])) <= handoff_cart_abs
                and handoff_bounds_satisfied(
                    state,
                    env.n,
                    angle_abs=handoff_angle_abs,
                    cart_velocity_abs=handoff_cart_velocity_abs,
                    hinge_velocity_rms=handoff_hinge_velocity_rms,
                )
            ):
                latched = True
                first_handoff_step = int(env.step_count)
            if not latched and env.step_count < len(controls):
                step = int(env.step_count)
                coordinate_state = dimensionless_wrapped_state(
                    env.data.qpos, env.data.qvel, transform
                )
                error = coordinate_state - nominal_states[step]
                action = float(
                    np.clip(controls[step] + feedback_gains[step] @ error, -1.0, 1.0)
                )
                mode = tracking_mode
            else:
                action = lqr_action(env, gain, scale=lqr_scale, cart_target=0.0)
                mode = "lqr"
            _, reward, terminated, truncated, info = env.step([action])
            episode_return += float(reward)
            row = row_from_env(
                env, step=env.step_count, action=action, reward=reward, info=info
            )
            row["controller_mode"] = mode
            row["dimensionless_lyapunov_value"] = lyapunov_value(
                data_state(env.data), transform, lyapunov
            )
            trajectory.append(row)
    finally:
        env.close()
    return {
        "success": bool(info.get("success", False)),
        "return": float(episode_return),
        "length": len(trajectory),
        "termination_reason": info.get("termination_reason"),
        "max_upright_streak_seconds": float(
            info.get("max_upright_streak_seconds", 0.0)
        ),
        "max_low_momentum_upright_streak_seconds": float(
            info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(info.get("max_cart_excursion", 0.0)),
        "minimum_lyapunov": float(minimum_value),
        "latched": bool(latched),
        "first_handoff_step": first_handoff_step,
        "first_handoff_time": None
        if first_handoff_step is None
        else first_handoff_step * env.dt,
        "trajectory": trajectory,
        "final_info": info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search a constrained nonlinear iLQR capture trajectory in exact MuJoCo"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--state-json", default="runs/p1_capture_envelope/validation.json"
    )
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--interpolate-from-state-index", default=None)
    parser.add_argument("--interpolation-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=63001)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon-seconds", type=float, default=3.0)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--initial-controller", default=None)
    parser.add_argument("--initial-lqr-scale", type=float, default=0.0)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--control-cost", type=float, default=0.1)
    parser.add_argument("--stage-weight", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--terminal-state-weight", type=float, default=0.0)
    parser.add_argument("--rail-soft-limit", type=float, default=2.4)
    parser.add_argument("--rail-weight", type=float, default=100_000_000.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--switch-lyapunov", type=float, default=None)
    parser.add_argument("--switch-angle-abs", type=float, default=None)
    parser.add_argument("--switch-cart-velocity-abs", type=float, default=None)
    parser.add_argument("--switch-hinge-velocity-rms", type=float, default=None)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if (
        min(
            args.horizon_seconds,
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
        or args.iterations < 1
    ):
        raise ValueError(
            "durations, weights, thresholds, and iterations must be positive"
        )
    if args.switch_lyapunov is not None and args.switch_lyapunov <= 0.0:
        raise ValueError("switch Lyapunov threshold must be positive")
    if any(
        value is not None and value <= 0.0
        for value in (
            args.switch_angle_abs,
            args.switch_cart_velocity_abs,
            args.switch_hinge_velocity_rms,
        )
    ):
        raise ValueError("switch state bounds must be positive")

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
    policy_dt = float(env.dt)
    horizon_steps = max(2, int(round(args.horizon_seconds / policy_dt)))
    start_state = transition.to_coordinates(data_state(env.data))
    if args.initial_controller is None:
        controls = initial_controls(
            transition,
            start_state,
            gain,
            horizon_steps=horizon_steps,
            lqr_scale=args.initial_lqr_scale,
        )
    else:
        controls = load_initial_controls(
            args.initial_controller, horizon_steps, policy_dt
        )
    stage_metric = args.stage_weight * np.eye(transform.shape[0], dtype=np.float64)
    terminal_metric = (
        args.terminal_weight * lyapunov / args.handoff_lyapunov
        + args.terminal_state_weight * np.eye(transform.shape[0], dtype=np.float64)
    )
    trajectory_cost = QuadraticTrajectoryCost(
        stage_state=stage_metric,
        terminal_state=terminal_metric,
        control=float(args.control_cost),
        rail_soft_limit=float(args.rail_soft_limit * transform[0, 0]),
        rail_limit=float(env.rail_limit * transform[0, 0]),
        rail_weight=float(args.rail_weight),
        wrap_angles=False,
    )
    started = time.time()
    search = optimize_ilqr(
        transition,
        start_state,
        controls,
        trajectory_cost,
        max_iterations=args.iterations,
    )
    search_seconds = time.time() - started
    env.close()
    nominal_values = [float(max(0.0, row @ lyapunov @ row)) for row in search.states]
    nominal_physical_states = np.asarray(
        [transition.to_physical(row) for row in search.states], dtype=np.float64
    )
    result = execute_controller(
        cfg,
        seed=args.seed,
        controls=search.controls,
        nominal_states=search.states,
        feedback_gains=search.feedback_gains,
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
        handoff_angle_abs=args.switch_angle_abs,
        handoff_cart_velocity_abs=args.switch_cart_velocity_abs,
        handoff_hinge_velocity_rms=args.switch_hinge_velocity_rms,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Single-state constrained nonlinear iLQR capture diagnostic; not P1 evidence.",
        "state_index": state_index,
        "selected_state": state,
        "state_interpolation": interpolation,
        "seed": int(args.seed),
        "controller": {
            "type": "exact_mujoco_box_constrained_ilqr_tracking_then_lqr",
            "control_constraint_method": "scalar_active_set",
            "horizon_steps": horizon_steps,
            "horizon_seconds": horizon_steps * policy_dt,
            "iterations": int(args.iterations),
            "initial_lqr_scale": float(args.initial_lqr_scale),
            "initial_controller": (
                None
                if args.initial_controller is None
                else file_metadata(args.initial_controller)
            ),
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
            "switch_angle_abs": args.switch_angle_abs,
            "switch_cart_velocity_abs": args.switch_cart_velocity_abs,
            "switch_hinge_velocity_rms": args.switch_hinge_velocity_rms,
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "controls": search.controls.astype(float).tolist(),
            "feedback_gains": search.feedback_gains.astype(float).tolist(),
        },
        "search": {
            "cost": float(search.cost),
            "iterations": int(search.iterations),
            "converged": bool(search.converged),
            "active_control_steps": int(search.active_control_steps),
            "wall_time_seconds": float(search_seconds),
            "initial_lyapunov": float(nominal_values[0]),
            "minimum_lyapunov": float(min(nominal_values)),
            "terminal_lyapunov": float(nominal_values[-1]),
            "terminal_dimensionless_state_norm": float(
                np.linalg.norm(search.states[-1])
            ),
            "max_cart_excursion": float(np.max(np.abs(nominal_physical_states[:, 0]))),
            "history": search.history,
            "nominal_coordinate_states": search.states.astype(float).tolist(),
            "nominal_physical_states": nominal_physical_states.astype(float).tolist(),
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
        f"search_min_v={min(nominal_values):.2f} terminal_v={nominal_values[-1]:.2f} "
        f"search_cart={payload['search']['max_cart_excursion']:.3f} wall={search_seconds:.1f}s"
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
