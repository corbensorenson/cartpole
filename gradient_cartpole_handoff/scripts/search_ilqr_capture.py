#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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
    dimensionless_wrapped_state,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from search_swingup_capture import lqr_action, lqr_gain


def lyapunov_value(state: np.ndarray, transform: np.ndarray, lyapunov: np.ndarray) -> float:
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
            (wrapped[1 : transition.env.n + 1] + np.pi) % (2.0 * np.pi) - np.pi
        )
        action = -float(lqr_scale) * float(gain @ wrapped)
        controls[step] = np.clip(action, -1.0, 1.0)
        current = transition(current, float(controls[step]))
    return controls


def load_initial_controls(path: str, horizon_steps: int, policy_dt: float) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    controls = np.asarray(payload["controller"]["controls"], dtype=np.float64)
    if controls.ndim != 1 or len(controls) < 2:
        raise ValueError("initial controller must contain at least two scalar controls")
    if len(controls) == horizon_steps:
        return np.clip(controls, -1.0, 1.0)
    source_seconds = float(payload["controller"]["horizon_seconds"])
    source_dt = source_seconds / len(controls)
    source_time = np.arange(len(controls), dtype=np.float64) * source_dt
    target_time = np.arange(horizon_steps, dtype=np.float64) * policy_dt
    return np.clip(np.interp(target_time, source_time, controls, right=0.0), -1.0, 1.0)


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
            if not latched and value <= handoff_lyapunov and abs(float(state[0])) <= handoff_cart_abs:
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
                mode = "ilqr_tracking"
            else:
                action = lqr_action(env, gain, scale=lqr_scale, cart_target=0.0)
                mode = "lqr"
            _, reward, terminated, truncated, info = env.step([action])
            episode_return += float(reward)
            row = row_from_env(env, step=env.step_count, action=action, reward=reward, info=info)
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
        "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(
            info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(info.get("max_cart_excursion", 0.0)),
        "minimum_lyapunov": float(minimum_value),
        "latched": bool(latched),
        "first_handoff_step": first_handoff_step,
        "first_handoff_time": None if first_handoff_step is None else first_handoff_step * env.dt,
        "trajectory": trajectory,
        "final_info": info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search a constrained nonlinear iLQR capture trajectory in exact MuJoCo"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--state-json", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--state-index", required=True)
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
    parser.add_argument("--rail-soft-limit", type=float, default=2.4)
    parser.add_argument("--rail-weight", type=float, default=100_000_000.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if min(
        args.horizon_seconds,
        args.control_cost,
        args.stage_weight,
        args.terminal_weight,
        args.rail_soft_limit,
        args.rail_weight,
        args.handoff_lyapunov,
        args.handoff_cart_abs,
    ) <= 0.0 or args.iterations < 1:
        raise ValueError("durations, weights, thresholds, and iterations must be positive")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    state, state_index = load_state(args.state_json, args.state_index)
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
        controls = load_initial_controls(args.initial_controller, horizon_steps, policy_dt)
    stage_metric = args.stage_weight * np.eye(transform.shape[0], dtype=np.float64)
    terminal_metric = args.terminal_weight * lyapunov / args.handoff_lyapunov
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
        handoff_lyapunov=args.handoff_lyapunov,
        handoff_cart_abs=args.handoff_cart_abs,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Single-state constrained nonlinear iLQR capture diagnostic; not P1 evidence.",
        "state_index": state_index,
        "selected_state": state,
        "seed": int(args.seed),
        "controller": {
            "type": "exact_mujoco_ilqr_tracking_then_lqr",
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
            "rail_soft_limit": float(args.rail_soft_limit),
            "rail_weight": float(args.rail_weight),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "controls": search.controls.astype(float).tolist(),
            "feedback_gains": search.feedback_gains.astype(float).tolist(),
        },
        "search": {
            "cost": float(search.cost),
            "iterations": int(search.iterations),
            "converged": bool(search.converged),
            "wall_time_seconds": float(search_seconds),
            "initial_lyapunov": float(nominal_values[0]),
            "minimum_lyapunov": float(min(nominal_values)),
            "terminal_lyapunov": float(nominal_values[-1]),
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
            "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(cfg)},
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
