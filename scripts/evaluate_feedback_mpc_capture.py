#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
    dimensionless_wrapped_state,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from search_swingup_capture import lqr_gain


def feedback_action(
    qpos: np.ndarray,
    qvel: np.ndarray,
    gain: np.ndarray,
    *,
    n_links: int,
    scale: float,
    cart_target: float,
    residual: float = 0.0,
    clip: bool = True,
) -> float:
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    d = n_links + 1
    if qpos.shape != (d,) or qvel.shape != (d,) or np.asarray(gain).shape != (2 * d,):
        raise ValueError("state or gain shape does not match n_links")
    state = np.r_[qpos, qvel].astype(np.float64)
    state[0] -= float(cart_target)
    state[1:d] = wrap_angle(state[1:d])
    action = -float(scale) * float(np.asarray(gain, dtype=np.float64) @ state) + float(residual)
    return float(np.clip(action, -1.0, 1.0)) if clip else action


def schedule_value(t: float, knots: np.ndarray, horizon_seconds: float) -> float:
    if t >= horizon_seconds:
        return 0.0
    times = np.linspace(0.0, horizon_seconds, len(knots), dtype=np.float64)
    return float(np.interp(max(0.0, t), times, knots))


def shift_schedule(knots: np.ndarray, elapsed_seconds: float, horizon_seconds: float) -> np.ndarray:
    knots = np.asarray(knots, dtype=np.float64)
    times = np.linspace(0.0, horizon_seconds, len(knots), dtype=np.float64)
    shifted = np.asarray(
        [schedule_value(float(t + elapsed_seconds), knots, horizon_seconds) for t in times],
        dtype=np.float64,
    )
    shifted[-1] = 0.0
    return shifted


def state_lyapunov(
    qpos: np.ndarray,
    qvel: np.ndarray,
    transform: np.ndarray,
    lyapunov: np.ndarray,
) -> float:
    state = dimensionless_wrapped_state(qpos, qvel, transform)
    if not np.all(np.isfinite(state)):
        return 1e30
    return float(max(0.0, state @ lyapunov @ state))


def reset_scratch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)


def controller_from_vector(
    vector: np.ndarray,
    *,
    target_count: int,
    residual_count: int,
    target_limit: float,
    residual_limit: float,
) -> dict[str, Any]:
    target = np.clip(vector[:target_count], -target_limit, target_limit)
    residual = np.clip(
        vector[target_count : target_count + residual_count],
        -residual_limit,
        residual_limit,
    )
    target[-1] = 0.0
    residual[-1] = 0.0
    return {
        "target_knots": target.astype(float).tolist(),
        "residual_knots": residual.astype(float).tolist(),
    }


def controller_vector(controller: dict[str, Any]) -> np.ndarray:
    return np.r_[
        np.asarray(controller["target_knots"], dtype=np.float64),
        np.asarray(controller["residual_knots"], dtype=np.float64),
    ]


def rollout_feedback_candidate(
    env: NLinkCartPoleEnv,
    scratch: mujoco.MjData,
    *,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    gain: np.ndarray,
    lqr_scale: float,
    planning_lqr_scale: float,
    transform: np.ndarray,
    lyapunov: np.ndarray,
    controller: dict[str, Any],
    horizon_steps: int,
    horizon_seconds: float,
    handoff_lyapunov: float,
    handoff_cart_abs: float,
) -> dict[str, Any]:
    reset_scratch(env.model, scratch, start_qpos, start_qvel)
    targets = np.asarray(controller["target_knots"], dtype=np.float64)
    residuals = np.asarray(controller["residual_knots"], dtype=np.float64)
    initial_value = state_lyapunov(scratch.qpos, scratch.qvel, transform, lyapunov)
    minimum_value = initial_value
    stage_log_value = 0.0
    rail_cost = 0.0
    action_energy = 0.0
    saturation_steps = 0
    latched = initial_value <= handoff_lyapunov and abs(float(scratch.qpos[0])) <= handoff_cart_abs
    first_handoff_step: int | None = 0 if latched else None
    rail_hit = False
    simulated_steps = 0

    for step in range(horizon_steps):
        t = float(step * env.dt)
        if latched:
            target = 0.0
            residual = 0.0
        else:
            target = schedule_value(t, targets, horizon_seconds)
            residual = schedule_value(t, residuals, horizon_seconds)
        raw_action = feedback_action(
            scratch.qpos,
            scratch.qvel,
            gain,
            n_links=env.n,
            scale=lqr_scale if latched else planning_lqr_scale,
            cart_target=target,
            residual=residual,
            clip=False,
        )
        action = float(np.clip(raw_action, -1.0, 1.0))
        saturation_steps += int(abs(raw_action) > 1.0)
        action_energy += action * action
        scratch.ctrl[0] = action * env.force_limit
        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, scratch)
        simulated_steps = step + 1
        if not np.all(np.isfinite(scratch.qpos)) or not np.all(np.isfinite(scratch.qvel)):
            rail_hit = True
            break
        cart_abs = abs(float(scratch.qpos[0]))
        if cart_abs > env.rail_limit:
            rail_hit = True
            break
        value = state_lyapunov(scratch.qpos, scratch.qvel, transform, lyapunov)
        minimum_value = min(minimum_value, value)
        stage_log_value += float(np.log1p(value))
        rail_ratio = cart_abs / env.rail_limit
        rail_cost += rail_ratio**8
        if not latched and value <= handoff_lyapunov and cart_abs <= handoff_cart_abs:
            latched = True
            first_handoff_step = step + 1

    terminal_value = state_lyapunov(scratch.qpos, scratch.qvel, transform, lyapunov)
    score_terms = {
        "rail_hit": 10_000_000.0 * float(rail_hit),
        "terminal_lyapunov": 3_000.0 * float(np.log1p(terminal_value)),
        "minimum_lyapunov": 300.0 * float(np.log1p(minimum_value)),
        "stage_lyapunov": 2.0 * stage_log_value,
        "rail_barrier": 5_000.0 * rail_cost,
        "action_energy": 2.0 * action_energy,
        "saturation": 200.0 * float(saturation_steps / max(1, simulated_steps)),
        "handoff": -50_000.0 * float(latched and not rail_hit),
    }
    return {
        "score": float(sum(score_terms.values())),
        "score_terms": score_terms,
        "initial_lyapunov": float(initial_value),
        "minimum_lyapunov": float(minimum_value),
        "terminal_lyapunov": float(terminal_value),
        "latched": bool(latched),
        "first_handoff_step": first_handoff_step,
        "rail_hit": bool(rail_hit),
        "simulated_steps": int(simulated_steps),
        "terminal_qpos": np.asarray(scratch.qpos, dtype=np.float64).astype(float).tolist(),
        "terminal_qvel": np.asarray(scratch.qvel, dtype=np.float64).astype(float).tolist(),
    }


def search_feedback_plan(
    env: NLinkCartPoleEnv,
    scratch: mujoco.MjData,
    *,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    gain: np.ndarray,
    lqr_scale: float,
    planning_lqr_scale: float,
    transform: np.ndarray,
    lyapunov: np.ndarray,
    initial_controller: dict[str, Any],
    rng: np.random.Generator,
    horizon_steps: int,
    horizon_seconds: float,
    handoff_lyapunov: float,
    handoff_cart_abs: float,
    iterations: int,
    population: int,
    elites: int,
    target_count: int,
    residual_count: int,
    target_limit: float,
    residual_limit: float,
    target_sigma: float,
    residual_sigma: float,
) -> dict[str, Any]:
    center = controller_vector(initial_controller)
    expected_shape = (target_count + residual_count,)
    if center.shape != expected_shape:
        raise ValueError("initial controller dimensions do not match feedback MPC")
    sigma = np.r_[
        np.full(target_count, target_sigma, dtype=np.float64),
        np.full(residual_count, residual_sigma, dtype=np.float64),
    ]
    floors = np.r_[
        np.full(target_count, 0.025, dtype=np.float64),
        np.full(residual_count, 0.01, dtype=np.float64),
    ]
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for iteration in range(iterations):
        candidates = [center.copy(), np.zeros_like(center)]
        candidates.extend(center + rng.normal(0.0, sigma) for _ in range(max(0, population - 2)))
        records = []
        for vector in candidates:
            controller = controller_from_vector(
                vector,
                target_count=target_count,
                residual_count=residual_count,
                target_limit=target_limit,
                residual_limit=residual_limit,
            )
            metrics = rollout_feedback_candidate(
                env,
                scratch,
                start_qpos=start_qpos,
                start_qvel=start_qvel,
                gain=gain,
                lqr_scale=lqr_scale,
                planning_lqr_scale=planning_lqr_scale,
                transform=transform,
                lyapunov=lyapunov,
                controller=controller,
                horizon_steps=horizon_steps,
                horizon_seconds=horizon_seconds,
                handoff_lyapunov=handoff_lyapunov,
                handoff_cart_abs=handoff_cart_abs,
            )
            records.append({"controller": controller, "metrics": metrics})
        records.sort(key=lambda row: float(row["metrics"]["score"]))
        if best is None or float(records[0]["metrics"]["score"]) < float(best["metrics"]["score"]):
            best = records[0]
        history.append(
            {
                "iteration": int(iteration + 1),
                "score": float(records[0]["metrics"]["score"]),
                "terminal_lyapunov": float(records[0]["metrics"]["terminal_lyapunov"]),
                "minimum_lyapunov": float(records[0]["metrics"]["minimum_lyapunov"]),
                "latched": bool(records[0]["metrics"]["latched"]),
                "rail_hit": bool(records[0]["metrics"]["rail_hit"]),
            }
        )
        elite = np.asarray(
            [controller_vector(row["controller"]) for row in records[:elites]],
            dtype=np.float64,
        )
        center = elite.mean(axis=0)
        sigma = np.maximum(0.90 * elite.std(axis=0), floors)

    assert best is not None
    return {"best": best, "history": history}


def shifted_controller(
    controller: dict[str, Any] | None,
    *,
    elapsed_seconds: float,
    horizon_seconds: float,
    target_count: int,
    residual_count: int,
) -> dict[str, Any]:
    if controller is None:
        return {
            "target_knots": np.zeros(target_count, dtype=np.float64).tolist(),
            "residual_knots": np.zeros(residual_count, dtype=np.float64).tolist(),
        }
    return {
        "target_knots": shift_schedule(
            np.asarray(controller["target_knots"], dtype=np.float64),
            elapsed_seconds,
            horizon_seconds,
        ).tolist(),
        "residual_knots": shift_schedule(
            np.asarray(controller["residual_knots"], dtype=np.float64),
            elapsed_seconds,
            horizon_seconds,
        ).tolist(),
    }


def evaluate_feedback_mpc(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    gain: np.ndarray,
    lqr_scale: float,
    planning_lqr_scale: float,
    transform: np.ndarray,
    lyapunov: np.ndarray,
    horizon_steps: int,
    replan_steps: int,
    mpc_steps: int,
    handoff_lyapunov: float,
    handoff_cart_abs: float,
    iterations: int,
    population: int,
    elites: int,
    target_count: int,
    residual_count: int,
    target_limit: float,
    residual_limit: float,
    target_sigma: float,
    residual_sigma: float,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    env.reset(seed=seed)
    scratch = mujoco.MjData(env.model)
    rng = np.random.default_rng(seed)
    horizon_seconds = float(horizon_steps * env.dt)
    controller: dict[str, Any] | None = None
    controller_step = 0
    latched = False
    first_handoff_step: int | None = None
    plan_events: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    episode_return = 0.0
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    initial_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    initial_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
    initial_observation = env._get_obs().copy()
    initial_value = state_lyapunov(env.data.qpos, env.data.qvel, transform, lyapunov)
    minimum_value = initial_value
    planning_started = time.time()

    try:
        while not (terminated or truncated):
            current_value = state_lyapunov(env.data.qpos, env.data.qvel, transform, lyapunov)
            minimum_value = min(minimum_value, current_value)
            if (
                not latched
                and current_value <= handoff_lyapunov
                and abs(float(env.data.qpos[0])) <= handoff_cart_abs
            ):
                latched = True
                first_handoff_step = int(env.step_count)

            use_mpc = not latched and env.step_count < mpc_steps
            if use_mpc and (controller is None or controller_step >= replan_steps):
                warm_start = shifted_controller(
                    controller,
                    elapsed_seconds=float(controller_step * env.dt),
                    horizon_seconds=horizon_seconds,
                    target_count=target_count,
                    residual_count=residual_count,
                )
                search = search_feedback_plan(
                    env,
                    scratch,
                    start_qpos=np.asarray(env.data.qpos, dtype=np.float64).copy(),
                    start_qvel=np.asarray(env.data.qvel, dtype=np.float64).copy(),
                    gain=gain,
                    lqr_scale=lqr_scale,
                    planning_lqr_scale=planning_lqr_scale,
                    transform=transform,
                    lyapunov=lyapunov,
                    initial_controller=warm_start,
                    rng=rng,
                    horizon_steps=horizon_steps,
                    horizon_seconds=horizon_seconds,
                    handoff_lyapunov=handoff_lyapunov,
                    handoff_cart_abs=handoff_cart_abs,
                    iterations=iterations,
                    population=population,
                    elites=elites,
                    target_count=target_count,
                    residual_count=residual_count,
                    target_limit=target_limit,
                    residual_limit=residual_limit,
                    target_sigma=target_sigma,
                    residual_sigma=residual_sigma,
                )
                controller = search["best"]["controller"]
                controller_step = 0
                plan_events.append(
                    {
                        "step": int(env.step_count),
                        "time_seconds": float(env.step_count * env.dt),
                        "controller": controller,
                        "best_metrics": search["best"]["metrics"],
                        "history": search["history"],
                    }
                )

            if use_mpc and controller is not None:
                t = float(controller_step * env.dt)
                target = schedule_value(
                    t,
                    np.asarray(controller["target_knots"], dtype=np.float64),
                    horizon_seconds,
                )
                residual = schedule_value(
                    t,
                    np.asarray(controller["residual_knots"], dtype=np.float64),
                    horizon_seconds,
                )
                controller_step += 1
            else:
                target = 0.0
                residual = 0.0
            action = feedback_action(
                env.data.qpos,
                env.data.qvel,
                gain,
                n_links=env.n,
                scale=planning_lqr_scale if use_mpc else lqr_scale,
                cart_target=target,
                residual=residual,
            )
            pre_action_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
            pre_action_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
            pre_action_observation = env._get_obs().copy()
            pre_action_value = state_lyapunov(
                pre_action_qpos,
                pre_action_qvel,
                transform,
                lyapunov,
            )
            _, reward, terminated, truncated, info = env.step([action])
            episode_return += float(reward)
            value = state_lyapunov(env.data.qpos, env.data.qvel, transform, lyapunov)
            minimum_value = min(minimum_value, value)
            row = row_from_env(env, step=env.step_count, action=action, reward=reward, info=info)
            row.update(
                {
                    "controller_mode": "lqr" if latched or not use_mpc else "feedback_mpc",
                    "cart_target": float(target),
                    "residual": float(residual),
                    "pre_action_qpos": pre_action_qpos.astype(float).tolist(),
                    "pre_action_qvel": pre_action_qvel.astype(float).tolist(),
                    "pre_action_observation": pre_action_observation.astype(float).tolist(),
                    "pre_action_dimensionless_lyapunov_value": float(pre_action_value),
                    "dimensionless_lyapunov_value": float(value),
                }
            )
            trajectory.append(row)
            final_info = dict(info)
    finally:
        env.close()

    success = bool(final_info.get("success", False))
    return {
        "success": success,
        "return": float(episode_return),
        "length": int(len(trajectory)),
        "termination_reason": final_info.get("termination_reason"),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "time_to_capture": final_info.get("time_to_capture"),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(
            final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(final_info.get("max_cart_excursion", 0.0)),
        "rail_hit": final_info.get("termination_reason") == "rail_violation",
        "initial_qpos": initial_qpos.astype(float).tolist(),
        "initial_qvel": initial_qvel.astype(float).tolist(),
        "initial_observation": initial_observation.astype(float).tolist(),
        "initial_lyapunov": float(initial_value),
        "minimum_lyapunov": float(minimum_value),
        "latched": bool(latched),
        "first_handoff_step": first_handoff_step,
        "first_handoff_time": (
            None if first_handoff_step is None else float(first_handoff_step * float(cfg["env"]["timestep"]) * int(cfg["env"].get("frame_skip", 1)))
        ),
        "planning_wall_time_seconds": float(time.time() - planning_started),
        "plan_event_count": int(len(plan_events)),
        "plan_events": plan_events,
        "trajectory": trajectory,
        "final_info": final_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate exact-model receding-horizon feedback MPC on one capture state"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--state-json", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--progress", type=float, default=0.065)
    parser.add_argument("--seed", type=int, default=63001)
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon-steps", type=int, default=75)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--mpc-seconds", type=float, default=2.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=2.4)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--target-count", type=int, default=6)
    parser.add_argument("--residual-count", type=int, default=8)
    parser.add_argument("--target-limit", type=float, default=1.5)
    parser.add_argument("--residual-limit", type=float, default=0.4)
    parser.add_argument("--target-sigma", type=float, default=0.5)
    parser.add_argument("--residual-sigma", type=float, default=0.15)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--planning-lqr-scale", type=float, default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.horizon_steps < 2 or args.replan_steps < 1:
        raise ValueError("horizon steps must be >= 2 and replan steps must be positive")
    if args.iterations < 1 or args.population < 2 or not 1 <= args.elites <= args.population:
        raise ValueError("invalid CEM iterations, population, or elite count")
    if min(args.target_count, args.residual_count) < 2:
        raise ValueError("target and residual schedules require at least two knots")
    if min(
        args.mpc_seconds,
        args.handoff_lyapunov,
        args.handoff_cart_abs,
        args.target_limit,
        args.residual_limit,
        args.target_sigma,
        args.residual_sigma,
    ) <= 0.0:
        raise ValueError("MPC durations, thresholds, limits, and sigmas must be positive")

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
    policy_dt = float(cfg["env"]["timestep"]) * int(cfg["env"].get("frame_skip", 1))
    result = evaluate_feedback_mpc(
        cfg,
        progress=args.progress,
        seed=args.seed,
        gain=gain,
        lqr_scale=args.lqr_scale,
        planning_lqr_scale=(
            args.lqr_scale if args.planning_lqr_scale is None else args.planning_lqr_scale
        ),
        transform=transform,
        lyapunov=lyapunov,
        horizon_steps=args.horizon_steps,
        replan_steps=args.replan_steps,
        mpc_steps=max(1, int(round(args.mpc_seconds / policy_dt))),
        handoff_lyapunov=args.handoff_lyapunov,
        handoff_cart_abs=args.handoff_cart_abs,
        iterations=args.iterations,
        population=args.population,
        elites=args.elites,
        target_count=args.target_count,
        residual_count=args.residual_count,
        target_limit=args.target_limit,
        residual_limit=args.residual_limit,
        target_sigma=args.target_sigma,
        residual_sigma=args.residual_sigma,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Single-state exact-model feedback MPC capture diagnostic; not P1 gate evidence.",
        "state_index": selected_index,
        "selected_state": selected_state,
        "progress": float(args.progress),
        "seed": int(args.seed),
        "controller": {
            "type": "receding_horizon_cem_target_residual_plus_lqr_feedback",
            "horizon_steps": int(args.horizon_steps),
            "horizon_seconds": float(args.horizon_steps * policy_dt),
            "replan_steps": int(args.replan_steps),
            "mpc_seconds": float(args.mpc_seconds),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "target_count": int(args.target_count),
            "residual_count": int(args.residual_count),
            "target_limit": float(args.target_limit),
            "residual_limit": float(args.residual_limit),
            "target_sigma": float(args.target_sigma),
            "residual_sigma": float(args.residual_sigma),
            "lqr_scale": float(args.lqr_scale),
            "planning_lqr_scale": float(
                args.lqr_scale if args.planning_lqr_scale is None else args.planning_lqr_scale
            ),
            "lqr_gain": gain.astype(float).tolist(),
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
        f"success={result['success']} latched={result['latched']} "
        f"initial_v={result['initial_lyapunov']:.2f} min_v={result['minimum_lyapunov']:.2f} "
        f"hold={result['max_upright_streak_seconds']:.3f}s cart={result['max_cart_excursion']:.3f} "
        f"plans={result['plan_event_count']} wall={result['planning_wall_time_seconds']:.1f}s"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
