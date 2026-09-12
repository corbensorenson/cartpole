#!/usr/bin/env python
"""Receding-horizon nonlinear capture search from a real MuJoCo state.

This is a component diagnostic.  It never overwrites the live rollout state:
MuJoCo batch rollouts are used only for planning, and the selected first
actions are applied to one uninterrupted environment instance.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from torch_runtime import prepare_runtime

prepare_runtime()

import mujoco
import numpy as np
from mujoco import rollout as mujoco_rollout

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp


def load_state(path: str, index: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(states, list) or not states:
        raise ValueError(f"{path} does not contain a non-empty states list")
    if index == "best":
        selected = min(
            states,
            key=lambda state: (
                float(state.get("max_abs_angle", np.inf)),
                float(state.get("hinge_velocity_rms", np.inf)),
            ),
        )
    else:
        selected = states[int(index)]
    return dict(selected)


def fixed_state_cfg(
    cfg: dict[str, Any], state: dict[str, Any], seconds: float
) -> dict[str, Any]:
    env_cfg = {
        **cfg["env"],
        "init_mode": "fixed_state",
        "init_qpos": state["qpos"],
        "init_qvel": state["qvel"],
        "episode_seconds": float(seconds),
        "terminate_abs_angle": None,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in (
        "init_cart_noise",
        "init_cart_vel_noise",
        "init_angle_noise",
        "init_vel_noise",
    ):
        env_cfg[f"{key}_start"] = 0.0
        env_cfg[f"{key}_end"] = 0.0
    return {**cfg, "env": env_cfg}


def interpolation_matrix(knot_count: int, step_count: int) -> np.ndarray:
    if knot_count < 2 or step_count < 2:
        raise ValueError("knot_count and step_count must be at least two")
    source = np.linspace(0.0, 1.0, knot_count, dtype=np.float64)
    target = np.linspace(0.0, 1.0, step_count, dtype=np.float64)
    right = np.searchsorted(source, target, side="right").clip(1, knot_count - 1)
    left = right - 1
    fraction = (target - source[left]) / (source[right] - source[left])
    matrix = np.zeros((step_count, knot_count), dtype=np.float64)
    rows = np.arange(step_count)
    matrix[rows, left] = 1.0 - fraction
    matrix[rows, right] += fraction
    return matrix


def rollout_metrics(
    states: np.ndarray,
    *,
    n_links: int,
    interpolation: np.ndarray,
    profiles: np.ndarray,
    env: NLinkCartPoleEnv,
    rail_soft_limit: float,
    angle_scale: float,
    hinge_scale: float,
    absolute_rate_scale: float,
    cart_scale: float,
    cart_velocity_scale: float,
    action_weight: float,
    slew_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    nq = n_links + 1
    qpos = states[..., 1 : 1 + nq]
    qvel = states[..., 1 + nq : 1 + nq + nq]
    absolute_angles = serial_absolute_angles(qpos[..., 1:])
    absolute_omega = np.cumsum(qvel[..., 1:], axis=-1)
    max_angle = np.max(np.abs(absolute_angles), axis=-1)
    hinge_rms = np.sqrt(np.mean(qvel[..., 1:] ** 2, axis=-1))
    absolute_rate_rms = np.sqrt(np.mean(absolute_omega**2, axis=-1))
    cart_abs = np.abs(qpos[..., 0])
    cart_velocity_abs = np.abs(qvel[..., 0])

    angle_cost = (max_angle / max(1e-9, angle_scale)) ** 2
    hinge_cost = (hinge_rms / max(1e-9, hinge_scale)) ** 2
    absolute_rate_cost = (absolute_rate_rms / max(1e-9, absolute_rate_scale)) ** 2
    cart_cost = (cart_abs / max(1e-9, cart_scale)) ** 2
    cart_velocity_cost = (
        cart_velocity_abs / max(1e-9, cart_velocity_scale)
    ) ** 2
    stage_cost = (
        18.0 * angle_cost
        + 3.0 * hinge_cost
        + 5.0 * absolute_rate_cost
        + 2.0 * cart_cost
        + 2.0 * cart_velocity_cost
    )
    late_start = max(0, stage_cost.shape[1] // 3)
    late = stage_cost[:, late_start:]
    best_cost = np.min(late, axis=1)
    best_index = late_start + np.argmin(late, axis=1)
    terminal_cost = stage_cost[:, -1]
    tail_cost = np.mean(stage_cost[:, -min(10, stage_cost.shape[1]) :], axis=1)

    profile_actions = np.clip(profiles @ interpolation.T, -1.0, 1.0)
    action_cost = np.mean(profile_actions**2, axis=1)
    slew_cost = np.mean(np.diff(profile_actions, axis=1) ** 2, axis=1)
    max_cart = np.max(cart_abs, axis=1)
    rail_over = np.maximum(0.0, max_cart / max(1e-9, rail_soft_limit) - 1.0)
    rail_penalty = 2.0e5 * rail_over**2
    finite = np.all(np.isfinite(states), axis=(1, 2))
    bounded = np.max(np.abs(np.nan_to_num(states, nan=np.inf)), axis=(1, 2)) < 1.0e6
    time_valid = np.all(np.diff(states[..., 0], axis=1) > 0.0, axis=1)
    invalid = ~(finite & bounded & time_valid)
    cost = (
        0.50 * best_cost
        + 0.25 * terminal_cost
        + 0.25 * tail_cost
        + float(action_weight) * action_cost
        + float(slew_weight) * slew_cost
        + rail_penalty
    )
    cost = np.where(invalid, 1.0e30, cost)
    metrics = {
        "qpos": qpos,
        "qvel": qvel,
        "absolute_angles": absolute_angles,
        "absolute_omega": absolute_omega,
        "max_angle": max_angle,
        "hinge_rms": hinge_rms,
        "absolute_rate_rms": absolute_rate_rms,
        "cart_abs": cart_abs,
        "cart_velocity_abs": cart_velocity_abs,
        "stage_cost": stage_cost,
        "best_cost": best_cost,
        "best_index": best_index,
        "terminal_cost": terminal_cost,
        "tail_cost": tail_cost,
        "profile_actions": profile_actions,
        "max_cart": max_cart,
        "invalid": invalid,
        "cost": cost,
    }
    del env
    return cost, metrics


def plan(
    env: NLinkCartPoleEnv,
    initial_state: np.ndarray,
    center: np.ndarray,
    *,
    interpolation: np.ndarray,
    pool: list[mujoco.MjData],
    population: int,
    elites: int,
    iterations: int,
    sigma: float,
    sigma_decay: float,
    sigma_floor: float,
    rng: np.random.Generator,
    rail_soft_limit: float,
    angle_scale: float,
    hinge_scale: float,
    absolute_rate_scale: float,
    cart_scale: float,
    cart_velocity_scale: float,
    action_weight: float,
    slew_weight: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    current = np.asarray(center, dtype=np.float64).copy()
    spread = np.full(current.shape, float(sigma), dtype=np.float64)
    best_profile = current.copy()
    best_record: dict[str, float] = {}
    best_score = float("inf")
    for _ in range(max(1, iterations)):
        candidates = np.clip(
            current[None, :]
            + rng.normal(0.0, spread, size=(population, current.size)),
            -1.0,
            1.0,
        )
        candidates[0] = current
        actions = np.clip(candidates @ interpolation.T, -1.0, 1.0)
        controls = np.repeat(actions, env.frame_skip, axis=1)[:, :, None]
        controls = controls * env.force_limit
        initial = np.repeat(initial_state[None, :], population, axis=0)
        states, _ = mujoco_rollout.rollout(
            env.model,
            pool,
            initial,
            controls,
            persistent_pool=True,
        )
        scores, metrics = rollout_metrics(
            states,
            n_links=env.n,
            interpolation=interpolation,
            profiles=candidates,
            env=env,
            rail_soft_limit=rail_soft_limit,
            angle_scale=angle_scale,
            hinge_scale=hinge_scale,
            absolute_rate_scale=absolute_rate_scale,
            cart_scale=cart_scale,
            cart_velocity_scale=cart_velocity_scale,
            action_weight=action_weight,
            slew_weight=slew_weight,
        )
        order = np.argsort(scores)
        elite = candidates[order[:elites]]
        current = np.mean(elite, axis=0)
        spread = np.maximum(np.std(elite, axis=0) * sigma_decay, sigma_floor)
        top = int(order[0])
        if float(scores[top]) < best_score:
            best_score = float(scores[top])
            best_profile = candidates[top].copy()
            index = int(metrics["best_index"][top])
            best_record = {
                "cost": best_score,
                "best_time_seconds": float((index + 1) * env.dt),
                "best_angle": float(metrics["max_angle"][top, index]),
                "best_hinge_rms": float(metrics["hinge_rms"][top, index]),
                "best_absolute_rate_rms": float(
                    metrics["absolute_rate_rms"][top, index]
                ),
                "best_cart": float(metrics["qpos"][top, index, 0]),
                "best_cart_velocity": float(metrics["qvel"][top, index, 0]),
                "max_cart": float(metrics["max_cart"][top]),
                "terminal_cost": float(metrics["terminal_cost"][top]),
                "invalid": bool(metrics["invalid"][top]),
            }
    return current, best_record, best_profile


def live_row(
    env: NLinkCartPoleEnv, *, step: int, action: float, info: dict[str, Any]
) -> dict[str, Any]:
    _, absolute_angles = env._angles()
    absolute_omega = env._absolute_angular_velocity()
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "action": float(action),
        "x": float(env.data.qpos[0]),
        "cart_velocity": float(env.data.qvel[0]),
        "max_abs_angle": float(info["max_abs_angle"]),
        "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
        "absolute_rate_rms": float(np.sqrt(np.mean(absolute_omega**2))),
        "absolute_angles": absolute_angles.astype(float).tolist(),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "is_upright": bool(info["is_upright"]),
        "upright_streak_seconds": float(info["upright_streak_seconds"]),
        "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
        "max_low_momentum_upright_streak_seconds": float(
            info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exact-MuJoCo receding-horizon capture search from a saved real state"
    )
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--state-index", default="0")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--horizon-steps", type=int, default=60)
    parser.add_argument("--knot-count", type=int, default=12)
    parser.add_argument("--replan-steps", type=int, default=3)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--action-sigma", type=float, default=0.55)
    parser.add_argument("--sigma-decay", type=float, default=0.85)
    parser.add_argument("--sigma-floor", type=float, default=0.03)
    parser.add_argument("--rail-soft-limit", type=float, default=2.85)
    parser.add_argument("--angle-scale", type=float, default=0.15)
    parser.add_argument("--hinge-scale", type=float, default=1.50)
    parser.add_argument("--absolute-rate-scale", type=float, default=2.0)
    parser.add_argument("--cart-scale", type=float, default=1.25)
    parser.add_argument("--cart-velocity-scale", type=float, default=0.75)
    parser.add_argument("--action-weight", type=float, default=0.01)
    parser.add_argument("--slew-weight", type=float, default=0.04)
    parser.add_argument("--seed", type=int, default=20770)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.horizon_steps < 2 or args.knot_count < 2 or args.replan_steps < 1:
        raise ValueError("horizon, knot count, and replan steps must be positive")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("elites must be in 1..population")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    state = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(base_cfg, state, args.seconds)
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    env.reset(seed=args.seed)
    state_size = mujoco.mj_stateSize(
        env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
    )
    live_state = np.empty(state_size, dtype=np.float64)
    interpolation = interpolation_matrix(args.knot_count, args.horizon_steps)
    pool = [
        mujoco.MjData(env.model)
        for _ in range(max(1, min(16, args.population // 16)))
    ]
    rng = np.random.default_rng(args.seed)
    center = np.zeros(args.knot_count, dtype=np.float64)
    buffer = np.zeros(args.horizon_steps, dtype=np.float64)
    next_replan = 0
    trajectory: list[dict[str, Any]] = []
    replans: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    global_best: dict[str, Any] | None = None
    started = time.time()
    total_steps = min(env.max_steps, int(args.seconds / env.dt))

    for step in range(total_steps):
        if step >= next_replan:
            mujoco.mj_getState(
                env.model,
                env.data,
                live_state,
                mujoco.mjtState.mjSTATE_FULLPHYSICS.value,
            )
            center, plan_metrics, best_profile = plan(
                env,
                live_state,
                center,
                interpolation=interpolation,
                pool=pool,
                population=args.population,
                elites=args.elites,
                iterations=args.iterations,
                sigma=args.action_sigma,
                sigma_decay=args.sigma_decay,
                sigma_floor=args.sigma_floor,
                rng=rng,
                rail_soft_limit=args.rail_soft_limit,
                angle_scale=args.angle_scale,
                hinge_scale=args.hinge_scale,
                absolute_rate_scale=args.absolute_rate_scale,
                cart_scale=args.cart_scale,
                cart_velocity_scale=args.cart_velocity_scale,
                action_weight=args.action_weight,
                slew_weight=args.slew_weight,
            )
            plan_actions = np.clip(best_profile @ interpolation.T, -1.0, 1.0)
            buffer = plan_actions.copy()
            center = np.r_[
                best_profile[args.replan_steps :],
                np.zeros(min(args.replan_steps, len(best_profile)), dtype=np.float64),
            ]
            plan_metrics["time_seconds"] = float(step * env.dt)
            plan_metrics["executed_cart"] = float(env.data.qpos[0])
            plan_metrics["executed_cart_velocity"] = float(env.data.qvel[0])
            replans.append(plan_metrics)
            if global_best is None or plan_metrics["cost"] < global_best["cost"]:
                global_best = dict(plan_metrics)
            next_replan = step + args.replan_steps

        action = float(buffer[0])
        buffer = np.r_[buffer[1:], 0.0]
        _, _, terminated, truncated, info = env.step([action])
        final_info = dict(info)
        trajectory.append(live_row(env, step=step + 1, action=action, info=info))
        if terminated or truncated:
            break

    elapsed = time.time() - started
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Reset-free exact-MuJoCo nonlinear receding-horizon capture diagnostic; not canonical evidence.",
        "config_path": str(Path(args.config)),
        "config_sha256": data_sha256(cfg),
        "state_json": str(Path(args.state_json)),
        "state_index": args.state_index,
        "selected_state": state,
        "progress": float(args.progress),
        "search": {
            "seconds": float(args.seconds),
            "horizon_steps": int(args.horizon_steps),
            "horizon_seconds": float(args.horizon_steps * env.dt),
            "knot_count": int(args.knot_count),
            "replan_steps": int(args.replan_steps),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "rail_soft_limit": float(args.rail_soft_limit),
            "angle_scale": float(args.angle_scale),
            "hinge_scale": float(args.hinge_scale),
            "absolute_rate_scale": float(args.absolute_rate_scale),
            "cart_scale": float(args.cart_scale),
            "cart_velocity_scale": float(args.cart_velocity_scale),
            "wall_time_seconds": float(elapsed),
        },
        "success": bool(final_info.get("success", False)),
        "max_upright_streak_seconds": float(
            final_info.get("max_upright_streak_seconds", 0.0)
        ),
        "max_low_momentum_upright_streak_seconds": float(
            final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "termination_reason": final_info.get("termination_reason"),
        "final_info": final_info,
        "global_best_plan": global_best,
        "replans": replans,
        "trajectory": trajectory,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(payload, Path(args.out))
    print(
        f"success={payload['success']} hold={payload['max_upright_streak_seconds']:.3f}s "
        f"low={payload['max_low_momentum_upright_streak_seconds']:.3f}s "
        f"replans={len(replans)} termination={payload['termination_reason']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
