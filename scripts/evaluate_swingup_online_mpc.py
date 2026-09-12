#!/usr/bin/env python
"""Evaluate a hanging-start swing followed by live nonlinear MPC capture.

This is a discovery controller, not final evidence. The swing and capture
stages share one MuJoCo state; the MPC replans from the state actually emitted
by the previous control step and never overwrites it with a planned state.
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

try:
    from scripts.search_swingup_tail_ilqr import load_swing_controller, source_action
except ModuleNotFoundError:
    from search_swingup_tail_ilqr import load_swing_controller, source_action


def physical_metrics(states: np.ndarray, n_links: int) -> dict[str, np.ndarray]:
    qpos = states[..., 1 : 1 + n_links + 1].copy()
    qvel = states[..., 1 + n_links + 1 : 1 + n_links + 1 + n_links + 1]
    # Sum the raw relative coordinates first.  A +pi hanging joint must not
    # become -pi before the serial sum, or the second wrap reports it as 0.
    absolute_angles = serial_absolute_angles(qpos[..., 1:])
    return {
        "qpos": qpos,
        "qvel": qvel,
        "absolute_angles": absolute_angles,
        "max_angle": np.max(np.abs(absolute_angles), axis=-1),
        "hinge_rms": np.sqrt(np.mean(qvel[..., 1:] ** 2, axis=-1)),
        "max_hinge": np.max(np.abs(qvel[..., 1:]), axis=-1),
        "cart_abs": np.abs(qpos[..., 0]),
        "cart_velocity_abs": np.abs(qvel[..., 0]),
    }


def interpolation_matrix(knot_count: int, step_count: int) -> np.ndarray:
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


def load_tail_actions(
    path: str | None,
    env: NLinkCartPoleEnv,
    record_key: str = "best",
) -> np.ndarray:
    if path is None:
        return np.zeros(0, dtype=np.float64)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    best = payload.get(record_key) if isinstance(payload, dict) else None
    if not isinstance(best, dict) or "knots" not in best:
        raise ValueError(f"could not load tail record {record_key!r} from {path}")
    seconds = float(payload.get("tail_horizon_seconds", 0.0))
    steps = int(payload.get("tail_horizon_steps", round(seconds / env.dt)))
    if steps < 2 or seconds <= 0.0:
        raise ValueError(f"tail artifact has no usable horizon: {path}")
    knots = np.asarray(best["knots"], dtype=np.float64)
    return np.clip(knots @ interpolation_matrix(len(knots), steps).T, -1.0, 1.0)


def score_sequences(
    env: NLinkCartPoleEnv,
    initial_state: np.ndarray,
    actions: np.ndarray,
    pool: list[mujoco.MjData],
    *,
    rail_soft_limit: float,
    action_limit: float,
    action_weight: float,
    action_slew_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    actions = np.clip(actions, -action_limit, action_limit).astype(np.float64)
    controls = np.repeat(actions, env.frame_skip, axis=1)[:, :, None] * env.force_limit
    initial = np.repeat(initial_state[None, :], len(actions), axis=0)
    states, _ = mujoco_rollout.rollout(
        env.model,
        pool,
        initial,
        controls,
        persistent_pool=True,
    )
    sampled = states[:, env.frame_skip - 1 :: env.frame_skip]
    metrics = physical_metrics(sampled, env.n)
    state_cost = (
        180.0 * (metrics["max_angle"] / 0.15) ** 2
        + 22.0 * (metrics["hinge_rms"] / 0.75) ** 2
        + 4.0 * (metrics["max_hinge"] / 1.50) ** 2
        + 4.0 * (metrics["cart_abs"] / 1.25) ** 2
        + 3.0 * (metrics["cart_velocity_abs"] / 0.50) ** 2
    )
    rail_penalty = 2.0e4 * np.maximum(0.0, metrics["cart_abs"] / rail_soft_limit - 1.0) ** 2
    half = max(1, state_cost.shape[1] // 2)
    cost = (
        0.55 * np.mean(state_cost[:, half:], axis=1)
        + 0.30 * state_cost[:, -1]
        + 0.15 * np.min(state_cost[:, half:], axis=1)
        + np.mean(rail_penalty, axis=1)
        + float(action_weight) * np.mean(actions * actions, axis=1)
        + float(action_slew_weight) * np.mean(np.diff(actions, axis=1) ** 2, axis=1)
    )
    metrics.update({"states": states, "state_cost": state_cost, "cost": cost})
    return cost, metrics


def plan(
    env: NLinkCartPoleEnv,
    initial_state: np.ndarray,
    center: np.ndarray,
    *,
    population: int,
    elites: int,
    iterations: int,
    sigma: float,
    sigma_floor: float,
    rng: np.random.Generator,
    rail_soft_limit: float,
    action_limit: float,
    action_weight: float,
    action_slew_weight: float,
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    pool = [mujoco.MjData(env.model) for _ in range(min(32, max(1, population // 16)))]
    current = np.asarray(center, dtype=np.float64).copy()
    spread = np.full(current.shape[0], float(sigma), dtype=np.float64)
    best_cost = float("inf")
    best_metrics: dict[str, float] = {}
    best_actions = current.copy()
    for _ in range(max(1, iterations)):
        candidates = np.clip(
            current[None, :] + rng.normal(0.0, spread, size=(population, current.size)),
            -1.0,
            1.0,
        )
        candidates[0] = current
        costs, metrics = score_sequences(
            env,
            initial_state,
            candidates,
            pool,
            rail_soft_limit=rail_soft_limit,
            action_limit=action_limit,
            action_weight=action_weight,
            action_slew_weight=action_slew_weight,
        )
        order = np.argsort(costs)
        elite = candidates[order[:elites]]
        current = np.mean(elite, axis=0)
        spread = np.maximum(np.std(elite, axis=0) * 0.92, sigma_floor)
        top = int(order[0])
        if float(costs[top]) < best_cost:
            idx = int(np.argmin(metrics["state_cost"][top]))
            best_cost = float(costs[top])
            best_actions = candidates[top].copy()
            best_metrics = {
                "cost": best_cost,
                "best_angle": float(metrics["max_angle"][top, idx]),
                "best_hinge_rms": float(metrics["hinge_rms"][top, idx]),
                "best_max_hinge": float(metrics["max_hinge"][top, idx]),
                "best_cart_abs": float(metrics["qpos"][top, idx, 0]),
                "best_cart_velocity": float(metrics["qvel"][top, idx, 0]),
                "best_time_seconds": float((idx + 1) * env.dt),
            }
    return current, best_metrics, best_actions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--swing-controller-json", required=True)
    parser.add_argument("--controller-key", default=None)
    parser.add_argument("--tail-json", default=None)
    parser.add_argument(
        "--tail-record",
        default="best",
        help="Record inside --tail-json to replay, such as best or best_feasible.",
    )
    parser.add_argument("--capture-checkpoint", default=None)
    parser.add_argument(
        "--capture-hidden-sizes",
        default=None,
        help="Comma-separated hidden sizes for the optional Torch capture checkpoint; empty means a linear actor.",
    )
    parser.add_argument("--switch-time", type=float, default=11.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--horizon-steps", type=int, default=50)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--action-sigma", type=float, default=0.35)
    parser.add_argument("--sigma-floor", type=float, default=0.015)
    parser.add_argument("--rail-soft-limit", type=float, default=2.5)
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=0.02)
    parser.add_argument("--action-slew-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.horizon_steps < 2 or args.replan_steps < 1 or args.elites > args.population:
        raise ValueError("invalid MPC dimensions")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"], "init_mode": "hanging"}
    for noise_key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        cfg["env"][noise_key] = 0.0
        cfg["env"][f"{noise_key}_start"] = 0.0
        cfg["env"][f"{noise_key}_end"] = 0.0
    controller = load_swing_controller(args.swing_controller_json, args.controller_key)
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    tail_actions = load_tail_actions(args.tail_json, env, args.tail_record)
    capture_model = None
    capture_sample_action = None
    if args.capture_checkpoint:
        from torch_runtime import prepare_runtime

        prepare_runtime()
        from gcartpole.ppo_torch import ActorCritic, load_model, sample_action

        hidden_sizes = []
        if args.capture_hidden_sizes:
            hidden_sizes = [int(value) for value in args.capture_hidden_sizes.split(",") if value.strip()]
        capture_model = ActorCritic(
            obs_dim=int(env.observation_space.shape[0]),
            act_dim=int(env.action_space.shape[0]),
            hidden_sizes=hidden_sizes,
            action_std_init=float(cfg["ppo"].get("action_std_init", 0.02)),
        )
        load_model(capture_model, args.capture_checkpoint)
        capture_sample_action = sample_action
    state_size = mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
    state = np.empty(state_size, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    center = np.zeros(args.horizon_steps, dtype=np.float64)
    trajectory: list[dict[str, Any]] = []
    replans: list[dict[str, Any]] = []
    mpc_buffer = np.zeros(args.horizon_steps, dtype=np.float64)
    switch_step = int(round(args.switch_time / env.dt))
    tail_start_step = switch_step
    tail_end_step = tail_start_step + len(tail_actions)
    # A residual capture policy is trained against the LQR teacher, while the
    # source swing/tail artifacts contain direct normalized force actions.
    # Keep the direct-action semantics until the live handoff, then enable the
    # teacher without changing qpos, qvel, or simulator time.
    capture_residual_cfg = env.env_cfg.get("action_lqr_residual")
    capture_residual_enabled = bool(
        capture_model is not None
        and isinstance(capture_residual_cfg, dict)
        and capture_residual_cfg.get("enabled", False)
    )
    if capture_residual_enabled:
        capture_residual_cfg["enabled"] = False
    capture_residual_activated = False
    next_replan_step = tail_end_step if len(tail_actions) else switch_step
    started = time.time()
    final_info: dict[str, Any] = {}
    total_steps = min(env.max_steps, int(args.seconds / env.dt))
    for step in range(total_steps):
        t = step * env.dt
        if step < switch_step:
            action = source_action(env, controller, t, step)
            mode = "source_swing"
        elif step < tail_end_step:
            action = float(tail_actions[step - tail_start_step])
            mode = "planned_tail"
        else:
            if capture_residual_enabled and not capture_residual_activated:
                capture_residual_cfg["enabled"] = True
                capture_residual_activated = True
            if capture_model is not None:
                obs = env._get_obs()
                action_array, _, _ = capture_sample_action(
                    capture_model,
                    obs[None, :],
                    deterministic=True,
                )
                action = float(action_array[0, 0])
                mode = "torch_capture"
            else:
                if step >= next_replan_step:
                    mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
                    center, plan_metrics, best_actions = plan(
                        env,
                        state,
                        center,
                        population=args.population,
                        elites=args.elites,
                        iterations=args.iterations,
                        sigma=args.action_sigma,
                        sigma_floor=args.sigma_floor,
                        rng=rng,
                        rail_soft_limit=args.rail_soft_limit,
                        action_limit=args.action_limit,
                        action_weight=args.action_weight,
                        action_slew_weight=args.action_slew_weight,
                    )
                    replans.append({"time_seconds": t, **plan_metrics})
                    mpc_buffer = best_actions.copy()
                    center = np.r_[
                        best_actions[args.replan_steps :],
                        np.zeros(min(args.replan_steps, len(best_actions)), dtype=np.float64),
                    ]
                    next_replan_step = step + args.replan_steps
                action = float(mpc_buffer[0])
                mpc_buffer = np.r_[mpc_buffer[1:], 0.0]
                mode = "online_nonlinear_mpc"
        _, reward, terminated, truncated, info = env.step([action])
        rel, abs_angles = env._angles()
        trajectory.append(
            {
                "step": step + 1,
                "time_seconds": (step + 1) * env.dt,
                "controller_mode": mode,
                "action": float(action),
                "reward": float(reward),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
                "x": float(info["x"]),
                "cart_velocity": float(env.data.qvel[0]),
                "relative_angles": rel.astype(float).tolist(),
                "absolute_angles": abs_angles.astype(float).tolist(),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "is_upright": bool(info["is_upright"]),
                "upright_streak_seconds": float(info["upright_streak_seconds"]),
                "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
            }
        )
        final_info = dict(info)
        if terminated or truncated:
            break

    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Reset-free hanging swing followed by receding-horizon exact-MuJoCo CEM capture search.",
        "source_controller": controller,
        "source_controller_json": str(args.swing_controller_json),
        "controller_key": args.controller_key,
        "tail_json": args.tail_json,
        "tail_record": args.tail_record if args.tail_json else None,
        "tail_steps": int(len(tail_actions)),
        "search": {
            "switch_time": float(args.switch_time),
            "horizon_steps": int(args.horizon_steps),
            "replan_steps": int(args.replan_steps),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "action_sigma": float(args.action_sigma),
            "rail_soft_limit": float(args.rail_soft_limit),
            "action_limit": float(args.action_limit),
            "action_weight": float(args.action_weight),
            "action_slew_weight": float(args.action_slew_weight),
            "seed": int(args.seed),
            "wall_time_seconds": float(time.time() - started),
        },
        "replans": replans,
        "success": bool(final_info.get("success", False)),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "termination_reason": final_info.get("termination_reason"),
        "simulated_seconds": float(len(trajectory) * env.dt),
        "final_info": final_info,
        "trajectory": trajectory,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(result, Path(args.out))
    print(
        f"success={result['success']} hold={result['max_upright_streak_seconds']:.3f}s "
        f"first_upright={result['time_to_first_upright']} replans={len(replans)}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
