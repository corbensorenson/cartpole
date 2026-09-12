#!/usr/bin/env python
"""Discover a hanging-start swing with live exact-MuJoCo receding-horizon CEM.

This is a discovery controller, not final evidence.  The planner searches a
short, low-dimensional force profile from the state actually emitted by the
previous action, applies only the first few actions, and replans.  The score
has a swing phase that rewards accumulated chain height and a capture phase
that penalizes whole-chain angular momentum near the top.
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


def physical_metrics(
    states: np.ndarray,
    *,
    n_links: int,
    potential_weights: np.ndarray,
) -> dict[str, np.ndarray]:
    """Extract batch metrics from MuJoCo full-physics rollout states."""
    nq = n_links + 1
    qpos = states[..., 1 : 1 + nq]
    qvel = states[..., 1 + nq : 1 + nq + nq]
    absolute_angles = serial_absolute_angles(qpos[..., 1:])
    absolute_omega = np.cumsum(qvel[..., 1:], axis=-1)
    # This is the exact potential-energy ordering for a serial chain, up to a
    # common positive gravity factor.  It is used as a dense swing-progress
    # signal, not as a substitute for the final MuJoCo energy measurement.
    cosine = np.cos(absolute_angles)
    potential_fraction = np.sum(
        potential_weights[None, None, :] * (cosine + 1.0) * 0.5,
        axis=-1,
    ) / max(1e-12, float(np.sum(potential_weights)))
    return {
        "qpos": qpos,
        "qvel": qvel,
        "absolute_angles": absolute_angles,
        "absolute_omega": absolute_omega,
        "potential_fraction": np.clip(potential_fraction, -1.0, 2.0),
        "max_angle": np.max(np.abs(absolute_angles), axis=-1),
        "hinge_rms": np.sqrt(np.mean(qvel[..., 1:] ** 2, axis=-1)),
        "absolute_omega_rms": np.sqrt(np.mean(absolute_omega**2, axis=-1)),
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


def score_sequences(
    env: NLinkCartPoleEnv,
    initial_state: np.ndarray,
    actions: np.ndarray,
    pool: list[mujoco.MjData],
    *,
    interpolation: np.ndarray,
    potential_weights: np.ndarray,
    rail_soft_limit: float,
    action_limit: float,
    action_weight: float,
    action_slew_weight: float,
    capture_potential_threshold: float,
    potential_weight: float,
    capture_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    profile_actions = np.clip(actions @ interpolation.T, -action_limit, action_limit)
    controls = np.repeat(profile_actions, env.frame_skip, axis=1)[:, :, None] * env.force_limit
    initial = np.repeat(initial_state[None, :], len(profile_actions), axis=0)
    states, _ = mujoco_rollout.rollout(
        env.model,
        pool,
        initial,
        controls,
        persistent_pool=True,
    )
    sampled = states[:, env.frame_skip - 1 :: env.frame_skip]
    metrics = physical_metrics(
        sampled,
        n_links=env.n,
        potential_weights=potential_weights,
    )
    late = max(1, metrics["potential_fraction"].shape[1] // 3)
    potential = metrics["potential_fraction"]
    capture_mask = potential >= float(capture_potential_threshold)
    capture_quality = (
        (metrics["max_angle"] / 0.30) ** 2
        + (metrics["hinge_rms"] / 1.50) ** 2
        + (metrics["absolute_omega_rms"] / 1.50) ** 2
        + (metrics["cart_abs"] / 2.50) ** 2
        + (metrics["cart_velocity_abs"] / 1.50) ** 2
    )
    # Before enough height is present, optimize the best height reached.  At
    # high potential, switch the local objective toward a quiet top crossing.
    swing_cost = (
        -float(potential_weight) * np.max(potential[:, late:], axis=1)
        -0.20 * float(potential_weight) * np.mean(potential[:, late:], axis=1)
    )
    capture_cost = np.where(
        capture_mask,
        180.0 * capture_quality,
        450.0 * np.maximum(0.0, capture_potential_threshold - potential) ** 2,
    )
    # A route that briefly gains height is useful, even if it is not yet
    # capture-ready.  Keep the best near-top cost separate from the swing score
    # so the planner does not select a zero-force equilibrium at the hanging
    # start.
    top_cost = np.min(capture_cost[:, late:], axis=1)
    rail_penalty = 3.0e4 * np.maximum(0.0, metrics["cart_abs"] / rail_soft_limit - 1.0) ** 2
    cost = (
        swing_cost
        + float(capture_weight) * top_cost
        + np.mean(rail_penalty, axis=1)
        + float(action_weight) * np.mean(profile_actions**2, axis=1)
        + float(action_slew_weight) * np.mean(np.diff(profile_actions, axis=1) ** 2, axis=1)
    )
    metrics.update(
        {
            "states": states,
            "profile_actions": profile_actions,
            "capture_quality": capture_quality,
            "cost": cost,
        }
    )
    return cost, metrics


def plan(
    env: NLinkCartPoleEnv,
    initial_state: np.ndarray,
    center: np.ndarray,
    *,
    interpolation: np.ndarray,
    potential_weights: np.ndarray,
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
    capture_potential_threshold: float,
    potential_weight: float,
    capture_weight: float,
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
            interpolation=interpolation,
            potential_weights=potential_weights,
            rail_soft_limit=rail_soft_limit,
            action_limit=action_limit,
            action_weight=action_weight,
            action_slew_weight=action_slew_weight,
            capture_potential_threshold=capture_potential_threshold,
            potential_weight=potential_weight,
            capture_weight=capture_weight,
        )
        order = np.argsort(costs)
        elite = candidates[order[:elites]]
        current = np.mean(elite, axis=0)
        spread = np.maximum(np.std(elite, axis=0) * 0.92, sigma_floor)
        top = int(order[0])
        if float(costs[top]) < best_cost:
            index = int(np.argmax(metrics["potential_fraction"][top]))
            capture_index = int(np.argmin(metrics["capture_quality"][top]))
            best_cost = float(costs[top])
            best_actions = candidates[top].copy()
            best_metrics = {
                "cost": best_cost,
                "max_potential_fraction": float(np.max(metrics["potential_fraction"][top])),
                "potential_time_seconds": float((index + 1) * env.dt),
                "best_angle": float(metrics["max_angle"][top, capture_index]),
                "best_hinge_rms": float(metrics["hinge_rms"][top, capture_index]),
                "best_absolute_omega_rms": float(metrics["absolute_omega_rms"][top, capture_index]),
                "best_cart_abs": float(metrics["qpos"][top, capture_index, 0]),
                "best_cart_velocity": float(metrics["qvel"][top, capture_index, 0]),
                "best_time_seconds": float((capture_index + 1) * env.dt),
            }
    return current, best_metrics, best_actions


def potential_weights(env: NLinkCartPoleEnv) -> np.ndarray:
    lengths = np.asarray(env.morphology.lengths, dtype=np.float64)
    masses = np.asarray(env.morphology.masses, dtype=np.float64)
    # Each absolute link angle contributes the height of every downstream
    # link COM.  The common gravity factor cancels in the normalized proxy.
    downstream_mass = np.cumsum(masses[::-1])[::-1] - 0.5 * masses
    return lengths * downstream_mass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--horizon-steps", type=int, default=70)
    parser.add_argument("--knot-count", type=int, default=12)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--action-sigma", type=float, default=0.55)
    parser.add_argument("--sigma-floor", type=float, default=0.04)
    parser.add_argument("--rail-soft-limit", type=float, default=2.75)
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument("--action-weight", type=float, default=0.01)
    parser.add_argument("--action-slew-weight", type=float, default=0.04)
    parser.add_argument("--capture-potential-threshold", type=float, default=0.72)
    parser.add_argument("--potential-weight", type=float, default=900.0)
    parser.add_argument("--capture-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=20732)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.horizon_steps < 2 or args.knot_count < 2 or args.replan_steps < 1:
        raise ValueError("horizon, knot count, and replan steps must be positive")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("elites must be in 1..population")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        # Discovery must own the applied action; curriculum teachers would
        # change the planner waveform and break attribution.
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for noise_key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        cfg["env"][noise_key] = 0.0
        cfg["env"][f"{noise_key}_start"] = 0.0
        cfg["env"][f"{noise_key}_end"] = 0.0
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=0)
    env.reset(seed=0)
    weights = potential_weights(env)
    interpolation = interpolation_matrix(args.knot_count, args.horizon_steps)
    state_size = mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
    state = np.empty(state_size, dtype=np.float64)
    rng = np.random.default_rng(args.seed)
    center = np.zeros(args.knot_count, dtype=np.float64)
    trajectory: list[dict[str, Any]] = []
    replans: list[dict[str, Any]] = []
    buffer = np.zeros(args.horizon_steps, dtype=np.float64)
    next_replan_step = 0
    final_info: dict[str, Any] = {}
    started = time.time()
    total_steps = min(env.max_steps, int(args.seconds / env.dt))

    for step in range(total_steps):
        t = step * env.dt
        if step >= next_replan_step:
            mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
            center, plan_metrics, best_profile = plan(
                env,
                state,
                center,
                interpolation=interpolation,
                potential_weights=weights,
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
                capture_potential_threshold=args.capture_potential_threshold,
                potential_weight=args.potential_weight,
                capture_weight=args.capture_weight,
            )
            best_actions = np.clip(best_profile @ interpolation.T, -1.0, 1.0)
            buffer = best_actions.copy()
            center = np.r_[
                best_profile[args.replan_steps :],
                np.zeros(min(args.replan_steps, len(best_profile)), dtype=np.float64),
            ]
            replans.append({"time_seconds": t, **plan_metrics})
            next_replan_step = step + args.replan_steps
        action = float(buffer[0])
        buffer = np.r_[buffer[1:], 0.0]
        _, reward, terminated, truncated, info = env.step([action])
        _, absolute = env._angles()
        absolute_omega = env._absolute_angular_velocity()
        trajectory.append(
            {
                "step": step + 1,
                "time_seconds": (step + 1) * env.dt,
                "action": action,
                "reward": float(reward),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
                "x": float(info["x"]),
                "cart_velocity": float(env.data.qvel[0]),
                "absolute_angles": absolute.astype(float).tolist(),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "absolute_angular_velocity_rms": float(np.sqrt(np.mean(absolute_omega**2))),
                "potential_fraction": float(
                    np.sum(weights * (np.cos(absolute) + 1.0) * 0.5) / max(1e-12, np.sum(weights))
                ),
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
        "summary": "Low-dimensional live receding-horizon CEM swing discovery from exact hanging state.",
        "config_path": str(Path(args.config)),
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
            "sigma_floor": float(args.sigma_floor),
            "rail_soft_limit": float(args.rail_soft_limit),
            "action_limit": float(args.action_limit),
            "action_weight": float(args.action_weight),
            "action_slew_weight": float(args.action_slew_weight),
            "capture_potential_threshold": float(args.capture_potential_threshold),
            "potential_weight": float(args.potential_weight),
            "capture_weight": float(args.capture_weight),
            "seed": int(args.seed),
            "wall_time_seconds": float(time.time() - started),
        },
        "potential_weights": weights.astype(float).tolist(),
        "success": bool(final_info.get("success", False)),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "termination_reason": final_info.get("termination_reason"),
        "simulated_seconds": float(len(trajectory) * env.dt),
        "max_potential_fraction": float(max((row["potential_fraction"] for row in trajectory), default=0.0)),
        "best_angle": min((row["max_abs_angle"] for row in trajectory), default=float("inf")),
        "max_cart_abs": float(max((abs(row["x"]) for row in trajectory), default=0.0)),
        "final_info": final_info,
        "replans": replans,
        "trajectory": trajectory,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(result, Path(args.out))
    print(
        f"success={result['success']} hold={result['max_upright_streak_seconds']:.3f}s "
        f"height={result['max_potential_fraction']:.3f} angle={result['best_angle']:.3f} "
        f"cart={result['max_cart_abs']:.3f} replans={len(replans)}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
