#!/usr/bin/env python
"""Batched exact-MuJoCo global action search for a low-momentum handoff.

This searches a low-dimensional piecewise-linear force waveform from hanging
start.  It is deliberately a proposal generator: every candidate is scored
with full state/rail metrics, and any result still requires reset-free
feedback replay and capture verification.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import rollout as mujoco_rollout

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles, wrap_angle
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from probe_swingup_trajectory import trajectory_action


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


def load_initial_center(
    env: NLinkCartPoleEnv,
    *,
    controller_path: str | None,
    seconds: float,
    knot_count: int,
) -> np.ndarray:
    if controller_path is None:
        return np.zeros(knot_count, dtype=np.float64)
    payload = json.loads(Path(controller_path).read_text(encoding="utf-8"))
    if isinstance(payload.get("trace"), list) and payload["trace"]:
        trace = payload["trace"]
        source_actions = np.asarray(
            [float(row["action"]) for row in trace if "action" in row],
            dtype=np.float64,
        )
        if source_actions.size:
            source_seconds = float(
                trace[-1].get("time_seconds", seconds)
            )
            source_t = np.linspace(0.0, source_seconds, source_actions.size, dtype=np.float64)
            knot_t = np.linspace(0.0, seconds, knot_count, dtype=np.float64)
            return np.clip(np.interp(knot_t, source_t, source_actions), -1.0, 1.0)
    # A prior global-CEM result is already expressed as normalized force
    # knots. Warm-start from its best route directly instead of replaying it
    # through the cart-position controller adapter.
    if isinstance(payload.get("best"), dict) and "knots" in payload["best"]:
        source_knots = np.asarray(payload["best"]["knots"], dtype=np.float64)
        source_seconds = float(payload.get("search", {}).get("seconds", seconds))
        source_t = np.linspace(0.0, source_seconds, len(source_knots), dtype=np.float64)
        knot_t = np.linspace(0.0, seconds, knot_count, dtype=np.float64)
        return np.clip(np.interp(knot_t, source_t, source_knots), -1.0, 1.0)
    if isinstance(payload.get("best"), dict):
        controller = payload["best"]["controller"]
    elif isinstance(payload.get("controller"), dict):
        controller = payload["controller"]
    else:
        raise ValueError(f"could not load controller from {controller_path}")
    env.reset(seed=0)
    source_knots = np.asarray(controller["knots"], dtype=np.float64)
    source_seconds = float(controller["trajectory_seconds"])
    source_actions: list[float] = []
    steps = min(env.max_steps, int(seconds / env.dt))
    for step in range(steps):
        t = step * env.dt
        source_actions.append(
            float(
                trajectory_action(
                    env,
                    t,
                    source_knots,
                    source_seconds,
                    float(controller["kp"]),
                    float(controller["kd"]),
                )
            )
        )
        _, _, terminated, truncated, _ = env.step([source_actions[-1]])
        if terminated or truncated:
            break
    if not source_actions:
        return np.zeros(knot_count, dtype=np.float64)
    source_t = np.linspace(0.0, seconds, len(source_actions), dtype=np.float64)
    knot_t = np.linspace(0.0, seconds, knot_count, dtype=np.float64)
    return np.clip(np.interp(knot_t, source_t, np.asarray(source_actions)), -1.0, 1.0)


def batch_metrics(
    env: NLinkCartPoleEnv,
    initial_state: np.ndarray,
    knots: np.ndarray,
    interpolation: np.ndarray,
    pool: list[mujoco.MjData],
    *,
    handoff_min_time: float,
    rolling_window: int,
    rail_penalty_limit: float,
    rail_penalty_weight: float,
    absolute_rate_weight: float,
    absolute_rate_scale: float,
    relative_angle_weight: float,
    relative_angle_scale: float,
    best_score_weight: float,
    terminal_score_weight: float,
    tail_average_weight: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Preserve float64 through candidate scoring.  The saved knot waveform is
    # replayed later by the exact evaluator; a float32 round-trip here can
    # diverge chaotically from that replay before the handoff.
    actions = np.clip(knots @ interpolation.T, -1.0, 1.0).astype(np.float64)
    controls = np.repeat(actions, env.frame_skip, axis=1)[:, :, None] * env.force_limit
    initial = np.repeat(initial_state[None, :], len(knots), axis=0)
    rollout_states, _ = mujoco_rollout.rollout(
        env.model,
        pool,
        initial,
        controls,
        persistent_pool=False,
    )
    finite = np.all(np.isfinite(rollout_states), axis=(1, 2))
    bounded = np.max(np.abs(np.nan_to_num(rollout_states, nan=np.inf)), axis=(1, 2)) < 1.0e6
    # MuJoCo's batched rollout can reset an unstable candidate's returned
    # state/time buffer to its first sample without producing NaNs. Reject any
    # trajectory whose time column jumps backward instead of ranking the reset
    # zero state as a perfect upright handoff.
    time_valid = np.all(np.diff(rollout_states[..., 0], axis=1) > 0.0, axis=1)
    sampled = rollout_states[:, env.frame_skip - 1 :: env.frame_skip]
    qpos = sampled[..., 1 : 1 + env.n + 1].copy()
    qvel = sampled[..., 1 + env.n + 1 : 1 + env.n + 1 + env.n + 1]
    # Keep the relative joint coordinates unwrapped until after the serial
    # sum.  Wrapping an exact hanging joint at +pi to -pi before summing can
    # turn the hanging pose into an apparent upright pose after the second
    # wrap, which would poison the CEM objective.
    angles = serial_absolute_angles(qpos[..., 1:])
    relative_angles = wrap_angle(qpos[..., 1:])
    max_angle = np.max(np.abs(angles), axis=-1)
    relative_angle_rms = np.sqrt(np.mean(relative_angles**2, axis=-1))
    max_relative_angle = np.max(np.abs(relative_angles), axis=-1)
    hinge_rms = np.sqrt(np.mean(qvel[..., 1:] ** 2, axis=-1))
    max_hinge = np.max(np.abs(qvel[..., 1:]), axis=-1)
    absolute_angular_velocity = np.cumsum(qvel[..., 1:], axis=-1)
    absolute_rate_rms = np.sqrt(np.mean(absolute_angular_velocity**2, axis=-1))
    cart_abs = np.abs(qpos[..., 0])
    cart_velocity_abs = np.abs(qvel[..., 0])
    ratio_score = (
        35.0 * (max_angle / 0.15) ** 2
        + 16.0 * (hinge_rms / 0.75) ** 2
        + 4.0 * (max_hinge / 1.50) ** 2
        + float(absolute_rate_weight)
        * (absolute_rate_rms / max(1e-9, float(absolute_rate_scale))) ** 2
        + float(relative_angle_weight)
        * (relative_angle_rms / max(1e-9, float(relative_angle_scale))) ** 2
        + 3.0 * (cart_abs / 1.25) ** 2
        + 3.0 * (cart_velocity_abs / 0.50) ** 2
    )
    first = max(1, int(round(handoff_min_time / env.dt)))
    late = ratio_score[:, first:]
    if rolling_window > 1:
        kernel = np.ones(rolling_window, dtype=np.float64) / rolling_window
        rolling = np.stack(
            [np.convolve(row, kernel, mode="valid") for row in late], axis=0
        )
        best_offset = np.argmin(rolling, axis=1)
        best_score = np.min(rolling, axis=1)
        best_index = first + best_offset + rolling_window - 1
    else:
        best_index = first + np.argmin(late, axis=1)
        best_score = np.min(late, axis=1)
    terminal_score = ratio_score[:, -1]
    tail_average = np.mean(ratio_score[:, -min(30, ratio_score.shape[1]) :], axis=1)
    rail = np.max(cart_abs, axis=1)
    # Batched MuJoCo rollouts do not call the environment's termination check,
    # so a candidate can keep integrating after it has reached the physical
    # rail. Reject those trajectories instead of allowing the joint-limit
    # dynamics to manufacture a fake return to the center.
    rail_contact = rail >= 0.995 * float(env.rail_limit)
    rail_penalty = float(rail_penalty_weight) * np.maximum(
        0.0, rail / max(1e-9, float(rail_penalty_limit)) - 1.0
    ) ** 2
    smooth = np.mean(np.diff(actions, axis=1) ** 2, axis=1)
    cost = (
        float(best_score_weight) * best_score
        + float(terminal_score_weight) * terminal_score
        + float(tail_average_weight) * tail_average
    )
    cost = cost + rail_penalty + 0.02 * np.mean(actions * actions, axis=1) + 0.10 * smooth
    invalid = ~(finite & bounded & time_valid & np.isfinite(cost)) | rail_contact
    cost[invalid] = 1.0e30
    metrics = {
        "cost": cost,
        "actions": actions,
        "qpos": qpos,
        "qvel": qvel,
        "angles": angles,
        "relative_angles": relative_angles,
        "max_angle": max_angle,
        "relative_angle_rms": relative_angle_rms,
        "max_relative_angle": max_relative_angle,
        "hinge_rms": hinge_rms,
        "max_hinge": max_hinge,
        "absolute_rate_rms": absolute_rate_rms,
        "cart_abs": cart_abs,
        "cart_velocity_abs": cart_velocity_abs,
        "rail": rail,
        "rail_contact": rail_contact,
        "best_index": best_index,
        "best_score": best_score,
        "terminal_score": terminal_score,
        "tail_average": tail_average,
        "invalid": invalid,
    }
    return cost, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--knot-count", type=int, default=32)
    parser.add_argument("--population", type=int, default=512)
    parser.add_argument("--elites", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--handoff-min-time", type=float, default=4.0)
    parser.add_argument("--rolling-window", type=int, default=5)
    parser.add_argument("--action-sigma", type=float, default=0.40)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.015)
    parser.add_argument(
        "--rail-penalty-limit",
        type=float,
        default=2.90,
        help="Rail magnitude before the discovery objective penalizes use (canonical default: 2.90 m).",
    )
    parser.add_argument(
        "--rail-penalty-weight",
        type=float,
        default=2.0e5,
        help="Weight for exceeding --rail-penalty-limit; set to 0 only for a declared wide-rail diagnostic.",
    )
    parser.add_argument(
        "--absolute-rate-weight",
        type=float,
        default=8.0,
        help="Weight on whole-chain absolute angular velocity RMS in the late-handoff objective.",
    )
    parser.add_argument(
        "--absolute-rate-scale",
        type=float,
        default=0.75,
        help="Scale for whole-chain absolute angular velocity RMS.",
    )
    parser.add_argument(
        "--relative-angle-weight",
        type=float,
        default=12.0,
        help="Weight on relative joint-angle RMS in the full-state handoff ranking.",
    )
    parser.add_argument(
        "--relative-angle-scale",
        type=float,
        default=0.15,
        help="Scale for relative joint-angle RMS in the full-state handoff ranking.",
    )
    parser.add_argument("--best-score-weight", type=float, default=0.85)
    parser.add_argument("--terminal-score-weight", type=float, default=0.10)
    parser.add_argument("--tail-average-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--progress",
        type=float,
        default=1.0,
        help="Fixed plant morphology progress for this diagnostic (1.0 is canonical uniform).",
    )
    parser.add_argument("--init-controller-json", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.knot_count < 2 or args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("--progress must be in [0, 1]")
    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"], "init_mode": "hanging"}
    # A forced hanging-start search must not inherit scheduled curriculum
    # noise. The scalar fields alone are insufficient because reset uses the
    # *_start/*_end schedules when they are present.
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
    state_size = mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
    initial_state = np.empty(state_size, dtype=np.float64)
    mujoco.mj_getState(env.model, env.data, initial_state, mujoco.mjtState.mjSTATE_FULLPHYSICS.value)
    interpolation = interpolation_matrix(args.knot_count, max(2, int(round(args.seconds / env.dt))))
    center = load_initial_center(
        env,
        controller_path=args.init_controller_json,
        seconds=args.seconds,
        knot_count=args.knot_count,
    )
    env.reset(seed=0)
    # Keep one MuJoCo data buffer per candidate whenever the population fits.
    # Do not use the backend's persistent thread pool here: discovery artifacts
    # must preserve candidate-to-state association across iterations, and the
    # persistent pool is difficult to audit when a search is interrupted.
    pool = [mujoco.MjData(env.model) for _ in range(min(32, max(1, args.population)))]
    rng = np.random.default_rng(args.seed)
    sigma = np.full(args.knot_count, args.action_sigma, dtype=np.float64)
    history: list[dict[str, Any]] = []
    best_record: dict[str, Any] | None = None
    started = time.time()

    for iteration in range(args.iterations):
        knots = np.clip(
            center[None, :] + rng.normal(0.0, sigma, size=(args.population, args.knot_count)),
            -1.0,
            1.0,
        )
        knots[0] = center
        cost, metrics = batch_metrics(
            env,
            initial_state,
            knots,
            interpolation,
            pool,
            handoff_min_time=args.handoff_min_time,
            rolling_window=max(1, args.rolling_window),
            rail_penalty_limit=args.rail_penalty_limit,
            rail_penalty_weight=args.rail_penalty_weight,
            absolute_rate_weight=args.absolute_rate_weight,
            absolute_rate_scale=args.absolute_rate_scale,
            relative_angle_weight=args.relative_angle_weight,
            relative_angle_scale=args.relative_angle_scale,
            best_score_weight=args.best_score_weight,
            terminal_score_weight=args.terminal_score_weight,
            tail_average_weight=args.tail_average_weight,
        )
        order = np.argsort(cost)
        top = int(order[0])
        elite_knots = knots[order[: args.elites]]
        center = elite_knots.mean(axis=0)
        sigma = np.maximum(elite_knots.std(axis=0) * args.sigma_decay, args.sigma_floor)
        best_index = int(metrics["best_index"][top])
        record = {
            "iteration": iteration + 1,
            "cost": float(cost[top]),
            "best_score": float(metrics["best_score"][top]),
            "best_time_seconds": float((best_index + 1) * env.dt),
            "max_angle": float(metrics["max_angle"][top, best_index]),
            "relative_angle_rms": float(metrics["relative_angle_rms"][top, best_index]),
            "max_relative_angle": float(metrics["max_relative_angle"][top, best_index]),
            "hinge_rms": float(metrics["hinge_rms"][top, best_index]),
            "max_hinge": float(metrics["max_hinge"][top, best_index]),
            "absolute_rate_rms": float(metrics["absolute_rate_rms"][top, best_index]),
            "cart": float(metrics["qpos"][top, best_index, 0]),
            "cart_velocity": float(metrics["qvel"][top, best_index, 0]),
            "rail": float(metrics["rail"][top]),
            "invalid": bool(metrics["invalid"][top]),
            "rail_contact": bool(metrics["rail_contact"][top]),
        }
        history.append(record)
        if best_record is None or record["cost"] < best_record["cost"]:
            best_record = {
                **record,
                "knots": knots[top].astype(float).tolist(),
                "best_state": {
                    "qpos": metrics["qpos"][top, best_index].astype(float).tolist(),
                    "qvel": metrics["qvel"][top, best_index].astype(float).tolist(),
                    "absolute_angles": metrics["angles"][top, best_index].astype(float).tolist(),
                    "relative_angles": metrics["relative_angles"][top, best_index].astype(float).tolist(),
                },
            }
        print(
            f"iter={iteration + 1:03d} cost={record['cost']:.4f} "
            f"angle={record['max_angle']:.4f} hinge={record['hinge_rms']:.4f} "
            f"max_hinge={record['max_hinge']:.4f} x={record['cart']:.3f} "
            f"xd={record['cart_velocity']:.3f} rail={record['rail']:.3f}",
            flush=True,
        )

    assert best_record is not None
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Batched exact-MuJoCo global force CEM; candidate requires feedback replay and capture verification.",
        "config_sha256": data_sha256(cfg),
        "search": {
            "seconds": float(args.seconds),
            "knot_count": int(args.knot_count),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "handoff_min_time": float(args.handoff_min_time),
            "rolling_window": int(args.rolling_window),
            "seed": int(args.seed),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "rail_penalty_limit": float(args.rail_penalty_limit),
            "rail_penalty_weight": float(args.rail_penalty_weight),
            "absolute_rate_weight": float(args.absolute_rate_weight),
            "absolute_rate_scale": float(args.absolute_rate_scale),
            "relative_angle_weight": float(args.relative_angle_weight),
            "relative_angle_scale": float(args.relative_angle_scale),
            "best_score_weight": float(args.best_score_weight),
            "terminal_score_weight": float(args.terminal_score_weight),
            "tail_average_weight": float(args.tail_average_weight),
            "plant_progress": float(args.progress),
            "wall_time_seconds": float(time.time() - started),
        },
        "morphology": {
            "alpha_length": float(env.morphology.alpha_length),
            "alpha_mass": float(env.morphology.alpha_mass),
            "alpha_damping": float(env.morphology.alpha_damping),
            "alpha_frictionloss": float(env.morphology.alpha_frictionloss),
            "total_damping": float(env.morphology.total_damping),
            "total_frictionloss": float(env.morphology.total_frictionloss),
            "lengths": env.morphology.lengths.astype(float).tolist(),
            "masses": env.morphology.masses.astype(float).tolist(),
            "damping": env.morphology.damping.astype(float).tolist(),
            "frictionloss": env.morphology.frictionloss.astype(float).tolist(),
        },
        "init_controller_json": args.init_controller_json,
        "best": best_record,
        "history": history,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(result, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
