#!/usr/bin/env python
"""Search an exact-MuJoCo action tail from a recorded swing crossing.

This is a discovery tool, not final evidence.  The source swing is replayed
from hanging start, then a batched CEM searches only the uninterrupted tail.
The objective ranks the complete capture envelope, including internal hinge
velocity, instead of selecting another short upright crossing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import rollout as mujoco_rollout

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
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


def load_controller(path: str, record_key: str | None = None) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload.get("best_by"), dict):
        record = payload["best_by"].get(record_key or "score")
        if isinstance(record, dict) and isinstance(record.get("controller"), dict):
            return dict(record["controller"])
    if isinstance(payload.get("best"), dict):
        if "knots" in payload["best"] and "controller" not in payload["best"]:
            return {
                "type": "normalized_force_knots",
                "knots": list(payload["best"]["knots"]),
                "trajectory_seconds": float(payload.get("search", {}).get("seconds", 0.0)),
            }
        return dict(payload["best"]["controller"])
    if isinstance(payload.get("controller"), dict):
        return dict(payload["controller"])
    raise ValueError(f"could not load swing controller from {path}")


def load_tail_center(
    path: str | None,
    knot_count: int,
    record_key: str = "best",
    *,
    tail_start_seconds: float | None = None,
    tail_seconds: float | None = None,
) -> np.ndarray:
    if path is None:
        return np.zeros(knot_count, dtype=np.float64)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # A replay trace is a valid warm start for a tail even though it is not a
    # CEM result.  Use the actions actually applied after the requested
    # handoff time; this preserves the downstream hybrid expert as the center
    # of the new open-loop search.
    trace_rows = payload.get("trace")
    if not isinstance(trace_rows, list) or not trace_rows:
        trace_rows = payload.get("trajectory")
    if isinstance(trace_rows, list):
        rows = [
            row
            for row in trace_rows
            if isinstance(row, dict)
            and "action" in row
            and "time_seconds" in row
            and (tail_start_seconds is None or float(row["time_seconds"]) > float(tail_start_seconds))
        ]
        if not rows:
            return np.zeros(knot_count, dtype=np.float64)
        source_times = np.asarray(
            [float(row["time_seconds"]) - float(tail_start_seconds or 0.0) for row in rows],
            dtype=np.float64,
        )
        source_actions = np.asarray([float(row["action"]) for row in rows], dtype=np.float64)
        target_times = np.linspace(
            0.0,
            float(tail_seconds if tail_seconds is not None else source_times[-1]),
            knot_count,
            dtype=np.float64,
        )
        return np.clip(
            np.interp(target_times, source_times, source_actions),
            -1.0,
            1.0,
        )
    record = payload.get(record_key)
    if not isinstance(record, dict) or "knots" not in record:
        raise ValueError(f"could not load tail knots from {path}")
    source = np.asarray(record["knots"], dtype=np.float64)
    source_duration = payload.get("search", {}).get("seconds")
    if source_duration is None:
        source_duration = payload.get("tail_horizon_seconds")
    source_start = float(payload.get("tail_start_seconds", 0.0))
    if (
        source_duration is not None
        and tail_start_seconds is not None
        and tail_seconds is not None
    ):
        source_duration = float(source_duration)
        target_start = float(tail_start_seconds) - source_start
        target_end = target_start + float(tail_seconds)
        if source_duration > 0.0 and 0.0 <= target_start < source_duration:
            source_times = np.linspace(0.0, source_duration, len(source), dtype=np.float64)
            target_times = np.linspace(
                max(0.0, target_start),
                min(source_duration, target_end),
                knot_count,
                dtype=np.float64,
            )
            return np.clip(np.interp(target_times, source_times, source), -1.0, 1.0)
    source_phase = np.linspace(0.0, 1.0, len(source), dtype=np.float64)
    target_phase = np.linspace(0.0, 1.0, knot_count, dtype=np.float64)
    return np.clip(np.interp(target_phase, source_phase, source), -1.0, 1.0)


def replay_to_tail(
    env: NLinkCartPoleEnv,
    controller: dict[str, Any],
    tail_start_seconds: float,
) -> np.ndarray:
    env.reset(seed=0)
    horizon = int(round(tail_start_seconds / env.dt))
    if controller.get("type") == "normalized_force_knots":
        knots = np.asarray(controller["knots"], dtype=np.float64)
        trajectory_seconds = float(controller["trajectory_seconds"])
        if trajectory_seconds <= 0.0:
            raise ValueError("normalized-force source has no positive trajectory_seconds")
        # Match the global CEM's endpoint-inclusive action grid.  Using
        # step*dt/trajectory_seconds here is endpoint-exclusive and creates a
        # one-sample phase drift that can diverge chaotically before handoff.
        action_count = max(2, int(round(trajectory_seconds / env.dt)))
        knot_phase = np.linspace(0.0, 1.0, len(knots), dtype=np.float64)
        for step in range(horizon):
            phase = float(np.clip(step, 0, action_count - 1)) / float(action_count - 1)
            action = float(np.interp(phase, knot_phase, knots))
            _, _, terminated, truncated, info = env.step([action])
            if terminated or truncated:
                raise RuntimeError(
                    f"source swing ended before tail start at step {step}: "
                    f"{info.get('termination_reason')} x={float(env.data.qpos[0]):.3f}"
                )
    elif "controls" in controller:
        source_controls = np.asarray(controller["controls"], dtype=np.float64)
        if source_controls.ndim != 1 or source_controls.size < 2:
            raise ValueError("FDDP source controller must contain at least two controls")
        source_seconds = float(
            controller.get("horizon_seconds", source_controls.size * env.dt)
        )
        source_times = np.linspace(0.0, max(source_seconds, 1e-9), source_controls.size)
        target_times = np.linspace(0.0, float(tail_start_seconds), horizon)
        for step, target_time in enumerate(target_times):
            action = float(
                np.clip(
                    np.interp(float(target_time), source_times, source_controls),
                    -1.0,
                    1.0,
                )
            )
            _, _, terminated, truncated, info = env.step([action])
            if terminated or truncated:
                raise RuntimeError(
                    f"source swing ended before tail start at step {step}: "
                    f"{info.get('termination_reason')} x={float(env.data.qpos[0]):.3f}"
                )
    else:
        knots = np.asarray(controller["knots"], dtype=np.float64)
        trajectory_seconds = float(controller["trajectory_seconds"])
        for step in range(horizon):
            action = trajectory_action(
                env,
                step * env.dt,
                knots,
                trajectory_seconds,
                float(controller["kp"]),
                float(controller["kd"]),
            )
            _, _, terminated, truncated, info = env.step([action])
            if terminated or truncated:
                raise RuntimeError(
                    f"source swing ended before tail start at step {step}: "
                    f"{info.get('termination_reason')} x={float(env.data.qpos[0]):.3f}"
                )
    state_size = mujoco.mj_stateSize(
        env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
    )
    state = np.empty(state_size, dtype=np.float64)
    mujoco.mj_getState(
        env.model,
        env.data,
        state,
        mujoco.mjtState.mjSTATE_FULLPHYSICS.value,
    )
    return state


def load_trace_state(
    env: NLinkCartPoleEnv,
    path: str,
    target_time: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("trace")
    if not isinstance(rows, list) or not rows:
        rows = payload.get("trajectory")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not contain a non-empty trace")
    row = min(
        (candidate for candidate in rows if isinstance(candidate, dict)),
        key=lambda candidate: abs(float(candidate["time_seconds"]) - float(target_time)),
    )
    qpos = np.asarray(row["qpos"], dtype=np.float64)
    qvel = np.asarray(row["qvel"], dtype=np.float64)
    if qpos.shape != (env.model.nq,) or qvel.shape != (env.model.nv,):
        raise ValueError(
            f"trace state dimensions do not match environment: qpos={qpos.shape} qvel={qvel.shape}"
        )
    env.reset(seed=0)
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.time = float(row["time_seconds"])
    mujoco.mj_forward(env.model, env.data)
    state_size = mujoco.mj_stateSize(
        env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
    )
    state = np.empty(state_size, dtype=np.float64)
    mujoco.mj_getState(
        env.model,
        env.data,
        state,
        mujoco.mjtState.mjSTATE_FULLPHYSICS.value,
    )
    return state, row


def physical_metrics(
    sampled: np.ndarray,
    *,
    n_links: int,
) -> dict[str, np.ndarray]:
    qpos = sampled[..., 1 : 1 + n_links + 1].copy()
    qvel = sampled[..., 1 + n_links + 1 : 1 + n_links + 1 + n_links + 1]
    # The serial angle is the wrapped cumulative relative coordinate.  Do not
    # wrap each relative joint first: +pi is the exact hanging state and
    # pre-wrapping it to -pi would create a false upright state.
    absolute_angles = serial_absolute_angles(qpos[..., 1:])
    max_angle = np.max(np.abs(absolute_angles), axis=-1)
    hinge_rms = np.sqrt(np.mean(qvel[..., 1:] ** 2, axis=-1))
    max_hinge = np.max(np.abs(qvel[..., 1:]), axis=-1)
    absolute_angular_velocity = np.cumsum(qvel[..., 1:], axis=-1)
    absolute_angular_velocity_rms = np.sqrt(
        np.mean(absolute_angular_velocity**2, axis=-1)
    )
    max_absolute_angular_velocity = np.max(
        np.abs(absolute_angular_velocity), axis=-1
    )
    return {
        "qpos": qpos,
        "qvel": qvel,
        "absolute_angles": absolute_angles,
        "max_angle": max_angle,
        "hinge_rms": hinge_rms,
        "max_hinge": max_hinge,
        "absolute_angular_velocity_rms": absolute_angular_velocity_rms,
        "max_absolute_angular_velocity": max_absolute_angular_velocity,
        "cart_abs": np.abs(qpos[..., 0]),
        "cart_velocity_abs": np.abs(qvel[..., 0]),
    }


def score_batch(
    env: NLinkCartPoleEnv,
    initial_state: np.ndarray,
    knots: np.ndarray,
    interpolation: np.ndarray,
    pool: list[mujoco.MjData],
    *,
    min_tail_steps: int,
    angle_weight: float,
    hinge_weight: float,
    max_hinge_weight: float,
    absolute_velocity_weight: float,
    cart_weight: float,
    cart_velocity_weight: float,
    rail_penalty_limit: float,
    rail_penalty_weight: float,
    best_score_weight: float,
    terminal_score_weight: float,
    tail_average_weight: float,
    robust_window_steps: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    # Preserve float64 through candidate scoring so a saved tail knot vector
    # has the same force sequence when it is replayed step by step.
    actions = np.clip(knots @ interpolation.T, -1.0, 1.0).astype(np.float64)
    controls = np.repeat(actions, env.frame_skip, axis=1)[:, :, None] * env.force_limit
    initial = np.repeat(initial_state[None, :], len(knots), axis=0)
    rollout_states, _ = mujoco_rollout.rollout(
        env.model,
        pool,
        initial,
        controls,
        persistent_pool=True,
    )
    finite = np.all(np.isfinite(rollout_states), axis=(1, 2))
    bounded = np.max(np.abs(np.nan_to_num(rollout_states, nan=np.inf)), axis=(1, 2)) < 1.0e6
    # An unstable batched rollout may reset its returned state/time buffer
    # instead of returning NaNs. Do not let that reset state become a false
    # low-cost handoff.
    time_valid = np.all(np.diff(rollout_states[..., 0], axis=1) > 0.0, axis=1)
    sampled = rollout_states[:, env.frame_skip - 1 :: env.frame_skip]
    metrics = physical_metrics(sampled, n_links=env.n)
    angle_ratio = metrics["max_angle"] / 0.15
    hinge_ratio = metrics["hinge_rms"] / 0.75
    max_hinge_ratio = metrics["max_hinge"] / 1.50
    absolute_velocity_ratio = metrics["absolute_angular_velocity_rms"] / 0.75
    cart_ratio = metrics["cart_abs"] / 1.25
    cart_velocity_ratio = metrics["cart_velocity_abs"] / 0.50
    state_score = (
        float(angle_weight) * angle_ratio**2
        + float(hinge_weight) * hinge_ratio**2
        + float(max_hinge_weight) * max_hinge_ratio**2
        + float(absolute_velocity_weight) * absolute_velocity_ratio**2
        + float(cart_weight) * cart_ratio**2
        + float(cart_velocity_weight) * cart_velocity_ratio**2
    )
    late = state_score[:, min_tail_steps:]
    best_index = min_tail_steps + np.argmin(late, axis=1)
    best_score = np.min(late, axis=1)
    terminal_score = state_score[:, -1]
    tail_average = np.mean(state_score[:, -min(20, state_score.shape[1]) :], axis=1)
    robust_window = min(
        state_score.shape[1], max(1, int(robust_window_steps))
    )
    robust_slice = slice(-robust_window, None)
    robust_score = np.max(state_score[:, robust_slice], axis=1)
    robust_mean_score = np.mean(state_score[:, robust_slice], axis=1)
    robust_max_angle = np.max(metrics["max_angle"][:, robust_slice], axis=1)
    robust_max_hinge = np.max(metrics["hinge_rms"][:, robust_slice], axis=1)
    robust_max_hinge_peak = np.max(metrics["max_hinge"][:, robust_slice], axis=1)
    robust_max_absolute_velocity = np.max(
        metrics["absolute_angular_velocity_rms"][:, robust_slice], axis=1
    )
    robust_max_absolute_velocity_peak = np.max(
        metrics["max_absolute_angular_velocity"][:, robust_slice], axis=1
    )
    robust_max_cart = np.max(metrics["cart_abs"][:, robust_slice], axis=1)
    robust_max_cart_velocity = np.max(
        metrics["cart_velocity_abs"][:, robust_slice], axis=1
    )
    rail = np.max(metrics["cart_abs"], axis=1)
    rail_penalty = float(rail_penalty_weight) * np.maximum(
        0.0, rail / float(rail_penalty_limit) - 1.0
    ) ** 2
    action_penalty = 0.02 * np.mean(actions * actions, axis=1)
    if robust_window_steps > 0:
        cost = (
            float(best_score_weight) * best_score
            + float(terminal_score_weight) * terminal_score
            + float(tail_average_weight) * tail_average
            + 0.35 * robust_score
        )
        handoff_index = np.full(len(knots), state_score.shape[1] - 1, dtype=int)
    else:
        cost = (
            float(best_score_weight) * best_score
            + float(terminal_score_weight) * terminal_score
            + float(tail_average_weight) * tail_average
        )
        handoff_index = best_index.copy()
    cost = cost + rail_penalty + action_penalty
    invalid = ~(finite & bounded & time_valid & np.isfinite(cost))
    cost[invalid] = 1.0e30
    metrics.update(
        {
            "actions": actions,
            "state_score": state_score,
            "best_index": best_index,
            "best_score": best_score,
            "terminal_score": terminal_score,
            "tail_average": tail_average,
            "robust_window_steps": np.full(
                len(knots), robust_window, dtype=np.int64
            ),
            "robust_score": robust_score,
            "robust_mean_score": robust_mean_score,
            "robust_max_angle": robust_max_angle,
            "robust_max_hinge": robust_max_hinge,
            "robust_max_hinge_peak": robust_max_hinge_peak,
            "robust_max_absolute_velocity": robust_max_absolute_velocity,
            "robust_max_absolute_velocity_peak": robust_max_absolute_velocity_peak,
            "robust_max_cart": robust_max_cart,
            "robust_max_cart_velocity": robust_max_cart_velocity,
            "handoff_index": handoff_index,
            "rail": rail,
            "invalid": invalid,
            "cost": cost,
            "rollout_states": rollout_states,
        }
    )
    return cost, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument(
        "--progress",
        type=float,
        default=1.0,
        help="fixed morphology progress for the replay plant; 1.0 is canonical uniform",
    )
    parser.add_argument(
        "--swing-controller-json",
        required=False,
        help="source swing controller used to replay the prefix; optional with --initial-trace-json",
    )
    parser.add_argument(
        "--initial-trace-json",
        default=None,
        help="exact replay trace whose post-state at --initial-trace-time seeds the tail",
    )
    parser.add_argument(
        "--initial-trace-time",
        type=float,
        default=None,
        help="trace time in seconds to use as the tail initial state; defaults to --tail-start-seconds",
    )
    parser.add_argument("--controller-key", default=None)
    parser.add_argument("--init-tail-json", default=None)
    parser.add_argument(
        "--init-tail-key",
        default="best",
        help="record key inside --init-tail-json, such as best_feasible",
    )
    parser.add_argument("--tail-start-seconds", type=float, default=5.86)
    parser.add_argument("--tail-seconds", type=float, default=5.0)
    parser.add_argument("--knot-count", type=int, default=32)
    parser.add_argument("--population", type=int, default=512)
    parser.add_argument("--elites", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--action-sigma", type=float, default=0.40)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.02)
    parser.add_argument("--angle-weight", type=float, default=30.0)
    parser.add_argument("--hinge-weight", type=float, default=15.0)
    parser.add_argument("--max-hinge-weight", type=float, default=3.0)
    parser.add_argument("--absolute-velocity-weight", type=float, default=0.0)
    parser.add_argument("--cart-weight", type=float, default=2.0)
    parser.add_argument("--cart-velocity-weight", type=float, default=2.0)
    parser.add_argument("--rail-penalty-limit", type=float, default=2.90)
    parser.add_argument("--rail-penalty-weight", type=float, default=1.0e5)
    parser.add_argument("--best-score-weight", type=float, default=0.55)
    parser.add_argument("--terminal-score-weight", type=float, default=0.25)
    parser.add_argument("--tail-average-weight", type=float, default=0.20)
    parser.add_argument("--handoff-angle-limit", type=float, default=0.15)
    parser.add_argument("--handoff-hinge-limit", type=float, default=0.75)
    parser.add_argument("--handoff-cart-limit", type=float, default=1.25)
    parser.add_argument("--handoff-cart-velocity-limit", type=float, default=0.50)
    parser.add_argument("--handoff-absolute-velocity-limit", type=float, default=None)
    parser.add_argument(
        "--robust-window-steps",
        type=int,
        default=0,
        help="require handoff limits across this many final policy steps; 0 keeps point handoff behavior",
    )
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.knot_count < 2 or args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("--progress must be in [0, 1]")
    cfg = apply_overrides(load_config(args.config), args.override)
    # Replay must start from the same deterministic hanging state used by the
    # global route search; residual cart noise would make tail scores describe
    # a different prefix than the one being refined.
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "init_angle_noise_start": 0.0,
        "init_angle_noise_end": 0.0,
        "init_vel_noise_start": 0.0,
        "init_vel_noise_end": 0.0,
        "init_cart_noise_start": 0.0,
        "init_cart_noise_end": 0.0,
        "init_cart_vel_noise_start": 0.0,
        "init_cart_vel_noise_end": 0.0,
    }
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=0)
    if args.initial_trace_json:
        trace_time = (
            float(args.initial_trace_time)
            if args.initial_trace_time is not None
            else float(args.tail_start_seconds)
        )
        initial_state, trace_row = load_trace_state(
            env,
            args.initial_trace_json,
            trace_time,
        )
        controller = None
    else:
        if not args.swing_controller_json:
            raise ValueError("--swing-controller-json is required unless --initial-trace-json is supplied")
        controller = load_controller(args.swing_controller_json, args.controller_key)
        trace_row = None
        initial_state = replay_to_tail(env, controller, args.tail_start_seconds)
    horizon_steps = max(2, int(round(args.tail_seconds / env.dt)))
    interpolation = interpolation_matrix(args.knot_count, horizon_steps)
    pool = [mujoco.MjData(env.model) for _ in range(min(32, max(1, args.population // 16)))]
    rng = np.random.default_rng(args.seed)
    center = load_tail_center(
        args.init_tail_json or args.initial_trace_json,
        args.knot_count,
        args.init_tail_key,
        tail_start_seconds=args.tail_start_seconds,
        tail_seconds=args.tail_seconds,
    )
    sigma = np.full(args.knot_count, args.action_sigma, dtype=np.float64)
    best_record: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = __import__("time").time()

    for iteration in range(args.iterations):
        knots = np.clip(
            center[None, :] + rng.normal(0.0, sigma, size=(args.population, args.knot_count)),
            -1.0,
            1.0,
        )
        knots[0] = center
        cost, metrics = score_batch(
            env,
            initial_state,
            knots,
            interpolation,
            pool,
            min_tail_steps=max(1, int(round(0.20 / env.dt))),
            angle_weight=args.angle_weight,
            hinge_weight=args.hinge_weight,
            max_hinge_weight=args.max_hinge_weight,
            absolute_velocity_weight=args.absolute_velocity_weight,
            cart_weight=args.cart_weight,
            cart_velocity_weight=args.cart_velocity_weight,
            rail_penalty_limit=args.rail_penalty_limit,
            rail_penalty_weight=args.rail_penalty_weight,
            best_score_weight=args.best_score_weight,
            terminal_score_weight=args.terminal_score_weight,
            tail_average_weight=args.tail_average_weight,
            robust_window_steps=args.robust_window_steps,
        )
        order = np.argsort(cost)
        top = int(order[0])
        elite_knots = knots[order[: args.elites]]
        center = np.mean(elite_knots, axis=0)
        sigma = np.maximum(np.std(elite_knots, axis=0) * args.sigma_decay, args.sigma_floor)
        best_index = int(metrics["best_index"][top])
        handoff_index = int(metrics["handoff_index"][top])
        record_index = handoff_index if args.robust_window_steps > 0 else best_index
        record = {
            "iteration": iteration + 1,
            "cost": float(cost[top]),
            "best_score": float(metrics["best_score"][top]),
            "best_time_seconds": float((best_index + 1) * env.dt),
            "handoff_time_seconds": float((handoff_index + 1) * env.dt),
            "max_angle": float(metrics["max_angle"][top, record_index]),
            "hinge_rms": float(metrics["hinge_rms"][top, record_index]),
            "max_hinge": float(metrics["max_hinge"][top, record_index]),
            "absolute_angular_velocity_rms": float(
                metrics["absolute_angular_velocity_rms"][top, record_index]
            ),
            "max_absolute_angular_velocity": float(
                metrics["max_absolute_angular_velocity"][top, record_index]
            ),
            "cart_abs": float(metrics["qpos"][top, record_index, 0]),
            "cart_velocity": float(metrics["qvel"][top, record_index, 0]),
            "robust_window_steps": int(metrics["robust_window_steps"][top]),
            "robust_max_angle": float(metrics["robust_max_angle"][top]),
            "robust_max_hinge": float(metrics["robust_max_hinge"][top]),
            "robust_max_hinge_peak": float(metrics["robust_max_hinge_peak"][top]),
            "robust_max_absolute_velocity": float(
                metrics["robust_max_absolute_velocity"][top]
            ),
            "robust_max_absolute_velocity_peak": float(
                metrics["robust_max_absolute_velocity_peak"][top]
            ),
            "robust_max_cart": float(metrics["robust_max_cart"][top]),
            "robust_max_cart_velocity": float(
                metrics["robust_max_cart_velocity"][top]
            ),
            "rail": float(metrics["rail"][top]),
        }
        history.append(record)
        if best_record is None or record["cost"] < best_record["cost"]:
            best_record = {
                **record,
                "knots": knots[top].astype(float).tolist(),
                "best_state": {
                    "qpos": metrics["qpos"][top, record_index].astype(float).tolist(),
                    "qvel": metrics["qvel"][top, record_index].astype(float).tolist(),
                    "absolute_angles": metrics["absolute_angles"][top, record_index].astype(float).tolist(),
                },
            }
        if args.robust_window_steps > 0:
            handoff_feasible = bool(
                metrics["robust_max_angle"][top] <= args.handoff_angle_limit
                and metrics["robust_max_hinge"][top] <= args.handoff_hinge_limit
                and (
                    args.handoff_absolute_velocity_limit is None
                    or metrics["robust_max_absolute_velocity"][top]
                    <= args.handoff_absolute_velocity_limit
                )
                and metrics["robust_max_cart"][top] <= args.handoff_cart_limit
                and metrics["robust_max_cart_velocity"][top]
                <= args.handoff_cart_velocity_limit
            )
        else:
            handoff_feasible = bool(
                metrics["max_angle"][top, best_index] <= args.handoff_angle_limit
                and metrics["hinge_rms"][top, best_index] <= args.handoff_hinge_limit
                and (
                    args.handoff_absolute_velocity_limit is None
                    or metrics["absolute_angular_velocity_rms"][top, best_index]
                    <= args.handoff_absolute_velocity_limit
                )
                and metrics["cart_abs"][top, best_index] <= args.handoff_cart_limit
                and metrics["cart_velocity_abs"][top, best_index]
                <= args.handoff_cart_velocity_limit
            )
        if handoff_feasible and (
            best_feasible is None or record["cost"] < best_feasible["cost"]
        ):
            best_feasible = {
                **record,
                "knots": knots[top].astype(float).tolist(),
                "best_state": {
                    "qpos": metrics["qpos"][top, record_index].astype(float).tolist(),
                    "qvel": metrics["qvel"][top, record_index].astype(float).tolist(),
                    "absolute_angles": metrics["absolute_angles"][top, record_index].astype(float).tolist(),
                },
            }
        print(
            f"iter={iteration + 1:03d} cost={record['cost']:.4f} "
            f"angle={record['max_angle']:.4f} hinge={record['hinge_rms']:.4f} "
            f"abs_omega={record['absolute_angular_velocity_rms']:.4f} "
            f"max_hinge={record['max_hinge']:.4f} x={record['cart_abs']:.3f} "
            f"xd={record['cart_velocity']:.3f} rail={record['rail']:.3f}",
            flush=True,
        )

    assert best_record is not None
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact-MuJoCo CEM tail search for a low-momentum handoff; final evidence still requires reset-free feedback replay.",
        "source_controller": controller,
        "source_controller_json": (
            None if args.swing_controller_json is None else str(args.swing_controller_json)
        ),
        "initial_trace_json": args.initial_trace_json,
        "initial_trace_time": (
            None if trace_row is None else float(trace_row["time_seconds"])
        ),
        "progress": float(args.progress),
        "controller_key": args.controller_key,
        "init_tail_json": args.init_tail_json,
        "init_tail_key": args.init_tail_key,
        "tail_start_seconds": float(args.tail_start_seconds),
        "tail_horizon_steps": int(horizon_steps),
        "tail_horizon_seconds": float(horizon_steps * env.dt),
        "search": {
            "seed": int(args.seed),
            "knot_count": int(args.knot_count),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "angle_weight": float(args.angle_weight),
            "hinge_weight": float(args.hinge_weight),
            "max_hinge_weight": float(args.max_hinge_weight),
            "absolute_velocity_weight": float(args.absolute_velocity_weight),
            "cart_weight": float(args.cart_weight),
            "cart_velocity_weight": float(args.cart_velocity_weight),
            "rail_penalty_limit": float(args.rail_penalty_limit),
            "rail_penalty_weight": float(args.rail_penalty_weight),
            "best_score_weight": float(args.best_score_weight),
            "terminal_score_weight": float(args.terminal_score_weight),
            "tail_average_weight": float(args.tail_average_weight),
            "handoff_angle_limit": float(args.handoff_angle_limit),
            "handoff_hinge_limit": float(args.handoff_hinge_limit),
            "handoff_cart_limit": float(args.handoff_cart_limit),
            "handoff_cart_velocity_limit": float(args.handoff_cart_velocity_limit),
            "handoff_absolute_velocity_limit": (
                None
                if args.handoff_absolute_velocity_limit is None
                else float(args.handoff_absolute_velocity_limit)
            ),
            "robust_window_steps": int(args.robust_window_steps),
            "wall_time_seconds": float(__import__("time").time() - started),
        },
        "best": best_record,
        "best_feasible": best_feasible,
        "history": history,
        "initial_state": {
            "qpos": initial_state[1 : 1 + env.model.nq].astype(float).tolist(),
            "qvel": initial_state[1 + env.model.nq : 1 + env.model.nq + env.model.nv].astype(float).tolist(),
            "time_seconds": float(initial_state[0]),
        },
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(result, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
