#!/usr/bin/env python
"""Replan around a transferred swing route on the exact target plant.

This is a development probe for the high-link frontier.  The controller uses
the transferred predecessor route as a moving reference, but searches a
bounded residual force profile around that reference at every replanning
point.  The score tracks the full serial chain in absolute-angle coordinates
and includes an optional upright capture tail.  It is deliberately not a
release evaluator: the saved artifact must still pass the project's exact
replay and noisy-gate checks before it can be considered evidence.
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
from gcartpole.generalized_solver import periodic_coordinate_error
from gcartpole.modal import StateScales, dimensionless_absolute_transform

try:
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from search_swingup_capture import lqr_gain


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


def load_route(path: str | Path, n_links: int) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    controller = payload.get("controller")
    search = payload.get("search")
    if not isinstance(controller, dict) or not isinstance(search, dict):
        raise ValueError("route must contain controller and search objects")
    controls = np.asarray(controller.get("controls", []), dtype=np.float64)
    nominal = np.asarray(search.get("nominal_coordinate_states", []), dtype=np.float64)
    if controls.ndim != 1 or controls.size < 2:
        raise ValueError("route controls must contain at least two samples")
    expected_dim = 2 * (int(n_links) + 1)
    if nominal.shape != (controls.size + 1, expected_dim):
        raise ValueError("route nominal states do not match the target dimension")
    transform = np.asarray(
        controller.get("coordinate_transform", []), dtype=np.float64
    )
    if transform.shape != (expected_dim, expected_dim):
        transform = dimensionless_absolute_transform(
            n_links,
            StateScales(
                cart_position=1.25,
                absolute_angle=0.15,
                cart_velocity=0.50,
                hinge_velocity=0.75,
            ),
        )
    return {
        "path": str(Path(path)),
        "sha256": data_sha256(payload),
        "controls": np.clip(controls, -1.0, 1.0),
        "nominal": nominal,
        "transform": transform,
    }


def state_coordinates(
    states: np.ndarray, transform: np.ndarray, n_links: int
) -> np.ndarray:
    """Map full physical states to the route's wrapped coordinate convention."""

    values = np.asarray(states, dtype=np.float64)
    d = int(n_links) + 1
    if values.shape[-1] != 2 * d:
        raise ValueError("state dimension does not match link count")
    physical = values.copy()
    physical[..., 1:d] = wrap_angle(physical[..., 1:d])
    return np.einsum("ij,...j->...i", transform, physical)


def physical_states_from_full(
    full_states: np.ndarray, env: NLinkCartPoleEnv
) -> np.ndarray:
    """Extract qpos+qvel from MuJoCo's full-physics state (which starts with time)."""

    values = np.asarray(full_states, dtype=np.float64)
    start = 1
    end = start + int(env.model.nq) + int(env.model.nv)
    if values.shape[-1] < end:
        raise ValueError("full-physics state is shorter than qpos+qvel")
    return values[..., start:end]


def periodic_error_batch(
    coordinates: np.ndarray,
    references: np.ndarray,
    transform: np.ndarray,
    n_links: int,
) -> np.ndarray:
    """Compute periodic route errors without crossing the +/-pi branch."""

    current = np.asarray(coordinates, dtype=np.float64)
    reference = np.asarray(references, dtype=np.float64)
    inverse = np.linalg.inv(transform)
    physical_current = np.einsum("ij,...j->...i", inverse, current)
    physical_reference = np.einsum("ij,...j->...i", inverse, reference)
    d = int(n_links) + 1
    delta = physical_current - physical_reference
    delta[..., 1:d] = wrap_angle(delta[..., 1:d])
    return np.einsum("ij,...j->...i", transform, delta)


def potential_weights(env: NLinkCartPoleEnv) -> np.ndarray:
    lengths = np.asarray(env.morphology.lengths, dtype=np.float64)
    masses = np.asarray(env.morphology.masses, dtype=np.float64)
    downstream_mass = np.cumsum(masses[::-1])[::-1] - 0.5 * masses
    return lengths * downstream_mass


def capture_quality(states: np.ndarray, n_links: int) -> np.ndarray:
    values = np.asarray(states, dtype=np.float64)
    d = int(n_links) + 1
    qpos = values[..., :d]
    qvel = values[..., d:]
    absolute = serial_absolute_angles(qpos[..., 1:])
    absolute_rate = np.cumsum(qvel[..., 1:], axis=-1)
    return (
        (np.max(np.abs(absolute), axis=-1) / 0.15) ** 2
        + (np.sqrt(np.mean(qvel[..., 1:] ** 2, axis=-1)) / 0.75) ** 2
        + (np.sqrt(np.mean(absolute_rate**2, axis=-1)) / 0.75) ** 2
        + (np.abs(qpos[..., 0]) / 1.25) ** 2
        + (np.abs(qvel[..., 0]) / 0.50) ** 2
    )


def lqr_tail_quality(
    env: NLinkCartPoleEnv,
    full_states: np.ndarray,
    *,
    gain: np.ndarray,
    scale: float,
    tail_steps: int,
) -> np.ndarray:
    """Roll the exact upright LQR from each candidate terminal state."""

    values = np.asarray(full_states, dtype=np.float64)
    qualities = np.full(values.shape[0], 1.0e9, dtype=np.float64)
    for index, state in enumerate(values):
        if not np.all(np.isfinite(state)):
            continue
        physical = physical_states_from_full(state, env)
        if abs(float(physical[0])) >= min(float(env.rail_limit), 8.0):
            continue
        data = mujoco.MjData(env.model)
        mujoco.mj_setState(
            env.model,
            data,
            state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS.value,
        )
        mujoco.mj_forward(env.model, data)
        tail_rows: list[float] = []
        for _ in range(max(1, int(tail_steps))):
            qpos = np.asarray(data.qpos, dtype=np.float64)
            qvel = np.asarray(data.qvel, dtype=np.float64)
            if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
                break
            absolute = serial_absolute_angles(qpos[1 : 1 + env.n])
            absolute_rate = np.cumsum(qvel[1 : 1 + env.n])
            tail_rows.append(
                (float(np.max(np.abs(absolute))) / 0.15) ** 2
                + (float(np.sqrt(np.mean(qvel[1:] ** 2))) / 0.75) ** 2
                + (float(np.sqrt(np.mean(absolute_rate**2))) / 0.75) ** 2
                + (abs(float(qpos[0])) / 1.25) ** 2
                + (abs(float(qvel[0])) / 0.50) ** 2
            )
            state_vector = np.r_[
                qpos[0],
                wrap_angle(qpos[1 : 1 + env.n]),
                qvel,
            ]
            data.ctrl[0] = float(
                np.clip(-float(scale) * float(gain @ state_vector), -1.0, 1.0)
                * env.force_limit
            )
            for _ in range(env.frame_skip):
                mujoco.mj_step(env.model, data)
            if abs(float(data.qpos[0])) >= float(env.rail_limit):
                break
        if tail_rows:
            tail_rows = tail_rows[-max(1, len(tail_rows) // 2) :]
            qualities[index] = float(np.mean(tail_rows))
    return qualities


def score_plan(
    env: NLinkCartPoleEnv,
    initial_state: np.ndarray,
    residual_knots: np.ndarray,
    base_actions: np.ndarray,
    route_references: np.ndarray,
    interpolation: np.ndarray,
    pool: list[mujoco.MjData],
    *,
    transform: np.ndarray,
    route_cursor: int,
    reference_weight: float,
    reference_angle_scale: float,
    reference_rate_scale: float,
    reference_cart_scale: float,
    potential_weight: float,
    terminal_weight: float,
    rail_soft_limit: float,
    rail_weight: float,
    action_weight: float,
    slew_weight: float,
    capture_gain: np.ndarray | None,
    capture_tail_scale: float,
    capture_tail_steps: int,
    capture_tail_weight: float,
    capture_potential_threshold: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    profile_residual = residual_knots @ interpolation.T
    actions = np.clip(base_actions[None, :] + profile_residual, -1.0, 1.0)
    controls = np.repeat(actions, env.frame_skip, axis=1)[:, :, None] * env.force_limit
    initial = np.repeat(np.asarray(initial_state, dtype=np.float64)[None, :], len(actions), axis=0)
    full_states, _ = mujoco_rollout.rollout(
        env.model, pool, initial, controls, persistent_pool=True
    )
    sampled_full = full_states[:, env.frame_skip - 1 :: env.frame_skip]
    if sampled_full.shape[1] != actions.shape[1]:
        sampled_full = full_states[:, env.frame_skip :: env.frame_skip]
    sampled = physical_states_from_full(sampled_full, env)
    coordinates = state_coordinates(sampled, transform, env.n)
    reference_indices = np.clip(
        route_cursor + np.arange(actions.shape[1], dtype=np.int64),
        0,
        route_references.shape[0] - 1,
    )
    references = route_references[reference_indices]
    errors = periodic_error_batch(
        coordinates,
        references[None, :, :],
        transform,
        env.n,
    )
    d = env.n + 1
    position_error = errors[..., :d]
    velocity_error = errors[..., d:]
    path_cost = (
        (position_error[..., 0] / max(1e-9, reference_cart_scale)) ** 2
        + np.mean(
            (position_error[..., 1:] / max(1e-9, reference_angle_scale)) ** 2,
            axis=-1,
        )
        + (velocity_error[..., 0] / max(1e-9, reference_cart_scale)) ** 2
        + np.mean(
            (velocity_error[..., 1:] / max(1e-9, reference_rate_scale)) ** 2,
            axis=-1,
        )
    )
    qpos = sampled[..., :d]
    absolute = serial_absolute_angles(qpos[..., 1:])
    weights = potential_weights(env)
    potential_fraction = np.sum(
        weights[None, None, :] * (np.cos(absolute) + 1.0) * 0.5,
        axis=-1,
    ) / max(1e-12, float(np.sum(weights)))
    tail = max(1, actions.shape[1] // 3)
    terminal_capture = capture_quality(sampled[:, -1], env.n)
    tail_quality = np.zeros(actions.shape[0], dtype=np.float64)
    tail_gate = np.zeros(actions.shape[0], dtype=np.float64)
    if capture_gain is not None and capture_tail_steps > 0:
        tail_quality = lqr_tail_quality(
            env,
            full_states[:, -1],
            gain=capture_gain,
            scale=capture_tail_scale,
            tail_steps=capture_tail_steps,
        )
        peak_potential = np.max(potential_fraction[:, tail:], axis=1)
        tail_gate = np.clip(
            (peak_potential - float(capture_potential_threshold))
            / max(1.0e-9, 1.0 - float(capture_potential_threshold)),
            0.0,
            1.0,
        )
    rail = np.max(np.abs(qpos[..., 0]), axis=-1)
    rail_penalty = rail_weight * np.maximum(0.0, rail / rail_soft_limit - 1.0) ** 2
    action_slew = np.mean(np.diff(actions, axis=1) ** 2, axis=1)
    cost = (
        reference_weight * np.mean(path_cost, axis=1)
        - potential_weight * np.max(potential_fraction[:, :tail], axis=1)
        + terminal_weight * terminal_capture
        + capture_tail_weight * tail_quality * tail_gate
        + rail_penalty
        + action_weight * np.mean(actions**2, axis=1)
        + slew_weight * action_slew
    )
    return cost, {
        "states": sampled,
        "coordinates": coordinates,
        "path_cost": path_cost,
        "potential_fraction": potential_fraction,
        "terminal_capture": terminal_capture,
        "capture_tail_quality": tail_quality,
        "capture_tail_gate": tail_gate,
        "rail": rail,
        "actions": actions,
        "cost": cost,
    }


def plan(
    env: NLinkCartPoleEnv,
    initial_state: np.ndarray,
    center: np.ndarray,
    base_actions: np.ndarray,
    references: np.ndarray,
    interpolation: np.ndarray,
    *,
    transform: np.ndarray,
    route_cursor: int,
    population: int,
    elites: int,
    iterations: int,
    sigma: float,
    sigma_floor: float,
    rng: np.random.Generator,
    score_kwargs: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float], np.ndarray]:
    pool = [mujoco.MjData(env.model) for _ in range(min(32, max(1, population // 16)))]
    current = np.asarray(center, dtype=np.float64).copy()
    spread = np.full(current.shape[0], float(sigma), dtype=np.float64)
    best_cost = float("inf")
    best_metrics: dict[str, float] = {}
    best_actions = np.clip(base_actions, -1.0, 1.0)
    for _ in range(max(1, iterations)):
        candidates = np.clip(
            current[None, :] + rng.normal(0.0, spread, size=(population, current.size)),
            -1.0,
            1.0,
        )
        candidates[0] = current
        costs, metrics = score_plan(
            env,
            initial_state,
            candidates,
            base_actions,
            references,
            interpolation,
            pool,
            transform=transform,
            route_cursor=route_cursor,
            **score_kwargs,
        )
        order = np.argsort(costs)
        elite = candidates[order[:elites]]
        current = np.mean(elite, axis=0)
        spread = np.maximum(np.std(elite, axis=0) * 0.92, sigma_floor)
        top = int(order[0])
        if float(costs[top]) < best_cost:
            best_cost = float(costs[top])
            best_actions = metrics["actions"][top].copy()
            best_metrics = {
                "cost": best_cost,
                "best_path_cost": float(np.min(metrics["path_cost"][top])),
                "max_potential_fraction": float(np.max(metrics["potential_fraction"][top])),
                "terminal_capture_quality": float(metrics["terminal_capture"][top]),
                "capture_tail_quality": float(metrics["capture_tail_quality"][top]),
                "capture_tail_gate": float(metrics["capture_tail_gate"][top]),
                "max_cart_abs": float(np.max(metrics["rail"][top])),
            }
    return current, best_metrics, best_actions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--initial-controller", required=True)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--horizon-steps", type=int, default=100)
    parser.add_argument("--knot-count", type=int, default=20)
    parser.add_argument("--replan-steps", type=int, default=4)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--action-sigma", type=float, default=0.30)
    parser.add_argument("--sigma-floor", type=float, default=0.02)
    parser.add_argument("--reference-weight", type=float, default=20.0)
    parser.add_argument("--reference-angle-scale", type=float, default=0.45)
    parser.add_argument("--reference-rate-scale", type=float, default=2.0)
    parser.add_argument("--reference-cart-scale", type=float, default=1.50)
    parser.add_argument("--potential-weight", type=float, default=100.0)
    parser.add_argument("--terminal-weight", type=float, default=0.10)
    parser.add_argument("--rail-soft-limit", type=float, default=2.75)
    parser.add_argument("--rail-weight", type=float, default=50000.0)
    parser.add_argument("--action-weight", type=float, default=0.02)
    parser.add_argument("--slew-weight", type=float, default=0.05)
    parser.add_argument("--capture-tail-steps", type=int, default=0)
    parser.add_argument("--capture-tail-scale", type=float, default=1.0)
    parser.add_argument("--capture-tail-weight", type=float, default=0.0)
    parser.add_argument("--capture-potential-threshold", type=float, default=0.90)
    parser.add_argument("--capture-lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--phase-window", type=int, default=16)
    parser.add_argument("--seed", type=int, default=21116)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.horizon_steps < 2 or args.knot_count < 2 or args.replan_steps < 1:
        raise ValueError("horizon, knot count, and replan steps must be positive")
    if not 1 <= args.elites <= args.population:
        raise ValueError("elites must be in 1..population")
    if min(args.seconds, args.action_sigma, args.sigma_floor, args.reference_weight) <= 0.0:
        raise ValueError("duration, action sigma, sigma floor, and reference weight must be positive")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for noise_key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][noise_key] = 0.0
        cfg["env"][f"{noise_key}_start"] = 0.0
        cfg["env"][f"{noise_key}_end"] = 0.0
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    capture_gain = None
    if args.capture_tail_steps > 0 and args.capture_tail_weight > 0.0:
        capture_gain = lqr_gain(
            cfg,
            progress=1.0,
            fd_eps=1.0e-7,
            control_cost=args.capture_lqr_control_cost,
        )
    route = load_route(args.initial_controller, env.n)
    controls = route["controls"]
    references = route["nominal"]
    transform = route["transform"]
    step_count = min(env.max_steps, max(2, round(args.seconds / env.dt)))
    horizon = min(args.horizon_steps, step_count)
    interpolation = interpolation_matrix(args.knot_count, horizon)
    rng = np.random.default_rng(args.seed)
    route_cursor = 0
    residual_center = np.zeros(args.knot_count, dtype=np.float64)
    trajectory: list[dict[str, Any]] = []
    replans: list[dict[str, Any]] = []
    started = time.time()
    final_info: dict[str, Any] = {}
    max_cart = 0.0
    best_angle = float("inf")
    best_hold = 0.0
    success = False

    for step in range(step_count):
        if step % args.replan_steps == 0:
            state_size = mujoco.mj_stateSize(
                env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
            )
            current_state = np.empty(state_size, dtype=np.float64)
            mujoco.mj_getState(
                env.model,
                env.data,
                current_state,
                mujoco.mjtState.mjSTATE_FULLPHYSICS.value,
            )
            current_physical = physical_states_from_full(current_state, env)
            current_coord = state_coordinates(current_physical, transform, env.n)
            search_end = min(references.shape[0], route_cursor + args.phase_window + 1)
            candidate_refs = references[route_cursor:search_end]
            if len(candidate_refs):
                errors = periodic_coordinate_error(
                    current_coord[None, :],
                    candidate_refs,
                    transform,
                )
                route_cursor += int(np.argmin(np.sum(errors**2, axis=1)))
            base_indices = np.clip(
                route_cursor + np.arange(horizon), 0, controls.size - 1
            )
            base = controls[base_indices]
            residual_center = np.asarray(residual_center, dtype=np.float64)
            residual_center = np.clip(residual_center, -1.0, 1.0)
            _, plan_metrics, best_profile = plan(
                env,
                current_state,
                residual_center,
                base,
                references,
                interpolation,
                transform=transform,
                route_cursor=route_cursor,
                population=args.population,
                elites=args.elites,
                iterations=args.iterations,
                sigma=args.action_sigma,
                sigma_floor=args.sigma_floor,
                rng=rng,
                score_kwargs={
                    "reference_weight": args.reference_weight,
                    "reference_angle_scale": args.reference_angle_scale,
                    "reference_rate_scale": args.reference_rate_scale,
                    "reference_cart_scale": args.reference_cart_scale,
                    "potential_weight": args.potential_weight,
                    "terminal_weight": args.terminal_weight,
                    "rail_soft_limit": args.rail_soft_limit,
                    "rail_weight": args.rail_weight,
                    "action_weight": args.action_weight,
                    "slew_weight": args.slew_weight,
                    "capture_gain": capture_gain,
                    "capture_tail_scale": args.capture_tail_scale,
                    "capture_tail_steps": args.capture_tail_steps,
                    "capture_tail_weight": args.capture_tail_weight,
                    "capture_potential_threshold": args.capture_potential_threshold,
                },
            )
            replans.append(
                {
                    "time_seconds": float(step * env.dt),
                    "route_cursor": int(route_cursor),
                    **plan_metrics,
                }
            )
            apply_count = min(args.replan_steps, step_count - step)
            planned_actions = best_profile[:apply_count]
        else:
            planned_actions = planned_actions[1:]
            apply_count = 1
        action = float(np.clip(planned_actions[0], -1.0, 1.0))
        _, reward, terminated, truncated, info = env.step([action])
        route_cursor = min(route_cursor + 1, controls.size - 1)
        qpos = np.asarray(env.data.qpos, dtype=np.float64)
        qvel = np.asarray(env.data.qvel, dtype=np.float64)
        absolute = serial_absolute_angles(qpos[1:])
        absolute_rate = np.cumsum(qvel[1:])
        max_cart = max(max_cart, abs(float(qpos[0])))
        best_angle = min(best_angle, float(np.max(np.abs(absolute))))
        best_hold = max(best_hold, float(info.get("max_upright_streak_seconds", 0.0)))
        trajectory.append(
            {
                "step": int(step + 1),
                "time_seconds": float((step + 1) * env.dt),
                "action": action,
                "reward": float(reward),
                "x": float(qpos[0]),
                "cart_velocity": float(qvel[0]),
                "absolute_angles": absolute.astype(float).tolist(),
                "absolute_angular_velocity_rms": float(np.sqrt(np.mean(absolute_rate**2))),
                "hinge_velocity_rms": float(np.sqrt(np.mean(qvel[1:] ** 2))),
                "max_abs_angle": float(np.max(np.abs(absolute))),
                "is_upright": bool(info.get("is_upright", False)),
                "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
                "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
                "route_cursor": int(route_cursor),
                "qpos": qpos.astype(float).tolist(),
                "qvel": qvel.astype(float).tolist(),
            }
        )
        final_info = dict(info)
        success = bool(info.get("success", False))
        if terminated or truncated:
            break

    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_route_tracking_mpc_not_solution",
        "not_solution": True,
        "summary": "Exact-MuJoCo receding-horizon residual search around the transferred predecessor route.",
        "config_path": str(Path(args.config)),
        "initial_controller": {
            "path": str(Path(args.initial_controller)),
            "sha256": route["sha256"],
        },
        "search": {
            "seconds": float(args.seconds),
            "horizon_steps": int(horizon),
            "knot_count": int(args.knot_count),
            "replan_steps": int(args.replan_steps),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "action_sigma": float(args.action_sigma),
            "sigma_floor": float(args.sigma_floor),
            "reference_weight": float(args.reference_weight),
            "reference_angle_scale": float(args.reference_angle_scale),
            "reference_rate_scale": float(args.reference_rate_scale),
            "reference_cart_scale": float(args.reference_cart_scale),
            "potential_weight": float(args.potential_weight),
            "terminal_weight": float(args.terminal_weight),
            "rail_soft_limit": float(args.rail_soft_limit),
            "rail_weight": float(args.rail_weight),
            "action_weight": float(args.action_weight),
            "slew_weight": float(args.slew_weight),
            "capture_tail_steps": int(args.capture_tail_steps),
            "capture_tail_scale": float(args.capture_tail_scale),
            "capture_tail_weight": float(args.capture_tail_weight),
            "capture_potential_threshold": float(args.capture_potential_threshold),
            "capture_lqr_control_cost": float(args.capture_lqr_control_cost),
            "phase_window": int(args.phase_window),
            "seed": int(args.seed),
            "wall_time_seconds": float(time.time() - started),
        },
        "success": bool(success),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", best_hold)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "termination_reason": final_info.get("termination_reason"),
        "simulated_seconds": float(len(trajectory) * env.dt),
        "best_angle": float(best_angle),
        "max_cart_abs": float(max_cart),
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
        f"angle={result['best_angle']:.3f} cart={result['max_cart_abs']:.3f} "
        f"replans={len(replans)}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
