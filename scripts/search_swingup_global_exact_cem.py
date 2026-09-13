#!/usr/bin/env python
"""Serial exact-MuJoCo CEM search for an n-link swing-up proposal.

This is intentionally slower than the batched proposal search.  Every
candidate is evaluated through the same ``NLinkCartPoleEnv.step`` loop used by
the final verifier, so a saved knot waveform can be replayed without relying
on batched-integrator equivalence.  The optional residual mode preserves a
morphology-derived deterministic controller exactly and searches only a
low-dimensional bounded correction around it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from probe_swingup_trajectory import trajectory_action

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import (
    data_sha256,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)


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
    path: str | None,
    knot_count: int,
    seconds: float,
    env: NLinkCartPoleEnv,
) -> np.ndarray:
    if path is None:
        return np.zeros(knot_count, dtype=np.float64)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    record = payload.get("best") if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        record = payload.get("controller") if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        raise TypeError(f"{path} does not contain a best record or controller")
    # Full-resolution controls are authoritative when a residual-search record
    # also contains its low-dimensional correction knots.
    if "controls" in record:
        source = np.asarray(record["controls"], dtype=np.float64)
        source_seconds = float(
            payload.get("controller", {}).get(
                "horizon_seconds",
                payload.get("search", {}).get("seconds", seconds),
            )
            if isinstance(payload, dict)
            else seconds
        )
        source_t = np.linspace(
            0.0, max(source_seconds, 1e-9), len(source), dtype=np.float64
        )
        target_t = np.linspace(0.0, seconds, knot_count, dtype=np.float64)
        return np.clip(
            np.interp(target_t, source_t, source, left=source[0], right=0.0),
            -1.0,
            1.0,
        )

    if "knots" in record:
        source = np.asarray(record["knots"], dtype=np.float64)
        source_seconds = float(payload.get("search", {}).get("seconds", seconds))
        source_t = np.linspace(0.0, source_seconds, len(source), dtype=np.float64)
        target_t = np.linspace(0.0, seconds, knot_count, dtype=np.float64)
        return np.clip(np.interp(target_t, source_t, source), -1.0, 1.0)

    final_eval = payload.get("final_eval") if isinstance(payload, dict) else None
    trace = final_eval.get("trace") if isinstance(final_eval, dict) else None
    if (
        isinstance(trace, list)
        and len(trace) >= 2
        and all("action" in row for row in trace)
    ):
        source = np.asarray([row["action"] for row in trace], dtype=np.float64)
        source_seconds = float(len(source) * env.dt)
        source_t = np.arange(source.size, dtype=np.float64) * env.dt
        target_t = np.linspace(0.0, seconds, knot_count, dtype=np.float64)
        return np.clip(
            np.interp(target_t, source_t, source, left=source[0], right=source[-1]),
            -1.0,
            1.0,
        )

    # A trajectory-CEM artifact stores cart-position targets under
    # best.controller. Convert that controller to the exact normalized force
    # waveform on the same serial plant before using it as a CEM center.
    controller = record.get("controller")
    if isinstance(controller, dict) and "knots" in controller:
        source = np.asarray(controller["knots"], dtype=np.float64)
        source_seconds = float(controller.get("trajectory_seconds", seconds))
        kp = float(controller.get("kp", 0.0))
        kd = float(controller.get("kd", 0.0))
        target_t = np.linspace(0.0, seconds, knot_count, dtype=np.float64)
        actions: list[float] = []
        sample_index = 0
        step_count = max(2, round(seconds / env.dt))
        env.reset(seed=0)
        last_action = 0.0
        for step in range(step_count):
            t = step * env.dt
            action = trajectory_action(env, t, source, source_seconds, kp, kd)
            while sample_index < len(target_t) and t >= target_t[sample_index]:
                actions.append(float(action))
                sample_index += 1
            last_action = float(action)
            _, _, terminated, truncated, _ = env.step([action])
            if terminated or truncated:
                break
        while len(actions) < knot_count:
            actions.append(last_action)
        return np.clip(np.asarray(actions[:knot_count], dtype=np.float64), -1.0, 1.0)
    raise ValueError(f"could not load force or cart-position knots from {path}")


def evaluate_candidate(
    env: NLinkCartPoleEnv,
    actions: np.ndarray,
    *,
    handoff_min_steps: int,
    rolling_window: int,
    rail_penalty_limit: float,
    rail_penalty_weight: float,
    angle_weight: float,
    hinge_weight: float,
    max_hinge_weight: float,
    absolute_velocity_weight: float,
    cart_weight: float,
    cart_velocity_weight: float,
    relative_angle_weight: float,
    relative_angle_scale: float,
    best_score_weight: float,
    terminal_score_weight: float,
    tail_average_weight: float,
    constraint_barrier_weight: float,
    handoff_angle_limit: float,
    handoff_hinge_limit: float,
    handoff_absolute_velocity_limit: float,
    handoff_cart_limit: float,
    handoff_cart_velocity_limit: float,
) -> tuple[float, dict[str, Any]]:
    env.reset(seed=0)
    rows: list[dict[str, Any]] = []
    terminated = False
    termination_reason: str | None = None
    for action in actions:
        _, _, terminated, truncated, info = env.step([float(action)])
        relative_angles, _ = env._angles()
        absolute_velocity = env._absolute_angular_velocity()
        row = {
            "qpos": np.asarray(env.data.qpos, dtype=np.float64).copy(),
            "qvel": np.asarray(env.data.qvel, dtype=np.float64).copy(),
            "max_angle": float(info["max_abs_angle"]),
            "relative_angle_rms": float(np.sqrt(np.mean(relative_angles**2))),
            "max_relative_angle": float(np.max(np.abs(relative_angles))),
            "hinge_rms": float(info["hinge_velocity_rms"]),
            "max_hinge": float(np.max(np.abs(env.data.qvel[1 : 1 + env.n]))),
            "absolute_velocity_rms": float(np.sqrt(np.mean(absolute_velocity**2))),
            "cart_abs": abs(float(env.data.qpos[0])),
            "cart_velocity_abs": abs(float(env.data.qvel[0])),
        }
        rows.append(row)
        if terminated or truncated:
            termination_reason = str(info.get("termination_reason"))
            break

    if not rows:
        return float("inf"), {"invalid": True, "termination_reason": termination_reason}
    finite = all(
        np.all(np.isfinite(row["qpos"])) and np.all(np.isfinite(row["qvel"]))
        for row in rows
    )
    complete = len(rows) == len(actions) and not terminated
    if not finite or not complete:
        return float("inf"), {
            "invalid": True,
            "termination_reason": termination_reason,
            "steps": len(rows),
            "max_cart": max(row["cart_abs"] for row in rows),
        }

    score = np.asarray(
        [
            float(angle_weight) * (row["max_angle"] / 0.15) ** 2
            + float(hinge_weight) * (row["hinge_rms"] / 0.75) ** 2
            + float(max_hinge_weight) * (row["max_hinge"] / 1.50) ** 2
            + float(cart_weight) * (row["cart_abs"] / 1.25) ** 2
            + float(cart_velocity_weight) * (row["cart_velocity_abs"] / 0.50) ** 2
            + float(absolute_velocity_weight)
            * (row["absolute_velocity_rms"] / 0.75) ** 2
            + float(relative_angle_weight)
            * (row["relative_angle_rms"] / max(1e-9, float(relative_angle_scale))) ** 2
            for row in rows
        ],
        dtype=np.float64,
    )
    late = score[handoff_min_steps:]
    if late.size == 0:
        return float("inf"), {"invalid": True, "termination_reason": "short_horizon"}
    constraint_violation = np.asarray(
        [
            max(0.0, row["max_angle"] / handoff_angle_limit - 1.0) ** 2
            + max(0.0, row["hinge_rms"] / handoff_hinge_limit - 1.0) ** 2
            + max(
                0.0,
                row["absolute_velocity_rms"] / handoff_absolute_velocity_limit - 1.0,
            )
            ** 2
            + max(0.0, row["cart_abs"] / handoff_cart_limit - 1.0) ** 2
            + max(
                0.0,
                row["cart_velocity_abs"] / handoff_cart_velocity_limit - 1.0,
            )
            ** 2
            for row in rows
        ],
        dtype=np.float64,
    )
    late_violation = constraint_violation[handoff_min_steps:]
    if constraint_barrier_weight > 0.0:
        window = min(max(1, rolling_window), late.size)
        robust_violation = np.asarray(
            [
                np.max(late_violation[index : index + window])
                for index in range(late.size - window + 1)
            ],
            dtype=np.float64,
        )
        best_offset = int(np.argmin(robust_violation))
        best_index = handoff_min_steps + best_offset + window - 1
        best_score = float(np.mean(late[best_offset : best_offset + window]))
        best_constraint_violation = float(robust_violation[best_offset])
    elif rolling_window > 1 and late.size >= rolling_window:
        rolling = np.asarray(
            [
                np.mean(late[index : index + rolling_window])
                for index in range(late.size - rolling_window + 1)
            ],
            dtype=np.float64,
        )
        best_offset = int(np.argmin(rolling))
        best_index = handoff_min_steps + best_offset + rolling_window - 1
        best_score = float(rolling[best_offset])
        best_constraint_violation = float(constraint_violation[best_index])
    else:
        best_index = handoff_min_steps + int(np.argmin(late))
        best_score = float(np.min(late))
        best_constraint_violation = float(constraint_violation[best_index])
    terminal_score = float(score[-1])
    tail_average = float(np.mean(score[-min(30, len(score)) :]))
    max_cart = max(row["cart_abs"] for row in rows)
    rail_penalty = (
        rail_penalty_weight
        * max(0.0, max_cart / max(1e-9, rail_penalty_limit) - 1.0) ** 2
    )
    cost = (
        float(best_score_weight) * best_score
        + float(terminal_score_weight) * terminal_score
        + float(tail_average_weight) * tail_average
        + rail_penalty
        + float(constraint_barrier_weight) * best_constraint_violation
    )
    cost += 0.02 * float(np.mean(np.asarray(actions) ** 2))
    cost += 0.10 * float(np.mean(np.diff(np.asarray(actions)) ** 2))
    selected = rows[best_index]
    return cost, {
        "invalid": False,
        "termination_reason": "time_limit"
        if len(rows) == len(actions)
        else termination_reason,
        "steps": len(rows),
        "best_index": best_index,
        "best_time_seconds": float((best_index + 1) * env.dt),
        "best_score": best_score,
        "capture_constraint_violation": best_constraint_violation,
        "capture_constraints_satisfied": bool(best_constraint_violation <= 1.0e-12),
        "terminal_score": terminal_score,
        "tail_average": tail_average,
        "max_angle": selected["max_angle"],
        "relative_angle_rms": selected["relative_angle_rms"],
        "max_relative_angle": selected["max_relative_angle"],
        "hinge_rms": selected["hinge_rms"],
        "max_hinge": selected["max_hinge"],
        "absolute_velocity_rms": selected["absolute_velocity_rms"],
        "cart": float(selected["qpos"][0]),
        "cart_velocity": float(selected["qvel"][0]),
        "rail": max_cart,
        "best_state": {
            "qpos": selected["qpos"].astype(float).tolist(),
            "qvel": selected["qvel"].astype(float).tolist(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--knot-count", type=int, default=32)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--handoff-min-time", type=float, default=4.0)
    parser.add_argument("--rolling-window", type=int, default=10)
    parser.add_argument("--action-sigma", type=float, default=0.30)
    parser.add_argument("--sigma-decay", type=float, default=0.92)
    parser.add_argument("--sigma-floor", type=float, default=0.008)
    parser.add_argument("--rail-penalty-limit", type=float, default=12.5)
    parser.add_argument("--rail-penalty-weight", type=float, default=20000.0)
    parser.add_argument("--angle-weight", type=float, default=35.0)
    parser.add_argument("--hinge-weight", type=float, default=16.0)
    parser.add_argument("--max-hinge-weight", type=float, default=4.0)
    parser.add_argument("--absolute-velocity-weight", type=float, default=8.0)
    parser.add_argument("--cart-weight", type=float, default=3.0)
    parser.add_argument("--cart-velocity-weight", type=float, default=3.0)
    parser.add_argument("--relative-angle-weight", type=float, default=12.0)
    parser.add_argument("--relative-angle-scale", type=float, default=0.15)
    parser.add_argument("--best-score-weight", type=float, default=0.60)
    parser.add_argument("--terminal-score-weight", type=float, default=0.25)
    parser.add_argument("--tail-average-weight", type=float, default=0.15)
    parser.add_argument("--constraint-barrier-weight", type=float, default=0.0)
    parser.add_argument("--handoff-angle-limit", type=float, default=0.15)
    parser.add_argument("--handoff-hinge-limit", type=float, default=0.75)
    parser.add_argument("--handoff-absolute-velocity-limit", type=float, default=0.75)
    parser.add_argument("--handoff-cart-limit", type=float, default=1.25)
    parser.add_argument("--handoff-cart-velocity-limit", type=float, default=0.50)
    parser.add_argument(
        "--progress",
        type=float,
        default=1.0,
        help="Fixed plant morphology progress for this diagnostic (1.0 is canonical uniform).",
    )
    parser.add_argument("--init-controller-json", default=None)
    parser.add_argument(
        "--residual-around-controller",
        action="store_true",
        help=(
            "preserve the initial controller at policy resolution and search "
            "bounded low-dimensional additive action corrections"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260965)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if (
        args.knot_count < 2
        or args.population < 2
        or not (1 <= args.elites <= args.population)
    ):
        raise ValueError("invalid CEM dimensions")
    if min(
        args.best_score_weight,
        args.terminal_score_weight,
        args.tail_average_weight,
    ) < 0.0 or np.isclose(
        args.best_score_weight + args.terminal_score_weight + args.tail_average_weight,
        0.0,
    ):
        raise ValueError(
            "trajectory score weights must be nonnegative and not all zero"
        )
    if (
        args.constraint_barrier_weight < 0.0
        or min(
            args.handoff_angle_limit,
            args.handoff_hinge_limit,
            args.handoff_absolute_velocity_limit,
            args.handoff_cart_limit,
            args.handoff_cart_velocity_limit,
        )
        <= 0.0
    ):
        raise ValueError("constraint barrier and handoff limits are invalid")
    if args.residual_around_controller and args.init_controller_json is None:
        raise ValueError("--residual-around-controller requires --init-controller-json")
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("--progress must be in [0, 1]")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"], "init_mode": "hanging"}
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        cfg["env"][key] = 0.0
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=0)
    step_count = max(2, round(args.seconds / env.dt))
    interpolation = interpolation_matrix(args.knot_count, step_count)
    if args.residual_around_controller:
        base_actions = load_initial_center(
            args.init_controller_json, step_count, args.seconds, env
        )
        center = np.zeros(args.knot_count, dtype=np.float64)
    else:
        base_actions = np.zeros(step_count, dtype=np.float64)
        center = load_initial_center(
            args.init_controller_json, args.knot_count, args.seconds, env
        )
    rng = np.random.default_rng(args.seed)
    sigma = np.full(args.knot_count, args.action_sigma, dtype=np.float64)
    best_record: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()

    for iteration in range(args.iterations):
        knots = np.clip(
            center[None, :]
            + rng.normal(0.0, sigma, size=(args.population, args.knot_count)),
            -1.0,
            1.0,
        )
        knots[0] = center
        records: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(knots):
            actions = np.clip(
                base_actions + candidate @ interpolation.T,
                -1.0,
                1.0,
            )
            cost, metrics = evaluate_candidate(
                env,
                actions,
                handoff_min_steps=max(1, round(args.handoff_min_time / env.dt)),
                rolling_window=max(1, args.rolling_window),
                rail_penalty_limit=args.rail_penalty_limit,
                rail_penalty_weight=args.rail_penalty_weight,
                angle_weight=args.angle_weight,
                hinge_weight=args.hinge_weight,
                max_hinge_weight=args.max_hinge_weight,
                absolute_velocity_weight=args.absolute_velocity_weight,
                cart_weight=args.cart_weight,
                cart_velocity_weight=args.cart_velocity_weight,
                relative_angle_weight=args.relative_angle_weight,
                relative_angle_scale=args.relative_angle_scale,
                best_score_weight=args.best_score_weight,
                terminal_score_weight=args.terminal_score_weight,
                tail_average_weight=args.tail_average_weight,
                constraint_barrier_weight=args.constraint_barrier_weight,
                handoff_angle_limit=args.handoff_angle_limit,
                handoff_hinge_limit=args.handoff_hinge_limit,
                handoff_absolute_velocity_limit=args.handoff_absolute_velocity_limit,
                handoff_cart_limit=args.handoff_cart_limit,
                handoff_cart_velocity_limit=args.handoff_cart_velocity_limit,
            )
            records.append({"index": candidate_index, "cost": cost, "metrics": metrics})
        records.sort(key=lambda record: float(record["cost"]))
        valid_records = [
            record for record in records if np.isfinite(float(record["cost"]))
        ]
        if not valid_records:
            sigma = np.maximum(sigma * 0.5, args.sigma_floor)
            metrics = records[0]["metrics"]
            history.append(
                {
                    "iteration": iteration + 1,
                    "cost": float("inf"),
                    "invalid": True,
                    "best_time_seconds": metrics.get("best_time_seconds"),
                    "max_angle": metrics.get("max_angle"),
                    "relative_angle_rms": metrics.get("relative_angle_rms"),
                    "hinge_rms": metrics.get("hinge_rms"),
                    "absolute_velocity_rms": metrics.get("absolute_velocity_rms"),
                    "capture_constraint_violation": metrics.get(
                        "capture_constraint_violation"
                    ),
                    "cart": metrics.get("cart"),
                    "rail": metrics.get("rail"),
                    "valid_population": 0,
                }
            )
            print(
                f"iter={iteration + 1:03d} cost=inf invalid=True "
                f"valid_population=0 sigma={float(np.max(sigma)):.4f}",
                flush=True,
            )
            continue
        top = valid_records[0]
        elite_knots = knots[
            [int(record["index"]) for record in valid_records[: args.elites]]
        ]
        center = np.mean(elite_knots, axis=0)
        sigma = np.maximum(
            np.std(elite_knots, axis=0) * args.sigma_decay, args.sigma_floor
        )
        metrics = top["metrics"]
        history.append(
            {
                "iteration": iteration + 1,
                "cost": float(top["cost"]),
                "invalid": bool(metrics.get("invalid", True)),
                "best_time_seconds": metrics.get("best_time_seconds"),
                "max_angle": metrics.get("max_angle"),
                "relative_angle_rms": metrics.get("relative_angle_rms"),
                "hinge_rms": metrics.get("hinge_rms"),
                "absolute_velocity_rms": metrics.get("absolute_velocity_rms"),
                "capture_constraint_violation": metrics.get(
                    "capture_constraint_violation"
                ),
                "cart": metrics.get("cart"),
                "rail": metrics.get("rail"),
                "valid_population": len(valid_records),
            }
        )
        if best_record is None or float(top["cost"]) < float(best_record["cost"]):
            best_actions = np.clip(
                base_actions + knots[int(top["index"])] @ interpolation.T,
                -1.0,
                1.0,
            )
            best_record = {
                "iteration": iteration + 1,
                "cost": float(top["cost"]),
                "knots": knots[int(top["index"])].astype(float).tolist(),
                "controls": best_actions.astype(np.float32).astype(float).tolist(),
                **metrics,
            }
            if args.residual_around_controller:
                best_record["residual_knots"] = list(best_record["knots"])
        print(
            f"iter={iteration + 1:03d} cost={float(top['cost']):.3f} "
            f"invalid={metrics.get('invalid', True)} "
            f"valid_population={len(valid_records)} "
            f"angle={metrics.get('max_angle')} hinge={metrics.get('hinge_rms')} "
            f"violation={metrics.get('capture_constraint_violation')} "
            f"x={metrics.get('cart')} rail={metrics.get('rail')}",
            flush=True,
        )

    assert best_record is not None
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Serial exact-MuJoCo CEM proposal; every saved state is generated by the final step loop.",
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
            "angle_weight": float(args.angle_weight),
            "hinge_weight": float(args.hinge_weight),
            "max_hinge_weight": float(args.max_hinge_weight),
            "absolute_velocity_weight": float(args.absolute_velocity_weight),
            "cart_weight": float(args.cart_weight),
            "cart_velocity_weight": float(args.cart_velocity_weight),
            "relative_angle_weight": float(args.relative_angle_weight),
            "relative_angle_scale": float(args.relative_angle_scale),
            "best_score_weight": float(args.best_score_weight),
            "terminal_score_weight": float(args.terminal_score_weight),
            "tail_average_weight": float(args.tail_average_weight),
            "constraint_barrier_weight": float(args.constraint_barrier_weight),
            "handoff_angle_limit": float(args.handoff_angle_limit),
            "handoff_hinge_limit": float(args.handoff_hinge_limit),
            "handoff_absolute_velocity_limit": float(
                args.handoff_absolute_velocity_limit
            ),
            "handoff_cart_limit": float(args.handoff_cart_limit),
            "handoff_cart_velocity_limit": float(args.handoff_cart_velocity_limit),
            "residual_around_controller": bool(args.residual_around_controller),
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
