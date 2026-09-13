#!/usr/bin/env python
"""Deterministically refine an n-link route toward an upright-rest endpoint.

The optimizer preserves a supplied exact action route and solves only for a
bounded, low-dimensional correction.  At each iteration it finite-differences
the ordinary float32 ``env.step`` path, computes the endpoint sensitivity to
the correction knots, and takes a damped minimum-norm Gauss--Newton step with
rail-aware line search.  This is a model-based warm-start refiner, not success
evidence; promotion still requires reset-free feedback replay and noisy gates.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
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


def load_controls(path: Path, record_key: str) -> tuple[np.ndarray, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload.get(record_key)
    if not isinstance(record, dict):
        record = payload.get("controller")
    if not isinstance(record, dict):
        raise TypeError(f"{path} has no {record_key!r} or controller record")
    controls = np.asarray(record.get("controls"), dtype=np.float64)
    if controls.ndim != 1 or controls.size < 2:
        raise ValueError("source controller must contain scalar interval controls")
    seconds = float(
        record.get(
            "horizon_seconds",
            payload.get("search", {}).get(
                "seconds",
                payload.get("search", {}).get(
                    "horizon_seconds",
                    payload.get("controller", {}).get("horizon_seconds", 0.0),
                ),
            ),
        )
    )
    if seconds <= 0.0:
        raise ValueError("source controller has no positive horizon")
    return np.clip(controls, -1.0, 1.0), seconds


def deterministic_config(base: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["env"]["init_mode"] = "hanging"
    cfg["env"]["terminate_abs_angle"] = None
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"]["action_lqr_switch"] = {"enabled": False}
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        cfg["env"][key] = 0.0
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    return cfg


def endpoint_residual(
    env: NLinkCartPoleEnv,
    *,
    cart_position_scale: float,
    angle_scale: float,
    cart_velocity_scale: float,
    absolute_velocity_scale: float,
) -> np.ndarray:
    absolute_angles = serial_absolute_angles(np.asarray(env.data.qpos[1:]))
    absolute_velocity = np.cumsum(np.asarray(env.data.qvel[1:], dtype=np.float64))
    return np.r_[
        float(env.data.qpos[0]) / cart_position_scale,
        absolute_angles / angle_scale,
        float(env.data.qvel[0]) / cart_velocity_scale,
        absolute_velocity / absolute_velocity_scale,
    ].astype(np.float64)


def replay(
    env: NLinkCartPoleEnv,
    actions: np.ndarray,
    *,
    cart_position_scale: float,
    angle_scale: float,
    cart_velocity_scale: float,
    absolute_velocity_scale: float,
    rail_soft_limit: float,
    rail_weight: float,
) -> dict[str, Any]:
    env.reset(seed=0)
    max_cart = abs(float(env.data.qpos[0]))
    trace: list[dict[str, Any]] = []
    termination_reason: str | None = None
    for step, action in enumerate(actions):
        _, _, terminated, truncated, info = env.step([float(action)])
        max_cart = max(max_cart, abs(float(env.data.qpos[0])))
        trace.append(
            {
                "step": step + 1,
                "time_seconds": float((step + 1) * env.dt),
                "action": float(info["applied_action_norm"]),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64)
                .astype(float)
                .tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64)
                .astype(float)
                .tolist(),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "absolute_angular_velocity_rms": float(
                    info["absolute_angular_velocity_rms"]
                ),
            }
        )
        if terminated or truncated:
            termination_reason = str(info.get("termination_reason"))
            break
    complete = len(trace) == len(actions) and termination_reason is None
    residual = endpoint_residual(
        env,
        cart_position_scale=cart_position_scale,
        angle_scale=angle_scale,
        cart_velocity_scale=cart_velocity_scale,
        absolute_velocity_scale=absolute_velocity_scale,
    )
    rail_violation = max(0.0, max_cart / rail_soft_limit - 1.0)
    cost = float(residual @ residual + rail_weight * rail_violation**2)
    if not complete or not np.isfinite(cost):
        cost = 1.0e30
    return {
        "cost": cost,
        "complete": bool(complete),
        "termination_reason": termination_reason,
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "max_cart_excursion": float(max_cart),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--record-key", default="best")
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--knot-count", type=int, default=48)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--fd-epsilon", type=float, default=0.002)
    parser.add_argument("--regularization", type=float, default=0.1)
    parser.add_argument("--trust-radius", type=float, default=0.04)
    parser.add_argument("--minimum-step", type=float, default=1.0 / 128.0)
    parser.add_argument("--cart-position-scale", type=float, default=1.25)
    parser.add_argument("--angle-scale", type=float, default=0.15)
    parser.add_argument("--cart-velocity-scale", type=float, default=0.50)
    parser.add_argument("--absolute-velocity-scale", type=float, default=0.75)
    parser.add_argument("--rail-soft-limit", type=float, default=5.0)
    parser.add_argument("--rail-weight", type=float, default=100.0)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (
        min(
            args.seconds,
            args.knot_count,
            args.iterations,
            args.fd_epsilon,
            args.regularization,
            args.trust_radius,
            args.minimum_step,
            args.cart_position_scale,
            args.angle_scale,
            args.cart_velocity_scale,
            args.absolute_velocity_scale,
            args.rail_soft_limit,
        )
        <= 0.0
    ):
        raise ValueError(
            "horizons, dimensions, scales, and solver values must be positive"
        )
    if args.knot_count < 2:
        raise ValueError("knot count must be at least two")
    if args.rail_weight < 0.0:
        raise ValueError("rail weight must be nonnegative")
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")

    cfg = deterministic_config(apply_overrides(load_config(args.config), args.override))
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=0)
    source_path = Path(args.controller)
    source_controls, source_seconds = load_controls(source_path, args.record_key)
    step_count = max(2, round(args.seconds / env.dt))
    source_dt = source_seconds / source_controls.size
    if not np.isclose(source_dt, env.dt, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            f"source policy period {source_dt} does not match target {env.dt}"
        )
    if source_controls.size < step_count:
        raise ValueError("source controller is shorter than the requested refinement")
    base_actions = source_controls[:step_count].copy()
    interpolation = interpolation_matrix(args.knot_count, step_count)
    correction = np.zeros(args.knot_count, dtype=np.float64)
    regularization = float(args.regularization)
    scales = {
        "cart_position_scale": float(args.cart_position_scale),
        "angle_scale": float(args.angle_scale),
        "cart_velocity_scale": float(args.cart_velocity_scale),
        "absolute_velocity_scale": float(args.absolute_velocity_scale),
        "rail_soft_limit": float(args.rail_soft_limit),
        "rail_weight": float(args.rail_weight),
    }

    def evaluate(knots: np.ndarray) -> dict[str, Any]:
        actions = np.clip(base_actions + knots @ interpolation.T, -1.0, 1.0)
        return replay(env, actions, **scales)

    current = evaluate(correction)
    best = current
    best_correction = correction.copy()
    history: list[dict[str, Any]] = []
    started = time.time()
    for iteration in range(args.iterations):
        residual = np.asarray(current["residual"], dtype=np.float64)
        jacobian = np.empty((residual.size, args.knot_count), dtype=np.float64)
        for column in range(args.knot_count):
            plus = correction.copy()
            minus = correction.copy()
            plus[column] += args.fd_epsilon
            minus[column] -= args.fd_epsilon
            plus_result = evaluate(plus)
            minus_result = evaluate(minus)
            if not plus_result["complete"] or not minus_result["complete"]:
                jacobian[:, column] = 0.0
            else:
                jacobian[:, column] = (
                    np.asarray(plus_result["residual"])
                    - np.asarray(minus_result["residual"])
                ) / (2.0 * args.fd_epsilon)
        system = jacobian.T @ jacobian + regularization * np.eye(args.knot_count)
        gradient = jacobian.T @ residual
        step = -np.linalg.solve(system, gradient)
        step = np.clip(step, -args.trust_radius, args.trust_radius)
        accepted = False
        step_scale = 1.0
        trial = current
        while step_scale >= args.minimum_step:
            candidate = np.clip(correction + step_scale * step, -1.0, 1.0)
            candidate_result = evaluate(candidate)
            if candidate_result["cost"] < current["cost"]:
                correction = candidate
                trial = candidate_result
                accepted = True
                break
            step_scale *= 0.5
        if accepted:
            current = trial
            regularization = max(1.0e-8, regularization * 0.5)
            if current["cost"] < best["cost"]:
                best = current
                best_correction = correction.copy()
        else:
            regularization = min(1.0e12, regularization * 10.0)
        history.append(
            {
                "iteration": iteration + 1,
                "accepted": accepted,
                "cost": float(current["cost"]),
                "residual_norm": float(current["residual_norm"]),
                "max_cart_excursion": float(current["max_cart_excursion"]),
                "regularization": regularization,
                "step_scale": float(step_scale),
                "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
                "jacobian_condition": float(np.linalg.cond(jacobian)),
                "maximum_knot_correction": float(np.max(np.abs(correction))),
            }
        )
        print(
            f"iter={iteration + 1:03d} accepted={accepted} "
            f"cost={current['cost']:.6f} residual={current['residual_norm']:.6f} "
            f"rail={current['max_cart_excursion']:.3f} "
            f"rank={history[-1]['jacobian_rank']} reg={regularization:.3e}",
            flush=True,
        )

    best_actions = np.clip(
        base_actions + best_correction @ interpolation.T, -1.0, 1.0
    ).astype(np.float32)
    endpoint = best["trace"][-1]
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "deterministic_endpoint_refinement_not_solution",
        "not_solution": True,
        "summary": "Damped exact-serial Gauss-Newton endpoint refinement; requires feedback and noisy-gate validation.",
        "source_controller": file_metadata(source_path),
        "source_record_key": args.record_key,
        "search": {
            "algorithm": "damped_minimum_norm_gauss_newton_with_rail_aware_line_search",
            "iterations": int(args.iterations),
            "horizon_seconds": float(step_count * env.dt),
            "knot_count": int(args.knot_count),
            "fd_epsilon": float(args.fd_epsilon),
            "trust_radius": float(args.trust_radius),
            "minimum_step": float(args.minimum_step),
            "scales": scales,
            "initial_cost": float(evaluate(np.zeros_like(correction))["cost"]),
            "final_cost": float(best["cost"]),
            "final_residual_norm": float(best["residual_norm"]),
            "max_cart_excursion": float(best["max_cart_excursion"]),
            "wall_time_seconds": float(time.time() - started),
            "history": history,
        },
        "best": {
            "horizon_seconds": float(step_count * env.dt),
            "controls": best_actions.astype(float).tolist(),
            "residual_knots": best_correction.astype(float).tolist(),
            "maximum_knot_correction": float(np.max(np.abs(best_correction))),
            "endpoint": endpoint,
            "trace": best["trace"],
        },
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(output, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
