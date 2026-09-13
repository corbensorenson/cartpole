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
from gcartpole.generalized_energy import upright_lqr_gain
from gcartpole.ilqr import MujocoTransition
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
    dimensionless_wrapped_state,
    linear_saturation_invariant_radius,
    transform_feedback_gain,
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


def damped_minimum_norm_step(
    jacobian: np.ndarray,
    residual: np.ndarray,
    regularization: float,
) -> np.ndarray:
    """Solve damped Gauss--Newton without squaring the Jacobian condition."""

    jacobian = np.asarray(jacobian, dtype=np.float64)
    residual = np.asarray(residual, dtype=np.float64)
    if jacobian.ndim != 2 or residual.shape != (jacobian.shape[0],):
        raise ValueError("Jacobian and residual shapes do not match")
    if not np.isfinite(regularization) or regularization < 0.0:
        raise ValueError("regularization must be finite and nonnegative")
    left, singular_values, right = np.linalg.svd(jacobian, full_matrices=False)
    filter_factors = singular_values / (singular_values**2 + regularization)
    return -(right.T @ (filter_factors * (left.T @ residual)))


def scale_step_to_trust_region(step: np.ndarray, trust_radius: float) -> np.ndarray:
    """Scale into an infinity-norm trust region without rotating the step."""

    step = np.asarray(step, dtype=np.float64)
    if not np.isfinite(trust_radius) or trust_radius <= 0.0:
        raise ValueError("trust_radius must be finite and positive")
    maximum = float(np.max(np.abs(step), initial=0.0))
    if maximum <= trust_radius:
        return step.copy()
    return step * (trust_radius / maximum)


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


def blend_controls(
    primary: np.ndarray, secondary: np.ndarray, *, alpha: float, step_count: int
) -> np.ndarray:
    """Blend compatible exact routes without extrapolating their horizons."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("blend alpha must be in [0, 1]")
    if primary.size < step_count or secondary.size < step_count:
        raise ValueError("controllers are shorter than the requested blend")
    return (
        (1.0 - alpha) * primary[:step_count] + alpha * secondary[:step_count]
    )


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
    state_transform: np.ndarray | None = None,
    residual_transform: np.ndarray | None = None,
    feedback_gain: np.ndarray | None = None,
    feedback_scale: float = 1.0,
    lqr_action_weight: float = 0.0,
) -> np.ndarray:
    if state_transform is not None or residual_transform is not None:
        if state_transform is None or residual_transform is None:
            raise ValueError("both invariant-set transforms must be supplied")
        state = dimensionless_wrapped_state(
            np.asarray(env.data.qpos, dtype=np.float64),
            np.asarray(env.data.qvel, dtype=np.float64),
            state_transform,
        )
        residual = np.asarray(residual_transform @ state, dtype=np.float64)
    else:
        absolute_angles = serial_absolute_angles(np.asarray(env.data.qpos[1:]))
        absolute_velocity = np.cumsum(np.asarray(env.data.qvel[1:], dtype=np.float64))
        residual = np.r_[
            float(env.data.qpos[0]) / cart_position_scale,
            absolute_angles / angle_scale,
            float(env.data.qvel[0]) / cart_velocity_scale,
            absolute_velocity / absolute_velocity_scale,
        ].astype(np.float64)
    if lqr_action_weight > 0.0:
        if feedback_gain is None:
            raise ValueError("lqr action residual requires a feedback gain")
        physical_state = np.r_[env.data.qpos, env.data.qvel].astype(np.float64)
        physical_state[1 : env.n + 1] = (
            physical_state[1 : env.n + 1] + np.pi
        ) % (2.0 * np.pi) - np.pi
        raw_action = float(feedback_scale * feedback_gain @ physical_state)
        residual = np.r_[
            residual,
            np.sqrt(float(lqr_action_weight)) * raw_action,
        ]
    return residual


def lqr_invariant_metric(
    env: NLinkCartPoleEnv,
    *,
    cart_position_scale: float,
    angle_scale: float,
    cart_velocity_scale: float,
    velocity_scale: float,
    control_cost: float,
    feedback_scale: float,
    action_limit: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Construct a saturation-safe terminal ellipsoid for the linear plant."""

    state_transform = dimensionless_absolute_transform(
        env.n,
        StateScales(
            cart_position=cart_position_scale,
            absolute_angle=angle_scale,
            cart_velocity=cart_velocity_scale,
            hinge_velocity=velocity_scale,
        ),
    )
    transition = MujocoTransition(env)
    equilibrium = np.zeros(2 * (env.n + 1), dtype=np.float64)
    state_matrix, input_matrix = transition.linearize(
        equilibrium,
        0.0,
        state_epsilon=1.0e-7,
        action_epsilon=1.0e-7,
    )
    physical_gain = upright_lqr_gain(env, control_cost=control_cost)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix,
        input_matrix,
        physical_gain,
        state_transform,
        feedback_scale,
    )
    dimensionless_gain = feedback_scale * transform_feedback_gain(
        physical_gain,
        state_transform,
    )
    radius = linear_saturation_invariant_radius(
        lyapunov,
        dimensionless_gain,
        action_limit=action_limit,
    )
    # np.linalg.cholesky returns L with P=L L.T, so ||L.T z||^2=z.T P z.
    residual_transform = np.linalg.cholesky(lyapunov).T / np.sqrt(radius)
    metadata = {
        "type": "linear_lqr_saturation_invariant",
        "scope": "certified_for_unsaturated_linearization_requires_exact_nonlinear_validation",
        "control_cost": float(control_cost),
        "feedback_scale": float(feedback_scale),
        "normalized_action_limit": float(action_limit),
        "closed_loop_spectral_radius": float(spectral_radius),
        "lyapunov_radius": float(radius),
        "physical_feedback_gain": physical_gain.astype(float).tolist(),
        "dimensionless_feedback_gain": dimensionless_gain.astype(float).tolist(),
        "dimensionless_state_transform": state_transform.astype(float).tolist(),
        "lyapunov_matrix": lyapunov.astype(float).tolist(),
    }
    return state_transform, residual_transform, physical_gain, metadata


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
    state_transform: np.ndarray | None = None,
    residual_transform: np.ndarray | None = None,
    feedback_gain: np.ndarray | None = None,
    feedback_scale: float = 1.0,
    lqr_action_weight: float = 0.0,
    terminal_feedback_steps: int = 0,
    rail_barrier_weight: float = 0.0,
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
    route_complete = len(trace) == len(actions) and termination_reason is None
    endpoint = endpoint_residual(
        env,
        cart_position_scale=cart_position_scale,
        angle_scale=angle_scale,
        cart_velocity_scale=cart_velocity_scale,
        absolute_velocity_scale=absolute_velocity_scale,
        state_transform=state_transform,
        residual_transform=residual_transform,
        feedback_gain=feedback_gain,
        feedback_scale=feedback_scale,
        lqr_action_weight=lqr_action_weight,
    )
    residual_blocks = [endpoint]
    feedback_trace: list[dict[str, Any]] = []
    maximum_raw_feedback_action = 0.0
    if terminal_feedback_steps > 0 and route_complete:
        if feedback_gain is None:
            raise ValueError("terminal feedback rollout requires an LQR gain")
        for step in range(terminal_feedback_steps):
            physical_state = np.r_[env.data.qpos, env.data.qvel].astype(np.float64)
            physical_state[1 : env.n + 1] = (
                physical_state[1 : env.n + 1] + np.pi
            ) % (2.0 * np.pi) - np.pi
            raw_action = float(-feedback_scale * feedback_gain @ physical_state)
            maximum_raw_feedback_action = max(
                maximum_raw_feedback_action,
                abs(raw_action),
            )
            _, _, terminated, truncated, info = env.step([np.clip(raw_action, -1.0, 1.0)])
            feedback_residual = endpoint_residual(
                env,
                cart_position_scale=cart_position_scale,
                angle_scale=angle_scale,
                cart_velocity_scale=cart_velocity_scale,
                absolute_velocity_scale=absolute_velocity_scale,
                state_transform=state_transform,
                residual_transform=residual_transform,
                feedback_gain=feedback_gain,
                feedback_scale=feedback_scale,
                lqr_action_weight=lqr_action_weight,
            )
            residual_blocks.append(feedback_residual)
            feedback_trace.append(
                {
                    "step": step + 1,
                    "action": float(np.clip(raw_action, -1.0, 1.0)),
                    "raw_action": raw_action,
                    "invariant_value": float(
                        feedback_residual @ feedback_residual
                    ),
                    "qpos": np.asarray(env.data.qpos, dtype=np.float64)
                    .astype(float)
                    .tolist(),
                    "qvel": np.asarray(env.data.qvel, dtype=np.float64)
                    .astype(float)
                    .tolist(),
                }
            )
            max_cart = max(max_cart, abs(float(env.data.qpos[0])))
            if terminated or truncated:
                termination_reason = str(info.get("termination_reason"))
                break
    feedback_complete = len(feedback_trace) == terminal_feedback_steps
    complete = route_complete and feedback_complete and termination_reason is None
    if rail_barrier_weight > 0.0:
        rail_positions = np.asarray(
            [row["qpos"][0] for row in trace]
            + [row["qpos"][0] for row in feedback_trace],
            dtype=np.float64,
        )
        rail_excess = np.maximum(
            0.0, np.abs(rail_positions) - float(rail_soft_limit)
        )
        residual_blocks.append(np.sqrt(float(rail_barrier_weight)) * rail_excess)
    # Average the exact feedback-rollout values so the objective scale does not
    # depend on the requested validation horizon.
    residual = np.concatenate(residual_blocks) / np.sqrt(len(residual_blocks))
    rail_violation = max(0.0, max_cart / rail_soft_limit - 1.0)
    cost = float(residual @ residual + rail_weight * rail_violation**2)
    if not complete or not np.isfinite(cost):
        cost = 1.0e30
    return {
        "cost": cost,
        "complete": bool(complete),
        "route_complete": bool(route_complete),
        "feedback_complete": bool(feedback_complete),
        "termination_reason": termination_reason,
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "terminal_metric_value": float(residual @ residual),
        "endpoint_metric_value": float(endpoint @ endpoint),
        "maximum_raw_feedback_action": float(maximum_raw_feedback_action),
        "terminal_feedback_trace": feedback_trace,
        "max_cart_excursion": float(max_cart),
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--record-key", default="best")
    parser.add_argument(
        "--secondary-controller",
        help="optional second exact route to blend into the primary warm start",
    )
    parser.add_argument("--secondary-record-key", default="best")
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.0,
        help="secondary-route fraction in [0, 1]",
    )
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
    parser.add_argument(
        "--terminal-metric",
        choices=("componentwise", "lqr-invariant"),
        default="componentwise",
        help="terminal residual geometry used by Gauss-Newton",
    )
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--lqr-feedback-scale", type=float, default=1.0)
    parser.add_argument(
        "--lqr-action-weight",
        type=float,
        default=0.0,
        help="weight an unsaturated terminal LQR action residual in the endpoint objective",
    )
    parser.add_argument(
        "--invariant-action-limit",
        type=float,
        default=1.0,
        help="normalized action bound used for the linear invariant ellipsoid",
    )
    parser.add_argument(
        "--terminal-feedback-steps",
        type=int,
        default=0,
        help="exact clipped-LQR steps included in the terminal objective",
    )
    parser.add_argument(
        "--rail-barrier-weight",
        type=float,
        default=0.0,
        help="per-step rail-excess residual weight in the endpoint objective",
    )
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
            args.lqr_control_cost,
            args.lqr_feedback_scale,
            args.invariant_action_limit,
            args.rail_soft_limit,
        )
        <= 0.0
    ):
        raise ValueError(
            "horizons, dimensions, scales, and solver values must be positive"
        )
    if args.knot_count < 2:
        raise ValueError("knot count must be at least two")
    if args.terminal_feedback_steps < 0:
        raise ValueError("terminal feedback steps must be nonnegative")
    if args.lqr_action_weight < 0.0:
        raise ValueError("lqr action weight must be nonnegative")
    if min(args.rail_weight, args.rail_barrier_weight) < 0.0:
        raise ValueError("rail weights must be nonnegative")
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    if not 0.0 <= args.blend_alpha <= 1.0:
        raise ValueError("blend alpha must be in [0, 1]")
    if args.blend_alpha > 0.0 and not args.secondary_controller:
        raise ValueError("positive blend alpha requires --secondary-controller")
    if args.terminal_feedback_steps and args.lqr_action_weight <= 0.0:
        raise ValueError(
            "terminal feedback rollout requires --lqr-action-weight so the gain is available"
        )

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
    secondary_path: Path | None = None
    if args.secondary_controller:
        secondary_path = Path(args.secondary_controller)
        secondary_controls, secondary_seconds = load_controls(
            secondary_path, args.secondary_record_key
        )
        secondary_dt = secondary_seconds / secondary_controls.size
        if not np.isclose(secondary_dt, env.dt, rtol=0.0, atol=1.0e-10):
            raise ValueError(
                f"secondary policy period {secondary_dt} does not match target {env.dt}"
            )
        if secondary_controls.size < step_count:
            raise ValueError(
                "secondary controller is shorter than the requested refinement"
            )
        base_actions = blend_controls(
            base_actions,
            secondary_controls,
            alpha=args.blend_alpha,
            step_count=step_count,
        )
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
        "rail_barrier_weight": float(args.rail_barrier_weight),
    }
    state_transform: np.ndarray | None = None
    residual_transform: np.ndarray | None = None
    feedback_gain: np.ndarray | None = None
    terminal_metric: dict[str, Any] = {
        "type": "componentwise_scaled_endpoint",
        "scope": "optimization_metric_not_an_invariant_set_certificate",
    }
    if args.terminal_metric == "lqr-invariant":
        (
            state_transform,
            residual_transform,
            feedback_gain,
            terminal_metric,
        ) = lqr_invariant_metric(
            env,
            cart_position_scale=args.cart_position_scale,
            angle_scale=args.angle_scale,
            cart_velocity_scale=args.cart_velocity_scale,
            velocity_scale=args.absolute_velocity_scale,
            control_cost=args.lqr_control_cost,
            feedback_scale=args.lqr_feedback_scale,
            action_limit=args.invariant_action_limit,
        )
    elif args.lqr_action_weight > 0.0:
        feedback_gain = upright_lqr_gain(env, control_cost=args.lqr_control_cost)
        terminal_metric["lqr_action_residual_weight"] = float(args.lqr_action_weight)

    def evaluate(knots: np.ndarray) -> dict[str, Any]:
        actions = np.clip(base_actions + knots @ interpolation.T, -1.0, 1.0)
        return replay(
            env,
            actions,
            **scales,
            state_transform=state_transform,
            residual_transform=residual_transform,
            feedback_gain=feedback_gain,
            feedback_scale=args.lqr_feedback_scale,
            lqr_action_weight=args.lqr_action_weight,
            terminal_feedback_steps=args.terminal_feedback_steps,
        )

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
        step = damped_minimum_norm_step(jacobian, residual, regularization)
        step = scale_step_to_trust_region(step, args.trust_radius)
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
    if feedback_gain is not None:
        endpoint_state = np.r_[endpoint["qpos"], endpoint["qvel"]].astype(np.float64)
        endpoint_state[1 : env.n + 1] = (
            endpoint_state[1 : env.n + 1] + np.pi
        ) % (2.0 * np.pi) - np.pi
        terminal_metric["endpoint_raw_normalized_action"] = float(
            -args.lqr_feedback_scale * feedback_gain @ endpoint_state
        )
        terminal_metric["endpoint_normalized_invariant_value"] = float(
            best["endpoint_metric_value"]
        )
        terminal_metric["feedback_rollout_steps"] = int(
            args.terminal_feedback_steps
        )
        terminal_metric["feedback_rollout_mean_invariant_value"] = float(
            best["terminal_metric_value"]
        )
        terminal_metric["feedback_rollout_maximum_raw_action"] = float(
            best["maximum_raw_feedback_action"]
        )
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "deterministic_endpoint_refinement_not_solution",
        "not_solution": True,
        "summary": "Damped exact-serial Gauss-Newton endpoint refinement; requires feedback and noisy-gate validation.",
        "source_controller": file_metadata(source_path),
        "source_record_key": args.record_key,
        "secondary_controller": (
            file_metadata(secondary_path) if secondary_path is not None else None
        ),
        "secondary_record_key": (
            args.secondary_record_key if secondary_path is not None else None
        ),
        "search": {
            "algorithm": "damped_minimum_norm_gauss_newton_with_rail_aware_line_search",
            "blend_alpha": float(args.blend_alpha),
            "iterations": int(args.iterations),
            "horizon_seconds": float(step_count * env.dt),
            "knot_count": int(args.knot_count),
            "fd_epsilon": float(args.fd_epsilon),
            "trust_radius": float(args.trust_radius),
            "minimum_step": float(args.minimum_step),
            "scales": scales,
            "terminal_metric": terminal_metric,
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
            "terminal_feedback_trace": best["terminal_feedback_trace"],
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
