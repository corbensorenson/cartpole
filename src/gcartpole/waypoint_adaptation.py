from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

Array = np.ndarray


@dataclass(frozen=True)
class WaypointAdaptationResult:
    controls: Array
    states: Array
    segments: list[dict[str, Any]]
    success: bool


def rollout_segment(transition: Any, initial_state: Array, controls: Array) -> Array:
    controls = np.asarray(controls, dtype=np.float64)
    states = np.empty((controls.size + 1, len(initial_state)), dtype=np.float64)
    states[0] = initial_state
    for step, action in enumerate(controls):
        states[step + 1] = transition(states[step], float(action))
    return states


def adapt_route_to_waypoints(
    transition: Any,
    initial_state: Array,
    reference_controls: Array,
    reference_states: Array,
    *,
    segment_steps: int,
    max_evaluations: int,
    endpoint_weight: float = 1_000.0,
    control_regularization: float = 1.0e-4,
    rail_soft_limit: float,
    rail_weight: float = 1_000.0,
    endpoint_tolerance: float = 0.05,
) -> WaypointAdaptationResult:
    """Retarget a route through short exact-model waypoint solves.

    Every segment begins from the preceding exact target-plant endpoint. This
    avoids the exponentially amplified continuity defects of one long
    infeasible rollout while keeping the neighboring route as the deterministic
    reference.
    """

    controls = np.asarray(reference_controls, dtype=np.float64)
    references = np.asarray(reference_states, dtype=np.float64)
    initial = np.asarray(initial_state, dtype=np.float64)
    if controls.ndim != 1 or references.shape != (controls.size + 1, initial.size):
        raise ValueError("reference controls and states have inconsistent dimensions")
    if min(
        segment_steps,
        max_evaluations,
        endpoint_weight,
        rail_soft_limit,
        endpoint_tolerance,
    ) <= 0.0:
        raise ValueError("counts, weights, rail limit, and tolerance must be positive")
    if min(control_regularization, rail_weight) < 0.0:
        raise ValueError("regularization and rail weight must be nonnegative")

    adapted_controls: list[float] = []
    adapted_states = [initial.copy()]
    records: list[dict[str, Any]] = []
    sqrt_endpoint = np.sqrt(endpoint_weight)
    sqrt_control = np.sqrt(control_regularization)
    sqrt_rail = np.sqrt(rail_weight)
    current = initial.copy()
    success = True

    for start in range(0, controls.size, segment_steps):
        stop = min(controls.size, start + segment_steps)
        base = np.clip(controls[start:stop], -1.0, 1.0)
        target = references[stop]

        def residual(
            actions: Array,
            current_state: Array,
            target_state: Array,
            base_controls: Array,
        ) -> Array:
            states = rollout_segment(transition, current_state, actions)
            endpoint = transition.difference(states[-1], target_state)
            rail = np.maximum(0.0, np.abs(states[1:, 0]) - rail_soft_limit)
            return np.r_[
                sqrt_endpoint * endpoint,
                sqrt_control * (actions - base_controls),
                sqrt_rail * rail,
            ]

        solve_options = {
            "args": (current.copy(), target.copy(), base.copy()),
            "bounds": (-1.0, 1.0),
            "jac": "2-point",
            "max_nfev": max_evaluations,
            "ftol": 1.0e-10,
            "xtol": 1.0e-10,
            "gtol": 1.0e-10,
        }
        optimizer_fallback: str | None = None
        try:
            optimized = least_squares(
                residual,
                base,
                x_scale="jac",
                **solve_options,
            )
        except np.linalg.LinAlgError:
            # Near a released count-continuation boundary, finite-difference
            # columns can become numerically rank deficient.  LSMR avoids the
            # dense SVD while preserving the same residual and box constraints.
            optimized = least_squares(
                residual,
                base,
                x_scale=1.0,
                tr_solver="lsmr",
                **solve_options,
            )
            optimizer_fallback = "dense_svd_to_lsmr"
        segment_controls = np.asarray(optimized.x, dtype=np.float64)
        segment_states = rollout_segment(transition, current, segment_controls)
        endpoint_error = transition.difference(segment_states[-1], target)
        endpoint_norm = float(np.linalg.norm(endpoint_error))
        maximum_cart = float(np.max(np.abs(segment_states[:, 0])))
        segment_passed = bool(
            endpoint_norm <= endpoint_tolerance
            and maximum_cart <= rail_soft_limit
        )
        success = success and segment_passed
        records.append(
            {
                "start_step": int(start),
                "stop_step": int(stop),
                "success": segment_passed,
                "endpoint_error_norm": endpoint_norm,
                "maximum_cart_coordinate": maximum_cart,
                "maximum_action_correction": float(
                    np.max(np.abs(segment_controls - base), initial=0.0)
                ),
                "evaluations": int(optimized.nfev),
                "optimizer_status": int(optimized.status),
                "optimizer_message": str(optimized.message),
                "optimizer_fallback": optimizer_fallback,
            }
        )
        adapted_controls.extend(segment_controls.astype(float).tolist())
        adapted_states.extend(segment_states[1:])
        current = segment_states[-1].copy()

    return WaypointAdaptationResult(
        controls=np.asarray(adapted_controls, dtype=np.float64),
        states=np.asarray(adapted_states, dtype=np.float64),
        segments=records,
        success=success,
    )
