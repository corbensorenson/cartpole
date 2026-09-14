"""Deterministic actuator-aware terminal geometry for arbitrary link counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class FeedbackHorizonMetric:
    """Quadratic terminal metric induced by a fixed linear feedback horizon.

    For ``x[k+1] = A x[k] + B u[k]`` and ``u[k] = -K x[k]``, the action map
    contains rows ``K (A-BK)^k``. The terminal map is a supplied
    dimensionless state transform followed by the same closed-loop dynamics.
    Their squared norms form a positive-semidefinite terminal cost that an
    exact nonlinear optimizer can consume without any link-count branch.
    """

    closed_loop: Array
    action_map: Array
    terminal_map: Array
    matrix: Array
    action_limit: float
    action_weight: float
    terminal_weight: float

    def residual_map(self) -> Array:
        """Return the rectangular factor an optimizer should differentiate."""

        return np.concatenate(
            (
                np.sqrt(self.action_weight) * self.action_map / self.action_limit,
                np.sqrt(self.terminal_weight) * self.terminal_map,
            ),
            axis=0,
        )

    def residual(self, state: Array) -> Array:
        vector = np.asarray(state, dtype=np.float64)
        if vector.shape != (self.closed_loop.shape[0],):
            raise ValueError("state shape does not match feedback metric")
        return self.residual_map() @ vector

    def residual_map_in_coordinates(self, coordinate_transform: Array) -> Array:
        """Express the residual in coordinates ``z = C x``.

        Optimizers commonly scale or mix the physical state before solving.
        If ``z = C x``, the identical residual is ``R C^-1 z``.  Returning the
        rectangular factor keeps that transformation out of count-specific
        search scripts and avoids forming normal equations prematurely.
        """

        transform = self._validated_coordinate_transform(coordinate_transform)
        inverse = np.linalg.solve(transform, np.eye(transform.shape[0]))
        return self.residual_map() @ inverse

    def matrix_in_coordinates(self, coordinate_transform: Array) -> Array:
        """Return the identical quadratic metric in coordinates ``z = C x``."""

        residual_map = self.residual_map_in_coordinates(coordinate_transform)
        matrix = residual_map.T @ residual_map
        return 0.5 * (matrix + matrix.T)

    def _validated_coordinate_transform(self, coordinate_transform: Array) -> Array:
        transform = np.asarray(coordinate_transform, dtype=np.float64)
        state_size = self.closed_loop.shape[0]
        if transform.shape != (state_size, state_size):
            raise ValueError("coordinate transform shape does not match feedback metric")
        if not np.all(np.isfinite(transform)):
            raise ValueError("coordinate transform must be finite")
        if np.linalg.matrix_rank(transform) != state_size:
            raise ValueError("coordinate transform must be invertible")
        return transform

    def evaluate(self, state: Array) -> dict[str, Any]:
        vector = np.asarray(state, dtype=np.float64)
        if vector.shape != (self.closed_loop.shape[0],):
            raise ValueError("state shape does not match feedback metric")
        raw_actions = -(self.action_map @ vector)
        terminal_state = self.terminal_map @ vector
        action_value = float(
            self.action_weight
            * np.sum((raw_actions / self.action_limit) ** 2)
        )
        terminal_value = float(
            self.terminal_weight * np.sum(terminal_state**2)
        )
        objective = float(vector @ self.matrix @ vector)
        maximum_action = float(np.max(np.abs(raw_actions), initial=0.0))
        return {
            "objective": objective,
            "action_value": action_value,
            "terminal_value": terminal_value,
            "raw_actions": raw_actions.astype(float).tolist(),
            "maximum_raw_action": maximum_action,
            "action_margin": (
                None
                if maximum_action == 0.0
                else float(self.action_limit / maximum_action)
            ),
            "predicted_saturation_steps": int(
                np.count_nonzero(np.abs(raw_actions) > self.action_limit)
            ),
            "terminal_dimensionless_state": terminal_state.astype(float).tolist(),
            "terminal_dimensionless_norm": float(np.linalg.norm(terminal_state)),
        }

    def to_dict(self) -> dict[str, Any]:
        eigenvalues = np.linalg.eigvalsh(self.matrix)
        positive = eigenvalues[eigenvalues > 0.0]
        residual_map = self.residual_map()
        singular_values = np.linalg.svd(residual_map, compute_uv=False)
        return {
            "type": "finite_horizon_linear_feedback_action_terminal_metric",
            "scope": (
                "optimizer terminal geometry from the exact upright "
                "linearization; exact nonlinear validation remains required"
            ),
            "horizon_steps": int(self.action_map.shape[0]),
            "action_limit": float(self.action_limit),
            "action_weight": float(self.action_weight),
            "terminal_weight": float(self.terminal_weight),
            "closed_loop_spectral_radius": float(
                np.max(np.abs(np.linalg.eigvals(self.closed_loop)))
            ),
            "residual_map": residual_map.astype(float).tolist(),
            "residual_map_rank": int(np.linalg.matrix_rank(residual_map)),
            "residual_map_condition": (
                float(singular_values[0] / singular_values[-1])
                if singular_values[-1] > 0.0
                else float("inf")
            ),
            "matrix": self.matrix.astype(float).tolist(),
            "matrix_rank": int(np.linalg.matrix_rank(self.matrix)),
            "matrix_minimum_eigenvalue": float(eigenvalues[0]),
            "matrix_maximum_eigenvalue": float(eigenvalues[-1]),
            "matrix_positive_condition": (
                float(positive[-1] / positive[0]) if positive.size else None
            ),
        }


def feedback_horizon_metric(
    state_matrix: Array,
    input_matrix: Array,
    feedback_gain: Array,
    state_transform: Array,
    *,
    horizon_steps: int,
    feedback_scale: float = 1.0,
    action_limit: float = 1.0,
    action_weight: float = 1.0,
    terminal_weight: float = 1.0,
) -> FeedbackHorizonMetric:
    """Build an exact algebraic terminal cost for a fixed feedback design."""

    a = np.asarray(state_matrix, dtype=np.float64)
    b = np.asarray(input_matrix, dtype=np.float64)
    gain = np.asarray(feedback_gain, dtype=np.float64).reshape(1, -1)
    transform = np.asarray(state_transform, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("state matrix must be square")
    state_size = a.shape[0]
    if b.shape != (state_size, 1):
        raise ValueError("input matrix must describe one actuator")
    if gain.shape != (1, state_size):
        raise ValueError("feedback gain shape does not match state matrix")
    if transform.shape != (state_size, state_size):
        raise ValueError("state transform shape does not match state matrix")
    if int(horizon_steps) != horizon_steps or horizon_steps < 1:
        raise ValueError("horizon_steps must be a positive integer")
    scalars = (feedback_scale, action_limit, action_weight, terminal_weight)
    if not all(np.isfinite(value) and value > 0.0 for value in scalars):
        raise ValueError("feedback and metric scales must be finite and positive")
    if not all(np.all(np.isfinite(array)) for array in (a, b, gain, transform)):
        raise ValueError("feedback metric arrays must be finite")

    scaled_gain = float(feedback_scale) * gain
    closed_loop = a - b @ scaled_gain
    transition = np.eye(state_size, dtype=np.float64)
    action_rows = []
    for _ in range(int(horizon_steps)):
        action_rows.append(scaled_gain @ transition)
        transition = closed_loop @ transition
    action_map = np.concatenate(action_rows, axis=0)
    terminal_map = transform @ transition
    matrix = (
        float(action_weight)
        * (action_map.T @ action_map)
        / float(action_limit) ** 2
        + float(terminal_weight) * (terminal_map.T @ terminal_map)
    )
    matrix = 0.5 * (matrix + matrix.T)
    return FeedbackHorizonMetric(
        closed_loop=closed_loop,
        action_map=action_map,
        terminal_map=terminal_map,
        matrix=matrix,
        action_limit=float(action_limit),
        action_weight=float(action_weight),
        terminal_weight=float(terminal_weight),
    )
