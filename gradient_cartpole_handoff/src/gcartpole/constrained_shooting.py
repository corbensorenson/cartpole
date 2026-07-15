from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, minimize

from gcartpole.ilqr import Array, MujocoTransition


@dataclass(frozen=True)
class ConstrainedShootingResult:
    controls: Array
    states: Array
    cost: float
    minimum_constraint_margin: float
    maximum_constraint_jacobian_abs: float
    maximum_objective_gradient_abs: float
    iterations: int
    evaluations: int
    status: int
    message: str
    success: bool


class _RolloutEvaluation:
    def __init__(
        self,
        transition: MujocoTransition,
        initial_state: Array,
        terminal_metric: Array,
        control_weight: float,
        lyapunov: Array,
        rail_limit: float,
        handoff_lyapunov: float,
        handoff_cart_ratio: float,
        handoff_angle_ratio: float,
        handoff_cart_velocity_ratio: float,
        handoff_hinge_velocity_ratio: float,
    ) -> None:
        self.transition = transition
        self.initial_state = np.asarray(initial_state, dtype=np.float64)
        self.terminal_metric = np.asarray(terminal_metric, dtype=np.float64)
        self.control_weight = float(control_weight)
        self.lyapunov = np.asarray(lyapunov, dtype=np.float64)
        self.rail_limit = float(rail_limit)
        self.handoff_lyapunov = float(handoff_lyapunov)
        self.handoff_cart_ratio = float(handoff_cart_ratio)
        self.handoff_angle_ratio = float(handoff_angle_ratio)
        self.handoff_cart_velocity_ratio = float(handoff_cart_velocity_ratio)
        self.handoff_hinge_velocity_ratio = float(handoff_hinge_velocity_ratio)
        self._controls: Array | None = None
        self._states: Array | None = None
        self._sensitivities: Array | None = None

    def rollout(self, controls: Array) -> tuple[Array, Array]:
        controls = np.asarray(controls, dtype=np.float64)
        if self._controls is not None and np.array_equal(controls, self._controls):
            assert self._states is not None
            assert self._sensitivities is not None
            return self._states, self._sensitivities
        horizon = controls.size
        nx = self.initial_state.size
        states = np.empty((horizon + 1, nx), dtype=np.float64)
        sensitivities = np.zeros((horizon + 1, nx, horizon), dtype=np.float64)
        states[0] = self.initial_state
        for step, action in enumerate(controls):
            state_matrix, input_matrix = self.transition.linearize(
                states[step],
                float(action),
                state_epsilon=1e-5,
                action_epsilon=1e-4,
            )
            sensitivities[step + 1] = state_matrix @ sensitivities[step]
            sensitivities[step + 1, :, step] = input_matrix[:, 0]
            states[step + 1] = self.transition(states[step], float(action))
        self._controls = controls.copy()
        self._states = states
        self._sensitivities = sensitivities
        return states, sensitivities

    def objective(self, controls: Array) -> float:
        states, _ = self.rollout(controls)
        terminal = states[-1]
        return float(
            0.5 * terminal @ self.terminal_metric @ terminal
            + 0.5 * self.control_weight * np.asarray(controls) @ controls
        )

    def objective_jacobian(self, controls: Array) -> Array:
        states, sensitivities = self.rollout(controls)
        return sensitivities[-1].T @ self.terminal_metric @ states[
            -1
        ] + self.control_weight * np.asarray(controls, dtype=np.float64)

    def constraints(self, controls: Array) -> Array:
        states, _ = self.rollout(controls)
        terminal = states[-1]
        n_links = (terminal.size // 2) - 1
        velocity_start = n_links + 1
        hinge_velocity = terminal[velocity_start + 1 :]
        maximum_cart_excursion = float(np.max(np.abs(states[:, 0])))
        return np.r_[
            1.0 - maximum_cart_excursion / self.rail_limit,
            1.0 - terminal[0] / self.handoff_cart_ratio,
            1.0 + terminal[0] / self.handoff_cart_ratio,
            1.0 - terminal[1 : n_links + 1] / self.handoff_angle_ratio,
            1.0 + terminal[1 : n_links + 1] / self.handoff_angle_ratio,
            1.0 - terminal[velocity_start] / self.handoff_cart_velocity_ratio,
            1.0 + terminal[velocity_start] / self.handoff_cart_velocity_ratio,
            1.0
            - float(np.mean(hinge_velocity**2)) / self.handoff_hinge_velocity_ratio**2,
            1.0 - terminal @ self.lyapunov @ terminal / self.handoff_lyapunov,
        ]

    def constraint_jacobian(self, controls: Array) -> Array:
        states, sensitivities = self.rollout(controls)
        terminal = states[-1]
        terminal_sensitivity = sensitivities[-1]
        n_links = (terminal.size // 2) - 1
        velocity_start = n_links + 1
        hinge_velocity = terminal[velocity_start + 1 :]
        hinge_sensitivity = terminal_sensitivity[velocity_start + 1 :]
        rail_step = int(np.argmax(np.abs(states[:, 0])))
        rail_sign = float(np.sign(states[rail_step, 0]))
        return np.vstack(
            (
                -rail_sign * sensitivities[rail_step, 0, :] / self.rail_limit,
                -terminal_sensitivity[0] / self.handoff_cart_ratio,
                terminal_sensitivity[0] / self.handoff_cart_ratio,
                -terminal_sensitivity[1 : n_links + 1] / self.handoff_angle_ratio,
                terminal_sensitivity[1 : n_links + 1] / self.handoff_angle_ratio,
                -terminal_sensitivity[velocity_start]
                / self.handoff_cart_velocity_ratio,
                terminal_sensitivity[velocity_start] / self.handoff_cart_velocity_ratio,
                -2.0
                * hinge_velocity
                @ hinge_sensitivity
                / (n_links * self.handoff_hinge_velocity_ratio**2),
                -2.0
                * terminal
                @ self.lyapunov
                @ terminal_sensitivity
                / self.handoff_lyapunov,
            )
        )


def optimize_constrained_shooting(
    transition: MujocoTransition,
    initial_state: Array,
    initial_controls: Array,
    terminal_metric: Array,
    lyapunov: Array,
    *,
    control_weight: float,
    rail_limit: float,
    handoff_lyapunov: float,
    handoff_cart_ratio: float,
    handoff_angle_ratio: float,
    handoff_cart_velocity_ratio: float,
    handoff_hinge_velocity_ratio: float,
    max_iterations: int,
) -> ConstrainedShootingResult:
    controls = np.clip(np.asarray(initial_controls, dtype=np.float64), -1.0, 1.0)
    if controls.ndim != 1 or controls.size < 1:
        raise ValueError("initial controls must be a nonempty vector")
    if (
        min(
            control_weight,
            rail_limit,
            handoff_lyapunov,
            handoff_cart_ratio,
            handoff_angle_ratio,
            handoff_cart_velocity_ratio,
            handoff_hinge_velocity_ratio,
            max_iterations,
        )
        <= 0.0
    ):
        raise ValueError("weights, bounds, and iteration count must be positive")
    evaluation = _RolloutEvaluation(
        transition,
        initial_state,
        terminal_metric,
        control_weight,
        lyapunov,
        rail_limit,
        handoff_lyapunov,
        handoff_cart_ratio,
        handoff_angle_ratio,
        handoff_cart_velocity_ratio,
        handoff_hinge_velocity_ratio,
    )
    best_controls = controls.copy()
    best_margin = float(np.min(evaluation.constraints(best_controls)))
    best_key = (max(0.0, -best_margin), evaluation.objective(best_controls))
    initial_constraint_jacobian_abs = float(
        np.max(np.abs(evaluation.constraint_jacobian(best_controls)))
    )
    initial_objective_gradient_abs = float(
        np.max(np.abs(evaluation.objective_jacobian(best_controls)))
    )

    def retain(values: Array) -> None:
        nonlocal best_controls, best_margin, best_key
        margin = float(np.min(evaluation.constraints(values)))
        key = (max(0.0, -margin), evaluation.objective(values))
        if np.isfinite(key[0]) and np.isfinite(key[1]) and key < best_key:
            best_controls = np.asarray(values, dtype=np.float64).copy()
            best_margin = margin
            best_key = key

    result = minimize(
        evaluation.objective,
        controls,
        method="SLSQP",
        jac=evaluation.objective_jacobian,
        bounds=Bounds(-np.ones_like(controls), np.ones_like(controls)),
        constraints=[
            {
                "type": "ineq",
                "fun": evaluation.constraints,
                "jac": evaluation.constraint_jacobian,
            }
        ],
        callback=retain,
        options={"maxiter": int(max_iterations), "ftol": 1e-9, "disp": False},
    )
    retain(result.x)
    states, _ = evaluation.rollout(best_controls)
    return ConstrainedShootingResult(
        controls=best_controls,
        states=states,
        cost=float(evaluation.objective(best_controls)),
        minimum_constraint_margin=float(best_margin),
        maximum_constraint_jacobian_abs=initial_constraint_jacobian_abs,
        maximum_objective_gradient_abs=initial_objective_gradient_abs,
        iterations=int(result.nit),
        evaluations=int(result.nfev),
        status=int(result.status),
        message=str(result.message),
        success=bool(result.success and best_margin >= -1e-6),
    )
