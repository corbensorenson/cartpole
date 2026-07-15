from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import mujoco
import numpy as np

from gcartpole.env import NLinkCartPoleEnv, wrap_angle


Array = np.ndarray


@dataclass(frozen=True)
class QuadraticTrajectoryCost:
    stage_state: Array
    terminal_state: Array
    control: float
    rail_soft_limit: float
    rail_limit: float
    rail_weight: float
    wrap_angles: bool = True


@dataclass(frozen=True)
class ILQRResult:
    controls: Array
    states: Array
    feedback_gains: Array
    cost: float
    iterations: int
    converged: bool
    history: list[dict[str, float | int | bool]]


def state_difference(first: Array, second: Array, n_links: int) -> Array:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    delta = delta.copy()
    delta[1 : n_links + 1] = wrap_angle(delta[1 : n_links + 1])
    return delta


def wrapped_state(state: Array, n_links: int) -> Array:
    result = np.asarray(state, dtype=np.float64).copy()
    result[1 : n_links + 1] = wrap_angle(result[1 : n_links + 1])
    return result


def data_state(data: mujoco.MjData) -> Array:
    return np.r_[
        np.asarray(data.qpos, dtype=np.float64),
        np.asarray(data.qvel, dtype=np.float64),
    ]


class MujocoTransition:
    def __init__(self, env: NLinkCartPoleEnv, coordinate_transform: Array | None = None) -> None:
        self.env = env
        self.data = mujoco.MjData(env.model)
        self.nx = int(env.model.nq + env.model.nv)
        self.coordinate_transform = (
            None
            if coordinate_transform is None
            else np.asarray(coordinate_transform, dtype=np.float64)
        )
        if self.coordinate_transform is not None and self.coordinate_transform.shape != (
            self.nx,
            self.nx,
        ):
            raise ValueError("coordinate transform does not match MuJoCo state dimension")
        self.inverse_transform = (
            None
            if self.coordinate_transform is None
            else np.linalg.inv(self.coordinate_transform)
        )

    def to_coordinates(self, physical_state: Array) -> Array:
        physical = wrapped_state(physical_state, self.env.n)
        if self.coordinate_transform is None:
            return physical
        return self.coordinate_transform @ physical

    def to_physical(self, coordinate_state: Array) -> Array:
        coordinates = np.asarray(coordinate_state, dtype=np.float64)
        if self.inverse_transform is None:
            return coordinates.copy()
        return self.inverse_transform @ coordinates

    def difference(self, first: Array, second: Array) -> Array:
        if self.coordinate_transform is None:
            return state_difference(first, second, self.env.n)
        return np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)

    def __call__(self, state: Array, action: float) -> Array:
        state = self.to_physical(state)
        nq = int(self.env.model.nq)
        mujoco.mj_resetData(self.env.model, self.data)
        self.data.qpos[:] = state[:nq]
        self.data.qvel[:] = state[nq:]
        self.data.ctrl[0] = float(np.clip(action, -1.0, 1.0)) * self.env.force_limit
        mujoco.mj_forward(self.env.model, self.data)
        for _ in range(self.env.frame_skip):
            mujoco.mj_step(self.env.model, self.data)
        return self.to_coordinates(data_state(self.data))

    def linearize(
        self,
        state: Array,
        action: float,
        *,
        state_epsilon: float,
        action_epsilon: float,
    ) -> tuple[Array, Array]:
        state = np.asarray(state, dtype=np.float64)
        a = np.empty((self.nx, self.nx), dtype=np.float64)
        for column in range(self.nx):
            offset = np.zeros(self.nx, dtype=np.float64)
            offset[column] = state_epsilon
            plus = self(state + offset, action)
            minus = self(state - offset, action)
            a[:, column] = self.difference(plus, minus) / (2.0 * state_epsilon)
        plus = self(state, min(1.0, action + action_epsilon))
        minus = self(state, max(-1.0, action - action_epsilon))
        denominator = min(1.0, action + action_epsilon) - max(-1.0, action - action_epsilon)
        b = (self.difference(plus, minus) / denominator)[:, None]
        return a, b


def _rail_cost_derivatives(x: float, cost: QuadraticTrajectoryCost) -> tuple[float, float, float]:
    distance = abs(float(x)) - cost.rail_soft_limit
    if distance <= 0.0:
        return 0.0, 0.0, 0.0
    width = max(1e-9, cost.rail_limit - cost.rail_soft_limit)
    ratio = distance / width
    value = cost.rail_weight * ratio**4
    gradient = 4.0 * cost.rail_weight * ratio**3 / width * np.sign(x)
    hessian = 12.0 * cost.rail_weight * ratio**2 / (width * width)
    return float(value), float(gradient), float(hessian)


def stage_cost(state: Array, action: float, cost: QuadraticTrajectoryCost, n_links: int) -> float:
    x = wrapped_state(state, n_links) if cost.wrap_angles else np.asarray(state, dtype=np.float64)
    rail, _, _ = _rail_cost_derivatives(float(x[0]), cost)
    return float(
        0.5 * x @ cost.stage_state @ x
        + 0.5 * cost.control * float(action) ** 2
        + rail
    )


def terminal_cost(state: Array, cost: QuadraticTrajectoryCost, n_links: int) -> float:
    x = wrapped_state(state, n_links) if cost.wrap_angles else np.asarray(state, dtype=np.float64)
    rail, _, _ = _rail_cost_derivatives(float(x[0]), cost)
    return float(0.5 * x @ cost.terminal_state @ x + rail)


def rollout(
    transition: Callable[[Array, float], Array],
    initial_state: Array,
    controls: Array,
    cost: QuadraticTrajectoryCost,
    n_links: int,
) -> tuple[Array, float]:
    controls = np.asarray(controls, dtype=np.float64)
    states = np.empty((len(controls) + 1, len(initial_state)), dtype=np.float64)
    states[0] = initial_state
    total = 0.0
    for step, action in enumerate(controls):
        total += stage_cost(states[step], float(action), cost, n_links)
        states[step + 1] = transition(states[step], float(action))
        if not np.all(np.isfinite(states[step + 1])):
            return states, 1e30
    total += terminal_cost(states[-1], cost, n_links)
    return states, float(total)


def _cost_derivatives(
    state: Array,
    action: float,
    cost: QuadraticTrajectoryCost,
    n_links: int,
) -> tuple[Array, float, Array, float]:
    x = wrapped_state(state, n_links) if cost.wrap_angles else np.asarray(state, dtype=np.float64)
    lx = cost.stage_state @ x
    lxx = cost.stage_state.copy()
    _, rail_gradient, rail_hessian = _rail_cost_derivatives(float(x[0]), cost)
    lx[0] += rail_gradient
    lxx[0, 0] += rail_hessian
    return lx, cost.control * float(action), lxx, float(cost.control)


def _terminal_derivatives(
    state: Array,
    cost: QuadraticTrajectoryCost,
    n_links: int,
) -> tuple[Array, Array]:
    x = wrapped_state(state, n_links) if cost.wrap_angles else np.asarray(state, dtype=np.float64)
    vx = cost.terminal_state @ x
    vxx = cost.terminal_state.copy()
    _, rail_gradient, rail_hessian = _rail_cost_derivatives(float(x[0]), cost)
    vx[0] += rail_gradient
    vxx[0, 0] += rail_hessian
    return vx, vxx


def _backward_pass(
    transition: MujocoTransition,
    states: Array,
    controls: Array,
    cost: QuadraticTrajectoryCost,
    *,
    regularization: float,
    state_epsilon: float,
    action_epsilon: float,
) -> tuple[Array, Array]:
    horizon = len(controls)
    nx = states.shape[1]
    feedforward = np.empty(horizon, dtype=np.float64)
    feedback = np.empty((horizon, nx), dtype=np.float64)
    vx, vxx = _terminal_derivatives(states[-1], cost, transition.env.n)

    for step in range(horizon - 1, -1, -1):
        a, b = transition.linearize(
            states[step],
            float(controls[step]),
            state_epsilon=state_epsilon,
            action_epsilon=action_epsilon,
        )
        lx, lu, lxx, luu = _cost_derivatives(
            states[step], float(controls[step]), cost, transition.env.n
        )
        qx = lx + a.T @ vx
        qu = float(lu + (b.T @ vx).item())
        qxx = lxx + a.T @ vxx @ a
        quu = float(luu + (b.T @ vxx @ b).item() + regularization)
        qux = (b.T @ vxx @ a).reshape(-1)
        if not np.isfinite(quu) or quu <= 0.0:
            raise np.linalg.LinAlgError("non-positive control curvature")
        k = -qu / quu
        gain = -qux / quu
        feedforward[step] = k
        feedback[step] = gain
        vx = qx + gain * (quu * k + qu) + qux * k
        vxx = qxx + quu * np.outer(gain, gain) + np.outer(gain, qux) + np.outer(qux, gain)
        vxx = 0.5 * (vxx + vxx.T)
    return feedforward, feedback


def optimize_ilqr(
    transition: MujocoTransition,
    initial_state: Array,
    initial_controls: Array,
    cost: QuadraticTrajectoryCost,
    *,
    max_iterations: int = 40,
    state_epsilon: float = 1e-5,
    action_epsilon: float = 1e-4,
    tolerance: float = 1e-5,
) -> ILQRResult:
    controls = np.clip(np.asarray(initial_controls, dtype=np.float64), -1.0, 1.0)
    states, current_cost = rollout(transition, initial_state, controls, cost, transition.env.n)
    regularization = 1e-4
    history: list[dict[str, float | int | bool]] = []
    converged = False
    feedback = np.zeros((len(controls), len(initial_state)), dtype=np.float64)

    for iteration in range(max_iterations):
        accepted = False
        try:
            feedforward, candidate_feedback = _backward_pass(
                transition,
                states,
                controls,
                cost,
                regularization=regularization,
                state_epsilon=state_epsilon,
                action_epsilon=action_epsilon,
            )
        except np.linalg.LinAlgError:
            regularization = min(1e12, regularization * 10.0)
            history.append(
                {
                    "iteration": iteration + 1,
                    "cost": float(current_cost),
                    "accepted": False,
                    "regularization": float(regularization),
                }
            )
            continue

        previous_cost = current_cost
        for alpha in (1.0, 0.5, 0.25, 0.1, 0.05, 0.01):
            candidate_controls = np.empty_like(controls)
            candidate_states = np.empty_like(states)
            candidate_states[0] = initial_state
            candidate_cost = 0.0
            for step in range(len(controls)):
                error = transition.difference(candidate_states[step], states[step])
                candidate_controls[step] = np.clip(
                    controls[step] + alpha * feedforward[step] + candidate_feedback[step] @ error,
                    -1.0,
                    1.0,
                )
                candidate_cost += stage_cost(
                    candidate_states[step],
                    float(candidate_controls[step]),
                    cost,
                    transition.env.n,
                )
                candidate_states[step + 1] = transition(
                    candidate_states[step], float(candidate_controls[step])
                )
            candidate_cost += terminal_cost(candidate_states[-1], cost, transition.env.n)
            if np.isfinite(candidate_cost) and candidate_cost < current_cost:
                controls = candidate_controls
                states = candidate_states
                current_cost = float(candidate_cost)
                feedback = candidate_feedback
                regularization = max(1e-9, regularization / 3.0)
                accepted = True
                break
        if not accepted:
            regularization = min(1e12, regularization * 10.0)
        relative_improvement = (previous_cost - current_cost) / max(1.0, abs(previous_cost))
        history.append(
            {
                "iteration": iteration + 1,
                "cost": float(current_cost),
                "accepted": bool(accepted),
                "relative_improvement": float(relative_improvement),
                "regularization": float(regularization),
            }
        )
        if accepted and relative_improvement < tolerance:
            converged = True
            break

    try:
        _, feedback = _backward_pass(
            transition,
            states,
            controls,
            cost,
            regularization=regularization,
            state_epsilon=state_epsilon,
            action_epsilon=action_epsilon,
        )
    except np.linalg.LinAlgError:
        pass
    return ILQRResult(
        controls=controls,
        states=states,
        feedback_gains=feedback,
        cost=float(current_cost),
        iterations=len(history),
        converged=converged,
        history=history,
    )
