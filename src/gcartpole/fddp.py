from __future__ import annotations

from typing import Any

import numpy as np

from gcartpole.ilqr import (
    MujocoTransition,
    QuadraticTrajectoryCost,
    _cost_derivatives,
    _terminal_derivatives,
    stage_cost,
    terminal_cost,
)

try:
    import crocoddyl
except ImportError as error:  # pragma: no cover - exercised without optional extra
    raise ImportError(
        "Crocoddyl is required for FDDP; install requirements-fddp.txt"
    ) from error


def rollout_controls(
    transition: MujocoTransition, start_state: np.ndarray, controls: np.ndarray
) -> np.ndarray:
    states = [np.asarray(start_state, dtype=np.float64).copy()]
    for action in np.asarray(controls, dtype=np.float64):
        states.append(transition(states[-1], float(action)))
    return np.asarray(states, dtype=np.float64)


class MujocoActionModel(crocoddyl.ActionModelAbstract):
    """Crocoddyl action model backed by one exact MuJoCo policy transition."""

    def __init__(
        self,
        transition: MujocoTransition,
        cost: QuadraticTrajectoryCost,
        *,
        terminal: bool = False,
        state_epsilon: float = 1e-5,
        action_epsilon: float = 1e-4,
    ) -> None:
        super().__init__(crocoddyl.StateVector(transition.nx), 1, 1)
        self.transition = transition
        self.trajectory_cost = cost
        self.terminal = bool(terminal)
        self.state_epsilon = float(state_epsilon)
        self.action_epsilon = float(action_epsilon)
        self.u_lb = np.asarray([-1.0], dtype=np.float64)
        self.u_ub = np.asarray([1.0], dtype=np.float64)

    def calc(
        self,
        data: Any,
        state: np.ndarray,
        control: np.ndarray | None = None,
    ) -> None:
        state = np.asarray(state, dtype=np.float64)
        if self.terminal or control is None:
            data.xnext[:] = state
            data.cost = terminal_cost(
                state, self.trajectory_cost, self.transition.env.n
            )
            return
        action = float(np.asarray(control, dtype=np.float64)[0])
        data.xnext[:] = self.transition(state, action)
        data.cost = stage_cost(
            state, action, self.trajectory_cost, self.transition.env.n
        )

    def calcDiff(
        self,
        data: Any,
        state: np.ndarray,
        control: np.ndarray | None = None,
    ) -> None:
        state = np.asarray(state, dtype=np.float64)
        if self.terminal or control is None:
            gradient, hessian = _terminal_derivatives(
                state, self.trajectory_cost, self.transition.env.n
            )
            data.Fx[:, :] = np.eye(self.state.ndx, dtype=np.float64)
            data.Fu[:] = 0.0
            data.Lx[:] = gradient
            data.Lu[:] = 0.0
            data.Lxx[:, :] = hessian
            data.Lxu[:] = 0.0
            data.Luu[:, :] = 0.0
            return
        action = float(np.asarray(control, dtype=np.float64)[0])
        state_matrix, input_matrix = self.transition.linearize(
            state,
            action,
            state_epsilon=self.state_epsilon,
            action_epsilon=self.action_epsilon,
        )
        lx, lu, lxx, luu = _cost_derivatives(
            state, action, self.trajectory_cost, self.transition.env.n
        )
        data.Fx[:, :] = state_matrix
        data.Fu[:] = input_matrix.reshape(-1)
        data.Lx[:] = lx
        data.Lu[:] = lu
        data.Lxx[:, :] = lxx
        data.Lxu[:] = 0.0
        data.Luu[:, :] = luu
