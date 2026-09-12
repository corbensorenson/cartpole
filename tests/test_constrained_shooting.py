from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from gcartpole.constrained_shooting import _RolloutEvaluation


class _LinearTransition:
    nx = 4
    env = SimpleNamespace(n=1)

    def __call__(self, state: np.ndarray, action: float) -> np.ndarray:
        matrix = np.diag([0.9, 1.0, 0.8, 1.1])
        input_vector = np.asarray([1.0, 0.5, -0.25, 0.75])
        return matrix @ np.asarray(state, dtype=np.float64) + input_vector * action

    def linearize(
        self,
        state: np.ndarray,
        action: float,
        *,
        state_epsilon: float,
        action_epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del state, action, state_epsilon, action_epsilon
        return (
            np.diag([0.9, 1.0, 0.8, 1.1]),
            np.asarray([[1.0], [0.5], [-0.25], [0.75]]),
        )


def finite_difference(function: object, controls: np.ndarray) -> np.ndarray:
    epsilon = 1e-6
    value = np.asarray(function(controls), dtype=np.float64)
    jacobian = np.empty((value.size, controls.size), dtype=np.float64)
    for column in range(controls.size):
        plus = controls.copy()
        minus = controls.copy()
        plus[column] += epsilon
        minus[column] -= epsilon
        jacobian[:, column] = (
            np.asarray(function(plus)).reshape(-1)
            - np.asarray(function(minus)).reshape(-1)
        ) / (2.0 * epsilon)
    return jacobian


class ConstrainedShootingTests(unittest.TestCase):
    def test_analytic_objective_and_constraint_jacobians(self) -> None:
        evaluation = _RolloutEvaluation(
            _LinearTransition(),
            np.asarray([0.1, -0.2, 0.3, -0.4]),
            2.0 * np.eye(4),
            0.3,
            np.eye(4),
            10.0,
            5.0,
            2.0,
            1.0,
            1.0,
            1.0,
        )
        controls = np.asarray([0.2, -0.1, 0.05])
        objective_fd = finite_difference(evaluation.objective, controls)
        constraint_fd = finite_difference(evaluation.constraints, controls)
        np.testing.assert_allclose(
            evaluation.objective_jacobian(controls),
            objective_fd.reshape(-1),
            rtol=1e-5,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            evaluation.constraint_jacobian(controls),
            constraint_fd,
            rtol=1e-5,
            atol=1e-7,
        )


if __name__ == "__main__":
    unittest.main()
