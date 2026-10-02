from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from gcartpole.ilqr import QuadraticTrajectoryCost, rollout
from gcartpole.sqrt_fddp import optimize_sqrt_fddp, trajectory_gaps
from gcartpole.sqrt_ilqr import square_root_backward_pass


class LinearPlant:
    env = SimpleNamespace(n=1)
    a = np.array([[1., .1], [0., .9]])
    b = np.array([[.1], [.5]])

    def __call__(self, x, u):
        return self.a @ x + self.b[:, 0] * u

    def linearize(self, x, u, **kwargs):
        return self.a, self.b


@pytest.mark.parametrize('scheduled',[False,True])
def test_defect_aware_step_solves_independent_dense_linear_horizon(scheduled):
    plant = LinearPlant()
    cost = QuadraticTrajectoryCost(np.diag([2., 3.]), np.diag([4., 5.]),
                                   .5, 100., 200., 0., wrap_angles=False)
    x0 = np.array([.2, .1])
    horizon = 6
    costs=([cost]*3+[replace(cost,stage_state=np.diag([5.,1.]),control=.8)]*3
           if scheduled else [cost]*horizon)
    exact, _ = rollout(plant, x0, np.zeros(horizon), cost, 1)
    nodes = exact + np.random.default_rng(12).normal(size=exact.shape) * .01
    nodes[0] = x0
    assert np.max(np.abs(trajectory_gaps(plant, nodes, np.zeros(horizon)))) > .001
    blocks, offsets = [], []
    mapping = np.zeros((2, horizon))
    for step in range(horizon + 1):
        root = np.linalg.cholesky(cost.terminal_state if step == horizon else costs[step].stage_state).T
        blocks.append(root @ mapping)
        offsets.append(root @ exact[step])
        if step < horizon:
            mapping = plant.a @ mapping
            mapping[:, step] += plant.b[:, 0]
    matrix = np.vstack(blocks + [np.diag(np.sqrt([c.control for c in costs]))])
    expected = np.linalg.lstsq(matrix, -np.r_[np.concatenate(offsets), np.zeros(horizon)], rcond=None)[0]
    result = optimize_sqrt_fddp(plant, x0, np.zeros(horizon), nodes, cost,
                               max_iterations=1, initial_regularization=0., defect_penalty=1e4,running_costs=costs)
    np.testing.assert_allclose(result.controls, expected, atol=1e-8)
    np.testing.assert_array_equal(trajectory_gaps(plant, result.states, result.controls), np.zeros((horizon, 2)))
    assert result.history[0]["alpha"] == 1.


def test_zero_defects_preserve_the_feasible_square_root_direction():
    plant = LinearPlant()
    cost = QuadraticTrajectoryCost(np.eye(2), np.eye(2), .5, 100., 200., 0., wrap_angles=False)
    controls = np.zeros(4)
    states, _ = rollout(plant, np.array([.2, .1]), controls, cost, 1)
    ordinary = square_root_backward_pass(plant, states, controls, cost)
    explicit = square_root_backward_pass(plant, states, controls, cost, gaps=np.zeros((4, 2)))
    for a, b in zip(ordinary, explicit):
        np.testing.assert_array_equal(a, b)
