"""A finite soft penalty can favor a cheaper rail-violating trajectory."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gcartpole.ilqr import QuadraticTrajectoryCost, rollout
from gcartpole.sqrt_fddp import optimize_sqrt_fddp, trajectory_gaps
from gcartpole.sqrt_ilqr import optimize_sqrt_ilqr


class Plant:
    env = SimpleNamespace(n=1)

    def __call__(self, x, u):
        return x + np.array([u, 0.])

    def difference(self, x, reference):
        return x-reference

    def linearize(self, x, u, **kwargs):
        return np.eye(2), np.array([[1.], [0.]])


def objective():
    return QuadraticTrajectoryCost(
        np.eye(2), np.eye(2)*1e8, .01, .25, .3, 1.,
        wrap_angles=False, terminal_target=np.array([1., 0.]))


def solve(kind, controls=None, cost=None, costs=None, enforce=True, nodes=None):
    plant = Plant()
    cost = objective() if cost is None else cost
    controls = np.zeros(1) if controls is None else controls
    options = dict(max_iterations=8, initial_regularization=0.,
                   enforce_rail=enforce, running_costs=costs)
    if kind == 'ilqr':
        return optimize_sqrt_ilqr(plant, np.zeros(2), controls, cost, **options)
    if nodes is None:
        nodes, _ = rollout(plant, np.zeros(2), controls, cost, 1, running_costs=costs)
    return optimize_sqrt_fddp(plant, np.zeros(2), controls, nodes, cost, **options)


@pytest.mark.parametrize('kind', ['ilqr', 'fddp'])
def test_hard_filter_rejects_cheaper_rail_violation_including_terminal_node(kind):
    unconstrained = solve(kind, enforce=False)
    constrained = solve(kind)
    assert unconstrained.states[-1, 0] > .9
    assert np.max(np.abs(constrained.states[:, 0])) <= .3
    assert .2 < constrained.states[-1, 0]
    assert unconstrained.cost < constrained.cost
    assert any(row['rail_rejected_trials'] > 0 for row in constrained.history)
    assert all(row['invalid_transition_trials'] == 0 for row in constrained.history)
    np.testing.assert_array_equal(trajectory_gaps(Plant(), constrained.states, constrained.controls), np.zeros((1, 2)))


@pytest.mark.parametrize('kind', ['ilqr', 'fddp'])
def test_phase_limits_apply_to_intermediate_nodes_even_when_terminal_is_wider(kind):
    base = replace(objective(), rail_soft_limit=1., rail_limit=2.)
    capture = replace(base, rail_soft_limit=.1, rail_limit=.15)
    result = solve(kind, np.zeros(2), base, [base, capture])
    assert result.states[1, 0] <= .15
    assert result.states[-1, 0] > .3


@pytest.mark.parametrize('kind', ['ilqr', 'fddp'])
def test_inadmissible_initializer_is_rejected_instead_of_reported_feasible(kind):
    with pytest.raises(ValueError, match='initial trajectory violates hard rail at node 1'):
        solve(kind, np.array([.5]))


@pytest.mark.parametrize('limit', [0., -1., float('nan'), float('inf')])
def test_hard_filter_rejects_invalid_bounds(limit):
    with pytest.raises(ValueError, match='finite positive limits'):
        solve('ilqr', cost=replace(objective(), rail_limit=limit))


def test_virtual_fddp_nodes_remain_explicitly_infeasible_after_small_filtered_step():
    nodes = np.array([[0., 0.], [.2, 0.]])
    result = solve('fddp', nodes=nodes)
    assert np.max(np.abs(result.states[:, 0])) <= .3
    assert any(row['rail_rejected_trials'] > 0 for row in result.history)
    assert np.max(np.abs(trajectory_gaps(Plant(), result.states, result.controls))) > .01
