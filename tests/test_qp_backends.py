import numpy as np
import pytest
from scipy import sparse

pytest.importorskip('clarabel')

from gcartpole.qp_backends import solve_clarabel_bounded_qp


def test_bounded_qp_preserves_equality_and_both_inequality_signs():
    # Unconstrained optimum (3,-2), equality x+y=0; bounds x<=1 and y>=-1
    # imply the unique optimum (1,-1). Include an entirely unbounded row.
    a = sparse.csc_matrix([[1., 1.], [1., 0.], [0., 1.], [1., -1.]])
    result = solve_clarabel_bounded_qp(sparse.eye(2,format='csc'), np.array([-3., 2.]),
        a, np.array([0., -np.inf, -1., -np.inf]), np.array([0., 1., np.inf, np.inf]),
        tolerance=1e-9, max_iterations=100)
    assert result.info.status_val in (1,2)
    np.testing.assert_allclose(result.x, [1., -1.], atol=2e-8)


def test_infeasible_native_status_cannot_be_used_as_inexact_candidate():
    result = solve_clarabel_bounded_qp(sparse.eye(1,format='csc'), np.zeros(1),
        sparse.csc_matrix([[1.], [1.]]), np.array([1., -np.inf]), np.array([np.inf, 0.]),
        tolerance=1e-9, max_iterations=100)
    assert result.info.status_val == -1
    assert 'Infeasible' in result.info.status


def test_clarabel_shooting_executes_actual_merit_and_hard_rails():
    from gcartpole.constrained_shooting import optimize_constrained_shooting
    from test_constrained_shooting import Plant, costs
    base, _ = costs()
    initial = np.array([.5, .2])
    controls = np.zeros(3)
    states = np.vstack([initial, np.zeros((3,2))])
    result = optimize_constrained_shooting(Plant(), initial, controls, states, base,
        max_iterations=5, state_trust=.5, control_trust=.5, qp_solver='clarabel',
        qp_max_iterations=100, qp_tolerance=1e-9)
    assert any(row['accepted'] for row in result.history)
    assert all(row['qp_solver']=='clarabel' and not row['inexact_qp_candidate'] for row in result.history)
    assert np.max(np.abs(result.states[:,0])) <= base.rail_limit
    accepted = [row['merit'] for row in result.history if row['accepted']]
    assert all(after < before for before,after in zip(accepted,accepted[1:]))
