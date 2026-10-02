import numpy as np
import pytest

from gcartpole.lqr_design import checked_discrete_lqr


def test_checked_lqr_accepts_a_stabilizing_solution():
    gain, p, diagnostics = checked_discrete_lqr(
        np.array([[1.2]]), np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1))
    )
    assert abs(1.2 - gain[0, 0]) < 1.0
    assert p[0, 0] > 0
    assert diagnostics["relative_riccati_residual"] < 1e-12


def test_returned_riccati_matrix_is_not_automatically_a_valid_controller(monkeypatch):
    monkeypatch.setattr("gcartpole.lqr_design.solve_discrete_are", lambda *args: np.ones((1, 1)))
    with pytest.raises(ValueError, match="Invalid discrete LQR"):
        checked_discrete_lqr(np.array([[2.0]]), np.ones((1, 1)), np.ones((1, 1)), np.ones((1, 1)))


def test_indefinite_returned_riccati_matrix_is_rejected(monkeypatch):
    monkeypatch.setattr("gcartpole.lqr_design.solve_discrete_are", lambda *args: -np.ones((1, 1)))
    with pytest.raises(ValueError, match="minimum Riccati eigenvalue"):
        checked_discrete_lqr(np.array([[0.5]]), np.zeros((1, 1)), np.ones((1, 1)), np.ones((1, 1)))


def test_high_precision_doubling_agrees_with_independent_dare_design():
    from gcartpole.lqr_design import high_precision_discrete_lqr
    args = (np.array([[1., 1.], [0., 1.]]), np.array([[0.], [1.]]),
            np.eye(2), np.array([[1.]]))
    expected_gain, expected_p, _ = checked_discrete_lqr(*args)
    gain, p, diagnostics = high_precision_discrete_lqr(*args, decimal_digits=50)
    np.testing.assert_allclose(gain, expected_gain, atol=1e-12)
    np.testing.assert_allclose(p, expected_p, atol=1e-12)
    assert float(diagnostics["relative_riccati_residual"]) < 1e-40
    assert float(diagnostics["rounded_gain_high_precision_spectral_radius"]) < 1
    assert not diagnostics["nonlinear_certified"]


def test_high_precision_doubling_rejects_unconverged_design():
    from gcartpole.lqr_design import high_precision_discrete_lqr
    with pytest.raises(ValueError, match="did not converge"):
        high_precision_discrete_lqr(np.array([[1.2]]), np.zeros((1, 1)),
                                    np.ones((1, 1)), np.ones((1, 1)), max_iterations=3)


def test_precision_input_design_retains_small_unstable_drift_lost_by_float_conversion():
    from gcartpole.lqr_design import high_precision_discrete_lqr
    inputs = ([["1.00000000000000000001"]], [["0.00000000000000000001"]], [[1.]], [[1.]])
    retained, _, diagnostic = high_precision_discrete_lqr(
        *inputs, decimal_digits=80, max_iterations=100, preserve_input_precision=True)
    rounded, _, _ = high_precision_discrete_lqr(*inputs, decimal_digits=80, max_iterations=100)
    np.testing.assert_allclose(retained, [[1+np.sqrt(2)]], rtol=1e-12)
    np.testing.assert_allclose(rounded, [[1.]], rtol=1e-12)
    assert diagnostic['preserved_input_precision']


def test_precision_plant_is_retained_when_factoring_the_rounded_gain_value():
    from decimal import Decimal
    from gcartpole.lqr_design import high_precision_lyapunov_factor
    p, root, diagnostic = high_precision_lyapunov_factor(
        [["0.900000000000000000000001"]], [[0.]], [0.], [[1.]],
        decimal_digits=80, preserve_input_precision=True)
    assert Decimal(diagnostic['spectral_radius'])-Decimal('.9') == Decimal('1e-24')
    np.testing.assert_allclose(p, [[1/(1-.9**2)]], rtol=1e-13)
    np.testing.assert_allclose(root.T@root, p)


def test_factored_promoted_lyapunov_matches_independent_equation_and_rejects_unstable_gain():
    from scipy.linalg import solve_discrete_lyapunov
    from gcartpole.lqr_design import high_precision_lyapunov_factor
    a = np.array([[.8, 2.], [0., .6]])
    args = (a, np.zeros((2, 1)), np.zeros(2), np.eye(2))
    p, root, diagnostics = high_precision_lyapunov_factor(*args, decimal_digits=50)
    expected = solve_discrete_lyapunov(a.T, np.eye(2))
    np.testing.assert_allclose(p, expected, rtol=1e-13)
    for state in [np.array([1., 0.]), np.array([0., 1.]), np.array([1., -1.])]:
        np.testing.assert_allclose(np.linalg.norm(root@state)**2, state@expected@state, rtol=1e-13)
    assert float(diagnostics['relative_equation_residual']) < 1e-40
    with pytest.raises(ValueError, match='unstable'):
        high_precision_lyapunov_factor(a+np.eye(2), *args[1:], decimal_digits=50)


def test_factored_terminal_value_and_gradient_retain_small_direction_hidden_by_gram_rounding():
    from gcartpole.ilqr import QuadraticTrajectoryCost, terminal_cost, _terminal_derivatives
    root = np.array([[1e12, -1e12], [0., 1.]])
    p = root.T@root
    state = np.ones(2)
    assert state@p@state == 0.
    cost = QuadraticTrajectoryCost(stage_state=np.eye(2), terminal_state=p,
        terminal_factor=root, control=1., rail_soft_limit=10., rail_limit=20., rail_weight=1., wrap_angles=False)
    assert terminal_cost(state, cost, 0) == .5
    gradient, _ = _terminal_derivatives(state, cost, 0)
    np.testing.assert_array_equal(gradient, [0., 1.])


def test_factored_running_value_and_gradient_retain_small_direction_hidden_by_gram_rounding():
    from gcartpole.ilqr import QuadraticTrajectoryCost,stage_cost,_cost_derivatives
    root=np.array([[1e12,-1e12],[0.,1.]]);p=root.T@root;state=np.ones(2)
    assert state@p@state==0.
    cost=QuadraticTrajectoryCost(p,np.eye(2),1.,10.,20.,0.,wrap_angles=False,stage_factor=root)
    assert stage_cost(state,0.,cost,0)==.5
    gradient,*_= _cost_derivatives(state,0.,cost,0)
    np.testing.assert_array_equal(gradient,[0.,1.])
