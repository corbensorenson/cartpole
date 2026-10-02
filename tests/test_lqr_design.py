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
