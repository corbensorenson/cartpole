import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov

from scripts.audit_capture_value_precision import precise_value_matrix


def test_precise_value_handles_nonnormal_transients_and_matches_equation():
    a = np.array([[.95, 20.], [0., .85]])
    t = np.diag([3., .5])
    ctx, value, root, diagnostics = precise_value_matrix(a, np.zeros((2, 1)), np.zeros(2), t, 1., 80)
    transformed = t@a@np.linalg.inv(t)
    expected = solve_discrete_lyapunov(transformed.T, np.eye(2))
    np.testing.assert_allclose(np.array(value.tolist(), dtype=float), expected, rtol=1e-11)
    factor = np.array(root.tolist(), dtype=float)
    np.testing.assert_allclose(factor.T@factor, expected, rtol=1e-11)
    assert ctx.mpf(diagnostics["absolute_equation_residual"]) < ctx.mpf("1e-50")


def test_precise_value_refuses_unstable_promoted_input_loop():
    with pytest.raises(ValueError, match="unstable"):
        precise_value_matrix(np.diag([1.01, .9]), np.zeros((2, 1)), np.zeros(2), np.eye(2), 1., 80)
