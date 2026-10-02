import numpy as np
import pytest

from gcartpole.high_order_dynamics import FourthOrderMujocoTransition
from gcartpole.ilqr import MujocoTransition


class AnalyticMap(FourthOrderMujocoTransition):
    def __init__(self):
        self.nx = 2
        self.continuous_angles = True

    def __call__(self, state, action):
        u = float(np.float32(action))
        return np.array([state[0] ** 5 + state[1] ** 3 + u ** 4,
                         np.sin(state[0]) + np.exp(state[1]) + u ** 3])


@pytest.mark.parametrize("action", [.7, -1., 1.])
def test_fourth_order_secants_agree_with_analytic_nonlinear_derivatives(action):
    plant = AnalyticMap()
    x = np.array([.7, -.2])
    a, b = plant.linearize(x, action, state_epsilon=.01, action_epsilon=.01)
    expected_a = np.array([[5 * x[0] ** 4, 3 * x[1] ** 2], [np.cos(x[0]), np.exp(x[1])]])
    u = float(np.float32(action))
    expected_b = np.array([[4 * u ** 3], [3 * u ** 2]])
    np.testing.assert_allclose(a, expected_a, atol=5e-8)
    np.testing.assert_allclose(b, expected_b, atol=1e-11)
    two_point, _ = MujocoTransition.linearize(plant, x, action, state_epsilon=.01, action_epsilon=.01)
    assert np.max(np.abs(a - expected_a)) < np.max(np.abs(two_point - expected_a)) / 1000


def test_fourth_order_secant_rejects_duplicate_quantized_knots():
    with pytest.raises(ValueError, match="coincide"):
        AnalyticMap().linearize(np.zeros(2), .7, state_epsilon=1e-3, action_epsilon=1e-12)
