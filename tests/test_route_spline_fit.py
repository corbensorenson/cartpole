import numpy as np
import pytest
from scipy.interpolate import BSpline

from scripts.fit_inverse_spline_from_route import fit_route_spline


def fixture():
    degree, count, seconds = 5, 20, 8.
    knots = np.r_[np.zeros(6), np.linspace(0, seconds, count-degree+1)[1:-1], np.full(6, seconds)]
    points = np.random.default_rng(29).normal(0, .2, (count, 4))
    points[:3] = [0, np.pi, 0, 0]; points[-3:] = 0
    times = np.linspace(0, seconds, 401)
    spline = BSpline(knots, points, degree)
    states = np.c_[spline(times), spline.derivative()(times)]
    return points, knots, states, times


def test_fit_recovers_known_curve_and_velocity_after_branch_alignment():
    points, knots, states, times = fixture()
    states[:, 1] -= 2*np.pi
    template = points.copy(); template[3:-3] = 0
    fitted, metrics = fit_route_spline(template, knots, states, times, 3., .02)
    np.testing.assert_allclose(fitted, points, atol=2e-12)
    assert metrics["max_absolute_link_angle_error"] < 1e-12
    assert metrics["relative_joint_velocity_rms_error"] < 1e-11


def test_fit_rejects_a_different_terminal_winding():
    points, knots, states, times = fixture()
    states[:, 1] += 2*np.pi*(times/times[-1])**3
    with pytest.raises(ValueError, match="winding"):
        fit_route_spline(points, knots, states, times, 3., .02)
