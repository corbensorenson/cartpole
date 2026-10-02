from pathlib import Path

import mujoco
import numpy as np
import pytest
from scipy.interpolate import BSpline

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.inverse_collocation import SplineInverseProblem
from scripts.search_inverse_spline import refine_saved_spline
from scripts.transfer_inverse_spline import transfer_coefficients, retime_spline_knots


def make_problem():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/swingup11_uniform.yaml")
    cfg["env"]["n_links"] = 2
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    initial = np.array([0.0, np.pi, 0.0])
    return env, SplineInverseProblem(env.model, initial, np.zeros(3), 8.0, 10,
                                     np.linspace(0, 8, 11), 80.0)


def test_spline_preserves_position_velocity_acceleration_boundary_conditions():
    env, problem = make_problem()
    q, v, a = problem.trajectory(problem.initial_values)
    np.testing.assert_allclose(q[[0, -1]], [problem.initial_position, problem.terminal_position], atol=1e-14)
    np.testing.assert_allclose(v[[0, -1]], 0, atol=1e-14)
    np.testing.assert_allclose(a[[0, -1]], 0, atol=1e-14)
    env.close()


def test_inverse_jacobian_matches_independent_directional_difference():
    env, problem = make_problem()
    x = problem.initial_values
    direction = np.random.default_rng(16).normal(size=len(x))
    epsilon = 1e-6
    numerical = (problem.residual(x + epsilon * direction) - problem.residual(x - epsilon * direction)) / (2 * epsilon)
    np.testing.assert_allclose(problem.jacobian(x) @ direction, numerical, rtol=2e-4, atol=2e-5)
    env.close()


def test_inverse_force_reproduces_requested_acceleration_when_fully_applied():
    env, problem = make_problem()
    q, v, a = np.array([0.2, 1.3, -0.4]), np.array([0.1, -0.5, 0.3]), np.array([1., -2., 0.4])
    force = problem.inverse(q, v, a)
    data = mujoco.MjData(env.model)
    data.qpos[:], data.qvel[:] = q, v
    data.qfrc_applied[:] = force
    mujoco.mj_forward(env.model, data)
    np.testing.assert_allclose(data.qacc, a, atol=1e-10)
    env.close()


def test_acceleration_derivative_includes_mass_and_preserves_rk4():
    env, problem = make_problem()
    q, v, a = np.array([0.2, 1.3, -0.4]), np.array([0.1, -0.5, 0.3]), np.array([1., -2., 0.4])
    integrator = env.model.opt.integrator
    _, _, mass = problem.inverse_derivatives(q, v, a)
    direction = np.array([0.7, -0.2, 0.6])
    numerical = (problem.inverse(q, v, a + 1e-6 * direction)
                 - problem.inverse(q, v, a - 1e-6 * direction)) / 2e-6
    np.testing.assert_allclose(mass @ direction, numerical, atol=1e-8)
    assert env.model.opt.integrator == integrator == mujoco.mjtIntegrator.mjINT_RK4
    env.close()


def test_acceleration_residual_matches_actual_missing_joint_force():
    env, problem = make_problem()
    q, v, a = np.array([0.2, 1.3, -0.4]), np.array([0.1, -0.5, 0.3]), np.array([1., -2., 0.4])
    force = problem.inverse(q, v, a)
    data = mujoco.MjData(env.model)
    data.qpos[:], data.qvel[:] = q, v
    data.qfrc_applied[0] = force[0]
    mujoco.mj_forward(env.model, data)
    np.testing.assert_allclose(problem.acceleration_residual(q, v, a), a - data.qacc, atol=1e-10)
    env.close()


def test_acceleration_spline_jacobian_matches_independent_directional_difference():
    env, original = make_problem()
    problem = SplineInverseProblem(env.model, original.initial_position, original.terminal_position,
                                   8., 10, np.linspace(0, 8, 11), 80., dynamics_residual="acceleration")
    x = problem.initial_values
    direction = np.random.default_rng(16).normal(size=len(x))
    numerical = (problem.residual(x + 1e-6 * direction) - problem.residual(x - 1e-6 * direction)) / 2e-6
    np.testing.assert_allclose(problem.jacobian(x) @ direction, numerical, rtol=2e-4, atol=2e-5)
    env.close()


def test_knot_refinement_preserves_full_saved_curve_and_endpoint_conditions():
    env, original = make_problem()
    source = dict(search=dict(spline_knots=original.knots.tolist(),
                             spline_control_points=original.reference.tolist()))
    spline = refine_saved_spline(source, 2)
    refined = SplineInverseProblem(env.model, original.initial_position, original.terminal_position,
                                   8., len(spline.c), original.sample_times, 80.,
                                   knots=spline.t, initial_control_points=spline.c)
    times = np.linspace(0, 8, 513)
    baseline = BSpline(original.knots, original.reference, 5)
    for order in range(3):
        np.testing.assert_allclose(spline.derivative(order)(times), baseline.derivative(order)(times), atol=1e-12)
    for first, second in zip(original.trajectory(original.initial_values), refined.trajectory(refined.initial_values)):
        np.testing.assert_allclose(first, second, atol=1e-12)
    env.close()


def test_spline_material_transfer_preserves_cart_and_constant_absolute_orientation():
    points = np.array([[.2, np.pi, 0.], [-.5, .7, 0.], [0., 0., 0.]])
    np.testing.assert_array_equal(transfer_coefficients(points, 2), points)
    mapped = transfer_coefficients(points, 11)
    np.testing.assert_array_equal(mapped[:, 0], points[:, 0])
    np.testing.assert_allclose(np.cumsum(mapped[:, 1:], axis=1), np.repeat(points[:, 1:2], 11, axis=1))


def test_spline_retiming_preserves_geometry_and_scales_velocity_acceleration():
    env, problem = make_problem()
    knots, points = problem.knots, problem.unpack(problem.initial_values)
    old = BSpline(knots, points, 5)
    new_knots = retime_spline_knots(knots, 8., 10.)
    new = BSpline(new_knots, points, 5)
    times = np.linspace(0., 8., 101)
    for order in (0, 1, 2):
        np.testing.assert_allclose(new.derivative(order)(times*1.25),
                                   old.derivative(order)(times)/1.25**order,
                                   rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(retime_spline_knots(knots, 8., 8.), knots)
    env.close()


def test_spline_retiming_rejects_inconsistent_time_domains():
    for knots, old, new in (([0., 1.], 2., 3.), ([0., float('nan')], 1., 2.),
                            ([0., 1.], 1., 0.), ([0., 1.], 1., float('inf')),
                            ([0., 1., .5], .5, 1.)):
        with pytest.raises(ValueError):
            retime_spline_knots(knots, old, new)
