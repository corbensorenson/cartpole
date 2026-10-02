import numpy as np

from scripts.build_route_ltv_feedback import ltv_feedback_gains


def test_ltv_gain_matches_independent_condensed_quadratic_optimum():
    a = np.array([[1.0, 1.0], [0.0, 1.0]])
    b = np.array([[0.0], [1.0]])

    class Transition:
        def linearize(self, state, action, **kwargs):
            return a, b

    steps = 5
    q = np.diag([1.0, 0.7])
    r = 0.9
    terminal_scale = 2.0
    gains, _ = ltv_feedback_gains(
        Transition(), np.zeros((steps + 1, 2)), np.zeros(steps),
        state_epsilon=1e-5, action_epsilon=1e-4, control_cost=r,
        position_cost=1.0, angle_cost=1.0, cart_velocity_cost=0.7,
        hinge_velocity_cost=1.0, terminal_scale=terminal_scale,
    )
    # Solve the whole finite-horizon quadratic directly, without a Riccati
    # recurrence, for both basis initial states.
    hessian = r * np.eye(steps)
    cross = np.zeros((steps, 2))
    for t in range(1, steps + 1):
        propagation = np.linalg.matrix_power(a, t)
        influence = np.zeros((2, steps))
        for j in range(t):
            influence[:, j] = (np.linalg.matrix_power(a, t - j - 1) @ b)[:, 0]
        weight = terminal_scale * q if t == steps else q
        hessian += influence.T @ weight @ influence
        cross += influence.T @ weight @ propagation
    optimal_controls = -np.linalg.solve(hessian, cross)
    np.testing.assert_allclose(gains[0], optimal_controls[0], atol=1e-12)


def test_square_root_ltv_matches_high_precision_for_a_weak_input_mode():
    import mpmath
    a = np.diag([1.2, 0.9])
    b = np.array([[1e-8], [1.0]])
    class Transition:
        def linearize(self, state, action, **kwargs):
            return a, b

    steps = 100
    gains, _ = ltv_feedback_gains(
        Transition(), np.zeros((steps + 1, 2)), np.zeros(steps),
        state_epsilon=1e-5, action_epsilon=1e-4, control_cost=1.0,
        position_cost=1.0, angle_cost=1.0, cart_velocity_cost=1.0,
        hinge_velocity_cost=1.0, terminal_scale=1.0,
    )
    ctx = mpmath.mp.clone()
    ctx.dps = 70
    am, bm = ctx.matrix(a.tolist()), ctx.matrix(b.tolist())
    p = ctx.eye(2)
    for _ in range(steps):
        hessian = ctx.mpf(1) + (bm.T * p * bm)[0]
        gain = bm.T * p * am / hessian
        p = ctx.eye(2) + am.T * p * am - am.T * p * bm * gain
    expected = -np.asarray(gain.tolist(), dtype=float)[0]
    np.testing.assert_allclose(gains[0], expected, rtol=2e-6, atol=1e-8)
