import numpy as np

from gcartpole.dynamics_restoration import RestorationProblem, restore_trajectory


class DoubleIntegrator:
    def __call__(self, state, action):
        return np.array([state[0] + state[1], state[1] + action])

    def linearize(self, state, action, **kwargs):
        return np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([[0.0], [1.0]])


def problem():
    reference = np.column_stack([np.linspace(1.0, 0.0, 11), np.zeros(11)])
    return RestorationProblem(
        DoubleIntegrator(), reference[0], reference, np.zeros(10), np.zeros(2),
        np.eye(2), 1e6, 1e4, 0.0, 1e-6,
    )


def test_sparse_restoration_jacobian_matches_directional_derivative():
    p = problem()
    p.reference_weight = 0.1
    x = np.r_[p.reference_states[1:].ravel(), p.reference_controls]
    direction = np.random.default_rng(13).normal(size=x.size)
    eps = 1e-6
    finite_difference = (p.residual(x + eps * direction) - p.residual(x - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(p.jacobian(x) @ direction, finite_difference, atol=1e-7)


def test_restoration_closes_gaps_and_reaches_a_bounded_reachable_target():
    p = problem()
    result, states, controls = restore_trajectory(p, rail_limit=2.0, max_evaluations=30)
    assert result.success
    assert np.max(np.abs(p.defects(states, controls))) < 1e-8
    assert np.linalg.norm(states[-1]) < 1e-7
    assert np.max(np.abs(controls)) <= 1.0
    actual = states[0].copy()
    for action in controls:
        actual = p.transition(actual, action)
    np.testing.assert_allclose(actual, states[-1], atol=1e-7)


def test_factored_residual_units_preserve_the_sparse_directional_jacobian():
    p = problem()
    p.defect_factor = np.array([[2., .3], [0., .4]])
    p.reference_factor = np.array([[.5, 0.], [.2, 3.]])
    p.reference_weight = .1
    values = np.r_[p.reference_states[1:].ravel(), p.reference_controls]
    direction = np.random.default_rng(8).normal(size=values.size)
    finite = (p.residual(values + 1e-6 * direction) - p.residual(values - 1e-6 * direction)) / 2e-6
    np.testing.assert_allclose(p.jacobian(values) @ direction, finite, atol=2e-7)


def test_l1_restoration_reaches_target_without_virtual_dynamics_gaps():
    p = problem()
    result, states, controls = restore_trajectory(
        p, rail_limit=2.0, max_evaluations=10, solver="l1-osqp"
    )
    assert np.max(np.abs(p.defects(states, controls))) < 1e-7
    assert np.linalg.norm(states[-1]) < 1e-6
    assert np.max(np.abs(controls)) <= 1.0
    assert any(row["accepted"] for row in result.qp_history)


def test_native_failure_reuses_derivatives_without_shrinking_trust(monkeypatch):
    import osqp
    from types import SimpleNamespace

    bounds = []
    class FailedQP:
        def setup(self, **kwargs):
            bounds.append((kwargs["l"].copy(), kwargs["u"].copy()))

        def solve(self, **kwargs):
            return SimpleNamespace(x=None, info=SimpleNamespace(
                status_val=7, status="maximum iterations reached", prim_res=1.0,
                dual_res=1.0, iter=10000))

    monkeypatch.setattr(osqp, "OSQP", FailedQP)
    p = problem()
    calls = []
    original = p.jacobian
    def counted_jacobian(values):
        calls.append(values.copy())
        return original(values)
    p.jacobian = counted_jacobian
    result, states, controls = restore_trajectory(
        p, rail_limit=2.0, max_evaluations=10, solver="l1-osqp"
    )
    assert len(result.qp_history) == 3
    assert len(calls) == 1
    assert not result.success
    np.testing.assert_array_equal(states, p.reference_states)
    np.testing.assert_array_equal(controls, p.reference_controls)
    for lower, upper in bounds[1:]:
        np.testing.assert_array_equal(lower, bounds[0][0])
        np.testing.assert_array_equal(upper, bounds[0][1])


def test_inexact_qp_step_is_checked_by_exact_nonlinear_merit(monkeypatch):
    import osqp
    original_qp = osqp.OSQP

    class InexactQP:
        def __init__(self):
            self.delegate = original_qp()

        def setup(self, **kwargs):
            self.delegate.setup(**kwargs)

        def solve(self, **kwargs):
            result = self.delegate.solve(**kwargs)
            result.info.status_val = 7
            result.info.status = "maximum iterations reached"
            result.info.prim_res = 1e-7
            result.info.dual_res = 1e-7
            return result

    monkeypatch.setattr(osqp, "OSQP", InexactQP)
    p = problem()
    result, states, controls = restore_trajectory(
        p, rail_limit=2.0, max_evaluations=1, solver="l1-osqp",
        inexact_qp_tolerance=1e-4, monotone_defects=True,
    )
    assert result.qp_history[0]["inexact_candidate"]
    assert result.qp_history[0]["accepted"]
    assert np.max(np.linalg.norm(p.defects(states, controls), axis=1)) < 1e-7


def test_restoration_enforces_a_reachable_bounded_terminal_interval():
    p = problem()
    p.reference_states[-3:, 0] = 0
    lower = np.full((p.steps, p.nx), -np.inf)
    upper = np.full_like(lower, np.inf)
    lower[-3:, 0] = -0.05
    upper[-3:, 0] = 0.05
    result, states, controls = restore_trajectory(
        p, rail_limit=2.0, max_evaluations=40,
        state_lower_bounds=lower, state_upper_bounds=upper,
    )
    assert result.success
    assert np.max(np.abs(p.defects(states, controls))) < 1e-8
    assert np.max(np.abs(states[-3:, 0])) <= 0.05
    actual = states[0].copy()
    actual_states = []
    for action in controls:
        actual = p.transition(actual, action)
        actual_states.append(actual)
    assert np.max(np.abs(np.asarray(actual_states)[-3:, 0])) <= 0.05


def test_monotone_guard_rejects_l2_improvement_that_increases_largest_component(monkeypatch):
    import osqp
    from types import SimpleNamespace

    class ProposedQP:
        def setup(self, **kwargs):
            pass

        def solve(self, **kwargs):
            decision = np.zeros(7)
            decision[:2] = [-0.0004, 0.0005]
            return SimpleNamespace(x=decision, info=SimpleNamespace(
                status_val=1, status="solved", prim_res=0.0, dual_res=0.0, iter=1))

    monkeypatch.setattr(osqp, "OSQP", ProposedQP)
    reference = np.array([[0.0, 0.0], [-0.006, -0.006]])
    p = RestorationProblem(DoubleIntegrator(), reference[0], reference, np.zeros(1),
                           reference[-1], np.eye(2), 1e6, 1000, 0.0, 0.0)
    result, states, _ = restore_trajectory(
        p, rail_limit=2.0, max_evaluations=1, solver="l1-osqp", monotone_defects=True,
    )
    assert not result.qp_history[0]["accepted"]
    assert result.qp_history[0]["gap_norm"] == "linf"
    np.testing.assert_array_equal(states, reference)
