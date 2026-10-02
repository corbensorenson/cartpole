from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gcartpole.constrained_shooting import FactoredShootingProblem, optimize_constrained_shooting
from gcartpole.ilqr import QuadraticTrajectoryCost, rollout
from gcartpole.sqrt_fddp import trajectory_cost, trajectory_gaps


class Plant:
    env = SimpleNamespace(n=1)
    def __init__(self, a=None, b=None):
        self.a = np.array([[1.1, .1], [0., .9]]) if a is None else a
        self.b = np.array([[.2], [.6]]) if b is None else b
        self.linearizations = 0
    def __call__(self, x, u):
        return self.a @ x + self.b[:, 0]*u
    def linearize(self, x, u, **kwargs):
        self.linearizations += 1
        return self.a, self.b


def costs():
    base = QuadraticTrajectoryCost(np.eye(2), np.diag([4., 5.]), .2, .25, 2., 3., wrap_angles=False,
                                   terminal_target=np.array([.02, .01]))
    factor = np.array([[3., -2.], [0., 2.]])
    capture = replace(base, stage_factor=factor, stage_state=factor.T@factor,
                      control=.7, stage_target=np.array([.03, -.02]))
    return base, capture


def test_factored_sparse_jacobians_match_independent_directional_differences_and_cost():
    plant = Plant()
    base, capture = costs()
    schedule = [base]*2+[capture]*2
    nodes, _ = rollout(plant, np.array([.3, .1]), np.zeros(4), base, 1)
    p = FactoredShootingProblem(plant, nodes[0], nodes, np.zeros(4), base, running_costs=schedule,
                               defect_factor=np.array([[2., .3], [0., .4]]))
    values = p.initial.copy()
    direction = np.random.default_rng(14).normal(size=values.size)
    gap, scaled, r, dynamics, objective = p.evaluate(values, derivatives=True)
    plus, minus = p.evaluate(values+1e-6*direction), p.evaluate(values-1e-6*direction)
    np.testing.assert_allclose(dynamics @ direction, (plus[1]-minus[1])/2e-6, atol=2e-9)
    np.testing.assert_allclose(objective @ direction, (plus[2]-minus[2])/2e-6, atol=2e-8)
    np.testing.assert_allclose(.5*r@r, trajectory_cost(nodes, np.zeros(4), base, 1, schedule), rtol=1e-14)


def test_simultaneous_residual_qp_matches_independent_dense_linear_horizon_solution():
    plant = Plant()
    base, capture = costs()
    base = replace(base, rail_soft_limit=100., rail_limit=200.)
    capture = replace(capture, rail_soft_limit=100., rail_limit=200.)
    schedule = [base]*2+[capture]*4
    x0 = np.array([.15, .1])
    states, _ = rollout(plant, x0, np.zeros(6), base, 1)
    blocks, rhs, mapping, x = [], [], np.zeros((2, 6)), x0.copy()
    for step in range(7):
        c = base if step == 6 else schedule[step]
        root = np.linalg.cholesky(c.terminal_state).T if step == 6 else (
            c.stage_factor if c.stage_factor is not None else np.linalg.cholesky(c.stage_state).T)
        target = c.terminal_target if step == 6 else c.stage_target
        blocks.append(root@mapping)
        rhs.append(root@(x if target is None else x-target))
        if step < 6:
            x = plant.a@x
            mapping = plant.a@mapping
            mapping[:, step] += plant.b[:, 0]
    expected = np.linalg.lstsq(np.vstack(blocks+[np.diag(np.sqrt([c.control for c in schedule]))]),
                               -np.r_[np.concatenate(rhs), np.zeros(6)], rcond=None)[0]
    nodes = states.copy()
    nodes[1:] += np.random.default_rng(12).normal(size=nodes[1:].shape)*.01
    result = optimize_constrained_shooting(plant, x0, np.zeros(6), nodes, base,
                                          running_costs=schedule, max_iterations=5, initial_regularization=0.,
                                          state_trust=1., control_trust=1., defect_penalty=100.,
                                          qp_tolerance=1e-10, qp_max_iterations=20000)
    np.testing.assert_allclose(result.controls, expected, atol=2e-6)
    assert np.max(np.abs(trajectory_gaps(plant, result.states, result.controls))) < 1e-7
    assert result.history[0]['accepted'] and result.history[0]['alpha'] == 1.


def test_linear_capture_node_bound_prevents_cheaper_unsafe_endpoint():
    plant = Plant(np.eye(2), np.array([[.1], [1.]]))
    base, _ = costs()
    base = replace(base, terminal_state=np.eye(2)*100, terminal_target=np.array([0., 1.]))
    result = optimize_constrained_shooting(plant, np.zeros(2), np.zeros(1), np.zeros((2, 2)), base,
                                          max_iterations=10, state_trust=1., control_trust=1.,
                                          initial_regularization=0., defect_penalty=1e3, qp_tolerance=1e-10,
                                          node_constraint_matrix=np.array([[0., 1.]]),
                                          node_lower=np.array([[-.3]]), node_upper=np.array([[.3]]))
    assert .25 < result.states[-1, 1] <= .3
    assert np.max(np.abs(result.controls)) <= 1
    assert np.max(np.abs(trajectory_gaps(plant, result.states, result.controls))) < 1e-7


def test_explicit_weak_residual_survives_ill_conditioned_gram():
    plant = Plant(np.eye(2), np.zeros((2, 1)))
    base, _ = costs()
    root = np.array([[1e12, -1e12], [0., 1.]])
    base = replace(base, terminal_state=root.T@root, terminal_factor=root,
                   terminal_target=np.zeros(2), rail_soft_limit=10., rail_limit=20.)
    states = np.ones((2, 2))
    p = FactoredShootingProblem(plant, states[0], states, np.zeros(1), base)
    _, _, residual, _, objective = p.evaluate(p.initial, derivatives=True)
    # Stage has two rows plus a rail row; the terminal weak residual follows
    # its cancelled strong row and stays visible instead of evaluating P.
    assert residual[3] == 0 and residual[4] == 1
    assert objective[4, 1] == 1
    assert float(states[-1] @ base.terminal_state @ states[-1]) == 0


def test_native_qp_failure_preserves_nodes_and_trust_and_reuses_derivatives(monkeypatch):
    import osqp
    class Failed:
        def setup(self, **kwargs):
            pass
        def warm_start(self, **kwargs):
            pass
        def solve(self, **kwargs):
            return SimpleNamespace(x=None, info=SimpleNamespace(status_val=7, status='maximum iterations reached',
                                   iter=10000, prim_res=1., dual_res=1.))
    monkeypatch.setattr(osqp, 'OSQP', Failed)
    plant = Plant()
    base, _ = costs()
    states, _ = rollout(plant, np.array([.1, .1]), np.zeros(2), base, 1)
    result = optimize_constrained_shooting(plant, states[0], np.zeros(2), states, base, max_iterations=20)
    assert len(result.history) == 3
    assert not any(r['accepted'] for r in result.history)
    assert len({r['maximum_state_trust'] for r in result.history}) == 1
    assert plant.linearizations == 4  # One sparse Jacobian and the final feedback.
    np.testing.assert_array_equal(result.states, states)


def test_inadmissible_virtual_capture_input_is_rejected_without_projection():
    base, _ = costs()
    with pytest.raises(ValueError, match='initial virtual nodes violate'):
        optimize_constrained_shooting(Plant(), np.zeros(2), np.zeros(1), np.array([[0., 0.], [0., .4]]), base,
                                      node_constraint_matrix=np.array([[0., 1.]]),
                                      node_lower=np.array([[-.3]]), node_upper=np.array([[.3]]))


@pytest.mark.parametrize('gap', [0., .002])
def test_native_accuracy_schedule_uses_actual_gaps_and_keeps_the_refinement_floor(monkeypatch, gap):
    import osqp
    requested = []
    class Failed:
        def setup(self, **kwargs):
            requested.append(kwargs['eps_abs'])
        def warm_start(self, **kwargs):
            pass
        def solve(self, **kwargs):
            return SimpleNamespace(x=None, info=SimpleNamespace(status_val=7, status='maximum iterations reached',
                                   iter=10000, prim_res=1., dual_res=1.))
    monkeypatch.setattr(osqp, 'OSQP', Failed)
    plant = Plant()
    base, _ = costs()
    nodes, _ = rollout(plant, np.array([.1, .1]), np.zeros(2), base, 1)
    nodes[-1, 1] += gap
    result = optimize_constrained_shooting(plant, nodes[0], np.zeros(2), nodes, base,
                                          qp_tolerance=1e-8, qp_initial_tolerance=1e-3)
    expected = max(1e-8, min(1e-3, .1*gap))
    np.testing.assert_allclose(requested, np.full(3, expected), rtol=1e-12)
    np.testing.assert_array_equal(result.states, nodes)
    assert not any(row['accepted'] for row in result.history)


def test_inexact_candidate_can_pass_nonlinear_merit_without_being_called_native_optimal(monkeypatch):
    import osqp
    original = osqp.OSQP
    class Inexact:
        def __init__(self):
            self.delegate = original()
        def setup(self, **kwargs):
            self.delegate.setup(**kwargs)
        def warm_start(self, **kwargs):
            self.delegate.warm_start(**kwargs)
        def solve(self, **kwargs):
            result = self.delegate.solve(**kwargs)
            result.info.status_val = 7
            result.info.status = 'maximum iterations reached'
            result.info.prim_res = 1e-9
            result.info.dual_res = 1e-4
            return result
    monkeypatch.setattr(osqp, 'OSQP', Inexact)
    plant = Plant()
    base, _ = costs()
    states, _ = rollout(plant, np.array([.1, .1]), np.zeros(2), base, 1)
    states[-1] += .01
    result = optimize_constrained_shooting(plant, states[0], np.zeros(2), states, base,
                                          max_iterations=1, state_trust=1., control_trust=1.,
                                          defect_penalty=100., qp_tolerance=1e-9,
                                          qp_inexact_dual_tolerance=1e-3)
    row = result.history[0]
    assert row['inexact_qp_candidate'] and row['accepted'] and row['qp_step_usable']
    assert not row['qp_native_accepted']
    assert row['maximum_dynamics_defect'] < 1e-7


def test_pure_tracking_feedback_matches_independent_riccati_with_phase_costs():
    from gcartpole.sqrt_ilqr import square_root_tracking_gains
    plant = Plant()
    base, capture = costs()
    base = replace(base, rail_weight=0.)
    capture = replace(capture, rail_weight=0.)
    schedule = [base]*2+[capture]*3
    controls = np.array([.1, -.1, .2, 0., -.2])
    states, _ = rollout(plant, np.array([.1, .2]), controls, base, 1)
    observed = square_root_tracking_gains(plant, states, controls, base, running_costs=schedule,
                                         regularization=.3)
    p = base.terminal_state.copy()
    expected = np.zeros_like(observed)
    for step in range(len(controls)-1, -1, -1):
        c = schedule[step]
        expected[step] = -np.linalg.solve(np.array([[c.control+.3]])+plant.b.T@p@plant.b,
                                          plant.b.T@p@plant.a).ravel()
        closed = plant.a+plant.b@expected[step:step+1]
        p = c.stage_state+closed.T@p@closed+(c.control+.3)*np.outer(expected[step], expected[step])
    np.testing.assert_allclose(observed, expected, rtol=2e-14, atol=2e-14)


def test_pure_tracking_does_not_disable_gains_for_an_unexecuted_clipped_descent():
    from gcartpole.sqrt_ilqr import square_root_backward_pass, square_root_tracking_gains
    plant = Plant(np.eye(2), np.array([[.1], [1.]]))
    base, _ = costs()
    base = replace(base, rail_weight=0., terminal_target=np.array([0., 100.]))
    controls, states = np.array([.2]), np.zeros((2, 2))
    delta, optimization_gains, active = square_root_backward_pass(plant, states, controls, base,
                                                                 regularization=0.)
    assert active[0] and delta[0] == .8
    assert not np.any(optimization_gains)
    tracking = square_root_tracking_gains(plant, states, controls, base)
    target_free = square_root_tracking_gains(plant, states, controls, replace(base, terminal_target=None))
    np.testing.assert_array_equal(tracking, target_free)
    assert np.linalg.norm(tracking[0]) > .1
    np.testing.assert_array_equal(controls, np.array([.2]))
