import numpy as np
from gcartpole.periodic_capture import antiperiodic_dynamics_matrix, linear_antiperiodic_nodes, refine_antiperiodic_orbit


class OddPlant:
    def __init__(self, nonlinear=False):
        self.a = np.array([[1.2, .1], [0., .8]])
        self.b = np.array([.2, .5])
        self.nonlinear = nonlinear
    def __call__(self, x, u):
        return self.a@x+self.b*u+(.1*x**3 if self.nonlinear else 0.)
    def linearize(self, x, u, **kwargs):
        return self.a+np.diag(.3*x**2 if self.nonlinear else np.zeros(2)), self.b[:, None]


def test_antiperiodic_sparse_jacobian_matches_actual_wrap_boundary():
    plant = OddPlant(True)
    nodes = np.random.default_rng(21).normal(size=(5, 2))*.1
    direction = np.random.default_rng(22).normal(size=nodes.shape)
    controls = np.linspace(-.1, .1, 5)
    def gaps(x):
        return np.asarray([plant(v,u) for v,u in zip(x, controls)])-np.vstack([x[1:], -x[:1]])
    matrix = antiperiodic_dynamics_matrix([plant.linearize(x, u)[0] for x,u in zip(nodes, controls)])
    np.testing.assert_allclose(matrix@direction.ravel(),
                               ((gaps(nodes+1e-6*direction)-gaps(nodes-1e-6*direction))/2e-6).ravel(), atol=1e-10)


def test_linear_boundary_solve_handles_an_unstable_mode_without_forward_shooting():
    plant = OddPlant()
    controls = (np.cos(np.arange(25)*np.pi/25)*.1).astype(np.float32).astype(float)
    nodes = linear_antiperiodic_nodes(plant.a, plant.b, controls)
    full, actions, diagnostics = refine_antiperiodic_orbit(plant, nodes, controls)
    assert diagnostics['converged'] and not diagnostics['hanging_start_evidence']
    np.testing.assert_array_equal(full[25:50], -full[:25])
    np.testing.assert_array_equal(actions[25:], -actions[:25])
    assert diagnostics['maximum_physical_gap'] < 1e-14


def test_nonlinear_refinement_verifies_both_halves_with_the_actual_step_map():
    plant = OddPlant(True)
    controls = (np.cos(np.arange(10)*np.pi/10)*.3).astype(np.float32).astype(float)
    nodes = linear_antiperiodic_nodes(plant.a, plant.b, controls)
    full, actions, diagnostics = refine_antiperiodic_orbit(plant, nodes, controls)
    assert diagnostics['converged'] and not diagnostics['exactly_feasible']
    assert diagnostics['iterations'] >= 1 and diagnostics['history'][0]['alpha'] == 1.
    assert diagnostics['maximum_physical_gap'] <= 1e-13
    np.testing.assert_array_equal(full[-1], full[0])


def test_single_node_boundary_accumulates_its_antiperiodic_identity():
    a = np.array([[1.2, .1], [0., .8]])
    np.testing.assert_array_equal(antiperiodic_dynamics_matrix(a[None]).toarray(), a+np.eye(2))
