import json

import numpy as np
import pytest

from scripts.search_fddp_capture import (
    load_terminal_target,
    lift_nominal_trajectory,
    rebuild_feedback_warm_start,
    warm_start_diagnostics,
    load_solver_feedback,
    capture_interval_cost,
    upright_branch_target,
    feedback_capture_tail,
    capture_node_constraints,
)


def test_explicit_solver_feedback_selection_does_not_use_scaled_applied_gains():
    payload = dict(controller=dict(feedback_gains=[[0., 0.]], solver_feedback_gains=[[-2., 3.]]))
    selected = load_solver_feedback(payload, (1, 2))
    np.testing.assert_array_equal(selected, [[-2., 3.]])
    selected[0, 0] = 8.
    assert payload["controller"]["solver_feedback_gains"][0][0] == -2.
    with pytest.raises(ValueError, match="shape"):
        load_solver_feedback(payload, (2, 2))
    payload["controller"]["solver_feedback_gains"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        load_solver_feedback(payload, (1, 2))


def test_continuous_nominal_lift_preserves_rotation_and_aligns_launch_branch():
    from types import SimpleNamespace
    from gcartpole.ilqr import MujocoTransition

    transition = MujocoTransition.__new__(MujocoTransition)
    transition.env = SimpleNamespace(n=1)
    transition.continuous_angles = True
    transition.coordinate_transform = np.diag([1.0, 2.0, 1.0, 1.0])
    transition.inverse_transform = np.linalg.inv(transition.coordinate_transform)
    states = np.zeros((4, 4))
    states[:, 1] = 2 * np.array([-np.pi, -2.0, 0.0, 2.0])
    start = np.array([0.0, 2 * np.pi, 0.0, 0.0])
    lifted = lift_nominal_trajectory(transition, states, start)
    np.testing.assert_allclose(lifted[0], start)
    assert np.all(np.diff(lifted[:, 1]) > 0)
    np.testing.assert_allclose(lifted[:, 1], states[:, 1] + 4 * np.pi)


def test_aligned_continuous_lift_preserves_exact_coordinate_states():
    from types import SimpleNamespace
    from gcartpole.ilqr import MujocoTransition

    transition = MujocoTransition.__new__(MujocoTransition)
    transition.env = SimpleNamespace(n=2)
    transition.continuous_angles = True
    block = np.array([[1.7, 0., 0.], [0., 6.7, 0.], [0., 6.7, 6.7]])
    transition.coordinate_transform = np.kron(np.eye(2), block)
    transition.inverse_transform = np.linalg.inv(transition.coordinate_transform)
    physical = np.random.default_rng(12).normal(size=(10, 6)) * .01
    states = physical @ transition.coordinate_transform.T
    lifted = lift_nominal_trajectory(transition, states, states[0])
    # A serially rolled coordinate trajectory must remain exactly feasible
    # when no angle branch needs changing, including its velocity entries.
    np.testing.assert_array_equal(lifted, states)


def test_physical_continuous_lift_aligns_branch_without_a_transform():
    from types import SimpleNamespace
    from gcartpole.ilqr import MujocoTransition

    transition = MujocoTransition.__new__(MujocoTransition)
    transition.env = SimpleNamespace(n=1)
    transition.continuous_angles = True
    transition.coordinate_transform = None
    transition.inverse_transform = None
    states = np.zeros((4, 4))
    states[:, 1] = [-np.pi, -2., 0., 2.]
    states[:, 3] = [.123, .456, .789, .321]
    start = states[0].copy()
    start[1] = np.pi
    lifted = lift_nominal_trajectory(transition, states, start)
    np.testing.assert_array_equal(lifted[:, [0, 2, 3]], states[:, [0, 2, 3]])
    np.testing.assert_allclose(lifted[:, 1], states[:, 1] + 2*np.pi)
    np.testing.assert_array_equal(lift_nominal_trajectory(transition, states, states[0]), states)


def test_capture_interval_factor_transports_pose_and_directional_value_without_gram_evaluation():
    from gcartpole.ilqr import QuadraticTrajectoryCost,stage_cost
    mapping=np.diag([.8,2.]);value=np.array([[1e12,-1e12],[0.,1.]])
    base=QuadraticTrajectoryCost(np.eye(2),np.eye(2),.1,100.,200.,0.,wrap_angles=False,
                                 terminal_target=np.array([.1,.1]))
    capture=capture_interval_cost(base,np.diag([1.,3.]),mapping,2.,value,4.,2.)
    state=np.array([1.1,1.1]);error=state-base.terminal_target
    expected=np.sum((np.sqrt(2.)*np.diag([1.,np.sqrt(3.)])@mapping@error)**2)+2*np.sum((value@error)**2)
    np.testing.assert_allclose(stage_cost(state,0.,capture,0),.5*expected,rtol=1e-14)
    with pytest.raises(ValueError,match='factored'):
        capture_interval_cost(base,np.eye(2),mapping,2.,None,1.,2.)


def test_upright_target_preserves_the_inherited_winding_before_free_tail_initialization():
    from types import SimpleNamespace
    from gcartpole.ilqr import MujocoTransition
    t=MujocoTransition.__new__(MujocoTransition);t.env=SimpleNamespace(n=1)
    t.continuous_angles=True;t.coordinate_transform=None;t.inverse_transform=None
    anchor=np.array([.1,-2*np.pi+.01,0.,.02])
    target=upright_branch_target(t,anchor)
    np.testing.assert_array_equal(target,[0.,-2*np.pi,0.,0.])
    # A freely spinning initialization tail cannot define the desired winding.
    assert not np.array_equal(target,upright_branch_target(t,[.1,-6*np.pi,0.,.02]))


def test_sparse_capture_constraints_map_absolute_angles_and_preserve_the_target_winding():
    from types import SimpleNamespace
    t = SimpleNamespace(env=SimpleNamespace(n=2), inverse_transform=np.diag([.5, 1/3, .25, .2, 1/6, 1/7]))
    target = np.array([0., 6*np.pi, 0., 0., 0., 0.])
    matrix, lower, upper = capture_node_constraints(t, target, 2, 4, .14)
    physical = np.array([0., 2*np.pi+.01, -.02, 0., 0., 0.])
    coordinates = physical/np.diag(t.inverse_transform)
    np.testing.assert_allclose(matrix @ coordinates, [2*np.pi+.01, 2*np.pi-.01])
    assert np.all(np.isneginf(lower[0])) and np.all(np.isposinf(upper[0]))
    np.testing.assert_allclose(lower[1:], np.full((3, 2), 2*np.pi-.14))
    np.testing.assert_allclose(upper[1:], np.full((3, 2), 2*np.pi+.14))


@pytest.mark.parametrize('normalized', [False, True])
def test_feedback_tail_saves_applied_float32_actions_and_unprojected_serial_states(normalized):
    from types import SimpleNamespace
    from gcartpole.angles import wrap_angle

    class Plant:
        env = SimpleNamespace(n=1)
        transform = np.diag([2., 3., 4., 5.]) if normalized else np.eye(4)
        inverse_transform = np.linalg.inv(transform) if normalized else None

        def to_physical(self, x):
            return np.linalg.solve(self.transform, x) if normalized else x.copy()

        def __call__(self, x, u):
            physical = self.to_physical(x)
            return self.transform @ (physical + np.array([.1, 0., .3, .02])*u)

    plant = Plant()
    physical = np.array([.2, 2*np.pi+.01, .1, .02])
    start = plant.transform @ physical
    gain = np.array([.5, 2., .1, .3])
    controls, nodes, gains = feedback_capture_tail(plant, start, gain, 3)
    np.testing.assert_array_equal(nodes[0], start)
    np.testing.assert_array_equal(controls, controls.astype(np.float32).astype(float))
    for step, action in enumerate(controls):
        expected_error = plant.to_physical(nodes[step])
        expected_error[1] = wrap_angle(expected_error[1])
        assert action == float(np.float32(-gain @ expected_error))
        np.testing.assert_array_equal(nodes[step+1], plant(nodes[step], action))
    np.testing.assert_allclose(gains, np.tile(-gain @ np.linalg.inv(plant.transform), (3, 1)))
    assert abs(plant.to_physical(nodes[-1])[1]) > 6.


def test_saturated_feedback_tail_has_zero_local_feedback_derivative():
    from types import SimpleNamespace
    class Plant:
        env = SimpleNamespace(n=1)
        inverse_transform = None
        def to_physical(self, x):
            return x.copy()
        def __call__(self, x, u):
            return x + np.array([u, 0., 0., 0.])
    controls, nodes, gains = feedback_capture_tail(Plant(), np.array([10., 0., 0., 0.]), np.array([1., 0., 0., 0.]), 2)
    np.testing.assert_array_equal(controls, [-1., -1.])
    np.testing.assert_array_equal(gains, np.zeros((2, 4)))
    np.testing.assert_array_equal(nodes[:, 0], [10., 9., 8.])


@pytest.mark.parametrize('terminal_weight', [0, 1000])
def test_hard_rail_cli_does_not_silently_ignore_unsupported_native_optimizer(monkeypatch, terminal_weight):
    from scripts.search_fddp_capture import main
    monkeypatch.setattr('sys.argv', ['search_fddp_capture.py', '--state-index', 'selected', '--enforce-rail-during-search', '--out', 'unused.json', '--terminal-weight', str(terminal_weight)])
    with pytest.raises(ValueError, match='square-root optimization run'):
        main()


@pytest.mark.parametrize('terminal_weight', ['-1', 'nan'])
def test_terminal_capture_weight_rejects_invalid_values(monkeypatch, terminal_weight):
    from scripts.search_fddp_capture import main
    monkeypatch.setattr('sys.argv', ['search_fddp_capture.py', '--state-index', 'selected', '--out', 'unused.json', '--terminal-weight', terminal_weight])
    with pytest.raises(ValueError, match='terminal capture value weight must be finite and nonnegative'):
        main()


@pytest.mark.parametrize('options', [[], ['--optimizer', 'sqrt-ilqr'], ['--optimizer', 'sqrt-fddp', '--initial-controller', 'unused.json', '--replay-only']])
def test_suffix_cli_rejects_unsupported_or_missing_inherited_prefix(monkeypatch, options):
    from scripts.search_fddp_capture import main
    monkeypatch.setattr('sys.argv', ['search_fddp_capture.py', '--state-index', 'selected', '--out', 'unused.json', '--optimize-suffix-start-seconds', '6', *options])
    with pytest.raises(ValueError, match='inherited square-root optimization run'):
        main()


def test_rebuilt_feedback_controls_reproduce_the_rebuilt_states():
    transition = lambda state, action: state + action
    controls, states = rebuild_feedback_warm_start(
        transition,
        np.array([1.0]),
        np.zeros(2),
        np.zeros((3, 1)),
        -np.ones((2, 1)),
    )
    np.testing.assert_allclose(controls, [-1.0, 0.0])
    np.testing.assert_allclose(states[:, 0], [1.0, 0.0, 0.0])
    assert warm_start_diagnostics(transition, states[0], controls, states)["is_feasible"]
    with pytest.raises(ValueError, match="state/control defect"):
        warm_start_diagnostics(
            transition, states[0], np.zeros(2), states, require_feasible=True
        )


def test_feasibility_checks_initial_state_and_euclidean_angle_branch():
    transition = lambda state, action: state.copy()
    start = np.zeros(2)
    states = np.array([[0.0, 0.0], [0.0, 2 * np.pi]])
    result = warm_start_diagnostics(transition, start, np.zeros(1), states)
    assert not result["is_feasible"]
    assert result["maximum_dynamics_defect"] == 2 * np.pi
    states[:] = 1.0
    result = warm_start_diagnostics(transition, start, np.zeros(1), states)
    assert not result["is_feasible"]
    assert result["initial_state_gap"] == 1.0


def test_warm_start_rejects_nonfinite_or_out_of_bounds_actions():
    transition = lambda state, action: state + action
    for controls in (np.array([np.nan]), np.array([1.01])):
        with pytest.raises(ValueError):
            warm_start_diagnostics(transition, np.zeros(1), controls, np.zeros((2, 1)))


def test_tolerated_defect_must_not_disable_solver_gap_restoration():
    transition = lambda state, action: 1000*state+action
    states = np.array([[0.], [1e-9], [1e-6], [1e-3]])
    result = warm_start_diagnostics(transition, states[0], np.zeros(3), states)
    assert result["is_feasible"]
    assert not result["is_exactly_feasible"]
    # The true serial trajectory stays zero, despite a tolerated local gap
    # and a nominal endpoint that differs by a million times that gap.
    with pytest.raises(ValueError, match="solver's flag requires zero defects"):
        warm_start_diagnostics(transition, states[0], np.zeros(3), states, require_feasible=True)


def test_load_terminal_target_selects_last_saved_handoff_state(tmp_path):
    artifact = tmp_path / "handoff.json"
    artifact.write_text(
        json.dumps(
            {
                "search": {
                    "nominal_coordinate_states": [
                        [0.0, 1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0, 7.0],
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    target, metadata = load_terminal_target(str(artifact), "last", 4)

    np.testing.assert_allclose(target, [4.0, 5.0, 6.0, 7.0])
    assert metadata["index"] == 1
    assert metadata["state_count"] == 2
