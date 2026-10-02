import json

import numpy as np
import pytest

from scripts.search_fddp_capture import (
    load_terminal_target,
    lift_nominal_trajectory,
    rebuild_feedback_warm_start,
    warm_start_diagnostics,
    load_solver_feedback,
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
