from __future__ import annotations

import numpy as np
import pytest

from gcartpole.capture_terminal import feedback_horizon_metric


def test_feedback_horizon_metric_matches_direct_closed_loop_rollout() -> None:
    a = np.array([[1.1, 0.2], [0.0, 0.9]])
    b = np.array([[0.5], [0.25]])
    gain = np.array([0.6, 0.3])
    transform = np.diag([2.0, 0.5])
    state = np.array([0.4, -0.2])
    metric = feedback_horizon_metric(
        a,
        b,
        gain,
        transform,
        horizon_steps=4,
        action_limit=0.8,
        action_weight=3.0,
        terminal_weight=2.0,
    )

    closed_loop = a - b @ gain[None, :]
    current = state.copy()
    actions = []
    for _ in range(4):
        actions.append(float(-(gain @ current)))
        current = closed_loop @ current
    expected = 3.0 * np.sum((np.asarray(actions) / 0.8) ** 2)
    expected += 2.0 * np.sum((transform @ current) ** 2)

    evaluation = metric.evaluate(state)
    np.testing.assert_allclose(evaluation["raw_actions"], actions)
    assert np.sum(metric.residual(state) ** 2) == pytest.approx(expected)
    assert evaluation["objective"] == pytest.approx(expected)
    assert evaluation["objective"] == pytest.approx(
        float(state @ metric.matrix @ state)
    )


def test_feedback_horizon_metric_is_symmetric_positive_semidefinite() -> None:
    metric = feedback_horizon_metric(
        np.array([[1.2]]),
        np.array([[1.0]]),
        np.array([0.4]),
        np.array([[2.0]]),
        horizon_steps=5,
    )

    np.testing.assert_allclose(metric.matrix, metric.matrix.T)
    assert np.min(np.linalg.eigvalsh(metric.matrix)) >= 0.0
    assert metric.to_dict()["horizon_steps"] == 5


def test_feedback_horizon_metric_preserves_objective_in_optimizer_coordinates() -> None:
    metric = feedback_horizon_metric(
        np.array([[1.0, 0.1], [0.0, 0.95]]),
        np.array([[0.0], [0.2]]),
        np.array([0.7, 0.25]),
        np.diag([0.5, 2.0]),
        horizon_steps=6,
        action_limit=1.5,
    )
    coordinate_transform = np.array([[2.0, 0.25], [0.0, 0.5]])
    state = np.array([0.3, -0.4])
    optimizer_state = coordinate_transform @ state

    coordinate_residual = metric.residual_map_in_coordinates(
        coordinate_transform
    )
    coordinate_matrix = metric.matrix_in_coordinates(coordinate_transform)

    np.testing.assert_allclose(coordinate_residual @ optimizer_state, metric.residual(state))
    assert float(optimizer_state @ coordinate_matrix @ optimizer_state) == pytest.approx(
        metric.evaluate(state)["objective"]
    )
    np.testing.assert_allclose(coordinate_matrix, coordinate_matrix.T)


def test_feedback_horizon_metric_rejects_shape_and_scale_errors() -> None:
    with pytest.raises(ValueError, match="one actuator"):
        feedback_horizon_metric(
            np.eye(2),
            np.eye(2),
            np.ones(2),
            np.eye(2),
            horizon_steps=2,
        )
    with pytest.raises(ValueError, match="positive integer"):
        feedback_horizon_metric(
            np.eye(1),
            np.ones((1, 1)),
            np.ones(1),
            np.eye(1),
            horizon_steps=0,
        )

    metric = feedback_horizon_metric(
        np.eye(2),
        np.ones((2, 1)),
        np.ones(2),
        np.eye(2),
        horizon_steps=1,
    )
    with pytest.raises(ValueError, match="shape"):
        metric.residual_map_in_coordinates(np.eye(3))
    with pytest.raises(ValueError, match="invertible"):
        metric.matrix_in_coordinates(np.zeros((2, 2)))
