from __future__ import annotations

import numpy as np

from scripts.export_route_handoff_dataset import handoff_metrics, split_name
from scripts.extract_search_handoff_state import wrap_hinge_coordinates
from scripts.extract_trajectory_states import make_state


def test_handoff_metrics_preserve_serial_angle_and_rate_semantics() -> None:
    metrics = handoff_metrics(
        {
            "qpos": [0.25, 0.10, -0.20, 0.10],
            "qvel": [0.05, 0.20, -0.10, 0.30],
        }
    )
    np.testing.assert_allclose(metrics["absolute_angles"], [0.10, -0.10, 0.0])
    np.testing.assert_allclose(metrics["absolute_angular_velocity_rms"], np.sqrt(np.mean([0.2**2, 0.1**2, 0.4**2])))
    assert metrics["cart_position"] == 0.25
    assert metrics["cart_velocity"] == 0.05


def test_split_policy_is_disjoint_and_exhaustive() -> None:
    labels = [split_name(i, train_count=8, validation_count=2, total=12) for i in range(12)]
    assert labels == ["train"] * 8 + ["validation"] * 2 + ["test"] * 2


def test_handoff_angle_wrap_preserves_rates_and_marks_normalization() -> None:
    state = {"qpos": [0.2, 2.0 * np.pi + 0.25, -2.0 * np.pi - 0.5], "qvel": [0.1, 0.2, -0.3]}

    normalized = wrap_hinge_coordinates(state)

    np.testing.assert_allclose(normalized["qpos"], [0.2, 0.25, -0.5])
    np.testing.assert_allclose(normalized["qvel"], state["qvel"])
    assert normalized["hinge_coordinates_wrapped"] is True
    np.testing.assert_allclose(state["qpos"], [0.2, 2.0 * np.pi + 0.25, -2.0 * np.pi - 0.5])


def test_trajectory_state_extractor_wraps_only_coordinates() -> None:
    state = make_state(
        {
            "qpos": [0.0, 2.0 * np.pi + 0.25, -2.0 * np.pi - 0.5],
            "qvel": [0.1, 0.2, -0.3],
        },
        "source.json",
        3,
        wrap_hinge_angles=True,
    )

    np.testing.assert_allclose(state["qpos"], [0.0, 0.25, -0.5])
    np.testing.assert_allclose(state["qvel"], [0.1, 0.2, -0.3])
    assert state["hinge_coordinates_wrapped"] is True
