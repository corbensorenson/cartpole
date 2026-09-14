from __future__ import annotations

import numpy as np
import pytest

from scripts.diagnose_capture_geometry import (
    capture_ray_boundary,
    controllability_metrics,
    gain_line_feasibility,
    scaled_capture_state,
)


def test_controllability_metrics_reports_full_rank_system() -> None:
    a = np.array([[1.0, 1.0], [0.0, 1.0]])
    b = np.array([[0.0], [1.0]])

    metrics = controllability_metrics(a, b)

    assert metrics["rank"] == 2
    assert metrics["state_dimension"] == 2
    assert np.isfinite(metrics["condition_number"])
    assert len(metrics["singular_values"]) == 2


def test_controllability_metrics_reports_rank_deficiency() -> None:
    metrics = controllability_metrics(np.eye(2), np.array([[1.0], [0.0]]))

    assert metrics["rank"] == 1
    assert metrics["condition_number"] == float("inf")


def test_gain_line_feasibility_separates_stability_from_action_bound() -> None:
    a = np.array([[1.2]])
    b = np.array([[1.0]])
    gain = np.array([0.4])

    feasible = gain_line_feasibility(a, b, gain, np.array([1.0]))
    infeasible = gain_line_feasibility(a, b, gain, np.array([10.0]))

    assert np.isclose(
        feasible["minimum_stabilizing_feedback_scale_on_gain_line"], 0.5
    )
    assert np.isclose(
        feasible["maximum_initially_nonsaturating_feedback_scale"], 2.5
    )
    assert feasible["stabilizing_and_initially_nonsaturating_scale_exists"]
    assert not infeasible[
        "stabilizing_and_initially_nonsaturating_scale_exists"
    ]
    assert np.isclose(infeasible["stability_to_action_scale_gap_ratio"], 2.0)


def test_scaled_capture_state_wraps_equivalent_upright_branch() -> None:
    qpos, qvel = scaled_capture_state(
        np.array([0.5, 4.0 * np.pi + 0.2, -2.0 * np.pi - 0.4]),
        np.array([0.3, -0.2, 0.1]),
        0.25,
        cart_target=0.1,
    )
    np.testing.assert_allclose(qpos, [0.2, 0.05, -0.1], atol=1.0e-12)
    np.testing.assert_allclose(qvel, [0.075, -0.05, 0.025], atol=1.0e-12)


def test_capture_ray_boundary_refines_first_pass_to_fail(monkeypatch) -> None:
    def fake_rollout(_cfg, *, qpos, **_kwargs):
        scale = float(qpos[0])
        passed = scale <= 0.2
        return {
            "requested_upright_hold_completed": passed,
            "termination_reason": None if passed else "rail_violation",
            "max_upright_streak_seconds": 5.0 if passed else 0.1,
            "max_raw_normalized_action": scale,
            "saturation_fraction": 0.0 if passed else 1.0,
            "max_cart_excursion": scale,
        }

    monkeypatch.setattr(
        "scripts.diagnose_capture_geometry.saturated_feedback_rollout",
        fake_rollout,
    )
    result = capture_ray_boundary(
        {},
        progress=1.0,
        qpos=np.array([1.0, 0.0]),
        qvel=np.zeros(2),
        gain=np.zeros(4),
        feedback_scale=1.0,
        seconds=5.0,
        cart_target=0.0,
        minimum_scale=1.0e-4,
        grid_points=9,
        bisection_steps=24,
    )
    assert not result["full_state_passed"]
    assert result["maximum_origin_connected_pass_scale"] == pytest.approx(
        0.2, rel=1.0e-6
    )
    assert result["required_radial_contraction_factor"] == pytest.approx(
        5.0, rel=1.0e-6
    )
    assert not result["nonmonotonic_pass_detected"]
