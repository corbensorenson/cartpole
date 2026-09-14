from __future__ import annotations

import numpy as np

from scripts.diagnose_capture_geometry import (
    controllability_metrics,
    gain_line_feasibility,
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
