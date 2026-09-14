from __future__ import annotations

from scripts.verify_generalized_similarity import (
    verify_refined_similarity_cases,
    verify_route_similarity_cases,
    verify_similarity_grid,
)


def test_published_similarity_grid_passes_closed_loop_contract() -> None:
    rows = verify_similarity_grid(minimum_episodes=20)

    assert [row["label"] for row in rows] == [
        "l05_m05",
        "l05_m2",
        "l2_m05",
        "l2_m2",
    ]
    assert all(row["passed"] for row in rows)
    assert sum(row["successes"] for row in rows) == 80


def test_published_two_link_similarity_transfer_preserves_feedback_and_passes() -> None:
    rows = verify_route_similarity_cases(minimum_episodes=20)

    assert len(rows) == 1
    assert rows[0]["passed"]
    assert rows[0]["successes"] == 20
    assert rows[0]["prediction_execution_agreements"] == 20
    assert rows[0]["coordinate_feedback_invariance_max_abs_error"] < 1e-12
    assert not rows[0]["exact_open_loop_warm_start_feasible"]


def test_published_seven_link_similarity_uses_generic_exact_target_repair() -> None:
    rows = verify_refined_similarity_cases(minimum_episodes=20)

    assert len(rows) == 1
    assert rows[0]["passed"]
    assert rows[0]["successes"] == 20
    assert rows[0]["prediction_execution_agreements"] == 20
    assert rows[0]["unrefined_transfer_success_rate"] == 0.0
    assert rows[0]["feedback_rms_normalization_scale"] > 0.0
