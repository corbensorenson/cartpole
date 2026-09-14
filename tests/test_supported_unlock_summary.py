from __future__ import annotations

from scripts.summarize_supported_unlock_frontier import select_boundary_trials


def test_select_boundary_trials_keeps_frontier_and_nearest_rejection() -> None:
    trials = [
        {"trial": 1, "proposed_progress": 0.5, "accepted": True},
        {"trial": 2, "proposed_progress": 0.8, "accepted": False},
        {"trial": 3, "proposed_progress": 0.7, "accepted": True},
        {"trial": 4, "proposed_progress": 0.75, "accepted": False},
    ]

    accepted, rejected = select_boundary_trials(trials, progress=0.7)

    assert accepted["trial"] == 3
    assert rejected is not None
    assert rejected["trial"] == 4
