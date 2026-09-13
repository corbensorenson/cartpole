from __future__ import annotations

from scripts.summarize_generalized_homotopy import summarize


def test_summary_separates_accepted_rail_demand_from_failed_excursion() -> None:
    base = {
        "from_progress": 0.0,
        "step": 0.1,
        "dimensionless": {
            "length_fractions": [0.5, 0.5],
            "mass_fractions": [0.5, 0.5],
        },
    }
    payload = {
        "progress": 0.1,
        "next_step": 0.05,
        "status": "running",
        "trials": [
            {
                **base,
                "trial": 1,
                "proposed_progress": 0.1,
                "accepted": True,
                "outcome": {
                    "termination_reason": "time_limit",
                    "max_upright_streak_seconds": 6.0,
                    "max_cart_excursion": 2.0,
                    "rail_requirement": {"required_rail_ratio": 0.8},
                },
            },
            {
                **base,
                "trial": 2,
                "proposed_progress": 0.2,
                "accepted": False,
                "outcome": {
                    "termination_reason": "rail_violation",
                    "max_upright_streak_seconds": 0.0,
                    "max_cart_excursion": 3.1,
                    "rail_requirement": {"required_rail_ratio": 1.1},
                },
            },
        ],
    }
    result = summarize(payload)
    assert result["frontier"]["latest_required_rail_ratio"] == 0.8
    assert result["frontier"]["rail_terminated_rejections"] == 1
    assert len(result["accepted_curve"]) == 1
    assert len(result["rejected_curve"]) == 1
