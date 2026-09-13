from __future__ import annotations

import pytest

from scripts.evaluate_generalized_adaptive_library import (
    conditioning_schedule,
    select_candidate,
)


def test_conditioning_schedule_is_natural_time_scaled_quantized_and_unique() -> None:
    assert conditioning_schedule(0.5, 0.02, [1.0, 0.0, 0.49, 0.5]) == (
        (0.0, 0.0, 0),
        (0.49, 0.24, 12),
        (1.0, 0.5, 25),
    )


def test_conditioning_schedule_rejects_negative_ratio() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        conditioning_schedule(0.5, 0.02, [0.0, -1.0])


def test_selector_prefers_shortest_success_then_rail() -> None:
    candidates = [
        {
            "conditioning_seconds": 0.0,
            "route_index": 0,
            "result": {
                "success": False,
                "max_upright_streak_seconds": 0.0,
                "max_cart_excursion": 4.0,
            },
        },
        {
            "conditioning_seconds": 0.2,
            "route_index": 1,
            "result": {
                "success": True,
                "max_upright_streak_seconds": 18.0,
                "max_cart_excursion": 3.5,
            },
        },
        {
            "conditioning_seconds": 0.4,
            "route_index": 0,
            "result": {
                "success": True,
                "max_upright_streak_seconds": 19.0,
                "max_cart_excursion": 3.0,
            },
        },
    ]
    assert select_candidate(candidates) == 1
