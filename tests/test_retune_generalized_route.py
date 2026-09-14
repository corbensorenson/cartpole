from __future__ import annotations

import pytest

from scripts.retune_generalized_route import retune_route


def route() -> dict[str, object]:
    return {
        "controller": {
            "lqr_scale": 1.0,
            "lqr_control_cost": 50.0,
            "feedback_gains": [[1.0, 2.0], [3.0, 4.0]],
            "materialized_swing_prefix_steps": 1,
            "materialized_lqr_tail_steps": 1,
        },
        "search": {"nominal_coordinate_states": [[0.0]]},
    }


def test_retune_route_changes_only_requested_capture_scalars() -> None:
    source = route()
    result = retune_route(source, lqr_scale=1.5, lqr_control_cost=None)
    assert result["controller"]["lqr_scale"] == pytest.approx(1.5)
    assert result["controller"]["lqr_control_cost"] == pytest.approx(50.0)
    assert source["controller"]["lqr_scale"] == pytest.approx(1.0)
    assert result["capture_retuning"]["settings"] == {"lqr_scale": 1.5}


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_retune_route_rejects_invalid_settings(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        retune_route(route(), lqr_scale=value, lqr_control_cost=None)


def test_retune_route_requires_a_setting() -> None:
    with pytest.raises(ValueError, match="at least one"):
        retune_route(route(), lqr_scale=None, lqr_control_cost=None)


def test_retune_route_scales_swing_and_tail_feedback_independently() -> None:
    result = retune_route(
        route(), swing_feedback_scale=0.5, tail_feedback_scale=2.0
    )
    assert result["controller"]["feedback_gains"] == [[0.5, 1.0], [6.0, 8.0]]
    assert result["capture_retuning"]["settings"] == {
        "swing_feedback_scale": 0.5,
        "tail_feedback_scale": 2.0,
    }


def test_segment_retuning_requires_a_materialized_boundary() -> None:
    source = route()
    source["controller"].pop("materialized_swing_prefix_steps")
    with pytest.raises(ValueError, match="materialized route boundary"):
        retune_route(source, swing_feedback_scale=0.75)
