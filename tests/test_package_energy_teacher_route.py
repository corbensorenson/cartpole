from __future__ import annotations

import numpy as np
import pytest

from scripts.package_energy_teacher_route import package_trace


def _payload(*, success: bool = True) -> dict[str, object]:
    rows = []
    for index, mode in enumerate(("hanging_lqr", "hanging_lqr", "energy", "lqr", "lqr")):
        rows.append(
            {
                "time_seconds": 0.02 * (index + 1),
                "mode": mode,
                "action": 0.1 * index,
                "qpos": [0.01 * index, np.pi if index < 2 else 0.1],
                "qvel": [0.0, 0.02 * index],
            }
        )
    return {
        "n_links": 1,
        "controller": {"parameters": {"lqr_scale": 0.8, "lqr_control_cost": 12.0}},
        "episode_results": [{"success": success, "seed": 7, "trace": rows}],
    }


def _spec() -> dict[str, object]:
    return {
        "distribution": {
            "cart_position_abs_max": 1.25,
            "absolute_link_angle_abs_max": 0.15,
            "cart_velocity_abs_max": 0.5,
            "hinge_velocity_rms_max": 0.75,
        }
    }


def test_package_trace_removes_conditioning_and_preserves_route_boundaries() -> None:
    result = package_trace(
        _payload(),
        _spec(),
        episode_index=0,
        post_switch_seconds=0.02,
    )

    assert result["not_solution"] is True
    assert result["teacher"] == {
        "episode_index": 0,
        "seed": 7,
        "source_success": True,
        "conditioning_steps_removed": 2,
        "energy_steps": 1,
        "capture_tail_steps": 1,
        "post_switch_seconds_requested": 0.02,
    }
    assert result["controller"]["controls"] == pytest.approx([0.2, 0.3])
    assert result["controller"]["horizon_steps"] == 2
    assert result["controller"]["lqr_scale"] == 0.8
    assert result["controller"]["lqr_control_cost"] == 12.0
    nominal = np.asarray(result["search"]["nominal_coordinate_states"])
    feedback = np.asarray(result["controller"]["feedback_gains"])
    assert nominal.shape == (3, 4)
    assert feedback.shape == (2, 4)
    np.testing.assert_allclose(feedback, 0.0)
    assert result["selected_state"]["qpos"] == pytest.approx([0.01, np.pi])


def test_package_trace_rejects_failed_or_untraced_teacher() -> None:
    with pytest.raises(ValueError, match="did not pass"):
        package_trace(
            _payload(success=False),
            _spec(),
            episode_index=0,
            post_switch_seconds=1.0,
        )
    payload = _payload()
    payload["episode_results"][0].pop("trace")
    with pytest.raises(ValueError, match="include-traces"):
        package_trace(
            payload,
            _spec(),
            episode_index=0,
            post_switch_seconds=1.0,
        )
