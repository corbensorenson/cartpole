from __future__ import annotations

import numpy as np

from scripts.materialize_generalized_rollout import materialize_rollout


def test_materialize_rollout_uses_executed_actions_states_and_lqr_tail() -> None:
    payload = {
        "selected_state": {"qpos": [1.0], "qvel": [2.0]},
        "controller": {
            "controls": [0.1],
            "feedback_gains": [[3.0, 4.0]],
            "horizon_steps": 1,
            "policy_dt": 0.02,
        },
        "result": {
            "success": True,
            "trajectory": [
                {"step": 1, "action": 0.25, "qpos": [1.5], "qvel": [2.5]},
                {"step": 2, "action": -0.5, "qpos": [2.0], "qvel": [3.0]},
            ],
        },
    }
    result = materialize_rollout(payload, np.eye(2), np.array([-5.0, -6.0]))
    assert result["controller"]["controls"] == [0.25, -0.5]
    assert result["controller"]["feedback_gains"] == [[3.0, 4.0], [-5.0, -6.0]]
    assert result["search"]["nominal_coordinate_states"] == [
        [1.0, 2.0],
        [1.5, 2.5],
        [2.0, 3.0],
    ]
    assert result["controller"]["horizon_steps"] == 2
    assert result["controller"]["horizon_seconds"] == 0.04
    assert result["controller"]["periodic_coordinate_errors"] is True
    assert result["controller"]["materialized_swing_prefix_steps"] == 1
    assert result["controller"]["materialized_lqr_tail_steps"] == 1
    assert "result" not in result
