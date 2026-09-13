from __future__ import annotations

import numpy as np

from scripts.search_multiple_shooting_capture import saved_shooting_nodes


def test_saved_fddp_trajectory_projects_to_segment_end_nodes() -> None:
    states = np.arange(11 * 4, dtype=np.float64).reshape(11, 4)
    nodes = saved_shooting_nodes(
        {"search": {"nominal_coordinate_states": states.tolist()}},
        horizon_steps=10,
        segment_steps=2,
        state_size=4,
    )
    assert nodes is not None
    np.testing.assert_array_equal(nodes, states[[2, 4, 6, 8, 10]])


def test_saved_native_nodes_take_precedence() -> None:
    nodes = np.arange(20, dtype=np.float64).reshape(5, 4)
    result = saved_shooting_nodes(
        {
            "search": {
                "node_states": nodes.tolist(),
                "nominal_coordinate_states": np.zeros((11, 4)).tolist(),
            }
        },
        horizon_steps=10,
        segment_steps=2,
        state_size=4,
    )
    assert result is not None
    np.testing.assert_array_equal(result, nodes)


def test_incompatible_saved_trajectory_is_ignored() -> None:
    assert (
        saved_shooting_nodes(
            {"search": {"nominal_coordinate_states": [[0.0, 1.0]]}},
            horizon_steps=10,
            segment_steps=2,
            state_size=4,
        )
        is None
    )
