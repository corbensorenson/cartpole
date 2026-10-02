import json
from types import SimpleNamespace

import numpy as np

from scripts.search_swingup_handoff_bank_cem import (
    handoff_distance,
    load_feedback_route,
    load_handoff_targets,
    load_full_resolution_center,
)


def test_handoff_distance_uses_the_nearest_measured_state(tmp_path):
    artifact = tmp_path / "bank.json"
    artifact.write_text(
        json.dumps(
            {
                "states": [
                    {
                        "qpos": [0.0, 0.0, 0.0],
                        "qvel": [0.0, 0.0, 0.0],
                        "absolute_angles": [0.0, 0.0],
                    },
                    {
                        "qpos": [1.0, 0.0, 0.0],
                        "qvel": [0.0, 0.0, 0.0],
                        "absolute_angles": [0.0, 0.0],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    targets, metadata = load_handoff_targets(str(artifact), 2)

    assert metadata["state_count"] == 2
    assert targets.shape == (2, 6)
    np.testing.assert_allclose(targets[0], 0.0)
    np.testing.assert_allclose(
        handoff_distance(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            targets,
            np.ones(6),
        ),
        0.0,
    )


def test_trace_backed_warm_start_keeps_post_route_actions(tmp_path):
    artifact = tmp_path / "route.json"
    artifact.write_text(
        json.dumps(
            {
                "result": {
                    "trajectory": [
                        {"time_seconds": 0.02, "action": -0.5},
                        {"time_seconds": 0.04, "action": 0.25},
                        {"time_seconds": 0.06, "action": 0.75},
                    ]
                }
            }
        ),
        encoding="utf-8",
    )

    warm_start = load_full_resolution_center(
        str(artifact),
        step_count=4,
        seconds=0.08,
        env=SimpleNamespace(dt=0.02),
    )

    np.testing.assert_allclose(warm_start, [-0.5, -0.5, 0.25, 0.75])


def test_feedback_route_loader_preserves_closed_loop_arrays(tmp_path):
    artifact = tmp_path / "route.json"
    controls = [0.1, -0.2]
    state_dim = 6
    artifact.write_text(
        json.dumps(
            {
                "controller": {
                    "controls": controls,
                    "feedback_gains": np.zeros((2, state_dim)).tolist(),
                    "periodic_coordinate_errors": True,
                },
                "search": {
                    "nominal_coordinate_states": np.zeros((3, state_dim)).tolist(),
                },
            }
        ),
        encoding="utf-8",
    )

    route = load_feedback_route(artifact, 2)

    assert route["controls"].shape == (2,)
    assert route["feedback_gains"].shape == (2, state_dim)
    assert route["nominal_coordinate_states"].shape == (3, state_dim)
    assert route["periodic_coordinate_errors"] is True
