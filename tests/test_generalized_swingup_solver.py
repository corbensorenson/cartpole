import json
from pathlib import Path

import numpy as np

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.generalized_solver import (
    setup_from_config,
    split_state_lift_matrix,
    split_state_projection,
)
from scripts.generalized_swingup_solver import (
    coordinate_transform,
    split_continuation_config,
    split_warm_start,
    uniform_config,
)
from scripts.run_split_count_homotopy import (
    controller_link_count,
    freeze_progress_config,
    prioritized_waypoint_steps,
)
from scripts.verify_split_count_homotopy import verify


def test_split_command_builders_make_valid_count_continuation_and_invariant_gain():
    base = load_config("configs/swingup7_uniform.yaml")
    source_cfg = uniform_config(base, 2)
    target_cfg = uniform_config(base, 3)
    continuation, embedding = split_continuation_config(source_cfg, target_cfg)
    assert continuation["env"]["joint_lock_impedance_schedule"] == "log_compliance"

    source = setup_from_config(source_cfg)
    start = setup_from_config(continuation, progress=0.0)
    end = setup_from_config(continuation, progress=1.0)
    np.testing.assert_array_equal(embedding["split_counts"], [1, 2])
    np.testing.assert_array_equal(embedding["source_joint_locks"], [0.0, 0.0, 1.0])
    assert embedding["compatibility"]["global_dynamic_similarity"]
    assert start.n_links == end.n_links == 3
    np.testing.assert_allclose(start.lengths, [1.5, 0.75, 0.75])
    np.testing.assert_allclose(end.lengths, [1.0, 1.0, 1.0])

    steps = 4
    source_dim = 2 * (source.n_links + 1)
    source_transform = coordinate_transform(source.n_links, None)
    target_transform = coordinate_transform(start.n_links, None)
    payload = {
        "controller": {
            "controls": np.linspace(-0.2, 0.2, steps).tolist(),
            "feedback_gains": np.arange(steps * source_dim, dtype=float)
            .reshape(steps, source_dim)
            .tolist(),
        },
        "search": {
            "nominal_coordinate_states": np.arange(
                (steps + 1) * source_dim, dtype=float
            )
            .reshape(steps + 1, source_dim)
            .tolist(),
        },
    }
    warm = split_warm_start(
        payload,
        source,
        start,
        embedding,
        source_transform,
        target_transform,
    )
    assert warm["embedding"]["feedback_invariance_max_abs_error"] < 1.0e-12
    assert len(warm["controller"]["feedback_gains"][0]) == 8
    assert len(warm["search"]["nominal_coordinate_states"][0]) == 8
    assert warm["selected_state"]["qpos"][3] == 0.0


def test_split_release_freezes_every_profile_and_scheduled_rail(tmp_path):
    base = load_config("configs/swingup7_uniform.yaml")
    source_cfg = uniform_config(base, 2)
    target_cfg = uniform_config(base, 3)
    target_cfg["env"]["rail_limit_start"] = 4.0
    target_cfg["env"]["rail_limit_end"] = 2.0
    continuation, _embedding = split_continuation_config(source_cfg, target_cfg)

    frozen = freeze_progress_config(continuation, 0.25, "quarter_release")
    assert frozen["env"]["rail_limit"] == 3.5
    assert "rail_limit_start" not in frozen["env"]
    assert "rail_limit_end" not in frozen["env"]
    assert "plant_progress" not in frozen["env"]
    assert frozen["experiment"]["name"] == "quarter_release"
    for profile in (
        "lengths",
        "masses",
        "damping",
        "frictionloss",
        "joint_stiffness",
        "joint_lock",
    ):
        np.testing.assert_allclose(
            frozen["morphology"][f"{profile}_start"],
            frozen["morphology"][f"{profile}_end"],
        )
    start = setup_from_config(frozen, progress=0.0)
    end = setup_from_config(frozen, progress=1.0)
    np.testing.assert_allclose(start.lengths, end.lengths)
    np.testing.assert_allclose(start.masses, end.masses)
    np.testing.assert_allclose(start.joint_damping, end.joint_damping)
    np.testing.assert_allclose(
        frozen["morphology"]["joint_lock_start"], [0.0, 0.0, 0.75]
    )

    controller = tmp_path / "controller.json"
    controller.write_text(
        json.dumps({"controller": {"feedback_gains": [[0.0] * 8]}}),
        encoding="utf-8",
    )
    assert controller_link_count(controller) == 3


def test_rigid_split_start_matches_source_one_step_dynamics():
    base = load_config("configs/swingup7_uniform.yaml")
    source_cfg = uniform_config(base, 2)
    target_cfg = uniform_config(base, 3)
    continuation, embedding = split_continuation_config(source_cfg, target_cfg)
    target_cfg = freeze_progress_config(continuation, 0.0, "rigid_split_probe")
    assignments = np.asarray(embedding["segment_source_links"], dtype=np.int64)
    lift = split_state_lift_matrix(2, assignments)
    projection = split_state_projection(
        lift, np.asarray(embedding["source_lengths"]) / 3.0
    )
    source_state = np.array([0.1, 0.3, -0.2, 0.2, -0.1, 0.4])
    target_state = lift @ source_state
    for cfg, state in ((source_cfg, source_state), (target_cfg, target_state)):
        nq = int(cfg["env"]["n_links"]) + 1
        cfg["env"].update(
            init_mode="fixed_state",
            init_qpos=state[:nq].tolist(),
            init_qvel=state[nq:].tolist(),
            init_cart_noise=0.0,
            init_cart_vel_noise=0.0,
            init_angle_noise=0.0,
            init_vel_noise=0.0,
        )
    source_env = NLinkCartPoleEnv(source_cfg, progress=1.0, seed=0)
    target_env = NLinkCartPoleEnv(target_cfg, progress=1.0, seed=0)
    try:
        source_env.reset(seed=0)
        target_env.reset(seed=0)
        source_env.step([0.2])
        target_env.step([0.2])
        source_next = np.r_[source_env.data.qpos, source_env.data.qvel]
        target_next = np.r_[target_env.data.qpos, target_env.data.qvel]
        np.testing.assert_allclose(
            projection @ target_next, source_next, rtol=0.0, atol=2.0e-5
        )
        inserted_dof = 2 + int(np.flatnonzero(np.diff(assignments) == 0)[0])
        assert target_env.model.dof_armature[inserted_dof] == 0.0
        assert "rigid_split_geom" in target_env.xml
    finally:
        source_env.close()
        target_env.close()

    unlocked_cfg = freeze_progress_config(continuation, 1.0, "unlocked_probe")
    unlocked_env = NLinkCartPoleEnv(unlocked_cfg, progress=1.0, seed=0)
    try:
        assert "rigid_split_geom" not in unlocked_env.xml
        np.testing.assert_allclose(
            unlocked_env.model.dof_armature[1:],
            target_cfg["env"]["joint_armature"],
        )
    finally:
        unlocked_env.close()


def test_published_split_count_checkpoint_verifies():
    ledger = "runs/generalized_solver/n2_to_n3_split_homotopy/continuation.json"
    assert verify(Path(ledger)) == []


def test_waypoint_priority_reuses_latest_successful_horizon():
    baseline = {
        "accepted": True,
        "waypoint_attempts": [
            {"segment_steps": 24, "refinement_passed": True}
        ],
    }
    trials = [
        {
            "accepted": True,
            "waypoint_attempts": [
                {"segment_steps": 24, "refinement_passed": False},
                {"segment_steps": 96, "refinement_passed": True},
            ],
        }
    ]
    assert prioritized_waypoint_steps([24, 48, 96], baseline, trials) == [
        96,
        48,
        24,
    ]
