from __future__ import annotations

import numpy as np

from gcartpole.config import load_config
from gcartpole.generalized_solver import setup_from_config
from gcartpole.morphology import build_morphology
from scripts.build_supported_unlock_continuations import (
    build_supported_unlock_continuations,
)


def test_supported_unlock_is_topology_safe_and_dimensionless() -> None:
    locked_cfg = load_config(
        "runs/generalized_solver/n2_to_n3_split_logcompliance_homotopy/"
        "configs/trial_0050_p0.999548750.yaml"
    )
    target_cfg = load_config("configs/generalized_n3_unequal.yaml")
    ramp, release, relaxation = build_supported_unlock_continuations(
        locked_cfg,
        target_cfg,
        stiffness_ratio=2.0,
        damping_ratio=0.25,
    )
    ramp_start = build_morphology(ramp["env"], ramp["morphology"], 0.0)
    ramp_end = build_morphology(ramp["env"], ramp["morphology"], 1.0)
    release_start = build_morphology(release["env"], release["morphology"], 0.0)
    release_end = build_morphology(release["env"], release["morphology"], 1.0)
    relaxation_start = build_morphology(
        relaxation["env"], relaxation["morphology"], 0.0
    )
    relaxation_end = build_morphology(
        relaxation["env"], relaxation["morphology"], 1.0
    )
    target = build_morphology(target_cfg["env"], target_cfg["morphology"], 1.0)
    setup = setup_from_config(target_cfg)
    joint = 1

    assert ramp_start.joint_lock[joint] > 0.0
    np.testing.assert_allclose(ramp_start.joint_stiffness, target.joint_stiffness)
    np.testing.assert_allclose(ramp_end.joint_stiffness, release_start.joint_stiffness)
    assert release_start.joint_lock[joint] > 0.0
    assert release_end.joint_lock[joint] == 0.0
    np.testing.assert_allclose(release_start.joint_stiffness, release_end.joint_stiffness)
    np.testing.assert_allclose(release_start.damping, release_end.damping)
    np.testing.assert_allclose(relaxation_start.joint_stiffness, release_end.joint_stiffness)
    np.testing.assert_allclose(relaxation_start.damping, release_end.damping)
    np.testing.assert_allclose(relaxation_end.joint_stiffness, target.joint_stiffness)
    np.testing.assert_allclose(relaxation_end.damping, target.damping)
    assert np.isclose(
        release_start.joint_stiffness[joint] - target.joint_stiffness[joint],
        2.0 * setup.system_mass * setup.gravity * setup.chain_length,
    )
    assert release["supported_unlock"]["releasing_joint_indices"] == [joint]


def test_supported_unlock_can_continue_from_existing_support() -> None:
    locked_cfg = load_config(
        "runs/generalized_solver/n2_to_n3_split_logcompliance_homotopy/"
        "configs/trial_0050_p0.999548750.yaml"
    )
    target_cfg = load_config("configs/generalized_n3_unequal.yaml")
    ramp, release, _ = build_supported_unlock_continuations(
        locked_cfg,
        target_cfg,
        stiffness_ratio=10.0,
        damping_ratio=0.1,
        initial_stiffness_ratio=1.0,
        initial_damping_ratio=0.01,
    )
    start = build_morphology(ramp["env"], ramp["morphology"], 0.0)
    end = build_morphology(ramp["env"], ramp["morphology"], 1.0)
    release_start = build_morphology(release["env"], release["morphology"], 0.0)
    target = build_morphology(target_cfg["env"], target_cfg["morphology"], 1.0)
    setup = setup_from_config(target_cfg)
    joint = 1

    assert np.isclose(
        start.joint_stiffness[joint] - target.joint_stiffness[joint],
        setup.system_mass * setup.gravity * setup.chain_length,
    )
    assert np.isclose(
        end.joint_stiffness[joint] - target.joint_stiffness[joint],
        10.0 * setup.system_mass * setup.gravity * setup.chain_length,
    )
    np.testing.assert_allclose(end.joint_stiffness, release_start.joint_stiffness)
    assert ramp["supported_unlock"]["initial_stiffness_ratio"] == 1.0
