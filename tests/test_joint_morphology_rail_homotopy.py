from __future__ import annotations

import copy

import numpy as np

from gcartpole.config import load_config
from gcartpole.generalized_solver import dimensionless_setup, setup_from_config
from scripts.run_joint_morphology_rail_homotopy import (
    config_at,
    rail_contraction_proposal,
    rail_rescue_ratio,
    target_rail_ratio,
)


def test_config_at_interpolates_morphology_and_sets_dimensionless_rail() -> None:
    source_cfg = load_config("runs/generalized_solver/n3_uniform_exact.yaml")
    target_cfg = load_config("configs/generalized_n3_unequal.yaml")
    source = setup_from_config(source_cfg)
    target = setup_from_config(target_cfg)
    cfg = config_at(
        target_cfg,
        source.lengths,
        source.masses,
        target.lengths,
        target.masses,
        progress=0.5,
        rail_ratio=1.8,
        name="joint_test",
    )
    setup = setup_from_config(cfg)
    groups = dimensionless_setup(setup)
    np.testing.assert_allclose(setup.lengths, 0.5 * (source.lengths + target.lengths))
    np.testing.assert_allclose(setup.masses, 0.5 * (source.masses + target.masses))
    assert groups.rail_ratio == 1.8


def test_rail_rescue_uses_measured_requirement_plus_clearance() -> None:
    outcome = {
        "termination_reason": "rail_violation",
        "rail_requirement": {"required_rail_ratio": 1.74},
    }
    assert rail_rescue_ratio(
        outcome,
        current_ratio=1.6,
        maximum_ratio=2.5,
        growth=1.08,
        clearance_ratio=0.05,
    ) == 1.79


def test_rail_rescue_requires_a_measured_rail_collision() -> None:
    outcome = {
        "termination_reason": "time_limit",
        "rail_requirement": {"required_rail_ratio": 2.0},
    }
    assert rail_rescue_ratio(
        outcome,
        current_ratio=1.6,
        maximum_ratio=2.5,
        growth=1.08,
        clearance_ratio=0.05,
    ) is None


def test_rail_rescue_is_capped_by_declared_diagnostic_limit() -> None:
    outcome = {
        "termination_reason": "rail_violation",
        "rail_requirement": {"required_rail_ratio": 2.7},
    }
    assert rail_rescue_ratio(
        outcome,
        current_ratio=1.6,
        maximum_ratio=2.0,
        growth=1.08,
        clearance_ratio=0.05,
    ) == 2.0


def test_rail_contraction_is_monotone_and_clamped() -> None:
    assert rail_contraction_proposal(2.0, 1.5, 0.1) == 1.9
    assert rail_contraction_proposal(1.52, 1.5, 0.1) == 1.5


def test_target_ratio_comes_from_physical_config() -> None:
    cfg = load_config("configs/generalized_n3_unequal.yaml")
    assert target_rail_ratio(copy.deepcopy(cfg)) == 5.0 / 3.0
