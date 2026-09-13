from __future__ import annotations

from pathlib import Path

from scripts.run_generalized_homotopy import (
    backfill_trial_outcomes,
    fddp_command,
    result_summary,
    waypoint_command,
    waypoint_lookaheads,
    waypoint_usable,
)


def test_fddp_warm_mode_distinguishes_feedback_and_exact_trajectory(
    tmp_path: Path,
) -> None:
    common = {
        "cfg": Path("configs/generalized_n3_unequal.yaml"),
        "state": tmp_path / "state.json",
        "controller": tmp_path / "controller.json",
        "output": tmp_path / "output.json",
        "iterations": 10,
        "regularization": 1e-6,
        "tracking_gain": 1.0,
    }
    inherited = fddp_command(**common)
    exact = fddp_command(**common, exact_initial_trajectory=True)
    assert "--rebuild-initial-feedback" in inherited
    assert "--initial-feasible" not in inherited
    assert "--initial-feasible" in exact
    assert "--rebuild-initial-feedback" not in exact


def test_fddp_can_use_physical_rail_margin() -> None:
    command = fddp_command(
        cfg=Path("configs/generalized_n3_unequal.yaml"),
        state=Path("state.json"),
        controller=Path("controller.json"),
        output=Path("output.json"),
        iterations=10,
        regularization=1e-6,
        tracking_gain=1.0,
        rail_soft_margin=0.25,
    )
    assert command[command.index("--rail-soft-limit") + 1] == "4.75"


def test_waypoint_command_uses_target_rail_and_generic_parameters(
    tmp_path: Path,
) -> None:
    command = waypoint_command(
        cfg=Path("configs/generalized_n3_unequal.yaml"),
        controller=tmp_path / "source.json",
        output=tmp_path / "adapted.json",
        segment_steps=24,
        max_evaluations=120,
        endpoint_weight=10_000.0,
        control_regularization=1e-6,
        rail_soft_margin=0.5,
        rail_weight=1_000_000.0,
        endpoint_tolerance=0.5,
    )
    assert command[command.index("--rail-soft-limit") + 1] == "4.5"
    assert command[command.index("--segment-steps") + 1] == "24"
    assert "adapt_generalized_route_waypoints.py" in command[1]


def test_rail_safe_exact_waypoint_is_usable_even_if_reference_gate_failed(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "waypoint.json"
    artifact.write_text(
        """{
          "controller": {
            "controls": [0.1, -0.1],
            "nominal_coordinate_states": [[0, 0], [0.1, 0], [0, 0]]
          },
          "search": {
            "success": false,
            "maximum_cart_excursion": 1.2,
            "rail_soft_limit": 1.5
          }
        }""",
        encoding="utf-8",
    )
    assert waypoint_usable(artifact)


def test_waypoint_lookaheads_expand_and_deduplicate() -> None:
    assert waypoint_lookaheads(24, [1, 2, 2, 4]) == (24, 48, 96)


def test_result_summary_records_body_aware_rail_demand(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    artifact.write_text(
        """{
          "result": {
            "success": true,
            "latched": true,
            "termination_reason": "time_limit",
            "max_upright_streak_seconds": 6.0,
            "max_cart_excursion": 2.0
          }
        }""",
        encoding="utf-8",
    )
    from gcartpole.config import load_config

    summary = result_summary(artifact, load_config("configs/generalized_n3_unequal.yaml"))
    assert summary["success"] is True
    assert summary["rail_requirement"]["max_cart_center_excursion"] == 2.0
    assert summary["rail_requirement"]["required_rail_half_length"] == 2.18


def test_backfill_trial_outcomes_skips_missing_public_traces() -> None:
    trials = [{"result": {"path": "missing.json"}, "config": {"path": "missing.yaml"}}]
    backfill_trial_outcomes(trials)
    assert "outcome" not in trials[0]
