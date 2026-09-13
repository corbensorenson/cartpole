from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_generalized_force_adaptation import calibration_dither
from scripts.verify_generalized_adaptation import verify_artifact


def trial(mode: str, seed: int, *, success: bool) -> dict:
    return {
        "mode": mode,
        "seed": seed,
        "success": success,
        "rail_requirement": {"required_rail_ratio": 1.0},
        "adaptation": {
            "maximum_action_correction": 0.2,
            "gain_absolute_error": 0.001,
            "bias_absolute_error": 0.001,
            "updates": 10 if mode == "adaptive" else 0,
        },
    }


def artifact() -> dict:
    return {
        "claim_status": "development_bounded_adaptation_diagnostic",
        "n_links": 3,
        "simulated_actuator": {"gain": 1.18, "bias": 0.06},
        "adaptation_contract": {
            "route_or_energy_parameters_changed": False,
            "estimate_frozen_after_calibration": True,
            "maximum_per_step_correction": 0.35,
        },
        "results": [
            trial("baseline", seed, success=False) for seed in range(3)
        ]
        + [trial("adaptive", seed, success=True) for seed in range(3)],
    }


def write_artifact(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_adaptation_verifier_accepts_paired_bounded_success(tmp_path: Path) -> None:
    path = tmp_path / "adaptation.json"
    write_artifact(path, artifact())
    result = verify_artifact(
        path,
        3,
        expected_gain=1.18,
        expected_bias=0.06,
        episodes=3,
        parameter_tolerance=0.01,
    )
    assert result["passed"]
    assert result["adaptive_successes"] == 3
    assert result["baseline_successes"] == 0


def test_adaptation_verifier_rejects_bound_violation(tmp_path: Path) -> None:
    payload = artifact()
    payload["results"][-1]["adaptation"]["maximum_action_correction"] = 0.36
    path = tmp_path / "adaptation.json"
    write_artifact(path, payload)
    result = verify_artifact(
        path,
        3,
        expected_gain=1.18,
        expected_bias=0.06,
        episodes=3,
        parameter_tolerance=0.01,
    )
    assert not result["passed"]
    assert "action correction exceeded its bound" in result["errors"]


def test_calibration_dither_is_bounded_balanced_and_stops() -> None:
    samples = [calibration_dither(step, 0.02, 1.28, 0.25) for step in range(64)]
    assert max(samples) == 0.25
    assert min(samples) == -0.25
    assert abs(sum(samples)) < 1e-12
    assert calibration_dither(64, 0.02, 1.28, 0.25) == 0.0
