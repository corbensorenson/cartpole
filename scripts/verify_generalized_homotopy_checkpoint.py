#!/usr/bin/env python
"""Verify a compact partial-morphology continuation checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import load_config
from gcartpole.evidence import file_metadata
from gcartpole.generalized_solver import (
    dimensionless_setup,
    mirror_feedback_route,
    setup_from_config,
)

ROOT = Path(__file__).resolve().parents[1]


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def verify_paired_gate(
    gate: dict[str, Any],
    declaration: dict[str, Any],
    controller_hashes: set[str | None],
    label: str,
) -> list[str]:
    errors: list[str] = []
    gate_hashes = {row.get("sha256") for row in gate.get("controllers", [])}
    if gate_hashes != controller_hashes:
        errors.append(f"{label} does not use the declared mirror pair")
    episode_rows = gate.get("episode_results", [])
    successes = sum(bool(row.get("success")) for row in episode_rows)
    agreements = sum(
        bool(row.get("prediction_execution_agreement"))
        if "prediction_execution_agreement" in row
        else bool(row.get("success"))
        == bool(row["predictions"][int(row["selected_route"])]["success"])
        for row in episode_rows
    )
    expected_counts = {
        "0": int(declaration.get("route_counts", {}).get("original", -1)),
        "1": int(declaration.get("route_counts", {}).get("mirror", -1)),
    }
    checks = (
        ("episodes", int(gate.get("episodes", 0)), int(declaration.get("episodes", -1))),
        ("successes", successes, int(declaration.get("successes", -1))),
        (
            "prediction agreements",
            agreements,
            int(declaration.get("prediction_execution_agreements", -1)),
        ),
        ("route counts", gate.get("route_counts", {}), expected_counts),
    )
    for metric, actual, expected in checks:
        if actual != expected:
            errors.append(f"declared {label} {metric} does not match gate")
    declared_conditioning = declaration.get("conditioning_ratio_counts")
    if (
        declared_conditioning is not None
        and gate.get("conditioning_ratio_counts") != declared_conditioning
    ):
        errors.append(f"declared {label} conditioning counts do not match gate")
    if successes != len(episode_rows):
        errors.append(f"{label} contains a failed episode")
    maximum_cart = max(
        (float(row.get("max_cart_excursion", np.inf)) for row in episode_rows),
        default=np.inf,
    )
    for metric, actual, expected in (
        (
            "maximum cart excursion",
            maximum_cart,
            float(declaration.get("maximum_cart_excursion", np.nan)),
        ),
        (
            "maximum required rail ratio",
            float(gate.get("max_required_rail_ratio", np.nan)),
            float(declaration.get("maximum_required_rail_ratio", np.nan)),
        ),
    ):
        if not np.isclose(actual, expected):
            errors.append(f"declared {label} {metric} does not match gate")
    return errors


def verify_checkpoint(path: Path, root: Path = ROOT) -> list[str]:
    checkpoint = load_object(path)
    errors: list[str] = []
    artifacts = checkpoint.get("artifacts", {})
    loaded: dict[str, dict[str, Any]] = {}
    required_artifacts = {"config", "controller", "mirror_controller", "exact_gate"}
    for missing in sorted(required_artifacts - set(artifacts)):
        errors.append(f"{missing} artifact is missing")
    for name, record in artifacts.items():
        artifact_path = root / str(record.get("path", ""))
        if not artifact_path.is_file():
            errors.append(f"{name} artifact is missing")
            continue
        if file_metadata(artifact_path)["sha256"] != record.get("sha256"):
            errors.append(f"{name} artifact hash mismatch")
            continue
        if name != "config":
            loaded[name] = load_object(artifact_path)

    progress = float(checkpoint.get("checkpoint", {}).get("progress", 0.0))
    if checkpoint.get("not_solution_of_full_target") is not True or not 0.0 < progress < 1.0:
        errors.append("checkpoint must remain explicitly partial")

    config_record = artifacts.get("config", {})
    config_path = root / str(config_record.get("path", ""))
    if config_path.is_file():
        setup = setup_from_config(load_config(config_path))
        expected = checkpoint.get("checkpoint", {})
        if not np.allclose(setup.lengths, expected.get("lengths", [])):
            errors.append("checkpoint lengths do not match config")
        if not np.allclose(setup.masses, expected.get("masses", [])):
            errors.append("checkpoint masses do not match config")
        pi = dimensionless_setup(setup).to_dict()
        declared_pi = checkpoint.get("dimensionless", {})
        for key in ("rail_ratio", "usable_rail_ratio", "force_authority", "natural_time"):
            if not np.isclose(float(pi[key]), float(declared_pi.get(key, np.nan))):
                errors.append(f"dimensionless {key} does not match config")

    controller = loaded.get("controller", {})
    mirror = loaded.get("mirror_controller", {})
    if controller and mirror:
        expected_controls, expected_states, expected_gains = mirror_feedback_route(
            np.asarray(controller["controller"]["controls"], dtype=np.float64),
            np.asarray(
                controller["search"]["nominal_coordinate_states"], dtype=np.float64
            ),
            np.asarray(
                controller["controller"]["feedback_gains"], dtype=np.float64
            ),
        )
        if mirror.get("controller", {}).get("mirror_symmetry") is not True:
            errors.append("mirror controller is not marked as analytic symmetry")
        for name, actual, expected in (
            ("controls", mirror["controller"]["controls"], expected_controls),
            (
                "nominal states",
                mirror["search"]["nominal_coordinate_states"],
                expected_states,
            ),
            ("feedback gains", mirror["controller"]["feedback_gains"], expected_gains),
        ):
            if not np.allclose(np.asarray(actual, dtype=np.float64), expected):
                errors.append(f"mirror {name} violate exact planar symmetry")

    exact = loaded.get("exact_gate", {})
    declared_exact = checkpoint.get("exact_gate", {})
    if int(exact.get("episodes", 0)) < 1 or float(exact.get("success_rate", 0.0)) != 1.0:
        errors.append("exact gate is not a complete pass")
    for key in ("episodes", "success_rate", "max_required_rail_ratio"):
        declared_key = "required_rail_ratio" if key == "max_required_rail_ratio" else key
        if not np.isclose(
            float(exact.get(key, np.nan)), float(declared_exact.get(declared_key, np.nan))
        ):
            errors.append(f"declared exact {declared_key} does not match gate")
    episode_results = exact.get("episode_results", [])
    if episode_results:
        row = episode_results[0]
        if not np.isclose(
            float(row.get("max_upright_streak_seconds", np.nan)),
            float(declared_exact.get("held_upright_seconds", np.nan)),
        ):
            errors.append("declared exact hold does not match gate")

    declared_noise = checkpoint.get("noise_boundary", {})
    if declared_noise:
        noisy = loaded.get("noisy_gate", {})
        for key in ("episodes", "success_rate"):
            if not np.isclose(
                float(noisy.get(key, np.nan)), float(declared_noise.get(key, np.nan))
            ):
                errors.append(f"declared noisy {key} does not match gate")
        successes = sum(
            bool(row.get("success")) for row in noisy.get("episode_results", [])
        )
        if successes != int(declared_noise.get("successes", -1)):
            errors.append("declared noisy successes do not match gate")
        rail_violations = int(
            noisy.get("termination_counts", {}).get("rail_violation", 0)
        )
        if rail_violations != int(declared_noise.get("rail_violations", -1)):
            errors.append("declared rail violations do not match gate")

    declared_selection = checkpoint.get("mirror_selection_gate", {})
    controller_hashes = {
        artifacts.get("controller", {}).get("sha256"),
        artifacts.get("mirror_controller", {}).get("sha256"),
    }
    if declared_selection:
        for artifact_name, declaration_name in (
            ("mirror_noisy20_gate", "twenty_episode_gate"),
            ("mirror_noisy100_gate", "hundred_episode_gate"),
        ):
            errors.extend(
                verify_paired_gate(
                    loaded.get(artifact_name, {}),
                    declared_selection.get(declaration_name, {}),
                    controller_hashes,
                    declaration_name,
                )
            )

    declared_adaptive = checkpoint.get("adaptive_selection_gate", {})
    if declared_adaptive:
        if int(declared_adaptive.get("learned_parameters", -1)) != 0:
            errors.append("adaptive selector must declare zero learned parameters")
        for artifact_name, declaration_name in (
            ("adaptive20_gate", "twenty_episode_gate"),
            ("adaptive100_gate", "hundred_episode_gate"),
        ):
            gate = loaded.get(artifact_name, {})
            errors.extend(
                verify_paired_gate(
                    gate,
                    declared_adaptive.get(declaration_name, {}),
                    controller_hashes,
                    declaration_name,
                )
            )
            adaptation = gate.get("adaptation", {})
            if int(adaptation.get("learned_parameters", -1)) != 0:
                errors.append(f"{declaration_name} adaptation is not deterministic")
            if adaptation.get("uses_measured_initial_state") is not True:
                errors.append(f"{declaration_name} does not declare measured-state use")
            if int(adaptation.get("execution_resets_after_measurement", -1)) != 0:
                errors.append(f"{declaration_name} resets after measuring the start")
            if any(int(row.get("resets_inside_episode", -1)) != 0 for row in gate.get("episode_results", [])):
                errors.append(f"{declaration_name} contains an in-episode reset")
            if not np.isclose(
                float(adaptation.get("natural_time_seconds", np.nan)),
                float(checkpoint.get("dimensionless", {}).get("natural_time", np.nan)),
            ):
                errors.append(f"{declaration_name} natural time does not match config")
            declared_ratios = declared_adaptive.get("conditioning_time_ratios", [])
            gate_ratios = [
                row.get("ratio") for row in adaptation.get("conditioning_schedule", [])
            ]
            if not np.allclose(gate_ratios, declared_ratios):
                errors.append(f"{declaration_name} conditioning grid does not match")

        immediate = loaded.get("adaptive_immediate_negative_gate", {})
        boundary = checkpoint.get("immediate_pair_boundary", {})
        immediate_hashes = {
            row.get("sha256") for row in immediate.get("controllers", [])
        }
        if immediate_hashes != controller_hashes:
            errors.append("immediate boundary does not use the declared mirror pair")
        immediate_successes = sum(
            bool(row.get("success")) for row in immediate.get("episode_results", [])
        )
        for metric, actual, expected in (
            ("episodes", int(immediate.get("episodes", 0)), int(boundary.get("episodes", -1))),
            ("successes", immediate_successes, int(boundary.get("successes", -1))),
            (
                "rail violations",
                int(immediate.get("termination_counts", {}).get("rail_violation", 0)),
                int(boundary.get("rail_violations", -1)),
            ),
        ):
            if actual != expected:
                errors.append(f"declared immediate boundary {metric} does not match gate")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    errors = verify_checkpoint(Path(args.checkpoint))
    print(f"passed={not errors} errors={len(errors)}")
    for error in errors:
        print(error)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
