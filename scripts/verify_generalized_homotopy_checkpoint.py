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
from gcartpole.generalized_solver import dimensionless_setup, setup_from_config

ROOT = Path(__file__).resolve().parents[1]


def load_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def verify_checkpoint(path: Path, root: Path = ROOT) -> list[str]:
    checkpoint = load_object(path)
    errors: list[str] = []
    artifacts = checkpoint.get("artifacts", {})
    loaded: dict[str, dict[str, Any]] = {}
    for name in ("config", "controller", "exact_gate", "noisy_gate"):
        record = artifacts.get(name, {})
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

    noisy = loaded.get("noisy_gate", {})
    declared_noise = checkpoint.get("noise_boundary", {})
    for key in ("episodes", "success_rate"):
        if not np.isclose(
            float(noisy.get(key, np.nan)), float(declared_noise.get(key, np.nan))
        ):
            errors.append(f"declared noisy {key} does not match gate")
    successes = sum(bool(row.get("success")) for row in noisy.get("episode_results", []))
    if successes != int(declared_noise.get("successes", -1)):
        errors.append("declared noisy successes do not match gate")
    rail_violations = int(noisy.get("termination_counts", {}).get("rail_violation", 0))
    if rail_violations != int(declared_noise.get("rail_violations", -1)):
        errors.append("declared rail violations do not match gate")
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
