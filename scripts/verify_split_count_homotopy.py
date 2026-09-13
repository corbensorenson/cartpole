#!/usr/bin/env python
"""Verify a locked-split continuation ledger and every referenced exact replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import load_config
from gcartpole.morphology import Morphology, build_morphology

try:
    from scripts.run_split_count_homotopy import scheduled_rail_limit
except ModuleNotFoundError:
    from run_split_count_homotopy import scheduled_rail_limit


ROOT = Path(__file__).resolve().parents[1]
PROFILE_NAMES = (
    "lengths",
    "masses",
    "damping",
    "frictionloss",
    "joint_stiffness",
    "joint_lock",
)


def resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def check_metadata(metadata: dict[str, Any], label: str, errors: list[str]) -> Path:
    path = resolve(str(metadata.get("path", "")))
    if not path.is_file():
        errors.append(f"{label}: missing file {path}")
        return path
    content = path.read_bytes()
    if int(metadata.get("bytes", -1)) != len(content):
        errors.append(f"{label}: byte count mismatch")
    if metadata.get("sha256") != hashlib.sha256(content).hexdigest():
        errors.append(f"{label}: sha256 mismatch")
    return path


def compare_morphology(
    expected: Morphology,
    actual: Morphology,
    label: str,
    errors: list[str],
) -> None:
    for name in PROFILE_NAMES:
        if not np.allclose(
            getattr(expected, name), getattr(actual, name), rtol=1e-10, atol=1e-12
        ):
            errors.append(f"{label}: {name} does not match continuation progress")


def check_exact_result(
    path: Path,
    cfg: dict[str, Any],
    label: str,
    errors: list[str],
) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result", {})
    if not result.get("success") or not result.get("latched"):
        errors.append(f"{label}: accepted result is not successful and latched")
    required_hold = float(cfg["env"]["success_sustain_seconds"])
    if float(result.get("max_upright_streak_seconds", 0.0)) < required_hold:
        errors.append(f"{label}: accepted result does not meet sustained-hold gate")
    return result


def check_waypoint_attempts(
    attempts: list[dict[str, Any]], label: str, errors: list[str]
) -> None:
    for index, attempt in enumerate(attempts, start=1):
        check_metadata(
            attempt.get("artifact", {}), f"{label} waypoint {index}", errors
        )
        refinement = attempt.get("refinement")
        if refinement is not None:
            check_metadata(
                refinement, f"{label} waypoint refinement {index}", errors
            )


def verify(path: Path) -> list[str]:
    errors: list[str] = []
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if ledger.get("claim_status") != "locked_split_continuation_not_record_evidence":
        errors.append("ledger: unexpected claim_status")
    if ledger.get("not_solution") is not True:
        errors.append("ledger: continuation evidence must remain not_solution")
    continuation_path = check_metadata(
        ledger.get("continuation_config", {}), "continuation", errors
    )
    source_path = check_metadata(
        ledger.get("source_controller", {}), "source controller", errors
    )
    if not continuation_path.is_file():
        return errors
    continuation = load_config(continuation_path)
    count = int(continuation["env"]["n_links"])
    if int(ledger.get("link_count", -1)) != count:
        errors.append("ledger: link count disagrees with continuation config")
    if source_path.is_file():
        source_payload = json.loads(source_path.read_text(encoding="utf-8"))
        gains = np.asarray(
            source_payload.get("controller", {}).get("feedback_gains"),
            dtype=np.float64,
        )
        if gains.ndim != 2 or gains.shape[1] != 2 * (count + 1):
            errors.append("source controller: feedback dimension disagrees with link count")

    baseline = ledger.get("locked_start_baseline")
    if not isinstance(baseline, dict) or not baseline.get("accepted"):
        errors.append("baseline: no accepted locked-start replay")
        return errors
    baseline_cfg_path = check_metadata(baseline.get("config", {}), "baseline config", errors)
    baseline_result_path = check_metadata(
        baseline.get("result", {}), "baseline result", errors
    )
    check_waypoint_attempts(
        baseline.get("waypoint_attempts", []), "baseline", errors
    )
    if baseline_cfg_path.is_file():
        baseline_cfg = load_config(baseline_cfg_path)
        compare_morphology(
            build_morphology(
                continuation["env"], continuation["morphology"], progress=0.0
            ),
            build_morphology(
                baseline_cfg["env"], baseline_cfg["morphology"], progress=1.0
            ),
            "baseline config",
            errors,
        )
        check_exact_result(
            baseline_result_path, baseline_cfg, "baseline result", errors
        )

    accepted_progress = 0.0
    current_result = baseline.get("result", {})
    for index, trial in enumerate(ledger.get("trials", []), start=1):
        label = f"trial {index}"
        proposed = float(trial.get("proposed_progress", -1.0))
        if not accepted_progress < proposed <= 1.0:
            errors.append(f"{label}: proposed progress is not ahead of frontier")
        if not np.isclose(float(trial.get("from_progress", -1.0)), accepted_progress):
            errors.append(f"{label}: from_progress disagrees with accepted frontier")
        cfg_path = check_metadata(trial.get("config", {}), f"{label} config", errors)
        result_path = check_metadata(trial.get("result", {}), f"{label} result", errors)
        check_waypoint_attempts(
            trial.get("waypoint_attempts", []), label, errors
        )
        if cfg_path.is_file():
            cfg = load_config(cfg_path)
            compare_morphology(
                build_morphology(
                    continuation["env"],
                    continuation["morphology"],
                    progress=proposed,
                ),
                build_morphology(cfg["env"], cfg["morphology"], progress=1.0),
                f"{label} config",
                errors,
            )
            expected_rail = scheduled_rail_limit(continuation["env"], proposed)
            if not np.isclose(float(cfg["env"]["rail_limit"]), expected_rail):
                errors.append(f"{label}: rail does not match continuation progress")
            if trial.get("accepted"):
                check_exact_result(result_path, cfg, f"{label} result", errors)
        if trial.get("accepted"):
            accepted_progress = proposed
            current_result = trial.get("result", {})

    if not np.isclose(float(ledger.get("progress", -1.0)), accepted_progress):
        errors.append("ledger: progress does not equal last accepted trial")
    current = ledger.get("current_controller", {})
    check_metadata(current, "current controller", errors)
    if current.get("sha256") != current_result.get("sha256"):
        errors.append("ledger: current controller is not the last accepted result")
    if ledger.get("status") == "unlocked_target_exact_replay_passed" and not np.isclose(
        accepted_progress, 1.0
    ):
        errors.append("ledger: unlocked status requires accepted progress p=1")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    args = parser.parse_args()
    errors = verify(Path(args.ledger))
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    print(f"verified split-count continuation: {args.ledger}")


if __name__ == "__main__":
    main()
