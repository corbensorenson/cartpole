#!/usr/bin/env python
"""Verify the paired n=1..7 bounded-actuator adaptation ladder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gcartpole.config import dump_json
from gcartpole.evidence import (
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)


def verify_artifact(
    path: Path,
    n_links: int,
    *,
    expected_gain: float,
    expected_bias: float,
    episodes: int,
    parameter_tolerance: float,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("claim_status") != "development_bounded_adaptation_diagnostic":
        errors.append("unexpected claim_status")
    if int(payload.get("n_links", -1)) != n_links:
        errors.append("link count mismatch")
    actuator = payload.get("simulated_actuator", {})
    if float(actuator.get("gain", float("nan"))) != expected_gain:
        errors.append("actuator gain mismatch")
    if float(actuator.get("bias", float("nan"))) != expected_bias:
        errors.append("actuator bias mismatch")
    contract = payload.get("adaptation_contract", {})
    if contract.get("route_or_energy_parameters_changed") is not False:
        errors.append("controller parameters were not declared frozen")
    if contract.get("estimate_frozen_after_calibration") is not True:
        errors.append("estimate was not frozen after calibration")

    results = payload.get("results", [])
    adaptive = [row for row in results if row.get("mode") == "adaptive"]
    baseline = [row for row in results if row.get("mode") == "baseline"]
    if len(adaptive) != episodes or len(baseline) != episodes:
        errors.append("paired episode count mismatch")
    if len({int(row["seed"]) for row in adaptive}) != episodes:
        errors.append("adaptive seeds are not unique")
    if {int(row["seed"]) for row in adaptive} != {
        int(row["seed"]) for row in baseline
    }:
        errors.append("baseline/adaptive seeds are not paired")
    if not all(bool(row.get("success")) for row in adaptive):
        errors.append("an adaptive trial failed")
    if any(bool(row.get("success")) for row in baseline):
        errors.append("baseline unexpectedly passed; perturbation is not discriminating")

    correction_bound = float(contract.get("maximum_per_step_correction", -1.0))
    maximum_correction = max(
        (float(row["adaptation"]["maximum_action_correction"]) for row in adaptive),
        default=float("inf"),
    )
    maximum_gain_error = max(
        (float(row["adaptation"]["gain_absolute_error"]) for row in adaptive),
        default=float("inf"),
    )
    maximum_bias_error = max(
        (float(row["adaptation"]["bias_absolute_error"]) for row in adaptive),
        default=float("inf"),
    )
    if maximum_correction > correction_bound + 1e-12:
        errors.append("action correction exceeded its bound")
    if max(maximum_gain_error, maximum_bias_error) > parameter_tolerance:
        errors.append("actuator estimate exceeded tolerance")
    if any(int(row["adaptation"]["updates"]) <= 0 for row in adaptive):
        errors.append("an adaptive trial made no accepted updates")

    return {
        "n_links": n_links,
        "passed": not errors,
        "errors": errors,
        "artifact": file_metadata(path),
        "paired_episodes": episodes,
        "baseline_successes": sum(bool(row.get("success")) for row in baseline),
        "adaptive_successes": sum(bool(row.get("success")) for row in adaptive),
        "maximum_action_correction": maximum_correction,
        "correction_bound": correction_bound,
        "maximum_gain_absolute_error": maximum_gain_error,
        "maximum_bias_absolute_error": maximum_bias_error,
        "maximum_required_rail_ratio": max(
            float(row["rail_requirement"]["required_rail_ratio"])
            for row in adaptive
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pattern",
        default="runs/generalized_solver/adaptation_n{n}_paired3.json",
    )
    parser.add_argument("--min-links", type=int, default=1)
    parser.add_argument("--max-links", type=int, default=7)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--expected-gain", type=float, default=1.18)
    parser.add_argument("--expected-bias", type=float, default=0.06)
    parser.add_argument("--parameter-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--out",
        default="runs/generalized_solver/adaptation_ladder_n1_n7.json",
    )
    args = parser.parse_args()
    if args.min_links < 1 or args.max_links < args.min_links:
        raise ValueError("link range must be positive and ordered")
    if args.episodes < 1 or args.parameter_tolerance <= 0.0:
        raise ValueError("episodes and parameter tolerance must be positive")

    rows = [
        verify_artifact(
            Path(args.pattern.format(n=n_links)),
            n_links,
            expected_gain=args.expected_gain,
            expected_bias=args.expected_bias,
            episodes=args.episodes,
            parameter_tolerance=args.parameter_tolerance,
        )
        for n_links in range(args.min_links, args.max_links + 1)
    ]
    passed = all(row["passed"] for row in rows)
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "verified_development_bounded_adaptation_ladder",
        "passed": passed,
        "summary": (
            "The same bounded projected-RLS actuator calibration recovered all "
            "paired n=1..7 deterministic controllers under a hidden affine "
            "actuator mismatch; this is not arbitrary-morphology evidence."
        ),
        "scope": {
            "link_range": [args.min_links, args.max_links],
            "episodes_per_mode_per_link": args.episodes,
            "expected_actuator_gain": args.expected_gain,
            "expected_actuator_bias": args.expected_bias,
            "parameter_tolerance": args.parameter_tolerance,
            "total_baseline_successes": sum(
                int(row["baseline_successes"]) for row in rows
            ),
            "total_adaptive_successes": sum(
                int(row["adaptive_successes"]) for row in rows
            ),
            "total_trials_per_mode": args.episodes * len(rows),
        },
        "rows": rows,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, args.out)
    print(
        f"passed={passed} baseline={payload['scope']['total_baseline_successes']}/"
        f"{payload['scope']['total_trials_per_mode']} "
        f"adaptive={payload['scope']['total_adaptive_successes']}/"
        f"{payload['scope']['total_trials_per_mode']}"
    )
    if not passed:
        for row in rows:
            for error in row["errors"]:
                print(f"n={row['n_links']}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
