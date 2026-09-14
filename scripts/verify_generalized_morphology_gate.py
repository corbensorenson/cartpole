#!/usr/bin/env python
"""Verify one transferred-and-refined target-morphology development gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import (
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.generalized_solver import dimensionless_setup, setup_from_config


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--warm", required=True)
    parser.add_argument("--negative", required=True)
    parser.add_argument("--optimizer", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--mirror", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--minimum-episodes", type=int, default=20)
    parser.add_argument("--require-warm-failure", action="store_true")
    parser.add_argument("--require-nonuniform-lengths", action="store_true")
    parser.add_argument("--require-nonuniform-masses", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = {
        name: Path(value)
        for name, value in {
            "config": args.config,
            "warm": args.warm,
            "negative": args.negative,
            "optimizer": args.optimizer,
            "route": args.route,
            "mirror": args.mirror,
            "gate": args.gate,
        }.items()
    }
    cfg = load_config(paths["config"])
    setup = setup_from_config(cfg)
    pi = dimensionless_setup(setup)
    warm = load(paths["warm"])
    negative = load(paths["negative"])
    optimizer = load(paths["optimizer"])
    route = load(paths["route"])
    mirror = load(paths["mirror"])
    gate = load(paths["gate"])
    errors: list[str] = []

    uniform_lengths = np.full(setup.n_links, 1.0 / setup.n_links)
    uniform_masses = np.full(setup.n_links, 1.0 / setup.n_links)
    if args.require_nonuniform_lengths and np.allclose(
        pi.length_fractions, uniform_lengths
    ):
        errors.append("length fractions are uniform")
    if args.require_nonuniform_masses and np.allclose(pi.mass_fractions, uniform_masses):
        errors.append("mass fractions are uniform")
    target = warm.get("target", {}).get("dimensionless", {})
    if not np.allclose(target.get("length_fractions", []), pi.length_fractions):
        errors.append("warm-start target length fractions do not match config")
    if not np.allclose(target.get("mass_fractions", []), pi.mass_fractions):
        errors.append("warm-start target mass fractions do not match config")
    if warm.get("not_solution") is not True:
        errors.append("transfer warm start is not explicitly marked not_solution")
    if (
        args.require_warm_failure
        and float(negative.get("success_rate", -1.0)) != 0.0
    ):
        errors.append("unrefined transfer negative control unexpectedly passed")
    if optimizer.get("search", {}).get("is_feasible") is not True:
        errors.append("exact optimizer did not report a feasible trajectory")
    if optimizer.get("result", {}).get("success") is not True:
        errors.append("exact optimized replay did not sustain upright")

    optimizer_sha = file_metadata(paths["optimizer"])["sha256"]
    for name, payload in (("route", route), ("mirror", mirror)):
        if payload.get("source", {}).get("sha256") != optimizer_sha:
            errors.append(f"{name} does not identify the optimized source")
    route_hashes = {
        file_metadata(paths["route"])["sha256"],
        file_metadata(paths["mirror"])["sha256"],
    }
    gate_hashes = {
        row.get("sha256") for row in gate.get("controllers", []) if isinstance(row, dict)
    }
    if gate_hashes != route_hashes:
        errors.append("gate controller hashes do not match the packaged mirror pair")
    if int(gate.get("episodes", 0)) < args.minimum_episodes:
        errors.append("gate contains too few episodes")
    if float(gate.get("success_rate", 0.0)) != 1.0:
        errors.append("noisy gate did not pass every episode")
    episode_results = gate.get("episode_results", [])
    if len({row.get("seed") for row in episode_results}) != len(episode_results):
        errors.append("gate seeds are not unique")
    if not all(
        row.get("success") is True
        and row["predictions"][int(row["selected_route"])]["success"] is True
        for row in episode_results
    ):
        errors.append("an uninterrupted gate result or prediction failed")
    required_ratio = float(gate.get("max_required_rail_ratio", float("inf")))
    if required_ratio >= pi.rail_ratio:
        errors.append("body-aware required rail exceeds configured rail ratio")

    passed = not errors
    if passed:
        summary_text = (
            "Arc-length transfer supplied the deterministic start, then the same "
            "exact target-plant Box-FDDP/Riccati/mirror architecture passed the "
            "declared target-morphology noisy gate. This verifies one morphology, "
            "not arbitrary setups."
        )
    else:
        summary_text = (
            "The declared unequal morphology did not pass the complete exact-refinement "
            "and noisy-gate contract. It remains a measured development frontier and "
            "must not be promoted as a solved setup."
        )
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "verified_development_generalized_morphology_gate",
        "passed": passed,
        "summary": summary_text,
        "errors": errors,
        "morphology": {
            "n_links": setup.n_links,
            "lengths": setup.lengths.tolist(),
            "masses": setup.masses.tolist(),
            "dimensionless": pi.to_dict(),
        },
        "evidence": {name: file_metadata(path) for name, path in paths.items()},
        "results": {
            "unrefined_transfer_success_rate": float(negative["success_rate"]),
            "optimizer_feasible": bool(optimizer["search"]["is_feasible"]),
            "optimizer_exact_replay_success": bool(optimizer["result"]["success"]),
            "optimizer_hold_seconds": float(
                optimizer["result"]["max_upright_streak_seconds"]
            ),
            "noisy_gate_episodes": int(gate["episodes"]),
            "noisy_gate_success_rate": float(gate["success_rate"]),
            "prediction_execution_agreements": int(
                sum(
                    bool(row["success"])
                    == bool(row["predictions"][int(row["selected_route"])]["success"])
                    for row in episode_results
                )
            ),
        },
        "rail_relationship": {
            "configured_half_length": setup.rail_half_length,
            "total_chain_length": setup.chain_length,
            "configured_ratio": pi.rail_ratio,
            "maximum_body_aware_required_ratio": required_ratio,
            "ratio_margin": pi.rail_ratio - required_ratio,
        },
        "boundary": (
            f"This verifies the architecture on this declared {setup.n_links}-link "
            f"plant with lengths {setup.lengths.tolist()} and masses "
            f"{setup.masses.tolist()} only. Broader morphology coverage and "
            "automatic rail minimization remain active work."
        ),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, args.out)
    print(
        f"passed={passed} warm={output['results']['unrefined_transfer_success_rate']:.3f} "
        f"gate={output['results']['noisy_gate_success_rate']:.3f} "
        f"required_rho={required_ratio:.3f}"
    )
    if not passed:
        for error in errors:
            print(error)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
