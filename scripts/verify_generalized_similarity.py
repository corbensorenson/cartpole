#!/usr/bin/env python
"""Verify exact dynamic similarity for analytic and transferred feedback laws."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json
from gcartpole.evidence import (
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.generalized_energy import EnergySwingParameters

try:
    from scripts.verify_generalized_uniform_ladder import verify_gate
except ModuleNotFoundError:
    from verify_generalized_uniform_ladder import verify_gate

DEFAULT_SOURCE_GATE = Path("runs/generalized_solver/energy_n1_noisy20_v2.json")
DEFAULT_CASES = {
    "l05_m05": (
        0.5,
        0.5,
        Path("runs/generalized_solver/similarity_n1_l05_m05.json"),
        Path("configs/generalized_n1_similar_l05_m05.yaml"),
        Path("runs/generalized_solver/similarity_n1_l05_m05_gate20.json"),
    ),
    "l05_m2": (
        0.5,
        2.0,
        Path("runs/generalized_solver/similarity_n1_l05_m2.json"),
        Path("configs/generalized_n1_similar_l05_m2.yaml"),
        Path("runs/generalized_solver/similarity_n1_l05_m2_gate20.json"),
    ),
    "l2_m05": (
        2.0,
        0.5,
        Path("runs/generalized_solver/similarity_n1_l2_m05.json"),
        Path("configs/generalized_n1_similar_l2_m05.yaml"),
        Path("runs/generalized_solver/similarity_n1_l2_m05_gate20.json"),
    ),
    "l2_m2": (
        2.0,
        2.0,
        Path("runs/generalized_solver/similarity_n1_l2_m2.json"),
        Path("configs/generalized_n1_similar_l2_m2.yaml"),
        Path("runs/generalized_solver/similarity_n1_l2_m2_gate20.json"),
    ),
}
DEFAULT_ROUTE_CASES = {
    "n2_l2_m05": (
        2,
        2.0,
        0.5,
        Path("runs/generalized_solver/similarity_n2_l2_m05.json"),
        Path("configs/generalized_n2_similar_l2_m05.yaml"),
        Path("runs/generalized_solver/similarity_n2_l2_m05_route.json"),
        Path("runs/generalized_solver/similarity_n2_l2_m05_route_mirror.json"),
        Path("runs/generalized_solver/similarity_n2_l2_m05_gate20.json"),
    )
}

INVARIANT_SCALARS = (
    "cart_damping_ratio",
    "cart_half_length_ratio",
    "cart_to_link_mass",
    "force_authority",
    "joint_armature_ratio",
    "link_radius_ratio",
    "policy_dt_ratio",
    "rail_ratio",
    "usable_rail_ratio",
)
INVARIANT_VECTORS = (
    "joint_damping_ratios",
    "length_fractions",
    "mass_fractions",
)


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def _check_close(
    errors: list[str], name: str, actual: float, expected: float, tolerance: float
) -> None:
    if not math.isfinite(actual) or not np.isclose(
        actual, expected, atol=tolerance, rtol=0.0
    ):
        errors.append(f"{name}: expected {expected}, got {actual}")


def verify_case(
    label: str,
    length_scale: float,
    mass_scale: float,
    similarity_path: Path,
    config_path: Path,
    gate_path: Path,
    source_gate: dict[str, Any],
    *,
    minimum_episodes: int,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    similarity = load(similarity_path)
    gate = load(gate_path)
    errors: list[str] = []

    if (
        similarity.get("claim_status")
        != "dynamic_similarity_plant_not_solution_evidence"
    ):
        errors.append("similarity artifact has an unexpected claim status")
    if similarity.get("not_solution") is not True:
        errors.append("plant artifact does not preserve the not-a-solution boundary")
    scales = similarity.get("scales", {})
    _check_close(
        errors,
        "length scale",
        float(scales.get("length", math.nan)),
        length_scale,
        tolerance,
    )
    _check_close(
        errors, "mass scale", float(scales.get("mass", math.nan)), mass_scale, tolerance
    )
    _check_close(
        errors,
        "natural-time scale",
        float(scales.get("time", math.nan)),
        math.sqrt(length_scale),
        tolerance,
    )
    if (
        float(similarity.get("dimensionless_errors", {}).get("maximum_abs", math.inf))
        > tolerance
    ):
        errors.append("emitted maximum dimensionless error exceeds tolerance")

    source_pi = similarity.get("source_dimensionless", {})
    target_pi = similarity.get("target_dimensionless", {})
    for name in INVARIANT_SCALARS:
        _check_close(
            errors,
            f"dimensionless scalar {name}",
            float(target_pi.get(name, math.nan)),
            float(source_pi.get(name, math.nan)),
            tolerance,
        )
    for name in INVARIANT_VECTORS:
        source = np.asarray(source_pi.get(name), dtype=np.float64)
        target = np.asarray(target_pi.get(name), dtype=np.float64)
        if source.shape != target.shape or not np.allclose(
            target, source, atol=tolerance, rtol=0.0
        ):
            errors.append(f"dimensionless vector {name} changed")
    _check_close(
        errors,
        "chain length",
        float(target_pi.get("chain_length", math.nan)),
        float(source_pi.get("chain_length", math.nan)) * length_scale,
        tolerance,
    )
    _check_close(
        errors,
        "system mass",
        float(target_pi.get("system_mass", math.nan)),
        float(source_pi.get("system_mass", math.nan)) * mass_scale,
        tolerance,
    )
    _check_close(
        errors,
        "natural time",
        float(target_pi.get("natural_time", math.nan)),
        float(source_pi.get("natural_time", math.nan)) * math.sqrt(length_scale),
        tolerance,
    )

    config_metadata = file_metadata(config_path)
    if similarity.get("target_config") != str(config_path):
        errors.append("similarity artifact points to a different target config")
    if gate.get("source_config", {}).get("sha256") != config_metadata["sha256"]:
        errors.append("gate config hash does not match the generated target config")
    if gate.get("dimensionless_setup") != target_pi:
        errors.append("gate plant does not match the similarity artifact")
    if gate.get("claim_status") != "development_deterministic_baseline":
        errors.append("gate claim status is not the deterministic baseline")
    if int(gate.get("n_links", -1)) != 1:
        errors.append("gate is not a one-link plant")
    controller = gate.get("controller", {})
    if controller.get("type") != "dimensionless_energy_pfl_then_exact_lqr":
        errors.append("gate did not use energy PFL followed by exact LQR")
    expected_parameters = EnergySwingParameters().to_dict()
    if controller.get("parameters") != expected_parameters:
        errors.append("dimensionless controller parameters changed")
    source_parameters = source_gate.get("controller", {}).get("parameters", {})
    if any(
        expected_parameters.get(name) != value
        for name, value in source_parameters.items()
    ):
        errors.append("current dimensionless controller changed from the source gate")

    episodes = gate.get("episode_results", [])
    declared_episodes = int(gate.get("episodes", -1))
    if declared_episodes < minimum_episodes or len(episodes) != declared_episodes:
        errors.append("gate episode count is incomplete")
    if float(gate.get("success_rate", 0.0)) != 1.0:
        errors.append("gate success rate is not one")
    if len({row.get("seed") for row in episodes}) != len(episodes):
        errors.append("gate episode seeds are not unique")
    if not all(
        row.get("success") is True and row.get("termination_reason") == "time_limit"
        for row in episodes
    ):
        errors.append("an uninterrupted similarity-gate episode failed")

    source_natural_time = float(source_gate["dimensionless_setup"]["natural_time"])
    target_natural_time = float(target_pi["natural_time"])
    expected_conditioning_ratio = (
        float(source_gate["episode_results"][0]["conditioning_seconds"])
        / source_natural_time
    )
    source_switches = {
        float(row["switch_time_after_conditioning"]) / source_natural_time
        for row in source_gate["episode_results"]
    }
    if len(source_switches) != 1:
        errors.append("source gate does not have one deterministic switch time")
    expected_switch_ratio = next(iter(source_switches), math.nan)
    episode_rail_ratios: list[float] = []
    switch_ratios: list[float] = []
    for row in episodes:
        conditioning_ratio = float(row["conditioning_seconds"]) / target_natural_time
        switch_ratio = (
            float(row["switch_time_after_conditioning"]) / target_natural_time
        )
        switch_ratios.append(switch_ratio)
        _check_close(
            errors,
            "dimensionless conditioning time",
            conditioning_ratio,
            expected_conditioning_ratio,
            tolerance,
        )
        _check_close(
            errors,
            "dimensionless capture-switch time",
            switch_ratio,
            expected_switch_ratio,
            2.0 * float(target_pi["policy_dt_ratio"]),
        )
        rail = row.get("rail_requirement", {})
        required_ratio = float(rail.get("required_rail_ratio", math.nan))
        required_half_length = float(rail.get("required_rail_half_length", math.nan))
        max_cart = float(rail.get("max_cart_center_excursion", math.nan))
        cart_half_length = float(target_pi["cart_half_length_ratio"]) * float(
            target_pi["chain_length"]
        )
        _check_close(
            errors,
            "body-aware required half rail",
            required_half_length,
            max_cart + cart_half_length,
            tolerance,
        )
        _check_close(
            errors,
            "body-aware rail ratio",
            required_ratio,
            required_half_length / float(target_pi["chain_length"]),
            tolerance,
        )
        if max_cart >= float(target_pi["rail_ratio"]) * float(
            target_pi["chain_length"]
        ):
            errors.append("cart center reached or exceeded the configured rail")
        episode_rail_ratios.append(required_ratio)
    maximum_rail_ratio = max(episode_rail_ratios, default=math.inf)
    _check_close(
        errors,
        "gate maximum rail ratio",
        float(gate.get("max_rail_ratio", math.nan)),
        maximum_rail_ratio,
        tolerance,
    )

    return {
        "label": label,
        "passed": not errors,
        "errors": errors,
        "scales": {"length": length_scale, "mass": mass_scale},
        "similarity": file_metadata(similarity_path),
        "config": config_metadata,
        "gate": file_metadata(gate_path),
        "episodes": declared_episodes,
        "successes": sum(bool(row.get("success")) for row in episodes),
        "maximum_body_aware_required_rail_ratio": maximum_rail_ratio,
        "dimensionless_switch_time_range": [
            min(switch_ratios, default=math.nan),
            max(switch_ratios, default=math.nan),
        ],
    }


def verify_similarity_grid(minimum_episodes: int = 20) -> list[dict[str, Any]]:
    source_gate = load(DEFAULT_SOURCE_GATE)
    return [
        verify_case(
            label,
            length_scale,
            mass_scale,
            similarity_path,
            config_path,
            gate_path,
            source_gate,
            minimum_episodes=minimum_episodes,
        )
        for label, (
            length_scale,
            mass_scale,
            similarity_path,
            config_path,
            gate_path,
        ) in DEFAULT_CASES.items()
    ]


def verify_route_similarity_cases(
    minimum_episodes: int = 20,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, (
        n_links,
        length_scale,
        mass_scale,
        similarity_path,
        config_path,
        route_path,
        mirror_path,
        gate_path,
    ) in DEFAULT_ROUTE_CASES.items():
        similarity = load(similarity_path)
        route = load(route_path)
        gate = load(gate_path)
        row = verify_gate(
            gate_path,
            route_path,
            mirror_path,
            n_links,
            minimum_episodes=minimum_episodes,
        )
        errors = row["errors"]
        scales = similarity.get("scales", {})
        _check_close(
            errors,
            "length scale",
            float(scales.get("length", math.nan)),
            length_scale,
            1e-12,
        )
        _check_close(
            errors,
            "mass scale",
            float(scales.get("mass", math.nan)),
            mass_scale,
            1e-12,
        )
        if (
            float(
                similarity.get("dimensionless_errors", {}).get(
                    "maximum_abs", math.inf
                )
            )
            > 1e-12
        ):
            errors.append("emitted maximum dimensionless error exceeds tolerance")
        target_pi = similarity.get("target_dimensionless")
        if route.get("target", {}).get("dimensionless") != target_pi:
            errors.append("transferred route target does not match the similarity plant")
        config_metadata = file_metadata(config_path)
        if gate.get("config", {}).get("sha256") != config_metadata["sha256"]:
            errors.append("gate config hash does not match the generated target config")
        feedback_error = float(
            route.get("transfer", {}).get(
                "coordinate_feedback_invariance_max_abs_error", math.inf
            )
        )
        if feedback_error > 1e-12:
            errors.append("coordinate feedback transfer changed the source law")
        exact_open_loop_feasible = bool(route.get("search", {}).get("is_feasible"))
        row.update(
            {
                "label": label,
                "passed": not errors,
                "scales": {"length": length_scale, "mass": mass_scale},
                "similarity": file_metadata(similarity_path),
                "config": config_metadata,
                "coordinate_feedback_invariance_max_abs_error": feedback_error,
                "exact_open_loop_warm_start_feasible": exact_open_loop_feasible,
            }
        )
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-episodes", type=int, default=20)
    parser.add_argument(
        "--out",
        default="runs/generalized_solver/similarity_ladder_n1_n2.json",
    )
    args = parser.parse_args()
    if args.minimum_episodes < 1:
        raise ValueError("minimum episodes must be positive")
    rows = verify_similarity_grid(args.minimum_episodes)
    route_rows = verify_route_similarity_cases(args.minimum_episodes)
    all_rows = [*rows, *route_rows]
    passed = all(row["passed"] for row in all_rows)
    total_episodes = sum(int(row["episodes"]) for row in all_rows)
    total_successes = sum(int(row["successes"]) for row in all_rows)
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "verified_dynamic_similarity_closed_loop_seed",
        "passed": passed,
        "summary": (
            "One dimensionless energy/PFL/exact-LQR controller passed the exact "
            "one-link 2x2 length/mass grid, and one transformed two-link "
            "feedback route passed its doubled-length/half-mass gate."
        ),
        "source_gate": file_metadata(DEFAULT_SOURCE_GATE),
        "scope": {
            "link_counts": [1, 2],
            "length_scales": [0.5, 2.0],
            "mass_scales": [0.5, 2.0],
            "minimum_episodes_per_case": args.minimum_episodes,
            "total_episodes": total_episodes,
            "total_successes": total_successes,
        },
        "verified_contract": [
            "exact Buckingham-pi plant invariance",
            "gravity-natural-time scaling",
            "force, damping, armature, rail, and body-geometry scaling",
            "unchanged dimensionless energy/PFL controller parameters",
            "exact-linearization LQR capture",
            "uninterrupted noisy hanging-start gates",
            "body-aware rail-ratio measurement",
        ],
        "rows": rows,
        "route_transfer_rows": route_rows,
        "boundary": (
            "This verifies the declared one-link family and one two-link feedback "
            "transfer. It does not prove arbitrary unequal length or mass fractions, "
            "arbitrary link count, or robust replay of one frozen open-loop force trace; "
            "higher-count fragile routes can still require exact-target refinement."
        ),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(f"passed={passed} successes={total_successes}/{total_episodes}")
    if not passed:
        for row in all_rows:
            for error in row["errors"]:
                print(f"{row['label']}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
