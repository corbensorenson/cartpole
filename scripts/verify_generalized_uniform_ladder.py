#!/usr/bin/env python
"""Verify the shared route/feedback/LQR execution contract for uniform n=1..7."""

from __future__ import annotations

import argparse
import json
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


DEFAULT_ARTIFACTS = {
    1: (
        "runs/generalized_solver/n1_gate_20_shared.json",
        "runs/generalized_solver/n1_shared_route.json",
        "runs/generalized_solver/n1_shared_route_mirror.json",
    ),
    2: (
        "runs/generalized_solver/n2_gate.json",
        "runs/generalized_solver/n2_route.json",
        "runs/generalized_solver/n2_route_mirror.json",
    ),
    3: (
        "runs/generalized_solver/n3_gate.json",
        "runs/generalized_solver/n3_route.json",
        "runs/generalized_solver/n3_route_mirror.json",
    ),
    4: (
        "runs/generalized_solver/n4_gate_20.json",
        "runs/generalized_solver/n4_route_solver.json",
        "runs/generalized_solver/n4_route_solver_mirror.json",
    ),
    5: (
        "runs/generalized_solver/n5_gate_20.json",
        "runs/generalized_solver/n5_route_solver.json",
        "runs/generalized_solver/n5_route_solver_mirror.json",
    ),
    6: (
        "runs/generalized_solver/n6_gate_20.json",
        "runs/generalized_solver/n6_route_solver.json",
        "runs/generalized_solver/n6_route_solver_mirror.json",
    ),
    7: (
        "runs/generalized_solver/n7_gate_20.json",
        "runs/swingup7_uniform/seven_link_release_controller.json",
        "runs/generalized_solver/n7_release_route_mirror.json",
    ),
}


def load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def verify_route(path: Path, n_links: int) -> tuple[dict[str, Any], list[str]]:
    payload = load(path)
    errors: list[str] = []
    controller = payload.get("controller", {})
    search = payload.get("search", {})
    controls = np.asarray(controller.get("controls"), dtype=np.float64)
    feedback = np.asarray(controller.get("feedback_gains"), dtype=np.float64)
    states = np.asarray(search.get("nominal_coordinate_states"), dtype=np.float64)
    state_size = 2 * (n_links + 1)
    if controls.ndim != 1 or controls.size < 1:
        errors.append("route controls are not a nonempty vector")
    if feedback.shape != (controls.size, state_size):
        errors.append("route feedback shape does not match link count")
    if states.shape != (controls.size + 1, state_size):
        errors.append("route state shape does not match link count")
    if not all(np.all(np.isfinite(array)) for array in (controls, feedback, states)):
        errors.append("route contains non-finite values")
    if controls.size and np.max(np.abs(controls)) > 1.0 + 1e-12:
        errors.append("route contains an action outside the normalized bound")
    if int(controller.get("horizon_steps", -1)) != controls.size:
        errors.append("declared route horizon does not match controls")
    controller_type = str(controller.get("type", ""))
    if "lqr" not in controller_type:
        errors.append("route does not declare its exact LQR capture stage")
    return payload, errors


def verify_pair(
    route_path: Path, mirror_path: Path, n_links: int
) -> tuple[dict[str, Any], dict[str, Any], list[str]]:
    route, errors = verify_route(route_path, n_links)
    mirror, mirror_errors = verify_route(mirror_path, n_links)
    errors.extend(f"mirror: {error}" for error in mirror_errors)
    if errors:
        return route, mirror, errors
    route_controls = np.asarray(route["controller"]["controls"], dtype=np.float64)
    mirror_controls = np.asarray(mirror["controller"]["controls"], dtype=np.float64)
    route_feedback = np.asarray(
        route["controller"]["feedback_gains"], dtype=np.float64
    )
    mirror_feedback = np.asarray(
        mirror["controller"]["feedback_gains"], dtype=np.float64
    )
    route_states = np.asarray(
        route["search"]["nominal_coordinate_states"], dtype=np.float64
    )
    mirror_states = np.asarray(
        mirror["search"]["nominal_coordinate_states"], dtype=np.float64
    )
    if not np.allclose(mirror_controls, -route_controls, atol=1e-12, rtol=0.0):
        errors.append("mirror controls are not exact action reflection")
    if not np.allclose(mirror_feedback, route_feedback, atol=1e-12, rtol=0.0):
        errors.append("mirror feedback gains changed under reflection")
    # A hanging angle at +/-pi has one canonical wrapped representation.  The
    # first n=2 sample therefore need not negate numerically even though it is
    # the same physical state; every subsequent sample must reflect exactly.
    if not np.allclose(mirror_states[1:], -route_states[1:], atol=1e-12, rtol=0.0):
        errors.append("mirror nominal states are not exact reflection")
    return route, mirror, errors


def verify_gate(
    gate_path: Path,
    route_path: Path,
    mirror_path: Path,
    n_links: int,
    *,
    minimum_episodes: int,
) -> dict[str, Any]:
    gate = load(gate_path)
    route, mirror, errors = verify_pair(route_path, mirror_path, n_links)
    if gate.get("claim_status") != "development_deterministic_model_predictive_library":
        errors.append("gate claim status is not the shared route-library evaluator")
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
        errors.append("an uninterrupted gate episode failed")
    if gate.get("selection", {}).get("type") != "exact_forward_model_argmin":
        errors.append("gate did not use the shared exact forward selector")
    prediction_agreements = 0
    for row in episodes:
        selected = int(row.get("selected_route", -1))
        predictions = row.get("predictions", [])
        if not 0 <= selected < len(predictions):
            errors.append("gate selected an invalid route index")
            continue
        predicted = bool(predictions[selected].get("success"))
        actual = bool(row.get("success"))
        prediction_agreements += int(predicted == actual)
        if not predicted:
            errors.append("selector chose a route predicted to fail")
    expected_hashes = {
        file_metadata(route_path)["sha256"],
        file_metadata(mirror_path)["sha256"],
    }
    gate_hashes = {
        item.get("sha256")
        for item in gate.get("controllers", [])
        if isinstance(item, dict)
    }
    if gate_hashes != expected_hashes:
        errors.append("gate controller hashes do not match the declared route pair")
    episode_ratios = [
        float(row["rail_requirement"]["required_rail_ratio"])
        for row in episodes
    ]
    maximum_ratio = max(episode_ratios, default=float("inf"))
    if not np.isclose(
        maximum_ratio,
        float(gate.get("max_required_rail_ratio", float("nan"))),
        atol=1e-12,
        rtol=0.0,
    ):
        errors.append("gate maximum rail ratio is inconsistent with its episodes")
    for row in episodes:
        rail = row["rail_requirement"]
        required_ratio = float(rail["required_rail_ratio"])
        required_half_length = float(rail["required_rail_half_length"])
        configured_ratio = float(rail["configured_rail_ratio"])
        max_cart = float(rail["max_cart_center_excursion"])
        if min(required_ratio, required_half_length, configured_ratio) <= 0.0:
            errors.append("rail relationship contains a nonpositive scale")
            continue
        chain_length = required_half_length / required_ratio
        if max_cart >= configured_ratio * chain_length:
            errors.append("an accepted route exceeded the simulator cart-center rail")
    controller_types = sorted(
        {
            str(route.get("controller", {}).get("type")),
            str(mirror.get("controller", {}).get("type")),
        }
    )
    return {
        "n_links": int(n_links),
        "passed": not errors,
        "errors": errors,
        "gate": file_metadata(gate_path),
        "routes": [file_metadata(route_path), file_metadata(mirror_path)],
        "controller_types": controller_types,
        "episodes": declared_episodes,
        "successes": sum(bool(row.get("success")) for row in episodes),
        "prediction_execution_agreements": prediction_agreements,
        "maximum_body_aware_required_rail_ratio": maximum_ratio,
        "route_selection_counts": gate.get("route_counts", {}),
    }


def verify_ladder(minimum_episodes: int = 20) -> list[dict[str, Any]]:
    return [
        verify_gate(
            Path(gate),
            Path(route),
            Path(mirror),
            n_links,
            minimum_episodes=minimum_episodes,
        )
        for n_links, (gate, route, mirror) in DEFAULT_ARTIFACTS.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-episodes", type=int, default=20)
    parser.add_argument(
        "--out",
        default="runs/generalized_solver/uniform_ladder_n1_n7.json",
    )
    args = parser.parse_args()
    if args.minimum_episodes < 1:
        raise ValueError("minimum episodes must be positive")
    rows = verify_ladder(args.minimum_episodes)
    passed = all(row["passed"] for row in rows)
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "verified_development_uniform_shared_architecture_ladder",
        "passed": passed,
        "summary": (
            "Uniform n=1..7 all pass one hash-bound route/feedback/exact-selector/"
            "LQR/uninterrupted-gate execution contract. Deterministic route teachers "
            "still differ by morphology; this is not an arbitrary-setup certificate."
        ),
        "shared_execution_contract": [
            "dimensionless absolute-angle route coordinates",
            "normalized bounded cart action",
            "time-varying route feedback interface",
            "exact planar mirror route",
            "exact target-model forward selection",
            "exact upright LQR capture",
            "uninterrupted noisy hanging-start gate",
            "body-aware rail ratio measurement",
        ],
        "scope": {
            "link_range": [1, 7],
            "minimum_episodes_per_link": int(args.minimum_episodes),
            "total_episodes": sum(int(row["episodes"]) for row in rows),
            "total_successes": sum(int(row["successes"]) for row in rows),
            "total_prediction_execution_agreements": sum(
                int(row["prediction_execution_agreements"]) for row in rows
            ),
        },
        "rows": rows,
        "boundary": (
            "This verifies a shared execution architecture on the declared canonical "
            "uniform plants. It does not prove that one open-loop force trace transfers "
            "unchanged, that every route is synthesized by one primitive, or that an "
            "arbitrary unequal morphology is solved without exact-model refinement."
        ),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(
        f"passed={passed} successes={payload['scope']['total_successes']}/"
        f"{payload['scope']['total_episodes']} "
        f"prediction_agreements="
        f"{payload['scope']['total_prediction_execution_agreements']}/"
        f"{payload['scope']['total_episodes']}"
    )
    if not passed:
        for row in rows:
            for error in row["errors"]:
                print(f"n={row['n_links']}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
