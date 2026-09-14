#!/usr/bin/env python
"""Verify and summarize a promoted physical-similarity continuation."""

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
from gcartpole.generalized_solver import (
    dimensionless_setup,
    mirror_feedback_route,
    setup_from_config,
)
from scripts.generalized_swingup_solver import uniform_config

ROOT = Path(__file__).resolve().parents[1]


def payload(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--source-controller", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--route", required=True)
    parser.add_argument("--mirror", required=True)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--continuation", required=True)
    parser.add_argument("--n-links", type=int, required=True)
    parser.add_argument("--length-scale", type=float, required=True)
    parser.add_argument("--mass-scale", type=float, required=True)
    parser.add_argument("--minimum-episodes", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    paths = {
        name: Path(value)
        for name, value in {
            "source_config": args.source_config,
            "source_controller": args.source_controller,
            "target_config": args.target_config,
            "route": args.route,
            "mirror": args.mirror,
            "gate": args.gate,
            "continuation": args.continuation,
        }.items()
    }
    source_cfg = uniform_config(load_config(paths["source_config"]), args.n_links)
    target_cfg = load_config(paths["target_config"])
    source_setup = setup_from_config(source_cfg)
    target_setup = setup_from_config(target_cfg)
    source_pi = dimensionless_setup(source_setup)
    target_pi = dimensionless_setup(target_setup)
    require(target_setup.n_links == args.n_links, "target link count mismatch")
    require(
        np.isclose(target_setup.chain_length / source_setup.chain_length, args.length_scale),
        "target length scale mismatch",
    )
    require(
        np.isclose(target_setup.system_mass / source_setup.system_mass, args.mass_scale),
        "target mass scale mismatch",
    )
    for name in (
        "length_fractions",
        "mass_fractions",
        "joint_damping_ratios",
    ):
        require(
            np.allclose(getattr(source_pi, name), getattr(target_pi, name), atol=1e-12),
            f"dimensionless {name} mismatch",
        )
    for name in (
        "cart_to_link_mass",
        "force_authority",
        "rail_ratio",
        "usable_rail_ratio",
        "cart_half_length_ratio",
        "link_radius_ratio",
        "policy_dt_ratio",
        "cart_damping_ratio",
        "joint_armature_ratio",
    ):
        require(
            np.isclose(getattr(source_pi, name), getattr(target_pi, name), atol=1e-12),
            f"dimensionless {name} mismatch",
        )

    route = payload(paths["route"])
    mirror = payload(paths["mirror"])
    route_controller = route["controller"]
    mirror_controller = mirror["controller"]
    route_search = route["search"]
    controls = np.asarray(route_controller["controls"], dtype=np.float64)
    states = np.asarray(route_search["nominal_coordinate_states"], dtype=np.float64)
    gains = np.asarray(route_controller["feedback_gains"], dtype=np.float64)
    require(states.shape == (controls.size + 1, 2 * (args.n_links + 1)), "route state shape mismatch")
    require(gains.shape == (controls.size, states.shape[1]), "route gain shape mismatch")
    require(bool(route_controller.get("periodic_coordinate_errors")), "periodic feedback is not enabled")
    prefix = int(route_controller.get("materialized_swing_prefix_steps", -1))
    tail = int(route_controller.get("materialized_lqr_tail_steps", -1))
    require(prefix >= 0 and tail > 0 and prefix + tail == controls.size, "materialized route boundary mismatch")
    expected_controls, expected_states, expected_gains = mirror_feedback_route(
        controls, states, gains
    )
    require(np.array_equal(expected_controls, np.asarray(mirror_controller["controls"])), "mirror controls mismatch")
    require(
        np.array_equal(expected_states, np.asarray(mirror["search"]["nominal_coordinate_states"])),
        "mirror states mismatch",
    )
    require(np.array_equal(expected_gains, np.asarray(mirror_controller["feedback_gains"])), "mirror gains mismatch")

    gate = payload(paths["gate"])
    episodes = gate.get("episode_results")
    require(isinstance(episodes, list) and len(episodes) >= args.minimum_episodes, "insufficient gate episodes")
    require(float(gate.get("success_rate", 0.0)) == 1.0, "gate did not pass")
    require(all(row.get("success") is True for row in episodes), "gate contains a failed episode")
    require(
        all(row["predictions"][int(row["selected_route"])]["success"] is True for row in episodes),
        "prediction/execution agreement failed",
    )
    gate_hashes = {item["sha256"] for item in gate.get("controllers", [])}
    require(file_metadata(paths["route"])["sha256"] in gate_hashes, "gate route hash mismatch")
    require(file_metadata(paths["mirror"])["sha256"] in gate_hashes, "gate mirror hash mismatch")

    continuation = payload(paths["continuation"])
    require(continuation.get("passed") is True, "continuation did not pass")
    require(float(continuation["accepted"]["progress"]) == 1.0, "continuation did not reach target")
    trace = [
        {
            "trial": int(row["trial"]),
            "progress": float(row["proposed_progress"]),
            "step": float(row["step"]),
            "accepted": bool(row["accepted"]),
            "transfer_passed": bool(row["transfer_passed"]),
            "route_passed": bool(row["materialized_route_passed"]),
            "global_feedback_scale": row.get("selected_feedback_gain_scale"),
            "segment_feedback_scales": row.get("selected_segment_feedback_gain_scales"),
        }
        for row in continuation["trials"]
    ]
    rail_ratio = max(float(row["rail_requirement"]["required_rail_ratio"]) for row in episodes)
    holds = [float(row["max_upright_streak_seconds"]) for row in episodes]
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "verified_generalized_similarity_continuation",
        "passed": True,
        "summary": (
            "A link-count-independent continuation transferred an executed swing-and-hold "
            "route through an exact dynamic-similarity family with bounded scalar adaptation."
        ),
        "source": {
            "config": file_metadata(paths["source_config"]),
            "controller": file_metadata(paths["source_controller"]),
            "n_links": args.n_links,
        },
        "target": {
            "config": file_metadata(paths["target_config"]),
            "route": file_metadata(paths["route"]),
            "mirror": file_metadata(paths["mirror"]),
            "length_scale": args.length_scale,
            "mass_scale": args.mass_scale,
            "maximum_dimensionless_error": 0.0,
        },
        "continuation": {
            "input_sha256": file_metadata(paths["continuation"])["sha256"],
            "trials": len(trace),
            "accepted_trials": sum(row["accepted"] for row in trace),
            "rejected_trials": sum(not row["accepted"] for row in trace),
            "trace": trace,
        },
        "gate": {
            "artifact": file_metadata(paths["gate"]),
            "episodes": len(episodes),
            "successes": len(episodes),
            "prediction_execution_agreement": len(episodes),
            "minimum_upright_hold_seconds": min(holds),
            "maximum_upright_hold_seconds": max(holds),
            "maximum_required_rail_ratio": rail_ratio,
            "configured_rail_ratio": target_pi.rail_ratio,
        },
        "route_structure": {
            "steps": int(controls.size),
            "swing_prefix_steps": prefix,
            "analytic_tail_steps": tail,
            "periodic_coordinate_errors": True,
            "mirror_exact": True,
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(ROOT),
    }
    dump_json(output, Path(args.out))
    print(
        f"verified {len(episodes)}/{len(episodes)} episodes, "
        f"rho={rail_ratio:.6f}, trials={len(trace)} -> {args.out}"
    )


if __name__ == "__main__":
    main()
