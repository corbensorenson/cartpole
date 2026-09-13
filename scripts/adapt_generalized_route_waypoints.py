#!/usr/bin/env python
"""Retarget a neighboring exact route with short deterministic waypoint solves."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform
from gcartpole.waypoint_adaptation import adapt_route_to_waypoints

try:
    from scripts.refine_ilqr_capture_chain import source_trajectory
except ModuleNotFoundError:
    from refine_ilqr_capture_chain import source_trajectory


def deterministic_config(config: dict) -> dict:
    result = copy.deepcopy(config)
    result["env"]["init_mode"] = "hanging"
    result["env"]["terminate_abs_angle"] = None
    result["env"].setdefault("action_lqr_residual", {})["enabled"] = False
    result["env"].setdefault("action_lqr_switch", {})["enabled"] = False
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        result["env"][key] = 0.0
        result["env"][f"{key}_start"] = 0.0
        result["env"][f"{key}_end"] = 0.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--segment-steps", type=int, default=12)
    parser.add_argument("--max-evaluations", type=int, default=40)
    parser.add_argument("--endpoint-weight", type=float, default=1000.0)
    parser.add_argument("--control-regularization", type=float, default=1.0e-4)
    parser.add_argument("--rail-soft-limit", type=float, required=True)
    parser.add_argument("--rail-weight", type=float, default=1000.0)
    parser.add_argument("--endpoint-tolerance", type=float, default=0.05)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = deterministic_config(
        apply_overrides(load_config(args.config), args.override)
    )
    source_path = Path(args.controller)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    controls, reference_states, _feedback = source_trajectory(source)
    distribution = load_config(args.spec)["distribution"]
    transform = dimensionless_absolute_transform(
        int(cfg["env"]["n_links"]),
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    transition = MujocoTransition(env, coordinate_transform=transform)
    initial_state = transition.to_coordinates(data_state(env.data))
    result = adapt_route_to_waypoints(
        transition,
        initial_state,
        controls,
        reference_states,
        segment_steps=args.segment_steps,
        max_evaluations=args.max_evaluations,
        endpoint_weight=args.endpoint_weight,
        control_regularization=args.control_regularization,
        rail_soft_limit=args.rail_soft_limit * transform[0, 0],
        rail_weight=args.rail_weight,
        endpoint_tolerance=args.endpoint_tolerance,
    )
    physical_states = np.asarray(
        [transition.to_physical(state) for state in result.states],
        dtype=np.float64,
    )
    env.close()
    artifact = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "deterministic_waypoint_adaptation_not_solution",
        "not_solution": True,
        "summary": (
            "Short-horizon exact-model waypoint adaptation; requires full FDDP, "
            "feedback, and noisy-gate validation."
        ),
        "source_controller": file_metadata(source_path),
        "controller": {
            "type": "exact_target_waypoint_adaptation",
            "horizon_seconds": float(controls.size * env.dt),
            "controls": result.controls.astype(float).tolist(),
            "nominal_coordinate_states": result.states.astype(float).tolist(),
            "feedback_gains": np.zeros(
                (controls.size, initial_state.size), dtype=np.float64
            ).tolist(),
        },
        "search": {
            "success": bool(result.success),
            "segment_steps": int(args.segment_steps),
            "segment_seconds": float(args.segment_steps * env.dt),
            "max_evaluations": int(args.max_evaluations),
            "endpoint_weight": float(args.endpoint_weight),
            "control_regularization": float(args.control_regularization),
            "rail_soft_limit": float(args.rail_soft_limit),
            "rail_weight": float(args.rail_weight),
            "endpoint_tolerance": float(args.endpoint_tolerance),
            "maximum_endpoint_error_norm": float(
                max(row["endpoint_error_norm"] for row in result.segments)
            ),
            "maximum_action_correction": float(
                max(row["maximum_action_correction"] for row in result.segments)
            ),
            "maximum_cart_excursion": float(np.max(np.abs(physical_states[:, 0]))),
            "segments": result.segments,
            "nominal_coordinate_states": result.states.astype(float).tolist(),
        },
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(artifact, args.out)
    print(
        f"success={result.success} segments={len(result.segments)} "
        f"endpoint_error={artifact['search']['maximum_endpoint_error_norm']:.3e} "
        f"action_correction={artifact['search']['maximum_action_correction']:.3f} "
        f"cart={artifact['search']['maximum_cart_excursion']:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
