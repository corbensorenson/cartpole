#!/usr/bin/env python
"""Stitch exact feedback segments and emit their planar symmetry partner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.generalized_solver import mirror_feedback_route
from gcartpole.ilqr import stitch_feedback_trajectories


ROOT = Path(__file__).resolve().parents[1]


def arrays(payload: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controller = payload.get("controller")
    search = payload.get("search")
    if not isinstance(controller, dict) or not isinstance(search, dict):
        raise ValueError("route segment must contain controller and search records")
    return (
        np.asarray(controller["controls"], dtype=np.float64),
        np.asarray(search["nominal_coordinate_states"], dtype=np.float64),
        np.asarray(controller["feedback_gains"], dtype=np.float64),
    )


def physical_mirror(state: dict[str, Any]) -> dict[str, Any]:
    mirrored = dict(state)
    mirrored["qpos"] = (-np.asarray(state["qpos"], dtype=np.float64)).astype(float).tolist()
    mirrored["qvel"] = (-np.asarray(state["qvel"], dtype=np.float64)).astype(float).tolist()
    return mirrored


def build_artifact(
    prefix_path: Path,
    tail_path: Path,
    *,
    mirrored: bool,
) -> dict[str, Any]:
    prefix = json.loads(prefix_path.read_text(encoding="utf-8"))
    tail = json.loads(tail_path.read_text(encoding="utf-8"))
    prefix_controls, prefix_states, prefix_gains = arrays(prefix)
    tail_controls, tail_states, tail_gains = arrays(tail)
    controls, states, gains = stitch_feedback_trajectories(
        prefix_controls,
        prefix_states,
        prefix_gains,
        tail_controls,
        tail_states,
        tail_gains,
        boundary_tolerance=1e-7,
    )
    selected_state = dict(prefix["selected_state"])
    if mirrored:
        controls, states, gains = mirror_feedback_route(controls, states, gains)
        selected_state = physical_mirror(selected_state)
    tail_controller = tail["controller"]
    policy_dt = float(
        tail_controller.get(
            "policy_dt",
            tail_controller["horizon_seconds"] / len(tail_controller["controls"]),
        )
    )
    return {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_exact_route_not_release_evidence",
        "not_solution": True,
        "summary": "Deterministic prefix, exact Box-FDDP arrest, and upright LQR; optionally reflected by exact planar symmetry.",
        "selected_state": selected_state,
        "controller": {
            "type": "generalized_deterministic_prefix_box_fddp_then_lqr",
            "mirror_symmetry": bool(mirrored),
            "reset_at_boundary": False,
            "prefix": file_metadata(prefix_path),
            "tail": file_metadata(tail_path),
            "prefix_horizon_steps": int(prefix_controls.size),
            "tail_horizon_steps": int(tail_controls.size),
            "controls": controls.astype(float).tolist(),
            "feedback_gains": gains.astype(float).tolist(),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(controls.size * policy_dt),
            "policy_dt": policy_dt,
            "lqr_scale": float(tail_controller.get("lqr_scale", 1.0)),
            "lqr_control_cost": float(tail_controller.get("lqr_control_cost", 1000.0)),
        },
        "search": {
            "nominal_coordinate_states": states.astype(float).tolist(),
            "is_feasible": True,
            "iterations": int(tail.get("search", {}).get("iterations", 0)),
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(ROOT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--tail", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mirror-out")
    args = parser.parse_args()
    prefix_path = Path(args.prefix)
    tail_path = Path(args.tail)
    dump_json(build_artifact(prefix_path, tail_path, mirrored=False), Path(args.out))
    print(f"wrote {args.out}")
    if args.mirror_out:
        dump_json(build_artifact(prefix_path, tail_path, mirrored=True), Path(args.mirror_out))
        print(f"wrote {args.mirror_out}")


if __name__ == "__main__":
    main()
