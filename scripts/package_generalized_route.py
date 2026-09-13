#!/usr/bin/env python
"""Strip diagnostic traces from a route artifact and optionally mirror it."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from gcartpole.config import dump_json
from gcartpole.evidence import file_metadata, utc_timestamp
from gcartpole.generalized_solver import mirror_feedback_route


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mirror", action="store_true")
    parser.add_argument(
        "--feedback-gain-source",
        choices=("applied", "solver"),
        default="applied",
        help="Select the already-scaled gains or the optimizer's unscaled gains.",
    )
    parser.add_argument("--feedback-gain-scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.feedback_gain_scale < 0.0:
        raise ValueError("--feedback-gain-scale must be nonnegative")
    source_path = Path(args.controller)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    packaged = {
        key: copy.deepcopy(source[key])
        for key in ("schema_version", "selected_state", "controller", "search")
        if key in source
    }
    packaged.update(
        {
            "generated_at": utc_timestamp(),
            "claim_status": "development_exact_route_not_record_evidence",
            "not_solution": True,
            "summary": "Curated deterministic swing route for the bottom-up generalized-solver ladder.",
            "source": file_metadata(source_path),
        }
    )
    controller = packaged["controller"]
    if args.feedback_gain_source == "solver":
        if "solver_feedback_gains" not in controller:
            raise ValueError("controller does not contain solver feedback gains")
        source_gains = np.asarray(controller["solver_feedback_gains"], dtype=np.float64)
    else:
        source_gains = np.asarray(controller["feedback_gains"], dtype=np.float64)
    controller["feedback_gains"] = (
        (args.feedback_gain_scale * source_gains).astype(float).tolist()
    )
    controller["source_tracking_gain_scale"] = float(
        controller.get("tracking_gain_scale", 1.0)
    )
    controller["tracking_gain_scale"] = 1.0
    controller["packaged_feedback_gain_source"] = args.feedback_gain_source
    controller["packaged_feedback_gain_scale"] = float(args.feedback_gain_scale)
    controller.pop("solver_feedback_gains", None)
    if args.mirror:
        search = packaged["search"]
        controls, states, gains = mirror_feedback_route(
            np.asarray(controller["controls"], dtype=np.float64),
            np.asarray(search["nominal_coordinate_states"], dtype=np.float64),
            np.asarray(controller["feedback_gains"], dtype=np.float64),
        )
        controller["controls"] = controls.astype(float).tolist()
        controller["feedback_gains"] = gains.astype(float).tolist()
        controller["mirror_symmetry"] = True
        search["nominal_coordinate_states"] = states.astype(float).tolist()
        selected = packaged.get("selected_state")
        if isinstance(selected, dict):
            selected["qpos"] = (
                (-np.asarray(selected["qpos"], dtype=np.float64)).astype(float).tolist()
            )
            selected["qvel"] = (
                (-np.asarray(selected["qvel"], dtype=np.float64)).astype(float).tolist()
            )
    dump_json(packaged, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
