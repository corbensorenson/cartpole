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
    args = parser.parse_args()
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
    if args.mirror:
        controller = packaged["controller"]
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
            selected["qpos"] = (-np.asarray(selected["qpos"], dtype=np.float64)).astype(float).tolist()
            selected["qvel"] = (-np.asarray(selected["qvel"], dtype=np.float64)).astype(float).tolist()
    dump_json(packaged, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
