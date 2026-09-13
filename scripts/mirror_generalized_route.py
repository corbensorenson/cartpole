#!/usr/bin/env python
"""Create the exact left/right symmetry partner of a feedback route."""

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
    args = parser.parse_args()
    source_path = Path(args.controller)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    result = copy.deepcopy(source)
    controller = result["controller"]
    search = result["search"]
    controls, states, gains = mirror_feedback_route(
        np.asarray(controller["controls"], dtype=np.float64),
        np.asarray(search["nominal_coordinate_states"], dtype=np.float64),
        np.asarray(controller["feedback_gains"], dtype=np.float64),
    )
    controller["controls"] = controls.astype(float).tolist()
    controller["feedback_gains"] = gains.astype(float).tolist()
    controller["mirror_symmetry"] = True
    controller["symmetry_source"] = file_metadata(source_path)
    search["nominal_coordinate_states"] = states.astype(float).tolist()
    selected = result.get("selected_state")
    if isinstance(selected, dict):
        selected["qpos"] = (-np.asarray(selected["qpos"], dtype=np.float64)).astype(float).tolist()
        selected["qvel"] = (-np.asarray(selected["qvel"], dtype=np.float64)).astype(float).tolist()
    result["generated_at"] = utc_timestamp()
    result["claim_status"] = "development_exact_route_not_release_evidence"
    result["not_solution"] = True
    result["summary"] = "Exact planar-symmetry partner of a deterministic feedback route."
    result.pop("result", None)
    dump_json(result, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
