#!/usr/bin/env python
"""Transfer spline geometry in material coordinates; feasibility is not inherited."""
import argparse
import json
from pathlib import Path

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from search_capture_sequence import fixed_state_cfg


def transfer_coefficients(points, target_links):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2:
        raise ValueError("two-dimensional spline coefficient matrix required")
    source_links = points.shape[1]-1
    if points.ndim != 2 or source_links < 1 or target_links < 1 or not np.all(np.isfinite(points)):
        raise ValueError("finite cart/joint spline coefficient matrix and positive target count required")
    source_centers = (np.arange(source_links)+.5)/source_links
    target_centers = (np.arange(target_links)+.5)/target_links
    absolute = np.cumsum(points[:, 1:], axis=1)
    mapped = np.array([np.interp(target_centers, source_centers, row) for row in absolute])
    relative = np.diff(np.column_stack([np.zeros(len(points)), mapped]), axis=1)
    return np.column_stack([points[:, 0], relative])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    source = json.loads(Path(args.source).read_text())
    cfg = load_config(args.config)
    n = cfg["env"]["n_links"]
    points = transfer_coefficients(source["search"]["spline_control_points"], n)
    selected = dict(qpos=points[0].tolist(), qvel=np.zeros(n+1).tolist())
    cfg = fixed_state_cfg(cfg, selected, 30.)
    dump_json(dict(schema_version=1, not_solution=True, initialization_only=True,
                   generated_at=utc_timestamp(), config=file_metadata(args.config),
                   effective_config=cfg, selected_state=selected, runtime=runtime_metadata(),
                   source=file_metadata(args.source),
                   controller=dict(horizon_seconds=source["controller"]["horizon_seconds"]),
                   search=dict(spline_control_points=points.tolist(),
                               spline_knots=source["search"]["spline_knots"]),
                   note="Linear interpolation of absolute-angle spline coefficients at normalized link centers, with clamped endpoint extrapolation. Cart curve is retained. Target dynamics and actuator feasibility must be optimized and independently replayed."), args.out)


if __name__ == "__main__":
    main()
