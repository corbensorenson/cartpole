#!/usr/bin/env python
"""Apply bounded scalar capture-LQR settings to a generalized route artifact."""

from __future__ import annotations

import argparse
import copy
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

ROOT = Path(__file__).resolve().parents[1]


def retune_route(
    payload: dict[str, Any],
    *,
    lqr_scale: float | None = None,
    lqr_control_cost: float | None = None,
    swing_feedback_scale: float | None = None,
    tail_feedback_scale: float | None = None,
) -> dict[str, Any]:
    """Return a copied route with only requested capture scalars replaced."""

    result = copy.deepcopy(payload)
    controller = result.get("controller")
    if not isinstance(controller, dict):
        raise TypeError("route must contain a controller object")
    settings: dict[str, float] = {}
    for name, value in (
        ("lqr_scale", lqr_scale),
        ("lqr_control_cost", lqr_control_cost),
    ):
        if value is None:
            continue
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
        controller[name] = float(value)
        settings[name] = float(value)
    if swing_feedback_scale is not None or tail_feedback_scale is not None:
        gains = np.asarray(controller.get("feedback_gains"), dtype=np.float64)
        prefix_steps = int(controller.get("materialized_swing_prefix_steps", -1))
        if gains.ndim != 2 or not 0 <= prefix_steps <= gains.shape[0]:
            raise ValueError(
                "segment feedback scaling requires a materialized route boundary"
            )
        for name, value in (
            ("swing_feedback_scale", swing_feedback_scale),
            ("tail_feedback_scale", tail_feedback_scale),
        ):
            if value is not None and (not math.isfinite(value) or value <= 0.0):
                raise ValueError(f"{name} must be finite and positive")
        swing_scale = 1.0 if swing_feedback_scale is None else swing_feedback_scale
        tail_scale = 1.0 if tail_feedback_scale is None else tail_feedback_scale
        gains[:prefix_steps] *= float(swing_scale)
        gains[prefix_steps:] *= float(tail_scale)
        controller["feedback_gains"] = gains.astype(float).tolist()
        if swing_feedback_scale is not None:
            settings["swing_feedback_scale"] = float(swing_feedback_scale)
        if tail_feedback_scale is not None:
            settings["tail_feedback_scale"] = float(tail_feedback_scale)
    if not settings:
        raise ValueError("at least one capture setting must be supplied")
    result["claim_status"] = "development_retuned_route_not_solution_evidence"
    result["not_solution"] = True
    result["capture_retuning"] = {
        "type": "bounded_scalar_capture_lqr_retuning",
        "settings": settings,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--lqr-scale", type=float)
    parser.add_argument("--lqr-control-cost", type=float)
    parser.add_argument("--swing-feedback-scale", type=float)
    parser.add_argument("--tail-feedback-scale", type=float)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_path = Path(args.controller)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("controller artifact must contain a JSON object")
    result = retune_route(
        payload,
        lqr_scale=args.lqr_scale,
        lqr_control_cost=args.lqr_control_cost,
        swing_feedback_scale=args.swing_feedback_scale,
        tail_feedback_scale=args.tail_feedback_scale,
    )
    result["source_controller"] = file_metadata(source_path)
    result["runtime"] = runtime_metadata()
    result["git"] = git_metadata(ROOT)
    result["generated_at"] = utc_timestamp()
    dump_json(result, Path(args.out))
    print(f"wrote {args.out}: {result['capture_retuning']['settings']}")


if __name__ == "__main__":
    main()
