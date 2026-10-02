#!/usr/bin/env python
"""Add a bounded residual that regulates a newly split joint.

The route itself remains unchanged.  In the absolute-angle coordinates used by
the exact replay helpers, the inserted relative angle is the difference
between the two duplicated absolute angles at the split.  This utility adds a
time-varying linear residual on that difference and on the inserted hinge rate
to an existing route feedback array.

The output is a development diagnostic.  It is useful for testing whether a
route's missing mode is a feedback-authority problem, but it is not evidence
until the resulting controller passes the target-plant gates.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp


ROOT = Path(__file__).resolve().parents[1]


def add_feedback(
    payload: dict[str, Any],
    *,
    split_link: int,
    angle_gain: float,
    rate_gain: float,
    start_step: int = 0,
    end_step: int | None = None,
) -> dict[str, Any]:
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("source artifact must contain a controller object")
    gains = np.asarray(controller.get("feedback_gains"), dtype=np.float64)
    if gains.ndim != 2 or gains.shape[1] % 2:
        raise ValueError("feedback_gains must be a two-block matrix")
    n_links = gains.shape[1] // 2 - 1
    if not 1 <= split_link < n_links:
        raise ValueError("split_link must identify an internal source joint")
    if not 0 <= start_step < gains.shape[0]:
        raise ValueError("start_step must lie within the route")
    if end_step is None:
        end_step = gains.shape[0]
    if not start_step < end_step <= gains.shape[0]:
        raise ValueError("end_step must be after start_step and within the route")
    if not np.isfinite(angle_gain) or not np.isfinite(rate_gain):
        raise ValueError("feedback gains must be finite")

    # Coordinate layout: [cart, absolute angles..., cart rate, hinge rates...].
    # Splitting source link `split_link` duplicates the absolute angle at the
    # next coordinate, while the inserted hinge rate is the next target rate.
    previous_angle = int(split_link)
    inserted_angle = int(split_link + 1)
    rate_block = n_links + 1
    inserted_rate = int(rate_block + split_link + 1)

    updated = copy.deepcopy(payload)
    updated_controller = updated["controller"]
    updated_gains = gains.copy()
    window = slice(start_step, end_step)
    updated_gains[window, previous_angle] -= float(angle_gain)
    updated_gains[window, inserted_angle] += float(angle_gain)
    updated_gains[window, inserted_rate] += float(rate_gain)
    updated_controller["feedback_gains"] = updated_gains.astype(float).tolist()
    if "solver_feedback_gains" in updated_controller:
        solver_gains = np.asarray(
            updated_controller["solver_feedback_gains"], dtype=np.float64
        )
        if solver_gains.shape != gains.shape:
            raise ValueError("solver_feedback_gains shape does not match feedback_gains")
        solver_gains[window, previous_angle] -= float(angle_gain)
        solver_gains[window, inserted_angle] += float(angle_gain)
        solver_gains[window, inserted_rate] += float(rate_gain)
        updated_controller["solver_feedback_gains"] = solver_gains.astype(float).tolist()

    updated["claim_status"] = "development_split_joint_feedback_not_solution"
    updated["not_solution"] = True
    updated["summary"] = (
        "Existing exact route with an explicit inserted-joint angle/rate residual; "
        "target-plant replay required."
    )
    updated["split_joint_feedback"] = {
        "split_link_one_based": int(split_link),
        "previous_absolute_angle_column": previous_angle,
        "inserted_absolute_angle_column": inserted_angle,
        "inserted_hinge_rate_column": inserted_rate,
        "angle_gain": float(angle_gain),
        "rate_gain": float(rate_gain),
        "start_step": int(start_step),
        "end_step": int(end_step),
    }
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--split-link", type=int, required=True)
    parser.add_argument("--angle-gain", type=float, required=True)
    parser.add_argument("--rate-gain", type=float, required=True)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_path = Path(args.controller)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    output = add_feedback(
        payload,
        split_link=args.split_link,
        angle_gain=args.angle_gain,
        rate_gain=args.rate_gain,
        start_step=args.start_step,
        end_step=args.end_step,
    )
    output["source_controller"] = file_metadata(source_path)
    output["runtime"] = runtime_metadata()
    output["git"] = git_metadata(ROOT)
    output["generated_at"] = utc_timestamp()
    dump_json(output, Path(args.out))
    print(
        f"wrote {args.out}: split={args.split_link} angle={args.angle_gain:.6g} "
        f"rate={args.rate_gain:.6g} steps=[{args.start_step},{output['split_joint_feedback']['end_step']})"
    )


if __name__ == "__main__":
    main()
