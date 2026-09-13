#!/usr/bin/env python
"""Pad a saved n-link FDDP route for a larger chain as a warm start only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json
from gcartpole.evidence import data_sha256, file_metadata, utc_timestamp


def pad_coordinate_state(state: np.ndarray, source_links: int, target_links: int) -> np.ndarray:
    source_d = source_links + 1
    target_d = target_links + 1
    qpos = state[:source_d]
    qvel = state[source_d : 2 * source_d]
    extra = target_links - source_links
    # Positions contain absolute link angles; velocities contain relative
    # hinge rates. A collinear added link has zero relative angle and rate.
    return np.r_[qpos, np.repeat(qpos[-1], extra), qvel, np.zeros(extra)]


def pad_feedback_gains(gains: np.ndarray, source_links: int, target_links: int) -> np.ndarray:
    source_d = source_links + 1
    target_d = target_links + 1
    gains = np.asarray(gains, dtype=np.float64)
    if gains.ndim != 2 or gains.shape[1] != 2 * source_d:
        raise ValueError("feedback columns do not match source link count")
    padded = np.zeros((gains.shape[0], 2 * target_d), dtype=np.float64)
    padded[:, :source_d] = gains[:, :source_d]
    padded[:, target_d : target_d + source_d] = gains[:, source_d:]
    return padded


def pad_physical_state(state: dict[str, Any], source_links: int, target_links: int) -> dict[str, Any]:
    qpos = np.asarray(state["qpos"], dtype=np.float64)
    qvel = np.asarray(state["qvel"], dtype=np.float64)
    extra = target_links - source_links
    return {
        **state,
        "qpos": np.r_[qpos, np.zeros(extra)].astype(float).tolist(),
        "qvel": np.r_[qvel, np.zeros(extra)].astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-links", type=int, required=True)
    parser.add_argument("--target-links", type=int, required=True)
    parser.add_argument("--extra-steps", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.source_links < 1 or args.target_links <= args.source_links:
        raise ValueError("target-links must be greater than source-links")
    if args.extra_steps < 0:
        raise ValueError("extra-steps must be nonnegative")

    source_path = Path(args.source)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    controller = copy.deepcopy(payload["controller"])
    search = payload["search"]
    nominal = np.asarray(search["nominal_coordinate_states"], dtype=np.float64)
    feedback = np.asarray(controller["feedback_gains"], dtype=np.float64)
    nominal_padded = np.asarray(
        [pad_coordinate_state(row, args.source_links, args.target_links) for row in nominal]
    )
    state_dim = 2 * (args.target_links + 1)
    feedback_padded = pad_feedback_gains(feedback, args.source_links, args.target_links)
    controls = np.asarray(controller["controls"], dtype=np.float64)
    policy_dt = float(controller["horizon_seconds"]) / controls.size
    if args.extra_steps:
        controls = np.r_[controls, np.zeros(args.extra_steps, dtype=np.float64)]
        terminal_state = nominal_padded[-1]
        nominal_padded = np.vstack(
            [nominal_padded, np.repeat(terminal_state[None, :], args.extra_steps, axis=0)]
        )
        feedback_padded = np.vstack(
            [feedback_padded, np.zeros((args.extra_steps, state_dim), dtype=np.float64)]
        )
    controller["controls"] = controls.astype(float).tolist()
    controller["horizon_steps"] = int(controls.size)
    controller["horizon_seconds"] = float(controls.size * policy_dt)
    controller["feedback_gains"] = feedback_padded.astype(float).tolist()
    controller.pop("solver_feedback_gains", None)
    search = {
        "converged": False,
        "is_feasible": False,
        "iterations": 0,
        "nominal_coordinate_states": nominal_padded.astype(float).tolist(),
    }
    selected = payload.get("selected_state")
    if isinstance(selected, dict):
        selected = pad_physical_state(selected, args.source_links, args.target_links)
    out = {
        "schema_version": 2,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Padded larger-chain FDDP warm start; requires exact target-chain re-optimization.",
        "selected_state": selected,
        "controller": {
            **controller,
            "source_padding": {
                "source": file_metadata(source_path),
                "source_links": int(args.source_links),
                "target_links": int(args.target_links),
                "extra_steps": int(args.extra_steps),
                "coordinate_rule": "duplicate terminal absolute angle; zero added relative rates; preserve position/velocity feedback blocks",
                "mapping_version": 2,
            },
        },
        "search": search,
        "evidence": {
            **payload.get("evidence", {}),
            "source_padding_sha256": data_sha256(
                {
                    "source_links": args.source_links,
                    "target_links": args.target_links,
                    "nominal_sha256": data_sha256(nominal_padded.tolist()),
                    "feedback_sha256": data_sha256(feedback_padded.tolist()),
                }
            ),
        },
    }
    dump_json(out, args.out)
    print(f"Wrote padded {args.source_links}->{args.target_links} warm start to {args.out}")


if __name__ == "__main__":
    main()
