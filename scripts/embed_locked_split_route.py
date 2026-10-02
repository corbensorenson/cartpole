#!/usr/bin/env python
"""Embed an n-link route in an exactly split, temporarily locked n+1 plant.

The generic morphology transfer interpolates absolute-angle samples by link
centres.  That is useful for arbitrary morphology changes, but it does not
preserve a locked split of one existing segment exactly.  This diagnostic
adapter does the exact kinematic embedding needed by the split curriculum:
the added joint starts at zero, and the original distal link becomes the
distal split link.
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


def embed_coordinate_states(states: np.ndarray, split_link: int) -> np.ndarray:
    """Copy absolute-angle/hinge-rate coordinates into the split target.

    ``split_link`` is one-based in the source chain.  The source distal link
    is represented by the new target distal link, while the inserted joint is
    zero-rate and the two split links share the same absolute angle.
    """

    states = np.asarray(states, dtype=np.float64)
    if states.ndim != 2 or states.shape[1] % 2:
        raise ValueError("coordinate states must be a two-block matrix")
    source_dim = states.shape[1]
    source_links = source_dim // 2 - 1
    if not 1 <= split_link <= source_links:
        raise ValueError("split_link must be a one-based source link index")
    target_dim = source_dim + 2
    source_d = source_links + 1
    target_d = source_d + 1
    result = np.zeros((states.shape[0], target_dim), dtype=np.float64)
    # Cart position and the absolute-angle field are copied with one duplicate
    # at the split.  The original distal link is represented by the second
    # half of the split segment.
    result[:, 0] = states[:, 0]
    result[:, 1 : split_link + 1] = states[:, 1 : split_link + 1]
    result[:, split_link + 1] = states[:, split_link]
    if split_link < source_links:
        result[:, split_link + 2 : source_links + 2] = states[
            :, split_link + 1 : source_links + 1
        ]
    # Cart velocity is copied directly.
    result[:, target_d] = states[:, source_d]
    # Hinge rates before and at the split are copied; the inserted joint is
    # locked at zero. Later rates shift by one target joint.
    result[:, target_d + 1 : target_d + split_link + 1] = states[
        :, source_d + 1 : source_d + split_link + 1
    ]
    result[:, target_d + split_link + 1] = 0.0
    if split_link < source_links:
        result[:, target_d + split_link + 2 :] = states[
            :, source_d + split_link + 1 :
        ]
    return result


def embed_feedback_gains(gains: np.ndarray, split_link: int) -> np.ndarray:
    """Project target coordinate errors back to the source route coordinates."""

    gains = np.asarray(gains, dtype=np.float64)
    if gains.ndim != 2 or gains.shape[1] % 2:
        raise ValueError("feedback gains must be a two-block matrix")
    source_dim = gains.shape[1]
    source_links = source_dim // 2 - 1
    if not 1 <= split_link <= source_links:
        raise ValueError("split_link must be a one-based source link index")
    source_d = source_links + 1
    target_d = source_d + 1
    result = np.zeros((gains.shape[0], source_dim + 2), dtype=np.float64)
    result[:, 0] = gains[:, 0]
    # Project target errors onto the source coordinates. The proximal target
    # angle/rate carries the source split coordinate; the inserted angle/rate
    # is a new zero-error coordinate while the equality is locked.
    result[:, 1 : split_link + 1] = gains[:, 1 : split_link + 1]
    if split_link < source_links:
        result[:, split_link + 2 : source_links + 2] = gains[
            :, split_link + 1 : source_links + 1
        ]
    result[:, target_d] = gains[:, source_d]
    result[:, target_d + 1 : target_d + split_link] = gains[
        :, source_d + 1 : source_d + split_link
    ]
    # The source split-joint rate is represented by the proximal target hinge
    # rate. The newly inserted hinge has zero nominal rate while locked.
    result[:, target_d + split_link] = gains[:, source_d + split_link]
    result[:, target_d + split_link + 1] = 0.0
    if split_link < source_links:
        result[:, target_d + split_link + 2 :] = gains[:, source_d + split_link + 1 :]
    return result


def embed_physical_state(qpos: np.ndarray, qvel: np.ndarray, split_link: int) -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    source_links = qpos.size - 1
    if qvel.shape != qpos.shape or not 1 <= split_link <= source_links:
        raise ValueError("physical state dimensions or split_link are invalid")
    target_qpos = np.zeros(source_links + 2, dtype=np.float64)
    target_qvel = np.zeros(source_links + 2, dtype=np.float64)
    target_qpos[0] = qpos[0]
    target_qvel[0] = qvel[0]
    target_qpos[1 : split_link + 1] = qpos[1 : split_link + 1]
    target_qpos[1 + split_link] = 0.0
    target_qpos[2 + split_link :] = qpos[1 + split_link :]
    target_qvel[1 : split_link + 1] = qvel[1 : split_link + 1]
    target_qvel[1 + split_link] = 0.0
    target_qvel[2 + split_link :] = qvel[1 + split_link :]
    return target_qpos, target_qvel


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--source-links", type=int, required=True)
    parser.add_argument("--split-link", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_path = Path(args.controller)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_controller = payload.get("controller")
    source_search = payload.get("search")
    if not isinstance(source_controller, dict) or not isinstance(source_search, dict):
        raise ValueError("source artifact must contain controller and search objects")
    controls = np.asarray(source_controller["controls"], dtype=np.float64)
    gains = np.asarray(source_controller["feedback_gains"], dtype=np.float64)
    states = np.asarray(source_search["nominal_coordinate_states"], dtype=np.float64)
    expected_dim = 2 * (args.source_links + 1)
    if controls.ndim != 1 or gains.shape != (controls.size, expected_dim):
        raise ValueError("source controls and feedback dimensions do not match")
    if states.shape != (controls.size + 1, expected_dim):
        raise ValueError("source nominal state dimensions do not match controls")
    selected = payload.get("selected_state")
    if not isinstance(selected, dict):
        raise ValueError("source artifact must contain selected_state")
    qpos, qvel = embed_physical_state(
        np.asarray(selected["qpos"], dtype=np.float64),
        np.asarray(selected["qvel"], dtype=np.float64),
        args.split_link,
    )
    controller = copy.deepcopy(source_controller)
    controller["type"] = "exact_locked_split_route_embedding"
    controller["controls"] = controls.astype(float).tolist()
    controller["feedback_gains"] = embed_feedback_gains(gains, args.split_link).astype(float).tolist()
    controller["horizon_steps"] = int(controls.size)
    search = copy.deepcopy(source_search)
    search["nominal_coordinate_states"] = embed_coordinate_states(states, args.split_link).astype(float).tolist()
    search["is_feasible"] = True
    output: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_locked_split_embedding_not_solution_evidence",
        "not_solution": True,
        "summary": "Exact kinematic embedding of a solved lower-count route in a temporarily locked split-link plant.",
        "source": file_metadata(source_path),
        "selected_state": {"qpos": qpos.astype(float).tolist(), "qvel": qvel.astype(float).tolist(), "state_index": 0},
        "controller": controller,
        "search": search,
        "embedding": {
            "source_links": int(args.source_links),
            "target_links": int(args.source_links + 1),
            "split_link": int(args.split_link),
            "inserted_joint_initially_locked": True,
            "exact_route_embedding": True,
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(ROOT),
    }
    dump_json(output, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
