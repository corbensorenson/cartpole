#!/usr/bin/env python
"""Materialize an executed swing-and-hold rollout as one transferable route."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import (
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.modal import transform_feedback_gain
from scripts.generalized_swingup_solver import coordinate_transform
from scripts.search_swingup_capture import lqr_gain

ROOT = Path(__file__).resolve().parents[1]


def materialize_rollout(
    payload: dict[str, Any],
    transform: np.ndarray,
    tail_feedback: np.ndarray,
) -> dict[str, Any]:
    """Replace planned states/actions with the exact executed rollout.

    The optimizer feedback remains on the swing prefix.  Once the source replay
    entered its analytic LQR tail, the equivalent coordinate-space LQR gain is
    repeated.  This produces one continuous route whose nominal state and
    feedforward action are the behavior that actually passed, not merely the
    optimizer trajectory around which that behavior was generated.
    """

    result = copy.deepcopy(payload)
    controller = result.get("controller")
    selected = result.get("selected_state")
    replay = result.get("result")
    if not isinstance(controller, dict):
        raise TypeError("replay must contain a controller object")
    if not isinstance(selected, dict):
        raise TypeError("replay must contain a selected_state object")
    if not isinstance(replay, dict) or not isinstance(replay.get("trajectory"), list):
        raise TypeError("replay must contain result.trajectory")

    source_gains = np.asarray(controller.get("feedback_gains"), dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    tail_feedback = np.asarray(tail_feedback, dtype=np.float64)
    if source_gains.ndim != 2 or source_gains.size == 0:
        raise ValueError("controller feedback_gains must be a nonempty matrix")
    state_size = source_gains.shape[1]
    if transform.shape != (state_size, state_size):
        raise ValueError("coordinate transform does not match controller state size")
    if tail_feedback.shape != (state_size,):
        raise ValueError("tail feedback does not match controller state size")

    initial = np.r_[
        np.asarray(selected.get("qpos"), dtype=np.float64),
        np.asarray(selected.get("qvel"), dtype=np.float64),
    ]
    if initial.shape != (state_size,) or not np.all(np.isfinite(initial)):
        raise ValueError("selected state does not match controller state size")
    states = [transform @ initial]
    controls: list[float] = []
    gains: list[np.ndarray] = []
    trajectory = replay["trajectory"]
    for index, row in enumerate(trajectory):
        if not isinstance(row, dict) or int(row.get("step", -1)) != index + 1:
            raise ValueError("trajectory steps must be contiguous and one-indexed")
        action = float(row.get("action"))
        physical = np.r_[
            np.asarray(row.get("qpos"), dtype=np.float64),
            np.asarray(row.get("qvel"), dtype=np.float64),
        ]
        if not np.isfinite(action) or physical.shape != (state_size,):
            raise ValueError("trajectory action/state has the wrong shape")
        if not np.all(np.isfinite(physical)):
            raise ValueError("trajectory state must be finite")
        controls.append(action)
        states.append(transform @ physical)
        gains.append(source_gains[index] if index < source_gains.shape[0] else tail_feedback)

    controller["controls"] = np.asarray(controls, dtype=np.float64).tolist()
    controller["feedback_gains"] = np.asarray(gains, dtype=np.float64).tolist()
    controller["horizon_steps"] = len(controls)
    policy_dt = float(controller.get("policy_dt", 0.0))
    if policy_dt > 0.0:
        controller["horizon_seconds"] = len(controls) * policy_dt
    controller["periodic_coordinate_errors"] = True
    controller["materialized_swing_prefix_steps"] = int(source_gains.shape[0])
    controller["materialized_lqr_tail_steps"] = int(
        max(0, len(controls) - source_gains.shape[0])
    )
    result["search"] = {
        "nominal_coordinate_states": np.asarray(states, dtype=np.float64).tolist(),
        "is_feasible": bool(replay.get("success", False)),
        "iterations": 0,
        "cost": None,
    }
    result.pop("result", None)
    result["claim_status"] = "development_executed_rollout_route_not_record_evidence"
    result["not_solution"] = True
    result["summary"] = (
        "Exact feedback-corrected swing and analytic LQR tail materialized as one "
        "topology-aware transferable trajectory."
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--replay", required=True)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    replay_path = Path(args.replay)
    cfg = load_config(config_path)
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("replay artifact must contain a JSON object")
    n_links = int(cfg["env"]["n_links"])
    transform = coordinate_transform(n_links, load_config(args.spec))
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        raise TypeError("replay must contain a controller object")
    gain = lqr_gain(
        cfg,
        progress=float(controller.get("lqr_progress", controller.get("progress", 1.0))),
        fd_eps=1.0e-7,
        control_cost=float(controller.get("lqr_control_cost", 1000.0)),
        q_weights=controller.get("lqr_weights"),
    )
    tail_feedback = -float(controller.get("lqr_scale", 1.0)) * transform_feedback_gain(
        gain, transform
    )
    result = materialize_rollout(payload, transform, tail_feedback)
    result["source"] = {
        "config": file_metadata(config_path),
        "replay": file_metadata(replay_path),
        "spec": file_metadata(Path(args.spec)),
    }
    result["runtime"] = runtime_metadata()
    result["git"] = git_metadata(ROOT)
    result["generated_at"] = utc_timestamp()
    dump_json(result, Path(args.out))
    print(
        f"wrote {args.out}: steps={result['controller']['horizon_steps']} "
        f"tail={result['controller']['materialized_lqr_tail_steps']}"
    )


if __name__ == "__main__":
    main()
