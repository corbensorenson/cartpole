#!/usr/bin/env python
"""Export measured route-terminal states as a reproducible capture dataset.

The source artifact is a reset-free exact-MuJoCo route cohort.  Each exported
state is the actual ``qpos/qvel`` observed when the swing route handed control
to the capture expert.  The result is suitable for ``env.init_mode=state_list``
experiments, but is deliberately marked as training evidence rather than a
benchmark solution: the source route uses a conditioning phase before launch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json
from gcartpole.env import serial_absolute_angles
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp


def _metric(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def handoff_metrics(handoff: dict[str, Any]) -> dict[str, Any]:
    qpos = np.asarray(handoff.get("qpos", []), dtype=np.float64)
    qvel = np.asarray(handoff.get("qvel", []), dtype=np.float64)
    if qpos.ndim != 1 or qvel.ndim != 1 or qpos.shape != qvel.shape or qpos.size < 2:
        raise ValueError("every handoff must contain equal one-dimensional qpos and qvel arrays")
    absolute_angles = serial_absolute_angles(qpos[1:])
    absolute_rates = np.cumsum(qvel[1:])
    return {
        "max_abs_angle": float(np.max(np.abs(absolute_angles))),
        "absolute_angle_rms": float(np.sqrt(np.mean(absolute_angles**2))),
        "hinge_velocity_rms": float(np.sqrt(np.mean(qvel[1:] ** 2))),
        "absolute_angular_velocity_rms": float(np.sqrt(np.mean(absolute_rates**2))),
        "max_absolute_angular_velocity": float(np.max(np.abs(absolute_rates))),
        "cart_position": float(qpos[0]),
        "cart_velocity": float(qvel[0]),
        "absolute_angles": absolute_angles.astype(float).tolist(),
    }


def split_name(index: int, *, train_count: int, validation_count: int, total: int) -> str:
    if index < train_count:
        return "train"
    if index < train_count + validation_count:
        return "validation"
    if index < total:
        return "test"
    raise IndexError(f"episode index {index} is outside the requested cohort")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="global discovery evidence JSON")
    parser.add_argument("--out", required=True, help="output state-list JSON")
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--validation-count", type=int, default=10)
    parser.add_argument("--test-count", type=int, default=10)
    args = parser.parse_args()

    source_path = Path(args.input)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source evidence must be a JSON object")
    episodes = payload.get("episodes_detail")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("source evidence must contain a non-empty episodes_detail list")
    counts = (int(args.train_count), int(args.validation_count), int(args.test_count))
    if any(count < 1 for count in counts):
        raise ValueError("all split counts must be positive")
    requested = sum(counts)
    if requested != len(episodes):
        raise ValueError(
            f"split counts total {requested}, but source has {len(episodes)} episodes; "
            "refuse to silently discard or duplicate handoffs"
        )

    states: list[dict[str, Any]] = []
    split_counts = {"train": 0, "validation": 0, "test": 0}
    successful_replays = 0
    for index, episode in enumerate(episodes):
        if not isinstance(episode, dict):
            raise ValueError(f"episode {index} is not an object")
        handoff = episode.get("handoff_state")
        if not isinstance(handoff, dict):
            raise ValueError(f"episode {index} does not contain a handoff_state")
        qpos = np.asarray(handoff.get("qpos", []), dtype=np.float64)
        qvel = np.asarray(handoff.get("qvel", []), dtype=np.float64)
        if qpos.ndim != 1 or qvel.ndim != 1 or qpos.shape != qvel.shape:
            raise ValueError(f"episode {index} has malformed handoff state")
        split = split_name(
            index,
            train_count=counts[0],
            validation_count=counts[1],
            total=requested,
        )
        replay = episode.get("capture_replay")
        if isinstance(replay, dict) and bool(replay.get("success", False)):
            successful_replays += 1
        metrics = handoff_metrics(handoff)
        split_counts[split] += 1
        states.append(
            {
                "state_id": f"route100_{index:03d}",
                "split": split,
                "source_episode": int(episode.get("episode", index)),
                "source_seed": int(episode["seed"]),
                "source_step": int(handoff.get("step", 0)),
                "source_time_seconds": _metric(handoff.get("time_seconds"), 0.0),
                "selected_route": int(episode.get("selected_route", -1)),
                "route_steps": int(handoff.get("route_steps", 0)),
                "qpos": qpos.astype(float).tolist(),
                "qvel": qvel.astype(float).tolist(),
                **metrics,
                "capture_replay": replay,
                "source_success": bool(episode.get("success", False)),
                "source_post_launch_reset_count": int(episode.get("post_launch_reset_count", -1)),
            }
        )

    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "route_handoff_training_data",
        "not_solution": True,
        "summary": (
            "Measured exact-MuJoCo route-terminal states exported from a "
            "reset-free six-link discovery cohort; the source uses a 15-second "
            "conditioning phase and therefore does not close P1, P2, or P3."
        ),
        "source_file": file_metadata(source_path),
        "source_sha256": data_sha256(payload),
        "source_claim_status": payload.get("claim_status"),
        "source_not_solution": bool(payload.get("not_solution", False)),
        "source_gate": payload.get("gate"),
        "source_config": payload.get("config"),
        "source_resolved_config_sha256": payload.get("resolved_config_sha256"),
        "source_controller_sha256": payload.get("controller_sha256"),
        "source_trace_sha256": payload.get("trace_sha256"),
        "split_policy": {
            "order": "source episode order",
            "train_count": counts[0],
            "validation_count": counts[1],
            "test_count": counts[2],
            "disjoint": True,
        },
        "split_counts": split_counts,
        "state_count": len(states),
        "capture_replay_successes": successful_replays,
        "all_source_post_launch_reset_counts_zero": all(
            int(state["source_post_launch_reset_count"]) == 0 for state in states
        ),
        "states": states,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, args.out)
    print(
        f"exported={len(states)} train={split_counts['train']} "
        f"validation={split_counts['validation']} test={split_counts['test']} "
        f"capture_replays={successful_replays}/{len(states)}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
