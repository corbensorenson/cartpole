#!/usr/bin/env python
"""Convert a successful deterministic energy trace into a generic route seed.

The energy controller is useful for discovering the one-link trajectory, but
promotion evidence should use the same route/feedback/LQR representation as
the higher-link ladder.  This adapter removes the already-replayed hanging
conditioning prefix, keeps the energy swing plus a short LQR settling tail,
and writes a zero-feedback warm start for exact target-plant optimization.
It is explicitly not solution evidence until refined and passed through the
shared noisy route-library evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, utc_timestamp
from gcartpole.modal import (
    StateScales,
    dimensionless_absolute_transform,
    dimensionless_wrapped_state,
)


def package_trace(
    payload: dict[str, Any],
    spec: dict[str, Any],
    *,
    episode_index: int,
    post_switch_seconds: float,
) -> dict[str, Any]:
    episodes = payload.get("episode_results", [])
    if not 0 <= episode_index < len(episodes):
        raise IndexError("episode index is outside the energy artifact")
    episode = episodes[episode_index]
    if episode.get("success") is not True:
        raise ValueError("energy teacher episode did not pass its exact hold gate")
    trace = episode.get("trace")
    if not isinstance(trace, list) or not trace:
        raise ValueError("energy artifact must be generated with --include-traces")
    if post_switch_seconds < 0.0:
        raise ValueError("post-switch duration must be nonnegative")

    route_start = next(
        (index for index, row in enumerate(trace) if row.get("mode") != "hanging_lqr"),
        None,
    )
    if route_start is None or route_start == 0:
        raise ValueError("trace does not contain a conditioning-to-swing boundary")
    capture_start = next(
        (
            index
            for index in range(route_start, len(trace))
            if trace[index].get("mode") == "lqr"
        ),
        None,
    )
    if capture_start is None:
        raise ValueError("successful teacher trace never entered exact LQR capture")

    first_dt = float(trace[0]["time_seconds"])
    if first_dt <= 0.0:
        raise ValueError("trace policy period must be positive")
    tail_steps = int(np.ceil(post_switch_seconds / first_dt))
    route_stop = min(len(trace), max(capture_start + tail_steps, capture_start + 1))
    route_rows = trace[route_start:route_stop]
    initial = trace[route_start - 1]
    n_links = int(payload["n_links"])
    state_size = 2 * (n_links + 1)
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        n_links,
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )

    state_rows = [initial, *route_rows]
    nominal = np.asarray(
        [
            dimensionless_wrapped_state(
                np.asarray(row["qpos"], dtype=np.float64),
                np.asarray(row["qvel"], dtype=np.float64),
                transform,
            )
            for row in state_rows
        ],
        dtype=np.float64,
    )
    controls = np.asarray([row["action"] for row in route_rows], dtype=np.float64)
    if nominal.shape != (controls.size + 1, state_size):
        raise ValueError("teacher trace state dimensions do not match its link count")
    if np.max(np.abs(controls), initial=0.0) > 1.0 + 1e-12:
        raise ValueError("teacher trace contains action outside normalized bounds")

    parameters = payload.get("controller", {}).get("parameters", {})
    return {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "energy_teacher_route_warm_start_not_solution_evidence",
        "not_solution": True,
        "summary": (
            "Successful deterministic energy trajectory converted to the shared "
            "route representation; exact route optimization and a fresh noisy gate "
            "remain required."
        ),
        "teacher": {
            "episode_index": int(episode_index),
            "seed": int(episode["seed"]),
            "source_success": True,
            "conditioning_steps_removed": int(route_start),
            "energy_steps": int(capture_start - route_start),
            "capture_tail_steps": int(route_stop - capture_start),
            "post_switch_seconds_requested": float(post_switch_seconds),
        },
        "selected_state": {
            "qpos": np.asarray(initial["qpos"], dtype=np.float64).astype(float).tolist(),
            "qvel": np.asarray(initial["qvel"], dtype=np.float64).astype(float).tolist(),
            "state_index": int(route_start - 1),
        },
        "controller": {
            "type": "energy_teacher_route_warm_start_then_exact_lqr",
            "controls": controls.astype(float).tolist(),
            "feedback_gains": np.zeros((controls.size, state_size)).tolist(),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(controls.size * first_dt),
            "policy_dt": first_dt,
            "lqr_scale": float(parameters.get("lqr_scale", 1.0)),
            "lqr_control_cost": float(parameters.get("lqr_control_cost", 10.0)),
        },
        "search": {
            "nominal_coordinate_states": nominal.astype(float).tolist(),
            "is_feasible": True,
            "iterations": 0,
            "cost": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--energy-artifact", required=True)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--post-switch-seconds", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source = Path(args.energy_artifact)
    payload = json.loads(source.read_text(encoding="utf-8"))
    result = package_trace(
        payload,
        load_config(args.spec),
        episode_index=args.episode_index,
        post_switch_seconds=args.post_switch_seconds,
    )
    result["source"] = file_metadata(source)
    result["coordinate_spec"] = file_metadata(Path(args.spec))
    dump_json(result, Path(args.out))
    print(
        f"wrote {args.out}: steps={result['controller']['horizon_steps']} "
        f"energy={result['teacher']['energy_steps']} "
        f"capture_tail={result['teacher']['capture_tail_steps']}"
    )


if __name__ == "__main__":
    main()
