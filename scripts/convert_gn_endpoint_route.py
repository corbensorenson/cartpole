#!/usr/bin/env python
"""Convert an exact-MuJoCo GN endpoint trace into an FDDP warm-start artifact.

The endpoint refiner stores physical states in its trace but the Box-FDDP
searcher consumes dimensionless coordinate states plus time-varying feedback.
This adapter preserves the exact GN controls and measured target-plant states,
while inheriting the transferred route feedback as a reproducible diagnostic
warm start.  It does not certify the route.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.modal import StateScales, dimensionless_absolute_transform, dimensionless_wrapped_state


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gn", required=True, help="GN endpoint-refinement artifact")
    parser.add_argument("--source-controller", required=True, help="transferred route with feedback gains")
    parser.add_argument("--config", required=True, help="exact target-plant config")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gn_path = Path(args.gn)
    source_path = Path(args.source_controller)
    gn = json.loads(gn_path.read_text(encoding="utf-8"))
    source = json.loads(source_path.read_text(encoding="utf-8"))
    best = gn.get("best")
    source_controller = source.get("controller")
    source_search = source.get("search")
    selected = source.get("selected_state")
    if not isinstance(best, dict) or not isinstance(source_controller, dict):
        raise ValueError("GN artifact and source controller must contain the expected objects")
    if not isinstance(source_search, dict) or not isinstance(selected, dict):
        raise ValueError("source controller must contain search and selected_state")

    controls = np.asarray(best.get("controls"), dtype=np.float64)
    trace = best.get("trace")
    feedback = np.asarray(source_controller.get("feedback_gains"), dtype=np.float64)
    if controls.ndim != 1 or not isinstance(trace, list) or len(trace) != controls.size:
        raise ValueError("GN controls and physical trace must have matching horizons")
    if feedback.shape != (controls.size, 2 * (int(load_config(args.config)["env"]["n_links"]) + 1)):
        raise ValueError("source feedback gains do not match the target link count")

    cfg = load_config(args.config)
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        cfg["env"][key] = 0.0
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    spec = load_config(args.spec)
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        int(cfg["env"]["n_links"]),
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    _, reset_info = env.reset(seed=0)
    initial_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    initial_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
    expected = int(cfg["env"]["n_links"]) + 1
    if initial_qpos.shape != (expected,) or initial_qvel.shape != (expected,):
        raise ValueError("selected initial state does not match target config")

    states = [dimensionless_wrapped_state(initial_qpos, initial_qvel, transform)]
    for row in trace:
        qpos = np.asarray(row["qpos"], dtype=np.float64)
        qvel = np.asarray(row["qvel"], dtype=np.float64)
        if qpos.shape != (expected,) or qvel.shape != (expected,):
            raise ValueError("GN trace state does not match target config")
        states.append(dimensionless_wrapped_state(qpos, qvel, transform))
    env.close()

    controller = copy.deepcopy(source_controller)
    controller.update(
        {
            "type": "exact_gn_endpoint_route_with_inherited_feedback",
            "controls": controls.astype(float).tolist(),
            "feedback_gains": feedback.astype(float).tolist(),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(best.get("horizon_seconds", controls.size * 0.02)),
        }
    )
    output: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_gn_endpoint_fddp_warm_start_not_solution",
        "not_solution": True,
        "summary": "Exact GN endpoint route with measured target-plant states and inherited feedback; FDDP warm start only.",
        "source_gn": file_metadata(gn_path),
        "source_controller": file_metadata(source_path),
        "selected_state": {
            "qpos": initial_qpos.astype(float).tolist(),
            "qvel": initial_qvel.astype(float).tolist(),
            "state_index": 0,
            "source": "exact_target_config_hanging_reset",
        },
        "controller": controller,
        "search": {
            "converged": False,
            "is_feasible": False,
            "iterations": 0,
            "nominal_coordinate_states": np.asarray(states, dtype=np.float64).tolist(),
        },
        "evidence": {
            "config": file_metadata(Path(args.config)),
            "spec": file_metadata(Path(args.spec)),
            "runtime": runtime_metadata(),
            "git": git_metadata(ROOT),
        },
    }
    dump_json(output, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
