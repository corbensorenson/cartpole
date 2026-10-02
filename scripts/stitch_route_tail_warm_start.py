#!/usr/bin/env python
"""Stitch an exact route and CEM tail into a feedback-optimization warm start."""

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
from gcartpole.fddp import rollout_controls
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform


ROOT = Path(__file__).resolve().parents[1]


def interpolation_matrix(knot_count: int, step_count: int) -> np.ndarray:
    source = np.linspace(0.0, 1.0, knot_count, dtype=np.float64)
    target = np.linspace(0.0, 1.0, step_count, dtype=np.float64)
    right = np.searchsorted(source, target, side="right").clip(1, knot_count - 1)
    left = right - 1
    fraction = (target - source[left]) / (source[right] - source[left])
    matrix = np.zeros((step_count, knot_count), dtype=np.float64)
    rows = np.arange(step_count)
    matrix[rows, left] = 1.0 - fraction
    matrix[rows, right] += fraction
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, help="route artifact containing best.controls")
    parser.add_argument("--tail", required=True, help="tail CEM artifact containing best.knots")
    parser.add_argument("--feedback-source", required=True, help="route feedback artifact")
    parser.add_argument("--state-json", required=True, help="exact selected-state artifact")
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    route_path = Path(args.route)
    tail_path = Path(args.tail)
    feedback_path = Path(args.feedback_source)
    state_path = Path(args.state_json)
    route = json.loads(route_path.read_text(encoding="utf-8"))
    tail = json.loads(tail_path.read_text(encoding="utf-8"))
    feedback_source = json.loads(feedback_path.read_text(encoding="utf-8"))
    state_payload = json.loads(state_path.read_text(encoding="utf-8"))
    route_best = route.get("best")
    tail_best = tail.get("best")
    source_controller = feedback_source.get("controller")
    selected = state_payload.get("selected_state")
    if not isinstance(route_best, dict) or not isinstance(tail_best, dict):
        raise ValueError("route and tail artifacts must contain best records")
    if not isinstance(source_controller, dict) or not isinstance(selected, dict):
        raise ValueError("feedback source and state artifact are missing required objects")

    route_controls = np.asarray(route_best["controls"], dtype=np.float64)
    knots = np.asarray(tail_best["knots"], dtype=np.float64)
    tail_steps = int(tail.get("tail_horizon_steps", 0))
    if route_controls.ndim != 1 or knots.ndim != 1 or tail_steps < 2:
        raise ValueError("route controls, tail knots, or tail horizon are invalid")
    tail_controls = knots @ interpolation_matrix(knots.size, tail_steps).T
    controls = np.concatenate((route_controls, np.clip(tail_controls, -1.0, 1.0)))

    cfg = load_config(args.config)
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
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
    qpos = np.asarray(selected["qpos"], dtype=np.float64)
    qvel = np.asarray(selected["qvel"], dtype=np.float64)
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    import mujoco

    mujoco.mj_forward(env.model, env.data)
    transition = MujocoTransition(env, coordinate_transform=transform)
    start = transition.to_coordinates(data_state(env.data))
    nominal = rollout_controls(transition, start, controls)
    env.close()

    source_feedback = np.asarray(source_controller["feedback_gains"], dtype=np.float64)
    expected_route_dim = 2 * (int(cfg["env"]["n_links"]) + 1)
    if source_feedback.shape != (route_controls.size, expected_route_dim):
        raise ValueError("feedback source does not match the route horizon and target dimension")
    feedback = np.vstack(
        (source_feedback, np.zeros((tail_steps, expected_route_dim), dtype=np.float64))
    )
    controller = copy.deepcopy(source_controller)
    controller.update(
        {
            "type": "exact_route_plus_cem_tail_fddp_warm_start",
            "controls": controls.astype(float).tolist(),
            "feedback_gains": feedback.astype(float).tolist(),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(controls.size * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]),
        }
    )
    output: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_stitched_route_tail_not_solution",
        "not_solution": True,
        "summary": "Exact GN route plus exact-MuJoCo CEM capture tail; Box-FDDP warm start only.",
        "selected_state": copy.deepcopy(selected),
        "controller": controller,
        "search": {
            "converged": False,
            "is_feasible": False,
            "iterations": 0,
            "nominal_coordinate_states": nominal.astype(float).tolist(),
        },
        "sources": {
            "route": file_metadata(route_path),
            "tail": file_metadata(tail_path),
            "feedback": file_metadata(feedback_path),
            "state": file_metadata(state_path),
            "config": file_metadata(Path(args.config)),
            "spec": file_metadata(Path(args.spec)),
        },
        "tail": {
            "steps": int(tail_steps),
            "seconds": float(tail_steps * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]),
            "knots": knots.astype(float).tolist(),
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(ROOT),
    }
    dump_json(output, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
