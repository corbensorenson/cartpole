#!/usr/bin/env python
"""Refine a recorded seven-link swing tail with exact multiple shooting.

The source swing is replayed from the true hanging state.  Only the tail is
optimized, with exact MuJoCo segment defects, a terminal absolute-angle and
velocity cost, and a soft rail penalty.  The emitted controls are replayed
again from the original prefix; no planned state is substituted at the
boundary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform
from gcartpole.multiple_shooting import optimize_multiple_shooting
from search_swingup_tail_action_cem import interpolation_matrix, load_controller, replay_to_tail


def load_tail_controls(path: str, key: str, step_count: int) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    record = payload.get(key)
    if not isinstance(record, dict) or "knots" not in record:
        raise ValueError(f"{path} does not contain tail record {key!r}")
    knots = np.asarray(record["knots"], dtype=np.float64)
    return np.clip(knots @ interpolation_matrix(len(knots), step_count).T, -1.0, 1.0)


def replay_tail(
    cfg: dict[str, Any],
    source_controller: dict[str, Any],
    tail_start_seconds: float,
    controls: np.ndarray,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    prefix_steps = int(round(tail_start_seconds / env.dt))
    for step in range(prefix_steps):
        from search_swingup_tail_action_cem import trajectory_action

        action = trajectory_action(
            env,
            step * env.dt,
            np.asarray(source_controller["knots"], dtype=np.float64),
            float(source_controller["trajectory_seconds"]),
            float(source_controller.get("kp", 0.0)),
            float(source_controller.get("kd", 0.0)),
        )
        _, _, terminated, truncated, info = env.step([action])
        if terminated or truncated:
            raise RuntimeError(f"prefix ended at step {step}: {info.get('termination_reason')}")
    rows: list[dict[str, Any]] = []
    for index, action in enumerate(controls):
        _, _, terminated, truncated, info = env.step([float(action)])
        _, absolute = env._angles()
        rows.append(
            {
                "step": int(index + 1),
                "time_seconds": float((index + 1) * env.dt),
                "action": float(action),
                "max_abs_angle": float(np.max(np.abs(absolute))),
                "hinge_velocity_rms": float(np.sqrt(np.mean(np.asarray(env.data.qvel[1 : 1 + env.n]) ** 2))),
                "x": float(env.data.qpos[0]),
                "cart_velocity": float(env.data.qvel[0]),
                "absolute_angles": absolute.astype(float).tolist(),
            }
        )
        if terminated or truncated:
            break
    result = {
        "rows": rows,
        "tail_steps": len(rows),
        "success": bool(rows and rows[-1]["max_abs_angle"] <= 0.15),
        "best": min(rows, key=lambda row: row["max_abs_angle"]) if rows else None,
        "endpoint": rows[-1] if rows else None,
        "max_cart_abs": max(abs(row["x"]) for row in rows) if rows else 0.0,
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine a seven-link swing tail with exact multiple shooting")
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument("--swing-controller-json", required=True)
    parser.add_argument("--tail-json", required=True)
    parser.add_argument("--tail-key", default="best_feasible")
    parser.add_argument("--tail-start-seconds", type=float, default=11.0)
    parser.add_argument("--tail-seconds", type=float, default=5.0)
    parser.add_argument("--segment-steps", type=int, default=10)
    parser.add_argument("--defect-weight", type=float, default=1.0e5)
    parser.add_argument("--terminal-weight", type=float, default=20.0)
    parser.add_argument("--control-weight", type=float, default=0.01)
    parser.add_argument("--rail-weight", type=float, default=1.0e4)
    parser.add_argument("--rail-soft-limit", type=float, default=3.0)
    parser.add_argument("--max-evaluations", type=int, default=30)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "episode_seconds": float(args.tail_start_seconds + args.tail_seconds + 1.0),
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "terminate_abs_angle": None,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    source = load_controller(args.swing_controller_json, None)
    env = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
    full_state = replay_to_tail(env, source, args.tail_start_seconds)
    nq = int(env.model.nq)
    physical_start = np.r_[full_state[1 : 1 + nq], full_state[1 + nq : 1 + nq + env.model.nv]]
    tail_steps = max(2, int(round(args.tail_seconds / env.dt)))
    controls = load_tail_controls(args.tail_json, args.tail_key, tail_steps)
    transition = MujocoTransition(
        env,
        coordinate_transform=dimensionless_absolute_transform(
            env.n,
            StateScales(3.0, 1.0, 3.0, 3.0),
        ),
    )
    initial_coordinates = transition.to_coordinates(physical_start)
    weights = np.r_[
        0.2,
        np.full(env.n, 120.0),
        0.2,
        np.full(env.n, 3.0),
    ]
    started = __import__("time").time()
    search = optimize_multiple_shooting(
        transition,
        initial_coordinates,
        controls,
        segment_steps=args.segment_steps,
        defect_weight=args.defect_weight,
        terminal_weight=args.terminal_weight,
        control_weight=args.control_weight,
        rail_weight=args.rail_weight,
        rail_soft_limit=args.rail_soft_limit / 3.0,
        rail_limit=float(env.rail_limit / 3.0),
        max_evaluations=args.max_evaluations,
    )
    env.close()
    replay = replay_tail(cfg, source, args.tail_start_seconds, search.controls)
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact-MuJoCo multiple-shooting tail refinement from a real hanging-start prefix.",
        "config_path": str(Path(args.config)),
        "swing_controller_json": str(Path(args.swing_controller_json)),
        "tail_json": str(Path(args.tail_json)),
        "tail_key": args.tail_key,
        "tail_start_seconds": float(args.tail_start_seconds),
        "tail_steps": int(tail_steps),
        "search": {
            "segment_steps": int(args.segment_steps),
            "defect_weight": float(args.defect_weight),
            "terminal_weight": float(args.terminal_weight),
            "control_weight": float(args.control_weight),
            "rail_weight": float(args.rail_weight),
            "rail_soft_limit": float(args.rail_soft_limit),
            "max_evaluations": int(args.max_evaluations),
            "cost": float(search.cost),
            "optimality": float(search.optimality),
            "evaluations": int(search.evaluations),
            "success": bool(search.success),
            "status": int(search.status),
            "message": search.message,
            "wall_time_seconds": float(__import__("time").time() - started),
        },
        "controller": {
            "type": "normalized_force_tail_multiple_shooting",
            "controls": search.controls.astype(float).tolist(),
            "node_states": search.node_states.astype(float).tolist(),
            "segment_defects_max_abs": float(np.max(np.abs(search.segment_defects))),
        },
        "replay": replay,
        "initial_state": {
            "qpos": physical_start[:nq].astype(float).tolist(),
            "qvel": physical_start[nq:].astype(float).tolist(),
        },
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(
        f"solver_success={search.success} cost={search.cost:.3f} "
        f"defect={np.max(np.abs(search.segment_defects)):.3e} "
        f"best_angle={replay['best']['max_abs_angle'] if replay['best'] else float('nan'):.4f} "
        f"endpoint_angle={replay['endpoint']['max_abs_angle'] if replay['endpoint'] else float('nan'):.4f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
