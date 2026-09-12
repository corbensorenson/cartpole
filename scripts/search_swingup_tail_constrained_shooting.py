#!/usr/bin/env python
"""Search a hard-constrained final tail from a real seven-link swing prefix."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.constrained_shooting import optimize_constrained_shooting
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform
from search_swingup_tail_action_cem import interpolation_matrix, load_controller, replay_to_tail
from probe_swingup_trajectory import trajectory_action


def load_tail_controls(
    path: str,
    key: str,
    steps: int,
    target_seconds: float,
) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    record = payload.get(key)
    if not isinstance(record, dict) or "knots" not in record:
        raise ValueError(f"{path} does not contain tail record {key!r}")
    knots = np.asarray(record["knots"], dtype=np.float64)
    source_seconds = float(payload.get("tail_horizon_seconds", 0.0))
    if source_seconds <= 0.0:
        source_seconds = float(target_seconds)
    source_times = np.linspace(0.0, source_seconds, len(knots), dtype=np.float64)
    target_times = np.linspace(0.0, float(target_seconds), steps, dtype=np.float64)
    return np.clip(np.interp(target_times, source_times, knots), -1.0, 1.0)


def source_action(env: NLinkCartPoleEnv, source: dict[str, Any], step: int) -> float:
    """Replay either a cart-position source or a normalized-force source."""
    if source.get("controls") is not None:
        controls = np.asarray(source["controls"], dtype=np.float64)
        if controls.ndim != 1 or controls.size < 2:
            raise ValueError("FDDP source controls must be a one-dimensional sequence")
        source_seconds = float(source.get("horizon_seconds", controls.size * env.dt))
        source_times = np.linspace(0.0, max(source_seconds, env.dt), controls.size)
        return float(
            np.clip(
                np.interp(step * env.dt, source_times, controls, left=controls[0], right=controls[-1]),
                -1.0,
                1.0,
            )
        )
    if source.get("type") == "normalized_force_knots":
        knots = np.asarray(source["knots"], dtype=np.float64)
        source_seconds = float(source["trajectory_seconds"])
        return float(
            np.interp(
                step * env.dt,
                np.linspace(0.0, source_seconds, len(knots), dtype=np.float64),
                knots,
            )
        )
    return float(
        trajectory_action(
            env,
            step * env.dt,
            np.asarray(source["knots"], dtype=np.float64),
            float(source["trajectory_seconds"]),
            float(source.get("kp", 0.0)),
            float(source.get("kd", 0.0)),
        )
    )


def setup_cfg(base: dict[str, Any], seconds: float) -> dict[str, Any]:
    env = {
        **base["env"],
        "init_mode": "hanging",
        "episode_seconds": float(seconds),
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "terminate_abs_angle": None,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        env[f"{key}_start"] = 0.0
        env[f"{key}_end"] = 0.0
    return {**base, "env": env}


def replay_prefix(
    env: NLinkCartPoleEnv,
    source: dict[str, Any],
    tail_start_seconds: float,
    prefix_controls: np.ndarray,
) -> None:
    prefix_steps = int(round(tail_start_seconds / env.dt))
    for step in range(prefix_steps):
        action = source_action(env, source, step)
        _, _, terminated, truncated, info = env.step([action])
        if terminated or truncated:
            raise RuntimeError(f"source ended at prefix step {step}: {info.get('termination_reason')}")
    for step, action in enumerate(prefix_controls):
        _, _, terminated, truncated, info = env.step([float(action)])
        if terminated or truncated:
            raise RuntimeError(f"tail prefix ended at step {step}: {info.get('termination_reason')}")


def replay_suffix(
    cfg: dict[str, Any],
    source: dict[str, Any],
    tail_start_seconds: float,
    prefix_controls: np.ndarray,
    controls: np.ndarray,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
    env.reset(seed=0)
    replay_prefix(env, source, tail_start_seconds, prefix_controls)
    rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    for step, action in enumerate(controls):
        _, _, terminated, truncated, info = env.step([float(action)])
        _, absolute = env._angles()
        hinge_velocity = np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64)
        absolute_angular_velocity = np.cumsum(hinge_velocity)
        rows.append(
            {
                "step": int(step + 1),
                "time_seconds": float((step + 1) * env.dt),
                "action": float(action),
                "max_abs_angle": float(np.max(np.abs(absolute))),
                "hinge_velocity_rms": float(np.sqrt(np.mean(np.asarray(env.data.qvel[1 : 1 + env.n]) ** 2))),
                "max_hinge": float(np.max(np.abs(np.asarray(env.data.qvel[1 : 1 + env.n])))),
                "absolute_angular_velocity_rms": float(np.sqrt(np.mean(absolute_angular_velocity**2))),
                "max_absolute_angular_velocity": float(np.max(np.abs(absolute_angular_velocity))),
                "x": float(env.data.qpos[0]),
                "cart_velocity": float(env.data.qvel[0]),
                "absolute_angles": absolute.astype(float).tolist(),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
            }
        )
        final_info = dict(info)
        if terminated or truncated:
            break
    env.close()
    return {
        "rows": rows,
        "endpoint": rows[-1] if rows else None,
        "best": min(rows, key=lambda row: row["max_abs_angle"]) if rows else None,
        "max_cart_abs": max((abs(row["x"]) for row in rows), default=0.0),
        "final_info": final_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Hard-constrained seven-link tail shooting")
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument("--swing-controller-json", required=True)
    parser.add_argument("--tail-json", required=True)
    parser.add_argument("--tail-key", default="best_feasible")
    parser.add_argument("--tail-start-seconds", type=float, default=11.0)
    parser.add_argument("--prefix-seconds", type=float, default=3.0)
    parser.add_argument("--suffix-seconds", type=float, default=2.0)
    parser.add_argument("--max-iterations", type=int, default=30)
    parser.add_argument("--control-weight", type=float, default=0.01)
    parser.add_argument("--rail-soft-limit", type=float, default=3.0)
    parser.add_argument("--handoff-angle", type=float, default=0.15)
    parser.add_argument("--handoff-hinge-rms", type=float, default=0.75)
    parser.add_argument("--handoff-cart", type=float, default=1.25)
    parser.add_argument("--handoff-cart-velocity", type=float, default=0.50)
    parser.add_argument(
        "--handoff-absolute-velocity",
        type=float,
        default=None,
        help="Optional RMS bound on cumulative absolute link angular velocity.",
    )
    parser.add_argument("--handoff-out", default=None, help="Optional state-list artifact for downstream capture replay")
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    base = apply_overrides(load_config(args.config), args.override)
    total_seconds = args.tail_start_seconds + args.prefix_seconds + args.suffix_seconds + 1.0
    cfg = setup_cfg(base, total_seconds)
    source = load_controller(args.swing_controller_json, None)
    env = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
    env.reset(seed=0)
    full_tail = load_tail_controls(
        args.tail_json,
        args.tail_key,
        max(2, int(round((args.prefix_seconds + args.suffix_seconds) / env.dt))),
        args.prefix_seconds + args.suffix_seconds,
    )
    prefix_steps = max(1, int(round(args.prefix_seconds / env.dt)))
    suffix_steps = max(2, int(round(args.suffix_seconds / env.dt)))
    prefix_controls = full_tail[:prefix_steps]
    env.reset(seed=0)
    replay_prefix(env, source, args.tail_start_seconds, prefix_controls)
    initial_physical = data_state(env.data)
    hinge_velocity_scale = 3.0
    transition = MujocoTransition(
        env,
        coordinate_transform=dimensionless_absolute_transform(
            env.n,
            StateScales(3.0, 1.0, 3.0, hinge_velocity_scale),
        ),
    )
    initial_coordinates = transition.to_coordinates(initial_physical)
    nx = initial_coordinates.size
    terminal_metric = np.diag(
        np.r_[
            0.25,
            np.full(env.n, 250.0),
            0.25,
            np.full(env.n, 8.0),
        ]
    )
    started = time.time()
    search = optimize_constrained_shooting(
        transition,
        initial_coordinates,
        full_tail[prefix_steps : prefix_steps + suffix_steps],
        terminal_metric,
        np.eye(nx, dtype=np.float64),
        control_weight=args.control_weight,
        rail_limit=float(args.rail_soft_limit / 3.0),
        handoff_lyapunov=1.0e6,
        handoff_cart_ratio=float(args.handoff_cart) / 3.0,
        handoff_angle_ratio=float(args.handoff_angle),
        handoff_cart_velocity_ratio=float(args.handoff_cart_velocity) / 3.0,
        handoff_hinge_velocity_ratio=float(args.handoff_hinge_rms) / 3.0,
        max_iterations=args.max_iterations,
        handoff_absolute_angular_velocity_ratio=args.handoff_absolute_velocity,
        absolute_angular_velocity_scale=hinge_velocity_scale,
    )
    env.close()
    replay = replay_suffix(cfg, source, args.tail_start_seconds, prefix_controls, search.controls)
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Hard-constrained exact-MuJoCo final-tail shooting from a real hanging-start prefix.",
        "config_path": str(Path(args.config)),
        "swing_controller_json": str(Path(args.swing_controller_json)),
        "tail_json": str(Path(args.tail_json)),
        "tail_key": args.tail_key,
        "tail_start_seconds": float(args.tail_start_seconds),
        "prefix_seconds": float(args.prefix_seconds),
        "suffix_seconds": float(args.suffix_seconds),
        "search": {
            "max_iterations": int(args.max_iterations),
            "control_weight": float(args.control_weight),
            "rail_soft_limit": float(args.rail_soft_limit),
            "handoff_angle": float(args.handoff_angle),
            "handoff_hinge_rms": float(args.handoff_hinge_rms),
            "handoff_cart": float(args.handoff_cart),
            "handoff_cart_velocity": float(args.handoff_cart_velocity),
            "handoff_absolute_velocity": (
                None
                if args.handoff_absolute_velocity is None
                else float(args.handoff_absolute_velocity)
            ),
            "success": bool(search.success),
            "status": int(search.status),
            "message": search.message,
            "cost": float(search.cost),
            "minimum_constraint_margin": float(search.minimum_constraint_margin),
            "evaluations": int(search.evaluations),
            "wall_time_seconds": float(time.time() - started),
        },
        "controller": {
            "type": "normalized_force_constrained_suffix",
            "prefix_controls": prefix_controls.astype(float).tolist(),
            "suffix_controls": search.controls.astype(float).tolist(),
            "suffix_states": search.states.astype(float).tolist(),
        },
        "replay": replay,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    if args.handoff_out and replay["endpoint"]:
        endpoint = replay["endpoint"]
        dump_json(
            {
                "schema_version": 1,
                "generated_at": utc_timestamp(),
                "not_solution": True,
                "summary": "Exact endpoint emitted by constrained seven-link tail search; capture component state only.",
                "source_run": str(Path(args.out)),
                "morphology_context": {
                    "plant_progress": float(cfg["env"].get("plant_progress", 0.0)),
                    "rail_limit": float(cfg["env"]["rail_limit"]),
                    "force_limit": float(cfg["env"]["force_limit"]),
                },
                "states": [
                    {
                        "state_index": 0,
                        "source_step": int(endpoint["step"]),
                        "source_time_seconds": float(args.tail_start_seconds + args.prefix_seconds + endpoint["time_seconds"]),
                        "qpos": endpoint["qpos"],
                        "qvel": endpoint["qvel"],
                        "absolute_angles": endpoint["absolute_angles"],
                        "max_abs_angle": endpoint["max_abs_angle"],
                        "hinge_velocity_rms": endpoint["hinge_velocity_rms"],
                        "absolute_angular_velocity_rms": endpoint["absolute_angular_velocity_rms"],
                        "max_absolute_angular_velocity": endpoint["max_absolute_angular_velocity"],
                        "cart_abs": abs(endpoint["x"]),
                        "cart_velocity_abs": abs(endpoint["cart_velocity"]),
                    }
                ],
            },
            Path(args.handoff_out),
        )
    print(
        f"solver_success={search.success} margin={search.minimum_constraint_margin:.3e} "
        f"endpoint_angle={replay['endpoint']['max_abs_angle'] if replay['endpoint'] else float('nan'):.4f} "
        f"endpoint_hinge={replay['endpoint']['hinge_velocity_rms'] if replay['endpoint'] else float('nan'):.4f} "
        f"endpoint_absolute_velocity={replay['endpoint']['absolute_angular_velocity_rms'] if replay['endpoint'] else float('nan'):.4f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
