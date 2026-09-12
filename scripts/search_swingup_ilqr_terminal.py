#!/usr/bin/env python
"""Refine a hanging-start force proposal into a terminal seven-link arrival.

The optimizer runs on the same exact MuJoCo action cadence used by the final
replay.  The trajectory is a proposal until its terminal state is handed to a
reset-free capture controller and independently verified.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles, wrap_angle
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.ilqr import MujocoTransition, QuadraticTrajectoryCost, data_state, optimize_ilqr


def absolute_rate_transform(n_links: int, *, cart_position: float, angle: float, cart_velocity: float, rate: float) -> np.ndarray:
    """Map relative MuJoCo coordinates to scaled absolute angles and rates."""
    n_links = int(n_links)
    d = n_links + 1
    transform = np.zeros((2 * d, 2 * d), dtype=np.float64)
    transform[0, 0] = 1.0 / float(cart_position)
    for link in range(n_links):
        transform[1 + link, 1 : 2 + link] = 1.0 / float(angle)
    transform[d, d] = 1.0 / float(cart_velocity)
    for link in range(n_links):
        transform[d + 1 + link, d + 1 : d + 2 + link] = 1.0 / float(rate)
    return transform


def load_initial_controls(path: str | None, *, seconds: float, steps: int, dt: float) -> np.ndarray:
    if path is None:
        return np.zeros(steps, dtype=np.float64)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    record = payload.get("best") if isinstance(payload, dict) else None
    if not isinstance(record, dict):
        record = payload.get("controller") if isinstance(payload, dict) else None
    source: np.ndarray | None = None
    source_seconds = float(payload.get("search", {}).get("seconds", seconds)) if isinstance(payload, dict) else seconds
    if isinstance(record, dict) and record.get("knots") is not None:
        source = np.asarray(record["knots"], dtype=np.float64)
    elif isinstance(record, dict) and record.get("controls") is not None:
        source = np.asarray(record["controls"], dtype=np.float64)
        source_seconds = float(payload.get("controller", {}).get("horizon_seconds", seconds))
    elif isinstance(payload, dict) and isinstance(payload.get("result"), dict):
        rows = payload["result"].get("trajectory")
        if isinstance(rows, list):
            source = np.asarray([float(row["action"]) for row in rows if "action" in row], dtype=np.float64)
            source_seconds = float(rows[-1].get("time_seconds", seconds)) if rows else seconds
    elif isinstance(payload, dict) and isinstance(payload.get("trajectory"), list):
        rows = payload["trajectory"]
        source = np.asarray([float(row["action"]) for row in rows if "action" in row], dtype=np.float64)
        source_seconds = float(rows[-1].get("time_seconds", seconds)) if rows else seconds
    if source is None or source.ndim != 1 or source.size < 2:
        raise ValueError(f"{path} does not contain force knots or controls")
    source_t = np.linspace(0.0, max(float(source_seconds), dt), source.size, dtype=np.float64)
    target_t = np.arange(steps, dtype=np.float64) * float(dt)
    return np.clip(np.interp(target_t, source_t, source, left=source[0], right=source[-1]), -1.0, 1.0)


def load_terminal_target(path: str | None, *, scale: float, n_links: int) -> tuple[np.ndarray | None, dict[str, Any] | None]:
    """Load a saved physical handoff and scale it toward the upright state."""
    if path is None:
        return None, None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(states, list) or not states:
        raise ValueError(f"{path} does not contain a non-empty states list")
    state = states[0]
    if not isinstance(state, dict):
        raise ValueError(f"{path} first state is not an object")
    qpos = np.asarray(state.get("qpos", []), dtype=np.float64)
    qvel = np.asarray(state.get("qvel", []), dtype=np.float64)
    expected = n_links + 1
    if qpos.shape != (expected,) or qvel.shape != (expected,):
        raise ValueError(f"{path} state must have qpos/qvel shape {(expected,)}")
    scale = float(scale)
    if not 0.0 < scale <= 1.0:
        raise ValueError("terminal target scale must be in (0, 1]")
    target_qpos = np.zeros(expected, dtype=np.float64)
    target_qpos[0] = scale * qpos[0]
    target_qpos[1:] = scale * wrap_angle(qpos[1:])
    target_qvel = scale * qvel
    target = np.r_[target_qpos, target_qvel]
    metadata = {
        "source": file_metadata(path),
        "scale": scale,
        "qpos": target_qpos.tolist(),
        "qvel": target_qvel.tolist(),
    }
    return target, metadata


def metrics(env: NLinkCartPoleEnv, state: np.ndarray, *, time_seconds: float, action: float) -> dict[str, Any]:
    n = env.n
    d = n + 1
    qpos = np.asarray(state[:d], dtype=np.float64)
    qvel = np.asarray(state[d:], dtype=np.float64)
    absolute_angles = serial_absolute_angles(qpos[1:])
    hinge_rates = qvel[1:]
    absolute_rates = np.cumsum(hinge_rates)
    return {
        "time_seconds": float(time_seconds),
        "action": float(action),
        "qpos": qpos.tolist(),
        "qvel": qvel.tolist(),
        "absolute_angles": absolute_angles.tolist(),
        "max_abs_angle": float(np.max(np.abs(absolute_angles))),
        "hinge_velocity_rms": float(np.sqrt(np.mean(hinge_rates**2))),
        "max_hinge_velocity": float(np.max(np.abs(hinge_rates))),
        "absolute_angular_velocity_rms": float(np.sqrt(np.mean(absolute_rates**2))),
        "max_absolute_angular_velocity": float(np.max(np.abs(absolute_rates))),
        "cart_abs": abs(float(qpos[0])),
        "cart_velocity_abs": abs(float(qvel[0])),
    }


def replay(env: NLinkCartPoleEnv, controls: np.ndarray) -> dict[str, Any]:
    env.reset(seed=0)
    rows: list[dict[str, Any]] = []
    for index, action in enumerate(controls, 1):
        _, _, terminated, truncated, info = env.step([float(action)])
        rows.append(metrics(env, data_state(env.data), time_seconds=index * env.dt, action=float(action)))
        if terminated or truncated:
            break
    if not rows:
        raise RuntimeError("replay emitted no states")
    best_composite = min(
        rows,
        key=lambda row: (
            (row["max_abs_angle"] / 0.15) ** 2
            + (row["hinge_velocity_rms"] / 0.75) ** 2
            + (row["absolute_angular_velocity_rms"] / 0.75) ** 2
            + (row["cart_abs"] / 1.25) ** 2
            + (row["cart_velocity_abs"] / 0.50) ** 2
        ),
    )
    terminal = rows[-1]
    return {
        "steps": len(rows),
        "termination_reason": "time_limit" if len(rows) == len(controls) else "terminated",
        "max_cart_excursion": float(max(row["cart_abs"] for row in rows)),
        "best_composite": best_composite,
        "terminal": terminal,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine a hanging-start seven-link force route with exact iLQR")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument(
        "--progress",
        type=float,
        default=1.0,
        help="fixed morphology progress for the proposal plant; 1.0 is canonical uniform",
    )
    parser.add_argument("--init-controller-json", default=None)
    parser.add_argument("--terminal-target-state", default=None)
    parser.add_argument("--terminal-target-scale", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=4.56)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--angle-terminal-weight", type=float, default=30.0)
    parser.add_argument("--rate-terminal-weight", type=float, default=20.0)
    parser.add_argument("--cart-terminal-weight", type=float, default=5.0)
    parser.add_argument("--cart-velocity-terminal-weight", type=float, default=10.0)
    parser.add_argument("--stage-angle-weight", type=float, default=0.002)
    parser.add_argument("--stage-rate-weight", type=float, default=0.001)
    parser.add_argument("--control-weight", type=float, default=0.01)
    parser.add_argument("--rail-soft-limit", type=float, default=2.85)
    parser.add_argument("--rail-weight", type=float, default=200000.0)
    parser.add_argument(
        "--terminal-value-mode",
        choices=("quadratic", "lqr"),
        default="quadratic",
        help="Use the configured terminal quadratic or a local upright DARE value matrix",
    )
    parser.add_argument(
        "--terminal-lqr-scale",
        type=float,
        default=1.0,
        help="Scale the DARE value matrix when --terminal-value-mode=lqr",
    )
    parser.add_argument("--handoff-out", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("--progress must be in [0, 1]")
    if min(args.seconds, args.iterations, args.angle_terminal_weight, args.rate_terminal_weight, args.cart_terminal_weight, args.cart_velocity_terminal_weight, args.control_weight, args.rail_soft_limit, args.rail_weight, args.terminal_lqr_scale) <= 0.0:
        raise ValueError("duration, weights, and rail bounds must be positive")
    if min(args.stage_angle_weight, args.stage_rate_weight) < 0.0:
        raise ValueError("stage weights must be nonnegative")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "episode_seconds": float(args.seconds + 1.0),
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0

    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"]["action_lqr_switch"] = {"enabled": False}
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=0)
    env.reset(seed=0)
    steps = max(2, int(round(args.seconds / env.dt)))
    controls = load_initial_controls(args.init_controller_json, seconds=args.seconds, steps=steps, dt=env.dt)
    n = env.n
    d = n + 1
    transform = absolute_rate_transform(n, cart_position=1.25, angle=0.15, cart_velocity=0.50, rate=0.75)
    transition = MujocoTransition(env, coordinate_transform=transform)
    initial_state = transition.to_coordinates(data_state(env.data))
    terminal_target_physical, terminal_target_metadata = load_terminal_target(
        args.terminal_target_state,
        scale=args.terminal_target_scale,
        n_links=n,
    )
    terminal_target = (
        None
        if terminal_target_physical is None
        else transition.to_coordinates(terminal_target_physical)
    )
    stage = np.diag(
        np.r_[
            np.asarray([0.02] + [args.stage_angle_weight] * n, dtype=np.float64),
            np.asarray([0.02] + [args.stage_rate_weight] * n, dtype=np.float64),
        ]
    )
    terminal = np.diag(
        np.r_[
            np.asarray([args.cart_terminal_weight] + [args.angle_terminal_weight] * n, dtype=np.float64),
            np.asarray([args.cart_velocity_terminal_weight] + [args.rate_terminal_weight] * n, dtype=np.float64),
        ]
    )
    terminal_value_mode = str(args.terminal_value_mode)
    if terminal_value_mode == "lqr":
        upright_coordinate_state = np.zeros_like(initial_state)
        upright_a, upright_b = transition.linearize(
            upright_coordinate_state,
            0.0,
            state_epsilon=2e-5,
            action_epsilon=2e-4,
        )
        try:
            terminal = float(args.terminal_lqr_scale) * solve_discrete_are(
                upright_a,
                upright_b,
                terminal,
                np.asarray([[float(args.control_weight)]], dtype=np.float64),
            )
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("could not solve the upright DARE terminal value") from exc
    cost = QuadraticTrajectoryCost(
        stage_state=stage,
        terminal_state=terminal,
        control=float(args.control_weight),
        rail_soft_limit=float(args.rail_soft_limit / 1.25),
        rail_limit=float(env.rail_limit / 1.25),
        rail_weight=float(args.rail_weight),
        wrap_angles=False,
        terminal_target=terminal_target,
    )
    started = time.time()
    search = optimize_ilqr(
        transition,
        initial_state,
        controls,
        cost,
        max_iterations=args.iterations,
        state_epsilon=2e-5,
        action_epsilon=2e-4,
        tolerance=1e-7,
    )
    refined_controls = np.asarray(search.controls, dtype=np.float64)
    replay_result = replay(env, refined_controls)
    terminal_metrics = replay_result["terminal"]
    best_metrics = replay_result["best_composite"]
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact-MuJoCo hanging-start iLQR terminal-arrival proposal; downstream capture verification required.",
        "config_path": str(Path(args.config)),
        "progress": float(args.progress),
            "init_controller_json": None if args.init_controller_json is None else file_metadata(args.init_controller_json),
            "terminal_target": terminal_target_metadata,
        "config_sha256": data_sha256(cfg),
        "search": {
            "seconds": float(args.seconds),
            "steps": int(steps),
            "iterations": int(search.iterations),
            "converged": bool(search.converged),
            "cost": float(search.cost),
            "active_control_steps": int(search.active_control_steps),
            "angle_terminal_weight": float(args.angle_terminal_weight),
            "rate_terminal_weight": float(args.rate_terminal_weight),
            "cart_terminal_weight": float(args.cart_terminal_weight),
            "cart_velocity_terminal_weight": float(args.cart_velocity_terminal_weight),
            "stage_angle_weight": float(args.stage_angle_weight),
            "stage_rate_weight": float(args.stage_rate_weight),
            "control_weight": float(args.control_weight),
            "rail_soft_limit": float(args.rail_soft_limit),
            "rail_weight": float(args.rail_weight),
            "terminal_value_mode": terminal_value_mode,
            "terminal_lqr_scale": float(args.terminal_lqr_scale),
            "wall_time_seconds": float(time.time() - started),
        },
        "controller": {
            "type": "exact_mujoco_ilqr_absolute_angle_rate_terminal",
            "horizon_seconds": float(len(refined_controls) * env.dt),
            "controls": refined_controls.tolist(),
            "nominal_coordinate_states": np.asarray(search.states, dtype=np.float64).tolist(),
            "feedback_gains": np.asarray(search.feedback_gains, dtype=np.float64).tolist(),
        },
        "terminal_metrics": terminal_metrics,
        "best_composite_metrics": best_metrics,
        "replay": replay_result,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(payload, Path(args.out))
    if args.handoff_out is not None:
        dump_json(
            {
                "schema_version": 1,
                "generated_at": utc_timestamp(),
                "not_solution": True,
                "summary": "Exact terminal state emitted by a hanging-start iLQR proposal; capture verification required.",
                "source_run": str(Path(args.out)),
                "config_sha256": data_sha256(cfg),
                "states": [
                    {
                        "state_index": 0,
                        "source_time_seconds": float(terminal_metrics["time_seconds"]),
                        "qpos": terminal_metrics["qpos"],
                        "qvel": terminal_metrics["qvel"],
                        "absolute_angles": terminal_metrics["absolute_angles"],
                        "max_abs_angle": terminal_metrics["max_abs_angle"],
                        "hinge_velocity_rms": terminal_metrics["hinge_velocity_rms"],
                        "absolute_angular_velocity_rms": terminal_metrics["absolute_angular_velocity_rms"],
                        "cart_abs": terminal_metrics["cart_abs"],
                        "cart_velocity_abs": terminal_metrics["cart_velocity_abs"],
                    }
                ],
            },
            Path(args.handoff_out),
        )
    print(
        f"cost={search.cost:.6f} converged={search.converged} "
        f"terminal_angle={terminal_metrics['max_abs_angle']:.6f} "
        f"terminal_hinge={terminal_metrics['hinge_velocity_rms']:.6f} "
        f"terminal_abs_rate={terminal_metrics['absolute_angular_velocity_rms']:.6f} "
        f"terminal_x={terminal_metrics['cart_abs']:.6f} "
        f"terminal_xd={terminal_metrics['cart_velocity_abs']:.6f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
