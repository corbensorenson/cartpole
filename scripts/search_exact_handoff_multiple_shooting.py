#!/usr/bin/env python
"""Search a terminal-rate-aware exact-MuJoCo tail from a real 7-link state.

This is a component diagnostic.  It starts from the recorded qpos/qvel without
resetting or replacing the state during the optimized tail, then replays the
optimized action sequence at the environment action cadence.  The terminal
residual penalizes both relative hinge rates and cumulative absolute link rates,
which prevents a quiet-looking relative state from being accepted as a quiet
physical chain.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform
from gcartpole.multiple_shooting import exact_rollout, optimize_multiple_shooting


def load_state(path: str, index: int) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(states, list) or not states:
        raise ValueError(f"{path} does not contain a non-empty states list")
    if index < 0 or index >= len(states):
        raise IndexError(f"state index {index} outside 0..{len(states) - 1}")
    state = states[index]
    if not isinstance(state, dict) or "qpos" not in state or "qvel" not in state:
        raise ValueError(f"state {index} in {path} is missing qpos/qvel")
    return dict(state)


def fixed_state_cfg(cfg: dict[str, Any], state: dict[str, Any], seconds: float) -> dict[str, Any]:
    env = {
        **cfg["env"],
        "init_mode": "fixed_state",
        "init_qpos": state["qpos"],
        "init_qvel": state["qvel"],
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
    return {**cfg, "env": env}


def initial_controls_from_json(path: str | None, steps: int) -> np.ndarray:
    if path is None:
        return np.zeros(steps, dtype=np.float64)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    candidates: list[Any] = []
    if isinstance(payload.get("controller"), dict):
        candidates.extend(
            [payload["controller"].get("controls"), payload["controller"].get("suffix_controls")]
        )
    if isinstance(payload.get("best"), dict):
        candidates.append(payload["best"].get("controls"))
        if isinstance(payload["best"].get("controller"), dict):
            candidates.append(payload["best"]["controller"].get("controls"))
    if isinstance(payload.get("best_feasible"), dict):
        candidates.append(payload["best_feasible"].get("controls"))
    source = next((value for value in candidates if value is not None), None)
    if source is None:
        raise ValueError(f"{path} does not contain a control sequence")
    source = np.clip(np.asarray(source, dtype=np.float64), -1.0, 1.0)
    if source.ndim != 1 or source.size < 2:
        raise ValueError("initial control sequence must contain at least two actions")
    source_times = np.linspace(0.0, 1.0, source.size)
    target_times = np.linspace(0.0, 1.0, steps)
    return np.interp(target_times, source_times, source)


def terminal_factor(n_links: int, *, angle_weight: float, relative_rate_weight: float, absolute_rate_weight: float,
                    cart_weight: float, cart_velocity_weight: float, hinge_scale: float) -> np.ndarray:
    """Return a residual factor in dimensionless absolute coordinates."""
    d = n_links + 1
    rows: list[np.ndarray] = []

    def row(index: int, weight: float) -> None:
        vector = np.zeros(2 * d, dtype=np.float64)
        vector[index] = np.sqrt(max(0.0, float(weight)))
        rows.append(vector)

    row(0, cart_weight)
    for index in range(1, d):
        row(index, angle_weight)
    row(d, cart_velocity_weight)
    for index in range(d + 1, 2 * d):
        row(index, relative_rate_weight)

    # Coordinates store relative hinge rates divided by hinge_scale.  This
    # lower-triangular map produces cumulative physical link rates.
    for link in range(n_links):
        vector = np.zeros(2 * d, dtype=np.float64)
        vector[d + 1 : d + 2 + link] = np.sqrt(max(0.0, float(absolute_rate_weight))) * hinge_scale
        rows.append(vector)
    return np.asarray(rows, dtype=np.float64)


def row(env: NLinkCartPoleEnv, step: int, action: float, info: dict[str, Any]) -> dict[str, Any]:
    _, absolute = env._angles()
    hinge = np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64)
    absolute_rate = np.cumsum(hinge)
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "action": float(action),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).tolist(),
        "absolute_angles": absolute.tolist(),
        "max_abs_angle": float(np.max(np.abs(absolute))),
        "hinge_velocity_rms": float(np.sqrt(np.mean(hinge**2))),
        "absolute_angular_velocity_rms": float(np.sqrt(np.mean(absolute_rate**2))),
        "max_absolute_angular_velocity": float(np.max(np.abs(absolute_rate))),
        "x": float(env.data.qpos[0]),
        "cart_velocity": float(env.data.qvel[0]),
        "is_upright": bool(info.get("is_upright", False)),
        "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
        "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
    }


def replay(cfg: dict[str, Any], controls: np.ndarray, state: dict[str, Any], seconds: float) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset()
    rows: list[dict[str, Any]] = []
    max_cart = 0.0
    final_info: dict[str, Any] = {}
    steps = min(env.max_steps, int(round(seconds / env.dt)))
    for step in range(steps):
        action = float(controls[min(step, len(controls) - 1)])
        _, _, terminated, truncated, info = env.step([action])
        rows.append(row(env, step + 1, action, info))
        max_cart = max(max_cart, abs(float(env.data.qpos[0])))
        final_info = dict(info)
        if terminated or truncated:
            break
    env.close()
    return {
        "rows": rows,
        "endpoint": rows[-1] if rows else None,
        "best_angle": min(rows, key=lambda item: item["max_abs_angle"]) if rows else None,
        "best_absolute_rate": min(rows, key=lambda item: item["absolute_angular_velocity_rms"]) if rows else None,
        "max_cart_abs": float(max_cart),
        "final_info": final_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact MuJoCo multiple-shooting tail from a recorded handoff")
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--initial-controls-json", default=None)
    parser.add_argument("--horizon-seconds", type=float, default=4.0)
    parser.add_argument("--replay-seconds", type=float, default=8.0)
    parser.add_argument("--segment-steps", type=int, default=10)
    parser.add_argument("--defect-weight", type=float, default=1.0e5)
    parser.add_argument("--terminal-weight", type=float, default=1.0)
    parser.add_argument("--control-weight", type=float, default=0.002)
    parser.add_argument("--rail-weight", type=float, default=1.0e4)
    parser.add_argument("--rail-soft-limit", type=float, default=3.0)
    parser.add_argument(
        "--optimizer-rail-limit",
        type=float,
        default=None,
        help="temporary optimizer state bound; replay still uses the configured rail",
    )
    parser.add_argument("--angle-weight", type=float, default=250.0)
    parser.add_argument("--relative-rate-weight", type=float, default=8.0)
    parser.add_argument("--absolute-rate-weight", type=float, default=80.0)
    parser.add_argument("--cart-weight", type=float, default=0.25)
    parser.add_argument("--cart-velocity-weight", type=float, default=0.25)
    parser.add_argument("--hinge-scale", type=float, default=1.0)
    parser.add_argument("--max-evaluations", type=int, default=30)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if min(args.horizon_seconds, args.replay_seconds, args.segment_steps, args.max_evaluations) <= 0:
        raise ValueError("horizon, replay, segment steps, and max evaluations must be positive")
    if int(round(args.horizon_seconds / 0.02)) % args.segment_steps != 0:
        raise ValueError("horizon_seconds must produce a step count divisible by segment_steps")

    state = load_state(args.state_json, args.state_index)
    base = apply_overrides(load_config(args.config), args.override)
    cfg = fixed_state_cfg(base, state, args.replay_seconds)
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset()
    transform = dimensionless_absolute_transform(
        env.n,
        StateScales(3.0, 1.0, 3.0, args.hinge_scale),
    )
    transition = MujocoTransition(env, coordinate_transform=transform)
    initial_state = transition.to_coordinates(data_state(env.data))
    horizon_steps = int(round(args.horizon_seconds / env.dt))
    controls = initial_controls_from_json(args.initial_controls_json, horizon_steps)
    factor = terminal_factor(
        env.n,
        angle_weight=args.angle_weight,
        relative_rate_weight=args.relative_rate_weight,
        absolute_rate_weight=args.absolute_rate_weight,
        cart_weight=args.cart_weight,
        cart_velocity_weight=args.cart_velocity_weight,
        hinge_scale=args.hinge_scale,
    )
    started = time.time()
    search = optimize_multiple_shooting(
        transition,
        initial_state,
        controls,
        segment_steps=args.segment_steps,
        defect_weight=args.defect_weight,
        terminal_weight=args.terminal_weight,
        control_weight=args.control_weight,
        rail_weight=args.rail_weight,
        rail_soft_limit=args.rail_soft_limit / 3.0,
        rail_limit=float(
            (env.rail_limit if args.optimizer_rail_limit is None else args.optimizer_rail_limit)
            / 3.0
        ),
        max_evaluations=args.max_evaluations,
        terminal_factor=factor,
    )
    env.close()
    replay_result = replay(cfg, search.controls, state, args.replay_seconds)
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact-MuJoCo terminal absolute-rate multiple-shooting tail from a recorded seven-link handoff.",
        "config_path": str(Path(args.config)),
        "state_json": str(Path(args.state_json)),
        "state_index": int(args.state_index),
        "initial_state": state,
        "search": {
            "horizon_seconds": float(args.horizon_seconds),
            "segment_steps": int(args.segment_steps),
            "defect_weight": float(args.defect_weight),
            "terminal_weight": float(args.terminal_weight),
            "control_weight": float(args.control_weight),
            "rail_weight": float(args.rail_weight),
            "rail_soft_limit": float(args.rail_soft_limit),
            "optimizer_rail_limit": (
                float(env.rail_limit)
                if args.optimizer_rail_limit is None
                else float(args.optimizer_rail_limit)
            ),
            "angle_weight": float(args.angle_weight),
            "relative_rate_weight": float(args.relative_rate_weight),
            "absolute_rate_weight": float(args.absolute_rate_weight),
            "cart_weight": float(args.cart_weight),
            "cart_velocity_weight": float(args.cart_velocity_weight),
            "max_evaluations": int(args.max_evaluations),
            "cost": float(search.cost),
            "optimality": float(search.optimality),
            "evaluations": int(search.evaluations),
            "success": bool(search.success),
            "status": int(search.status),
            "message": search.message,
            "wall_time_seconds": float(time.time() - started),
        },
        "controller": {
            "type": "normalized_force_terminal_absolute_rate_multiple_shooting",
            "controls": search.controls.tolist(),
            "node_states": search.node_states.tolist(),
            "segment_defects_max_abs": float(np.max(np.abs(search.segment_defects))),
        },
        "replay": replay_result,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    best = replay_result["best_angle"]
    print(
        f"solver_success={search.success} cost={search.cost:.6f} "
        f"defect={np.max(np.abs(search.segment_defects)):.3e} "
        f"best_angle={best['max_abs_angle'] if best else float('nan'):.6f} "
        f"best_abs_rate={replay_result['best_absolute_rate']['absolute_angular_velocity_rms'] if replay_result['best_absolute_rate'] else float('nan'):.6f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
