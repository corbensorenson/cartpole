#!/usr/bin/env python
"""Search a local seven-link capture policy from one real handoff state.

This is a discovery diagnostic. It uses exact MuJoCo transitions, optimizes a
finite-horizon control sequence with iLQR, then replays the resulting
time-varying feedback without resetting the state at the handoff. It does not
claim a benchmark result or a globally valid capture basin.
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
from gcartpole.ilqr import (
    MujocoTransition,
    QuadraticTrajectoryCost,
    data_state,
    optimize_ilqr,
)
from gcartpole.modal import StateScales, dimensionless_absolute_transform

try:
    from scripts.search_capture_sequence import fixed_state_cfg, load_state
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg, load_state


def load_initial_controls(path: str | None, horizon_steps: int) -> np.ndarray:
    if path is None:
        return np.zeros(horizon_steps, dtype=np.float64)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    record: Any = payload.get("best_feasible") or payload.get("best")
    if record is None and isinstance(payload.get("controller"), dict):
        controls = payload["controller"].get("suffix_controls")
        if controls is not None:
            source_phase = np.linspace(0.0, 1.0, len(controls), dtype=np.float64)
            target_phase = np.linspace(0.0, 1.0, horizon_steps, dtype=np.float64)
            return np.clip(
                np.interp(target_phase, source_phase, np.asarray(controls, dtype=np.float64)),
                -1.0,
                1.0,
            )
        controls = payload["controller"].get("controls")
        if controls is not None:
            source_phase = np.linspace(0.0, 1.0, len(controls), dtype=np.float64)
            target_phase = np.linspace(0.0, 1.0, horizon_steps, dtype=np.float64)
            return np.clip(
                np.interp(target_phase, source_phase, np.asarray(controls, dtype=np.float64)),
                -1.0,
                1.0,
            )
    if not isinstance(record, dict):
        raise ValueError(f"{path} has no best or best_feasible record")
    if "knots" in record:
        knots = np.asarray(record["knots"], dtype=np.float64)
    elif "action_knots" in record:
        knots = np.asarray(record["action_knots"], dtype=np.float64)
    else:
        raise ValueError(f"{path} has no action knots in the selected record")
    source_phase = np.linspace(0.0, 1.0, len(knots), dtype=np.float64)
    target_phase = np.linspace(0.0, 1.0, horizon_steps, dtype=np.float64)
    return np.clip(np.interp(target_phase, source_phase, knots), -1.0, 1.0)


def row_from_env(
    env: NLinkCartPoleEnv,
    *,
    step: int,
    action: float,
    reward: float,
    info: dict[str, Any],
    mode: str,
) -> dict[str, Any]:
    rel, absolute = env._angles()
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "controller_mode": mode,
        "action": float(action),
        "reward": float(reward),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "x": float(info["x"]),
        "cart_velocity": float(env.data.qvel[0]),
        "relative_angles": rel.astype(float).tolist(),
        "absolute_angles": absolute.astype(float).tolist(),
        "max_abs_angle": float(info["max_abs_angle"]),
        "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
        "is_upright": bool(info["is_upright"]),
        "upright_streak_seconds": float(info["upright_streak_seconds"]),
        "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact-MuJoCo fixed-state iLQR capture search")
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--state-index", default="0")
    parser.add_argument(
        "--progress",
        type=float,
        default=1.0,
        help="Plant curriculum progress used for the fixed-state replay (0 preserves discovery morphology).",
    )
    parser.add_argument("--initial-controls-json", default=None)
    parser.add_argument(
        "--initial-arrest-action",
        type=float,
        default=None,
        help="Optional normalized force applied during the first short capture interval.",
    )
    parser.add_argument(
        "--initial-arrest-seconds",
        type=float,
        default=0.0,
        help="Duration of --initial-arrest-action before iLQR feedback takes over.",
    )
    parser.add_argument("--horizon-seconds", type=float, default=4.0)
    parser.add_argument("--replay-seconds", type=float, default=8.0)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--control-cost", type=float, default=0.02)
    parser.add_argument("--stage-angle-weight", type=float, default=12.0)
    parser.add_argument("--stage-hinge-weight", type=float, default=2.0)
    parser.add_argument("--stage-cart-velocity-weight", type=float, default=0.10)
    parser.add_argument("--terminal-angle-weight", type=float, default=160.0)
    parser.add_argument("--terminal-hinge-weight", type=float, default=12.0)
    parser.add_argument(
        "--terminal-absolute-rate-weight",
        type=float,
        default=0.0,
        help="Additional terminal quadratic weight on cumulative physical link rates.",
    )
    parser.add_argument("--terminal-cart-weight", type=float, default=4.0)
    parser.add_argument("--terminal-cart-velocity-weight", type=float, default=8.0)
    parser.add_argument("--state-angle-scale", type=float, default=0.15)
    parser.add_argument("--state-hinge-scale", type=float, default=0.75)
    parser.add_argument("--state-cart-scale", type=float, default=1.25)
    parser.add_argument("--state-cart-velocity-scale", type=float, default=0.50)
    parser.add_argument("--rail-soft-limit", type=float, default=10.5)
    parser.add_argument("--rail-weight", type=float, default=100_000.0)
    parser.add_argument("--seed", type=int, default=20260933)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if min(args.horizon_seconds, args.replay_seconds, args.iterations, args.control_cost) <= 0.0:
        raise ValueError("horizon, replay, iterations, and control cost must be positive")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"], "action_lqr_residual": {"enabled": False}}
    selected_state, state_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(cfg, selected_state, args.replay_seconds)
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("--progress must be inside [0, 1]")
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    env.reset(seed=args.seed)
    transform = dimensionless_absolute_transform(
        env.n,
        StateScales(
            args.state_cart_scale,
            args.state_angle_scale,
            args.state_cart_velocity_scale,
            args.state_hinge_scale,
        ),
    )
    transition = MujocoTransition(env, coordinate_transform=transform)
    horizon_steps = max(2, int(round(args.horizon_seconds / env.dt)))
    replay_steps = min(env.max_steps, int(round(args.replay_seconds / env.dt)))
    initial_state = transition.to_coordinates(data_state(env.data))
    initial_controls = load_initial_controls(args.initial_controls_json, horizon_steps)
    if args.initial_arrest_action is not None:
        if args.initial_arrest_seconds < 0.0:
            raise ValueError("--initial-arrest-seconds must be nonnegative")
        arrest_steps = min(
            horizon_steps,
            max(0, int(round(args.initial_arrest_seconds / env.dt))),
        )
        initial_controls[:arrest_steps] = float(np.clip(args.initial_arrest_action, -1.0, 1.0))
    n_links = env.n
    d = n_links + 1
    stage_weights = np.r_[
        0.10,
        np.full(n_links, args.stage_angle_weight),
        args.stage_cart_velocity_weight,
        np.full(n_links, args.stage_hinge_weight),
    ]
    terminal_weights = np.r_[
        args.terminal_cart_weight,
        np.full(n_links, args.terminal_angle_weight),
        args.terminal_cart_velocity_weight,
        np.full(n_links, args.terminal_hinge_weight),
    ]
    terminal_state_matrix = np.diag(terminal_weights)
    if args.terminal_absolute_rate_weight > 0.0:
        cumulative = np.tril(np.ones((n_links, n_links), dtype=np.float64))
        rate_start = d + 1
        terminal_state_matrix[rate_start:, rate_start:] += (
            float(args.terminal_absolute_rate_weight) * cumulative.T @ cumulative
        )
    cost = QuadraticTrajectoryCost(
        stage_state=np.diag(stage_weights),
        terminal_state=terminal_state_matrix,
        control=float(args.control_cost),
        rail_soft_limit=float(args.rail_soft_limit / args.state_cart_scale),
        rail_limit=float(env.rail_limit / args.state_cart_scale),
        rail_weight=float(args.rail_weight),
        wrap_angles=False,
    )
    started = time.time()
    search = optimize_ilqr(
        transition,
        initial_state,
        initial_controls,
        cost,
        max_iterations=args.iterations,
    )
    search_seconds = time.time() - started

    # Replay with the actual MuJoCo state. After the optimized horizon, hold
    # the final local feedback law around the optimized terminal state.
    env.close()
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    env.reset(seed=args.seed)
    transition = MujocoTransition(env, coordinate_transform=transform)
    trajectory: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    max_cart_abs = 0.0
    for step in range(replay_steps):
        coordinate_state = transition.to_coordinates(data_state(env.data))
        if step < horizon_steps:
            nominal = search.states[step]
            gain = search.feedback_gains[step]
            action = float(search.controls[step] + gain @ transition.difference(coordinate_state, nominal))
            mode = "ilqr_feedback"
        else:
            nominal = search.states[-1]
            gain = search.feedback_gains[-1]
            action = float(search.controls[-1] + gain @ transition.difference(coordinate_state, nominal))
            mode = "terminal_feedback"
        action = float(np.clip(action, -1.0, 1.0))
        _, reward, terminated, truncated, info = env.step([action])
        final_info = dict(info)
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        trajectory.append(
            row_from_env(
                env,
                step=step + 1,
                action=action,
                reward=reward,
                info=info,
                mode=mode,
            )
        )
        if terminated or truncated:
            break
    env.close()
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Fixed-state exact-MuJoCo iLQR capture diagnostic with terminal feedback; not canonical evidence.",
        "state_json": str(Path(args.state_json)),
        "state_index": int(state_index),
        "progress": float(args.progress),
        "selected_state": selected_state,
        "controller": {
            "type": "fixed_state_exact_mujoco_ilqr_terminal_feedback",
            "horizon_steps": int(horizon_steps),
            "horizon_seconds": float(horizon_steps * 0.02),
            "replay_seconds": float(replay_steps * 0.02),
            "controls": search.controls.astype(float).tolist(),
            "feedback_gains": search.feedback_gains.astype(float).tolist(),
        },
        "search": {
            "cost": float(search.cost),
            "iterations": int(search.iterations),
            "converged": bool(search.converged),
            "active_control_steps": int(search.active_control_steps),
            "terminal_absolute_rate_weight": float(args.terminal_absolute_rate_weight),
            "initial_arrest_action": None if args.initial_arrest_action is None else float(args.initial_arrest_action),
            "initial_arrest_seconds": float(args.initial_arrest_seconds),
            "wall_time_seconds": float(search_seconds),
            "history": search.history,
            "initial_state_coordinates": initial_state.astype(float).tolist(),
            "terminal_state_coordinates": search.states[-1].astype(float).tolist(),
        },
        "success": bool(final_info.get("success", False)),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "termination_reason": final_info.get("termination_reason"),
        "max_cart_excursion": float(max_cart_abs),
        "simulated_seconds": float(len(trajectory) * 0.02),
        "final_info": final_info,
        "trajectory": trajectory,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(
        f"success={payload['success']} hold={payload['max_upright_streak_seconds']:.3f}s "
        f"first_upright={payload['time_to_first_upright']} cost={search.cost:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
