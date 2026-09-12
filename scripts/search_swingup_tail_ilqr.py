#!/usr/bin/env python
"""Refine the terminal part of a hanging-start swing in exact MuJoCo.

The source swing controller supplies a real approach state. iLQR changes only
the action tail, and the replay keeps the original approach and simulator
state continuous. This is a trajectory-discovery diagnostic; held-out gate
evaluation remains a separate step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import MujocoTransition, QuadraticTrajectoryCost, data_state, optimize_ilqr
from gcartpole.modal import StateScales, dimensionless_absolute_transform

try:
    from scripts.probe_swingup_trajectory import (
        DEFAULT_KD,
        DEFAULT_KNOTS,
        DEFAULT_KP,
        DEFAULT_TRAJECTORY_SECONDS,
        trajectory_action,
    )
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from probe_swingup_trajectory import (
        DEFAULT_KD,
        DEFAULT_KNOTS,
        DEFAULT_KP,
        DEFAULT_TRAJECTORY_SECONDS,
        trajectory_action,
    )
    from search_swingup_capture import lqr_action, lqr_gain


def load_swing_controller(path: str | None, record_key: str | None = None) -> dict[str, Any]:
    if path is None:
        return {
            "type": "cart_position_pd_fixed_knots",
            "trajectory_seconds": float(DEFAULT_TRAJECTORY_SECONDS),
            "kp": float(DEFAULT_KP),
            "kd": float(DEFAULT_KD),
            "knots": DEFAULT_KNOTS.astype(float).tolist(),
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    # Global force-CEM artifacts store normalized force knots directly under
    # best.knots. Preserve that controller type so the tail optimizer does not
    # misinterpret force values as cart-position targets.
    if isinstance(payload, dict) and isinstance(payload.get("best"), dict):
        best = payload["best"]
        if "knots" in best and "controller" not in best:
            return {
                "type": "normalized_force_knots",
                "trajectory_seconds": float(payload.get("search", {}).get("seconds", 0.0)),
                "knots": list(best["knots"]),
            }
    if isinstance(payload, dict) and isinstance(payload.get("best_by"), dict):
        record = payload["best_by"].get(record_key or "score")
        if isinstance(record, dict) and isinstance(record.get("controller"), dict):
            return dict(record["controller"])
    if isinstance(payload, dict) and isinstance(payload.get("best"), dict):
        record = payload["best"]
        if isinstance(record.get("controller"), dict):
            return dict(record["controller"])
    if isinstance(payload, dict) and isinstance(payload.get("controller"), dict):
        return dict(payload["controller"])
    if isinstance(payload, dict) and "knots" in payload:
        return dict(payload)
    raise ValueError(f"Could not load a swing controller from {path}")


def source_action(
    env: NLinkCartPoleEnv,
    controller: dict[str, Any],
    t: float,
    step: int | None = None,
) -> float:
    if controller.get("controls") is not None:
        controls = np.asarray(controller["controls"], dtype=np.float64)
        if controls.ndim != 1 or controls.size < 2:
            raise ValueError("FDDP source controls must be a one-dimensional sequence")
        seconds = float(controller.get("horizon_seconds", controls.size * env.dt))
        source_times = np.linspace(0.0, max(seconds, env.dt), controls.size)
        return float(
            np.clip(
                np.interp(
                    step * env.dt if step is not None else t,
                    source_times,
                    controls,
                    left=controls[0],
                    right=controls[-1],
                ),
                -1.0,
                1.0,
            )
        )
    knots = np.asarray(controller["knots"], dtype=np.float64)
    seconds = float(controller["trajectory_seconds"])
    if str(controller.get("type", "")).lower() == "normalized_force_knots":
        # Match search_swingup_global_cem exactly: its interpolation grid is
        # indexed over the action count, not over physical t=i*dt. Chaotic
        # swing dynamics make even this small distinction materially relevant.
        action_count = max(2, int(round(seconds / env.dt)))
        phase = (
            float(np.clip(step, 0, action_count - 1)) / float(action_count - 1)
            if step is not None
            else float(np.clip(t / max(seconds, 1e-9), 0.0, 1.0))
        )
        knot_phase = np.linspace(0.0, 1.0, len(knots), dtype=np.float64)
        return float(np.clip(np.interp(phase, knot_phase, knots), -1.0, 1.0))
    return float(
        trajectory_action(
            env,
            t,
            knots,
            seconds,
            float(controller["kp"]),
            float(controller["kd"]),
        )
    )


def load_tail_actions(path: str | None, step_count: int) -> np.ndarray | None:
    if path is None:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    best = payload.get("best") if isinstance(payload, dict) else None
    if not isinstance(best, dict) or "knots" not in best:
        raise ValueError(f"could not load tail knots from {path}")
    knots = np.asarray(best["knots"], dtype=np.float64)
    source_steps = int(payload.get("tail_horizon_steps", step_count))
    source_phase = np.linspace(0.0, 1.0, source_steps, dtype=np.float64)
    target_phase = np.linspace(0.0, 1.0, step_count, dtype=np.float64)
    knot_phase = np.linspace(0.0, 1.0, len(knots), dtype=np.float64)
    source_actions = np.interp(source_phase, knot_phase, knots)
    return np.interp(target_phase, source_phase, source_actions)


def state_array(env: NLinkCartPoleEnv) -> np.ndarray:
    return data_state(env.data).astype(np.float64)


def initial_tail_problem(
    cfg: dict[str, Any],
    *,
    controller: dict[str, Any],
    tail_start_seconds: float,
    tail_steps: int,
    zero_noise: bool,
    initial_tail_actions: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, float]:
    cfg = {**cfg, "env": {**cfg["env"]}}
    if zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
        for noise_key in (
            "init_angle_noise",
            "init_vel_noise",
            "init_cart_noise",
            "init_cart_vel_noise",
        ):
            cfg["env"][noise_key] = 0.0
            cfg["env"][f"{noise_key}_start"] = 0.0
            cfg["env"][f"{noise_key}_end"] = 0.0
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    pre_steps = int(round(tail_start_seconds / env.dt))
    for step in range(pre_steps):
        action = source_action(env, controller, step * env.dt, step)
        _, _, terminated, truncated, info = env.step([action])
        if terminated or truncated:
            env.close()
            raise RuntimeError(
                f"source swing terminates before tail start at step {step}: {info.get('termination_reason')}"
            )
    start = state_array(env)
    if initial_tail_actions is not None:
        env.close()
        return start, np.asarray(initial_tail_actions[:tail_steps], dtype=np.float64), env.dt
    initial_controls: list[float] = []
    for offset in range(tail_steps):
        action = source_action(env, controller, (pre_steps + offset) * env.dt, pre_steps + offset)
        initial_controls.append(float(action))
        _, _, terminated, truncated, info = env.step([action])
        if terminated or truncated:
            # Keep the optimization horizon fixed. A terminated nominal tail
            # is still a useful negative warm start, but pad it deterministically.
            initial_controls.extend([0.0] * (tail_steps - len(initial_controls)))
            break
    env.close()
    return start, np.asarray(initial_controls[:tail_steps], dtype=np.float64), env.dt


def replay_controller(
    cfg: dict[str, Any],
    *,
    source_controller: dict[str, Any],
    tail_start_seconds: float,
    tail_controls: np.ndarray,
    tail_states: np.ndarray,
    tail_feedback: np.ndarray,
    transform: np.ndarray,
    gain: np.ndarray,
    lqr_scale: float,
    seconds: float,
    zero_noise: bool,
) -> dict[str, Any]:
    cfg = {**cfg, "env": {**cfg["env"]}}
    if zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    obs, _ = env.reset(seed=0)
    del obs
    transition = MujocoTransition(env, coordinate_transform=transform)
    tail_start_step = int(round(tail_start_seconds / env.dt))
    total_steps = min(env.max_steps, int(seconds / env.dt))
    trajectory: list[dict[str, Any]] = []
    total_return = 0.0
    max_cart_abs = 0.0
    final_info: dict[str, Any] = {}
    for step in range(total_steps):
        t = step * env.dt
        if step < tail_start_step:
            action = source_action(env, source_controller, t, step)
            mode = "source_swing"
        elif step < tail_start_step + len(tail_controls):
            tail_step = step - tail_start_step
            coordinate_state = transition.to_coordinates(state_array(env))
            error = transition.difference(coordinate_state, tail_states[tail_step])
            action = float(
                np.clip(
                    tail_controls[tail_step] + tail_feedback[tail_step] @ error,
                    -1.0,
                    1.0,
                )
            )
            mode = "ilqr_tail"
        else:
            action = lqr_action(env, gain, scale=lqr_scale, cart_target=0.0)
            mode = "lqr_settle"
        _, reward, terminated, truncated, info = env.step([action])
        total_return += float(reward)
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        relative_angles, absolute_angles = env._angles()
        trajectory.append(
            {
                "step": int(step + 1),
                "time_seconds": float((step + 1) * env.dt),
                "controller_mode": mode,
                "action": float(action),
                "reward": float(reward),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
                "x": float(info["x"]),
                "cart_velocity": float(env.data.qvel[0]),
                "relative_angles": relative_angles.astype(float).tolist(),
                "absolute_angles": absolute_angles.astype(float).tolist(),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "is_upright": bool(info["is_upright"]),
                "upright_streak_seconds": float(info["upright_streak_seconds"]),
                "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
            }
        )
        final_info = dict(info)
        if terminated or truncated:
            break
    env.close()
    return {
        "success": bool(final_info.get("success", False)),
        "termination_reason": final_info.get("termination_reason"),
        "episode_return": float(total_return),
        "simulated_steps": int(len(trajectory)),
        "simulated_seconds": float(len(trajectory) * env.dt),
        "max_cart_excursion": float(max_cart_abs),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "time_to_capture": final_info.get("time_to_capture"),
        "final_info": final_info,
        "trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Refine a hanging-start swing tail with exact-MuJoCo iLQR")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--swing-controller-json", required=True)
    parser.add_argument(
        "--controller-key",
        default=None,
        help="Optional best_by record to replay, such as max_upright_streak or min_best_pass_angle.",
    )
    parser.add_argument("--init-tail-json", default=None)
    parser.add_argument("--tail-start-seconds", type=float, default=4.0)
    parser.add_argument("--tail-seconds", type=float, default=4.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--lqr-scale", type=float, default=1.0)
    parser.add_argument("--control-cost", type=float, default=0.05)
    parser.add_argument("--rail-soft-limit", type=float, default=2.4)
    parser.add_argument("--rail-weight", type=float, default=1e8)
    parser.add_argument("--out", required=True)
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if min(args.tail_start_seconds, args.tail_seconds, args.seconds, args.control_cost, args.rail_soft_limit, args.rail_weight) <= 0.0:
        raise ValueError("tail durations, control cost, rail limit, and rail weight must be positive")
    if args.iterations < 1:
        raise ValueError("--iterations must be positive")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"] = {**base_cfg["env"], "init_mode": "hanging"}
    controller = load_swing_controller(args.swing_controller_json, args.controller_key)
    policy_dt = float(base_cfg["env"]["timestep"]) * int(base_cfg["env"].get("frame_skip", 1))
    tail_steps = max(2, int(round(args.tail_seconds / policy_dt)))
    initial_tail_actions = load_tail_actions(args.init_tail_json, tail_steps)
    start_physical, initial_controls, policy_dt = initial_tail_problem(
        base_cfg,
        controller=controller,
        tail_start_seconds=args.tail_start_seconds,
        tail_steps=tail_steps,
        zero_noise=args.zero_noise,
        initial_tail_actions=initial_tail_actions,
    )
    env = NLinkCartPoleEnv(base_cfg, progress=1.0, seed=0)
    transition = MujocoTransition(
        env,
        coordinate_transform=dimensionless_absolute_transform(
            int(base_cfg["env"]["n_links"]),
            StateScales(3.0, 1.0, 3.0, 3.0),
        ),
    )
    transform = transition.coordinate_transform
    assert transform is not None
    start_coordinates = transition.to_coordinates(start_physical)
    n_links = int(base_cfg["env"]["n_links"])
    nx = 2 * (n_links + 1)
    stage_weights = np.r_[0.10, np.full(n_links, 6.0), 0.10, np.full(n_links, 0.20)]
    terminal_weights = np.r_[2.0, np.full(n_links, 180.0), 0.8, np.full(n_links, 8.0)]
    cost = QuadraticTrajectoryCost(
        stage_state=np.diag(stage_weights),
        terminal_state=np.diag(terminal_weights),
        control=float(args.control_cost),
        rail_soft_limit=float(args.rail_soft_limit / 3.0),
        rail_limit=float(env.rail_limit / 3.0),
        rail_weight=float(args.rail_weight),
        wrap_angles=False,
    )
    started = time.time()
    search = optimize_ilqr(
        transition,
        start_coordinates,
        initial_controls,
        cost,
        max_iterations=args.iterations,
    )
    search_seconds = time.time() - started
    gain = lqr_gain(base_cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    replay = replay_controller(
        base_cfg,
        source_controller=controller,
        tail_start_seconds=args.tail_start_seconds,
        tail_controls=search.controls,
        tail_states=search.states,
        tail_feedback=search.feedback_gains,
        transform=transform,
        gain=gain,
        lqr_scale=args.lqr_scale,
        seconds=args.seconds,
        zero_noise=args.zero_noise,
    )
    env.close()
    physical_states = np.asarray([transition.to_physical(row) for row in search.states], dtype=np.float64)
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact-MuJoCo hanging-start swing tail refinement; requires held-out evaluation before acceptance.",
        "source_swing_controller": file_metadata(args.swing_controller_json),
        "init_tail_json": args.init_tail_json,
        "source_controller": controller,
        "selected_state": {
            "qpos": start_physical[: n_links + 1].astype(float).tolist(),
            "qvel": start_physical[n_links + 1 :].astype(float).tolist(),
            "tail_start_seconds": float(args.tail_start_seconds),
        },
        "controller": {
            "type": "source_swing_then_exact_mujoco_ilqr_tail_then_lqr",
            "reset_at_boundary": False,
            "tail_start_seconds": float(args.tail_start_seconds),
            "tail_horizon_steps": int(search.controls.size),
            "tail_horizon_seconds": float(search.controls.size * policy_dt),
            "controls": search.controls.astype(float).tolist(),
            "feedback_gains": search.feedback_gains.astype(float).tolist(),
            "nominal_coordinate_states": search.states.astype(float).tolist(),
            "lqr_scale": float(args.lqr_scale),
        },
        "search": {
            "cost": float(search.cost),
            "iterations": int(search.iterations),
            "converged": bool(search.converged),
            "active_control_steps": int(search.active_control_steps),
            "wall_time_seconds": float(search_seconds),
            "initial_state_dimension": int(nx),
            "initial_controls": initial_controls.astype(float).tolist(),
            "nominal_physical_states": physical_states.astype(float).tolist(),
            "history": search.history,
        },
        "replay": replay,
        "config_sha256": data_sha256(base_cfg),
        "controller_sha256": data_sha256({"source": controller, "tail": search.controls.astype(float).tolist()}),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(result, Path(args.out))
    print(
        f"search_cost={search.cost:.3f} terminal_angle="
        f"{replay['trajectory'][-1]['max_abs_angle'] if replay['trajectory'] else float('nan'):.6f} "
        f"success={replay['success']} hold={replay['max_upright_streak_seconds']:.3f}s"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
