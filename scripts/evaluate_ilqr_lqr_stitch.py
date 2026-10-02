#!/usr/bin/env python
"""Evaluate an exact iLQR swing-up route followed by a local upright tail.

The route is replayed from the canonical hanging reset without a state reset.
At the route boundary, the measured MuJoCo state is handed directly to either
zero control or a finite-difference LQR controller.  This keeps terminal
arrival quality separate from actual capture and hold evidence.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp

try:
    from scripts.evaluate_lqr_teacher import build_gain, coordinate_transform
    from scripts.search_swingup_ilqr_terminal import absolute_rate_transform
except ModuleNotFoundError:
    from evaluate_lqr_teacher import build_gain, coordinate_transform
    from search_swingup_ilqr_terminal import absolute_rate_transform


def exact_episode_config(
    base_cfg: dict[str, Any],
    *,
    seconds: float,
    state: dict[str, Any] | None = None,
    noise_angle: float = 0.0,
    noise_velocity: float = 0.0,
    noise_cart: float = 0.0,
    noise_cart_velocity: float = 0.0,
) -> dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    env_cfg = cfg["env"]
    env_cfg["episode_seconds"] = float(seconds)
    env_cfg["terminate_abs_angle"] = None
    env_cfg["init_angle_noise"] = float(noise_angle)
    env_cfg["init_angle_noise_start"] = float(noise_angle)
    env_cfg["init_angle_noise_end"] = float(noise_angle)
    env_cfg["init_vel_noise"] = float(noise_velocity)
    env_cfg["init_vel_noise_start"] = float(noise_velocity)
    env_cfg["init_vel_noise_end"] = float(noise_velocity)
    env_cfg["init_cart_noise"] = float(noise_cart)
    env_cfg["init_cart_noise_start"] = float(noise_cart)
    env_cfg["init_cart_noise_end"] = float(noise_cart)
    env_cfg["init_cart_vel_noise"] = float(noise_cart_velocity)
    env_cfg["init_cart_vel_noise_start"] = float(noise_cart_velocity)
    env_cfg["init_cart_vel_noise_end"] = float(noise_cart_velocity)
    env_cfg["action_lqr_residual"] = {"enabled": False}
    env_cfg["action_lqr_switch"] = {"enabled": False}
    if state is None:
        env_cfg["init_mode"] = "hanging"
    else:
        env_cfg["init_mode"] = "fixed_state"
        env_cfg["init_qpos"] = list(state["qpos"])
        env_cfg["init_qvel"] = list(state["qvel"])
    return cfg


def load_state(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if isinstance(states, list) and states and isinstance(states[0], dict):
        return dict(states[0])
    if isinstance(payload, dict) and isinstance(payload.get("selected_state"), dict):
        return dict(payload["selected_state"])
    raise ValueError(f"{path} does not contain a usable qpos/qvel state")


def load_route(path: str, n_links: int) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("route artifact must contain a controller object")
    controls = np.asarray(controller.get("controls", []), dtype=np.float64)
    nominal = np.asarray(controller.get("nominal_coordinate_states", []), dtype=np.float64)
    feedback = np.asarray(controller.get("feedback_gains", []), dtype=np.float64)
    state_dim = 2 * (int(n_links) + 1)
    if controls.ndim != 1 or controls.size < 1:
        raise ValueError("route controls must be a non-empty vector")
    if nominal.shape != (controls.size + 1, state_dim):
        raise ValueError(
            f"route nominal states must have shape {(controls.size + 1, state_dim)}; got {nominal.shape}"
        )
    if feedback.shape != (controls.size, state_dim):
        raise ValueError(
            f"route feedback gains must have shape {(controls.size, state_dim)}; got {feedback.shape}"
        )
    return {
        "payload": payload,
        "controls": controls,
        "nominal": nominal,
        "feedback": feedback,
        "path": str(Path(path)),
    }


def route_coordinate_state(env: NLinkCartPoleEnv, transform: np.ndarray) -> np.ndarray:
    qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    qpos[1:] = wrap_angle(qpos[1:])
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    return transform @ np.r_[qpos, qvel]


def tail_coordinate_state(
    env: NLinkCartPoleEnv,
    *,
    state_coordinates: str,
    transform: np.ndarray,
) -> np.ndarray:
    qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    qpos[1:] = wrap_angle(qpos[1:])
    state = np.r_[qpos, np.asarray(env.data.qvel, dtype=np.float64)]
    if state_coordinates == "relative":
        return state
    return transform @ state


def row(env: NLinkCartPoleEnv, *, step: int, action: float, mode: str, info: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "action": float(action),
        "mode": mode,
        "x": float(info.get("x", env.data.qpos[0])),
        "cart_velocity": float(env.data.qvel[0]),
        "max_abs_angle": float(info.get("max_abs_angle", np.nan)),
        "hinge_velocity_rms": float(info.get("hinge_velocity_rms", np.nan)),
        "absolute_angular_velocity_rms": float(info.get("absolute_angular_velocity_rms", np.nan)),
        "upright": bool(info.get("is_upright", False)),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
    }


def run_case(
    base_cfg: dict[str, Any],
    route: dict[str, Any] | None,
    *,
    route_path: str | None,
    start_state: dict[str, Any] | None,
    seconds: float,
    seed: int,
    state_coordinates: str,
    control_cost: float,
    gain_scale: float,
    route_feedback_scale: float,
    noise_angle: float,
    noise_velocity: float,
    noise_cart: float,
    noise_cart_velocity: float,
) -> dict[str, Any]:
    cfg = exact_episode_config(
        base_cfg,
        seconds=seconds,
        state=start_state,
        noise_angle=noise_angle,
        noise_velocity=noise_velocity,
        noise_cart=noise_cart,
        noise_cart_velocity=noise_cart_velocity,
    )
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
    env.reset(seed=seed)
    n_links = int(cfg["env"]["n_links"])
    tail_transform = coordinate_transform(cfg, progress=1.0, state_coordinates=state_coordinates)
    tail_gain, closed_loop_radius, open_loop_radius, _ = build_gain(
        cfg,
        progress=1.0,
        fd_eps=1e-7,
        control_cost=control_cost,
        cart_position_cost=0.1,
        absolute_angle_cost_value=100.0,
        cart_velocity_cost=0.1,
        absolute_angular_velocity_cost=1.0,
        relative_angle_cost=1.0,
        relative_angular_velocity_cost=0.01,
        state_coordinates=state_coordinates,
    )
    route_transform = absolute_rate_transform(
        n_links,
        cart_position=1.25,
        angle=0.15,
        cart_velocity=0.50,
        rate=0.75,
    )
    route_steps = 0 if route is None else int(route["controls"].size)
    tail_start_step = None
    rows: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    for step in range(env.max_steps):
        if route is not None and step < route_steps:
            action = float(route["controls"][step])
            if route_feedback_scale:
                current = route_coordinate_state(env, route_transform)
                action += float(route_feedback_scale) * float(
                    route["feedback"][step] @ (current - route["nominal"][step])
                )
            mode = "ilqr_route"
        else:
            if tail_start_step is None:
                tail_start_step = int(step)
            if control_cost < 0.0:
                action = 0.0
            else:
                tail_state = tail_coordinate_state(
                    env,
                    state_coordinates=state_coordinates,
                    transform=tail_transform,
                )
                action = -float(gain_scale) * float(tail_gain @ tail_state)
            mode = "zero_tail" if control_cost < 0.0 else f"{state_coordinates}_lqr_tail"
        action = float(np.clip(action, -1.0, 1.0))
        _, _, terminated, truncated, info = env.step([action])
        final_info = dict(info)
        rows.append(row(env, step=step + 1, action=action, mode=mode, info=info))
        if terminated or truncated:
            break
    result = {
        "success": bool(final_info.get("success", False)),
        "termination_reason": final_info.get("termination_reason"),
        "steps": int(len(rows)),
        "simulated_seconds": float(len(rows) * env.dt),
        "route_steps": int(route_steps),
        "tail_start_step": tail_start_step,
        "max_cart_excursion": float(final_info.get("max_cart_excursion", 0.0)),
        "max_abs_angle": float(final_info.get("max_abs_angle", np.nan)),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(
            final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "closed_loop_radius": float(closed_loop_radius),
        "open_loop_radius": float(open_loop_radius),
        "gain_max_abs": float(np.max(np.abs(tail_gain * gain_scale))),
        "final_info": final_info,
    }
    env.close()
    return {
        "state_coordinates": state_coordinates,
        "control_cost": float(control_cost),
        "gain_scale": float(gain_scale),
        "route_feedback_scale": float(route_feedback_scale),
        "seed": int(seed),
        "initial_state_source": start_state is not None,
        "noise": {
            "angle": float(noise_angle),
            "velocity": float(noise_velocity),
            "cart": float(noise_cart),
            "cart_velocity": float(noise_cart_velocity),
        },
        "result": result,
        "trajectory": rows,
        "config_sha256": data_sha256(cfg),
        "route": route_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--route", default=None)
    parser.add_argument("--start-state", default=None)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-coordinates", choices=["relative", "absolute", "modal"], default="relative")
    parser.add_argument("--control-cost", type=float, action="append", default=[])
    parser.add_argument("--gain-scale", type=float, action="append", default=[])
    parser.add_argument("--route-feedback-scale", type=float, default=0.0)
    parser.add_argument("--noise-angle", type=float, default=0.0)
    parser.add_argument("--noise-velocity", type=float, default=0.0)
    parser.add_argument("--noise-cart", type=float, default=0.0)
    parser.add_argument("--noise-cart-velocity", type=float, default=0.0)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.seconds <= 0.0:
        raise ValueError("seconds must be positive")
    if any(value < 0.0 for value in args.gain_scale):
        raise ValueError("gain scales must be nonnegative")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    n_links = int(base_cfg["env"]["n_links"])
    route = None if args.route is None else load_route(args.route, n_links)
    start_state = None if args.start_state is None else load_state(args.start_state)
    control_costs = list(args.control_cost) or [-1.0]
    gain_scales = list(args.gain_scale) or [1.0]
    cases = [
        run_case(
            base_cfg,
            route,
            route_path=args.route,
            start_state=start_state,
            seconds=args.seconds,
            seed=args.seed,
            state_coordinates=args.state_coordinates,
            control_cost=float(control_cost),
            gain_scale=float(gain_scale),
            route_feedback_scale=args.route_feedback_scale,
            noise_angle=args.noise_angle,
            noise_velocity=args.noise_velocity,
            noise_cart=args.noise_cart,
            noise_cart_velocity=args.noise_cart_velocity,
        )
        for control_cost in control_costs
        for gain_scale in gain_scales
    ]
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "diagnostic_not_solution_evidence",
        "not_solution": True,
        "summary": "Reset-free exact-MuJoCo iLQR route followed by a local tail; capture and hold must be independently gated.",
        "config": str(Path(args.config)),
        "route": None if args.route is None else str(Path(args.route)),
        "start_state": None if args.start_state is None else str(Path(args.start_state)),
        "overrides": list(args.override),
        "cases": cases,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, Path(args.out))
    for case in cases:
        result = case["result"]
        print(
            f"coords={case['state_coordinates']} cost={case['control_cost']:g} scale={case['gain_scale']:g} "
            f"success={result['success']} steps={result['steps']} "
            f"streak={result['max_upright_streak_seconds']:.3f}s "
            f"rail={result['max_cart_excursion']:.3f} "
            f"reason={result['termination_reason']}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
