#!/usr/bin/env python
"""Evaluate a feedback swing route after parking the hanging cart.

This is the generalized eight-link development evaluator.  It keeps the
three phases explicit: settle the hanging chain at a translated cart target,
run the saved Box-FDDP route with state feedback, and hand off to the exact
upright LQR around that same cart target.

The route is translated only in the cart-position coordinate.  The dynamics
are translationally invariant, so this gives the swing room required by the
rail without changing the link trajectory.  The artifact remains a
development result until the repository's canonical noisy gates and
reproduction checks are promoted.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    text_sha256,
    utc_timestamp,
)
from gcartpole.generalized_energy import hanging_lqr_gain
from gcartpole.ilqr import data_state
from gcartpole.modal import dimensionless_wrapped_state

try:
    from scripts.evaluate_fddp_two_expert import load_controller
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from evaluate_fddp_two_expert import load_controller
    from search_swingup_capture import lqr_action, lqr_gain


def zero_noise_config(cfg: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        result["env"][key] = 0.0
        result["env"][f"{key}_start"] = 0.0
        result["env"][f"{key}_end"] = 0.0
    return result


def hanging_target_action(
    env: NLinkCartPoleEnv, gain: np.ndarray, cart_target: float
) -> float:
    state = data_state(env.data)
    state[0] = float(env.data.qpos[0]) - float(cart_target)
    state[1] = wrap_angle(float(env.data.qpos[1]) - np.pi)
    if env.n > 1:
        state[2 : env.n + 1] = wrap_angle(
            np.asarray(env.data.qpos[2 : env.n + 1], dtype=np.float64)
        )
    return float(np.clip(-np.asarray(gain, dtype=np.float64) @ state, -1.0, 1.0))


def run_episode(
    cfg: dict[str, Any],
    controller: dict[str, Any],
    capture_gain: np.ndarray,
    settle_gain: np.ndarray,
    *,
    seed: int,
    park_seconds: float,
    cart_target: float,
    tracking_gain_scale: float,
    phase_adaptive: bool,
    phase_window: int,
    include_trace: bool,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
    _, reset_info = env.reset(seed=seed)

    controls = controller["controls"]
    nominal_states = controller["nominal_states"]
    feedback_gains = controller["feedback_gains"]
    transform = controller["transform"]
    translated_nominal_states = nominal_states.copy()
    park_steps = round(float(park_seconds) / env.dt)
    route_steps = int(controls.size)
    cart_nominal_shift: float | None = None
    phase_cursor = 0
    max_cart = abs(float(reset_info.get("x", env.data.qpos[0])))
    first_upright: float | None = None
    park_state: dict[str, Any] | None = None
    route_state: dict[str, Any] | None = None
    final_info: dict[str, Any] = dict(reset_info)
    episode_return = 0.0
    trajectory: list[dict[str, Any]] = []

    for step in range(env.max_steps):
        if step < park_steps:
            phase = "park_hanging"
            route_index: int | None = None
            action = hanging_target_action(env, settle_gain, cart_target)
        else:
            route_elapsed = step - park_steps
            route_index = route_elapsed
            coordinate_state = dimensionless_wrapped_state(
                np.asarray(env.data.qpos, dtype=np.float64),
                np.asarray(env.data.qvel, dtype=np.float64),
                transform,
            )
            if cart_nominal_shift is None:
                cart_nominal_shift = float(
                    coordinate_state[0] - nominal_states[0, 0]
                )
                translated_nominal_states[:, 0] += cart_nominal_shift

            in_route = route_elapsed < route_steps
            if in_route and phase_adaptive:
                if phase_cursor >= route_steps:
                    in_route = False
                else:
                    candidates = np.arange(
                        phase_cursor,
                        min(route_steps, phase_cursor + phase_window + 1),
                        dtype=np.int64,
                    )
                    errors = translated_nominal_states[candidates] - coordinate_state
                    route_index = int(
                        candidates[int(np.argmin(np.einsum("ij,ij->i", errors, errors)))]
                    )
                    phase_cursor = route_index + 1

            if in_route:
                phase = "swing_route_feedback"
                action = float(
                    np.clip(
                        controls[route_index]
                        + tracking_gain_scale
                        * feedback_gains[route_index]
                        @ (coordinate_state - translated_nominal_states[route_index]),
                        -1.0,
                        1.0,
                    )
                )
            else:
                phase = "capture_lqr"
                action = lqr_action(
                    env,
                    capture_gain,
                    scale=controller["lqr_scale"],
                    cart_target=cart_target,
                )

        _, reward, terminated, truncated, info = env.step([action])
        final_info = dict(info)
        episode_return += float(reward)
        max_cart = max(max_cart, abs(float(info.get("x", env.data.qpos[0]))))
        if first_upright is None and bool(info.get("is_upright", False)):
            first_upright = float((step + 1) * env.dt)

        row = {
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "phase": phase,
            "route_index": route_index,
            "action": float(action),
            "reward": float(reward),
            "x": float(info["x"]),
            "max_abs_angle": float(info["max_abs_angle"]),
            "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
            "absolute_angular_velocity_rms": float(
                info["absolute_angular_velocity_rms"]
            ),
            "is_upright": bool(info["is_upright"]),
            "max_upright_streak_seconds": float(
                info.get("max_upright_streak_seconds", 0.0)
            ),
            "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        }
        if step == park_steps - 1:
            park_state = dict(row)
        if step == park_steps + route_steps - 1:
            route_state = dict(row)
        if include_trace:
            trajectory.append(row)
        if terminated or truncated:
            break

    env.close()
    return {
        "seed": int(seed),
        "success": bool(final_info.get("success", False)),
        "termination_reason": final_info.get("termination_reason"),
        "return": float(episode_return),
        "park_seconds": float(park_steps * env.dt),
        "route_seconds": float(route_steps * env.dt),
        "first_upright_time": first_upright,
        "max_upright_streak_seconds": float(
            final_info.get("max_upright_streak_seconds", 0.0)
        ),
        "max_low_momentum_upright_streak_seconds": float(
            final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(max_cart),
        "park_state": park_state,
        "route_state": route_state,
        "final_info": final_info,
        "trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup8_uniform.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--controller", default="runs/generalized_solver/n8_capture_fddp_feedback120.json"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=80801)
    parser.add_argument("--park-seconds", type=float, default=17.5)
    parser.add_argument("--cart-target", type=float, default=-0.10)
    parser.add_argument("--tracking-gain-scale", type=float, default=1.0)
    parser.add_argument("--phase-adaptive", action="store_true")
    parser.add_argument("--phase-window", type=int, default=12)
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--include-traces", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.episodes < 1 or args.park_seconds < 0.0:
        raise ValueError("episodes must be positive and park duration nonnegative")
    if args.cart_target >= 0.0:
        raise ValueError("cart target must be negative for the current rail-safe route")
    if args.tracking_gain_scale < 0.0 or args.phase_window < 0:
        raise ValueError("tracking gain and phase window must be nonnegative")

    source_cfg = load_config(args.config)
    cfg = copy.deepcopy(source_cfg)
    cfg["env"]["init_mode"] = "hanging"
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"].setdefault("action_lqr_switch", {"enabled": False})["enabled"] = False
    if args.zero_noise:
        cfg = zero_noise_config(cfg)

    spec = load_config(args.spec)
    controller_path = Path(args.controller)
    controller = load_controller(controller_path, int(cfg["env"]["n_links"]), spec)
    capture_gain = lqr_gain(
        cfg,
        progress=1.0,
        fd_eps=1.0e-7,
        control_cost=controller["lqr_control_cost"],
        q_weights=controller["lqr_weights"],
    )
    probe = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    settle_gain = hanging_lqr_gain(probe, control_cost=1000.0)
    generated_xml_sha256 = text_sha256(probe.xml)
    observation_dim = int(probe.observation_space.shape[0])
    action_dim = int(probe.action_space.shape[0])
    action_frequency_hz = float(1.0 / probe.dt)
    probe.close()

    episodes = [
        run_episode(
            cfg,
            controller,
            capture_gain,
            settle_gain,
            seed=args.seed + index,
            park_seconds=args.park_seconds,
            cart_target=args.cart_target,
            tracking_gain_scale=args.tracking_gain_scale,
            phase_adaptive=args.phase_adaptive,
            phase_window=args.phase_window,
            include_trace=args.include_traces and index == 0,
        )
        for index in range(args.episodes)
    ]
    successes = sum(bool(row["success"]) for row in episodes)
    source_git = {
        key: value
        for key, value in git_metadata(Path(__file__).resolve().parents[1]).items()
        if key != "root"
    }
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_eight_link_fddp_parked_route",
        "not_solution": True,
        "summary": (
            "Eight-link parked-cart launch with Box-FDDP feedback and "
            "parked-target upright LQR capture."
        ),
        "config": file_metadata(Path(args.config)),
        "spec": file_metadata(Path(args.spec)),
        "controller": controller["source"],
        "resolved_config_sha256": data_sha256(cfg),
        "controller_sha256": controller["source"]["sha256"],
        "generated_xml_sha256": generated_xml_sha256,
        "n_links": int(cfg["env"]["n_links"]),
        "episodes": int(args.episodes),
        "seed_start": int(args.seed),
        "zero_noise": bool(args.zero_noise),
        "park_seconds": float(args.park_seconds),
        "cart_target": float(args.cart_target),
        "tracking_gain_scale": float(args.tracking_gain_scale),
        "phase_adaptive": bool(args.phase_adaptive),
        "phase_window": int(args.phase_window),
        "route_steps": int(controller["horizon_steps"]),
        "route_seconds": float(controller["horizon_seconds"]),
        "successes": int(successes),
        "success_rate": float(successes / args.episodes),
        "ever_upright_rate": float(
            np.mean([row["first_upright_time"] is not None for row in episodes])
        ),
        "max_upright_streak_mean": float(
            np.mean([row["max_upright_streak_seconds"] for row in episodes])
        ),
        "max_upright_streak_max": float(
            np.max([row["max_upright_streak_seconds"] for row in episodes])
        ),
        "max_cart_excursion_max": float(
            np.max([row["max_cart_excursion"] for row in episodes])
        ),
        "termination_counts": dict(
            Counter(str(row["termination_reason"]) for row in episodes)
        ),
        "environment": {
            "n_links": int(cfg["env"]["n_links"]),
            "init_mode": str(cfg["env"]["init_mode"]),
            "force_limit": float(cfg["env"]["force_limit"]),
            "rail_limit": float(cfg["env"]["rail_limit"]),
            "observation_dim": observation_dim,
            "action_dim": action_dim,
            "action_frequency_hz": action_frequency_hz,
            "episode_seconds": float(cfg["env"]["episode_seconds"]),
            "init_angle_noise": float(cfg["env"]["init_angle_noise"]),
            "init_velocity_noise": float(cfg["env"]["init_vel_noise"]),
        },
        "runtime": runtime_metadata(),
        "git": source_git,
        "episode_results": episodes,
    }
    dump_json(output, args.out)
    print(
        f"episodes={args.episodes} success={successes}/{args.episodes} "
        f"hold_max={output['max_upright_streak_max']:.3f} "
        f"max_cart={output['max_cart_excursion_max']:.4f} "
        f"zero_noise={args.zero_noise} phase_adaptive={args.phase_adaptive}"
    )


if __name__ == "__main__":
    main()
