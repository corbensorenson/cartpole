#!/usr/bin/env python
"""Evaluate the eight-link route with a settled cart-parking prelude.

The controller has three explicit phases: park the cart while the chain is
hanging, replay the saved swing route, then use the exact upright LQR with the
parked cart position as its temporary cart target.  This is development
evidence until the noisy multi-episode result and fresh-clone checks pass.
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
    utc_timestamp,
)
from gcartpole.generalized_energy import hanging_lqr_gain
from gcartpole.ilqr import data_state

try:
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from search_swingup_capture import lqr_action, lqr_gain


def load_route(path: Path) -> tuple[np.ndarray, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload.get("best", payload)
    if not isinstance(record, dict) or "controls" not in record:
        raise ValueError(f"{path} has no saved route controls")
    controls = np.asarray(record["controls"], dtype=np.float64)
    seconds = float(record.get("horizon_seconds", 0.0))
    if controls.ndim != 1 or controls.size < 2 or seconds <= 0.0:
        raise ValueError(f"{path} has an invalid route")
    return np.clip(controls, -1.0, 1.0), seconds


def target_hanging_action(
    env: NLinkCartPoleEnv, gain: np.ndarray, cart_target: float
) -> float:
    state = data_state(env.data)
    state[0] = float(env.data.qpos[0]) - float(cart_target)
    state[1] = wrap_angle(float(env.data.qpos[1]) - np.pi)
    if env.n > 1:
        state[2 : env.n + 1] = wrap_angle(
            np.asarray(env.data.qpos[2 : env.n + 1], dtype=np.float64)
        )
    return float(np.clip(-gain @ state, -1.0, 1.0))


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


def run_episode(
    cfg: dict[str, Any],
    route: np.ndarray,
    *,
    seed: int,
    park_seconds: float,
    cart_target: float,
    include_trace: bool,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
    _, reset_info = env.reset(seed=seed)
    settle_gain = hanging_lqr_gain(env, control_cost=1000.0)
    capture_gain = lqr_gain(
        cfg, progress=1.0, fd_eps=1.0e-7, control_cost=1000.0
    )
    park_steps = round(float(park_seconds) / env.dt)
    route_steps = int(route.size)
    trace: list[dict[str, Any]] = []
    max_cart = abs(float(env.data.qpos[0]))
    park_state: dict[str, Any] | None = None
    route_state: dict[str, Any] | None = None
    final_info = dict(reset_info)

    for step in range(env.max_steps):
        if step < park_steps:
            phase = "park_hanging"
            action = target_hanging_action(env, settle_gain, cart_target)
        elif step < park_steps + route_steps:
            phase = "swing_route"
            action = float(route[step - park_steps])
        else:
            phase = "capture_lqr"
            action = lqr_action(
                env, capture_gain, scale=1.0, cart_target=cart_target
            )

        _, reward, terminated, truncated, info = env.step([action])
        final_info = dict(info)
        max_cart = max(max_cart, abs(float(env.data.qpos[0])))
        row = {
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "phase": phase,
            "action": float(action),
            "reward": float(reward),
            "x": float(env.data.qpos[0]),
            "qpos": np.asarray(env.data.qpos).astype(float).tolist(),
            "qvel": np.asarray(env.data.qvel).astype(float).tolist(),
            "max_abs_angle": float(info["max_abs_angle"]),
            "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
            "absolute_angular_velocity_rms": float(
                info["absolute_angular_velocity_rms"]
            ),
            "is_upright": bool(info["is_upright"]),
            "max_upright_streak_seconds": float(
                info.get("max_upright_streak_seconds", 0.0)
            ),
        }
        if step == park_steps - 1:
            park_state = dict(row)
        if step == park_steps + route_steps - 1:
            route_state = dict(row)
        if include_trace:
            trace.append(row)
        if terminated or truncated:
            break

    env.close()
    return {
        "seed": int(seed),
        "success": bool(final_info.get("success", False)),
        "termination_reason": final_info.get("termination_reason"),
        "park_seconds": float(park_steps * env.dt),
        "route_seconds": float(route_steps * env.dt),
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
        "trace": trace,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup8_uniform.yaml")
    parser.add_argument(
        "--route", default="runs/generalized_solver/n8_gn_capture_preimage10.json"
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=80801)
    parser.add_argument("--park-seconds", type=float, default=17.5)
    parser.add_argument("--cart-target", type=float, default=-0.10)
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--include-traces", action="store_true")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.episodes, args.park_seconds) < 1:
        raise ValueError("episodes and park duration must be positive")
    if args.cart_target >= 0.0:
        raise ValueError("cart target must be negative for this left-park route")

    cfg = load_config(args.config)
    if args.zero_noise:
        cfg = zero_noise_config(cfg)
    route_path = Path(args.route)
    route, route_seconds = load_route(route_path)
    route_dt = route_seconds / route.size
    target_dt = float(cfg["env"]["timestep"] * cfg["env"]["frame_skip"])
    if not np.isclose(route_dt, target_dt, atol=1.0e-10, rtol=0.0):
        raise ValueError("route period does not match configured policy period")

    episodes = [
        run_episode(
            cfg,
            route,
            seed=args.seed + index,
            park_seconds=args.park_seconds,
            cart_target=args.cart_target,
            include_trace=args.include_traces and index == 0,
        )
        for index in range(args.episodes)
    ]
    successes = sum(bool(row["success"]) for row in episodes)
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_eight_link_parked_route",
        "not_solution": True,
        "summary": "Three-phase eight-link route: settled cart park, exact swing route, parked-target upright LQR.",
        "config": file_metadata(Path(args.config)),
        "route": file_metadata(route_path),
        "resolved_config_sha256": data_sha256(cfg),
        "route_sha256": data_sha256(route.tolist()),
        "n_links": int(cfg["env"]["n_links"]),
        "episodes": int(args.episodes),
        "seed_start": int(args.seed),
        "zero_noise": bool(args.zero_noise),
        "park_seconds": float(args.park_seconds),
        "cart_target": float(args.cart_target),
        "route_seconds": float(route_seconds),
        "successes": int(successes),
        "success_rate": float(successes / args.episodes),
        "termination_counts": dict(
            Counter(str(row["termination_reason"]) for row in episodes)
        ),
        "max_cart_excursion": float(
            max(row["max_cart_excursion"] for row in episodes)
        ),
        "episode_results": episodes,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, args.out)
    print(
        f"episodes={args.episodes} success={successes}/{args.episodes} "
        f"max_cart={output['max_cart_excursion']:.4f} "
        f"zero_noise={args.zero_noise}"
    )


if __name__ == "__main__":
    main()
