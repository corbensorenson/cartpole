#!/usr/bin/env python
"""Sweep local linear capture teachers from exact saved seven-link states.

This is a diagnostic search for a usable downstream capture teacher.  It keeps
the MuJoCo plant, rail, action cadence, and saved state fixed while varying
the discrete-LQR control cost, angle/rate priorities, action scale, and final
action squash.  A candidate only counts as interesting if exact serial replay
holds the upright condition; a small terminal angle alone is not promoted.
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
from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def load_state(path: str, index: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(states, list) or not states:
        raise ValueError(f"{path} does not contain a non-empty states list")
    state = states[int(index)]
    qpos = np.asarray(state.get("qpos", []), dtype=np.float64)
    qvel = np.asarray(state.get("qvel", []), dtype=np.float64)
    return qpos, qvel, {"state_index": int(index), "source": file_metadata(path), "source_state": state}


def make_gain(
    a: np.ndarray,
    b: np.ndarray,
    n_links: int,
    *,
    control_cost: float,
    angle_cost: float,
    rate_cost: float,
    feedback_space: str,
) -> np.ndarray:
    q = absolute_angle_cost(
        n_links,
        {
            "cart_position": 0.1,
            "absolute_angle": float(angle_cost),
            "cart_velocity": 0.1,
            "absolute_angular_velocity": float(rate_cost),
            "relative_angle": 1.0 if feedback_space == "relative" else 0.0,
            "relative_angular_velocity": 0.01 if feedback_space == "relative" else 0.0,
        },
    )
    r = np.array([[float(control_cost)]], dtype=np.float64)
    p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(b.T @ p @ b + r, b.T @ p @ a).reshape(-1)
    if feedback_space == "absolute":
        d = n_links + 1
        relative_from_absolute = np.zeros((2 * d, 2 * d), dtype=np.float64)
        relative_from_absolute[0, 0] = 1.0
        relative_from_absolute[d, d] = 1.0
        for offset in (1, d + 1):
            for index in range(n_links):
                relative_from_absolute[offset + index, offset + index] = 1.0
                if index > 0:
                    relative_from_absolute[offset + index, offset + index - 1] = -1.0
        gain = gain @ relative_from_absolute
    return gain


def run_candidate(
    cfg: dict[str, Any],
    qpos: np.ndarray,
    qvel: np.ndarray,
    gain: np.ndarray,
    *,
    seconds: float,
    scale: float,
    squash: str,
    slew: float,
    feedback_space: str,
) -> dict[str, Any]:
    env_cfg = {
        **cfg["env"],
        "init_mode": "fixed_state",
        "init_qpos": qpos.astype(float).tolist(),
        "init_qvel": qvel.astype(float).tolist(),
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
        env_cfg[f"{key}_start"] = 0.0
        env_cfg[f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv({**cfg, "env": env_cfg}, progress=1.0, seed=0)
    env.reset(seed=0)
    max_streak = 0.0
    best_angle = float("inf")
    best_cost = float("inf")
    max_cart = abs(float(env.data.qpos[0]))
    action_sq = 0.0
    previous_action = 0.0
    simulated_steps = 0
    final_info: dict[str, Any] = {}
    steps = min(env.max_steps, int(round(seconds / env.dt)))
    for _ in range(steps):
        qnow = np.asarray(env.data.qpos, dtype=np.float64)
        vnow = np.asarray(env.data.qvel, dtype=np.float64)
        if feedback_space == "absolute":
            state = np.r_[
                qnow[0],
                serial_absolute_angles(qnow[1:]),
                vnow[0],
                np.cumsum(vnow[1:]),
            ]
        else:
            state = np.r_[qnow[0], wrap_angle(qnow[1:]), vnow]
        raw = -float(scale) * float(gain @ state)
        requested = float(np.tanh(raw) if squash == "tanh" else np.clip(raw, -1.0, 1.0))
        action = float(np.clip(requested, previous_action - slew, previous_action + slew))
        _, _, terminated, truncated, info = env.step([action])
        previous_action = action
        simulated_steps += 1
        max_streak = max(max_streak, float(info.get("max_upright_streak_seconds", 0.0)))
        angle = float(info.get("max_abs_angle", np.inf))
        hinge = float(info.get("hinge_velocity_rms", np.inf))
        absolute_rate = float(info.get("absolute_angular_velocity_rms", np.inf))
        x = abs(float(info.get("x", np.inf)))
        xdot = abs(float(env.data.qvel[0]))
        best_angle = min(best_angle, angle)
        local = (
            (angle / 0.15) ** 2
            + 0.25 * (hinge / 0.75) ** 2
            + 0.50 * (absolute_rate / 0.75) ** 2
            + 0.10 * (x / 1.25) ** 2
            + 0.10 * (xdot / 0.50) ** 2
        )
        best_cost = min(best_cost, local)
        max_cart = max(max_cart, x)
        action_sq += action * action
        final_info = dict(info)
        if terminated or truncated:
            break
    result = {
        "control_scale": float(scale),
        "squash": squash,
        "action_slew_limit": float(slew),
        "feedback_space": feedback_space,
        "success": bool(final_info.get("success", False)),
        "max_upright_streak_seconds": float(max_streak),
        "best_angle": float(best_angle),
        "best_cost": float(best_cost),
        "max_cart_excursion": float(max_cart),
        "action_rms": float(np.sqrt(action_sq / max(1, simulated_steps))),
        "simulated_seconds": float(simulated_steps) * env.dt,
        "termination_reason": final_info.get("termination_reason"),
        "final_info": final_info,
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep exact seven-link local LQR capture teachers")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--control-costs", default="30,100,300,1000,3000,10000,30000")
    parser.add_argument("--angle-costs", default="30,100,300,1000")
    parser.add_argument("--rate-costs", default="0.1,1,10,100")
    parser.add_argument("--scales", default="0.05,0.1,0.2,0.4,0.7,1,1.5,2")
    parser.add_argument("--squashes", default="clip,tanh")
    parser.add_argument("--slews", default="0.005,0.01,0.02,0.05,0.1,0.2,1.0")
    parser.add_argument("--feedback-spaces", default="relative,absolute")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "rail_limit_start": float(cfg["env"].get("rail_limit", 3.0)),
        "rail_limit_end": float(cfg["env"].get("rail_limit", 3.0)),
        "plant_progress": 1.0,
    }
    qpos, qvel, state_metadata = load_state(args.state_json, args.state_index)
    n_links = int(cfg["env"]["n_links"])
    a, b = finite_difference_dynamics(cfg, 1.0, 1e-7)
    control_costs = parse_floats(args.control_costs)
    angle_costs = parse_floats(args.angle_costs)
    rate_costs = parse_floats(args.rate_costs)
    scales = parse_floats(args.scales)
    slews = parse_floats(args.slews)
    feedback_spaces = [item.strip() for item in args.feedback_spaces.split(",") if item.strip()]
    squashes = [item.strip() for item in args.squashes.split(",") if item.strip()]
    records: list[dict[str, Any]] = []
    gains: dict[tuple[float, float, float], np.ndarray] = {}
    started = time.time()
    for control_cost in control_costs:
        for angle_cost in angle_costs:
            for rate_cost in rate_costs:
                for feedback_space in feedback_spaces:
                    key = (control_cost, angle_cost, rate_cost, feedback_space)
                    gains[key] = make_gain(
                        a,
                        b,
                        n_links,
                        control_cost=control_cost,
                        angle_cost=angle_cost,
                        rate_cost=rate_cost,
                        feedback_space=feedback_space,
                    )
                    for scale in scales:
                        for squash in squashes:
                            for slew in slews:
                                result = run_candidate(
                                    cfg,
                                    qpos,
                                    qvel,
                                    gains[key],
                                    seconds=args.seconds,
                                    scale=scale,
                                    squash=squash,
                                    slew=slew,
                                    feedback_space=feedback_space,
                                )
                                records.append(
                                    {
                                        "control_cost": float(control_cost),
                                        "angle_cost": float(angle_cost),
                                        "rate_cost": float(rate_cost),
                                        **result,
                                    }
                                )
    records.sort(
        key=lambda row: (
            -int(row["success"]),
            -float(row["max_upright_streak_seconds"]),
            float(row["best_cost"]),
            float(row["max_cart_excursion"]),
        )
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact serial seven-link local LQR teacher sweep from a saved state; capture diagnostic only.",
        "config_path": str(Path(args.config)),
        "config_sha256": data_sha256(cfg),
        "state": state_metadata,
        "sweep": {
            "seconds": float(args.seconds),
            "control_costs": control_costs,
            "angle_costs": angle_costs,
            "rate_costs": rate_costs,
            "scales": scales,
            "squashes": squashes,
            "slews": slews,
            "feedback_spaces": feedback_spaces,
            "candidate_count": len(records),
            "wall_time_seconds": float(time.time() - started),
        },
        "top_records": records[: max(1, int(args.top_k))],
        "records": records,
        "git": git_metadata(Path(__file__).resolve().parents[1]),
        "runtime": runtime_metadata(),
    }
    dump_json(payload, Path(args.out))
    top = records[0]
    print(
        f"candidates={len(records)} best_hold={top['max_upright_streak_seconds']:.3f}s "
        f"best_angle={top['best_angle']:.6f} best_cost={top['best_cost']:.3f} "
        f"rail={top['max_cart_excursion']:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
