#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from probe_swingup_trajectory import DEFAULT_KD, DEFAULT_KNOTS, DEFAULT_KP, DEFAULT_TRAJECTORY_SECONDS, trajectory_action


def load_controller(path: str | None) -> dict[str, Any]:
    if not path:
        return {
            "type": "cart_position_pd_fixed_knots",
            "trajectory_seconds": float(DEFAULT_TRAJECTORY_SECONDS),
            "kp": float(DEFAULT_KP),
            "kd": float(DEFAULT_KD),
            "knots": DEFAULT_KNOTS.astype(float).tolist(),
        }
    with open(Path(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "best" in payload:
        return dict(payload["best"]["controller"])
    if isinstance(payload, dict) and "best_by" in payload and "score" in payload["best_by"]:
        return dict(payload["best_by"]["score"]["controller"])
    if isinstance(payload, dict) and "controller" in payload:
        return dict(payload["controller"])
    if isinstance(payload, dict) and "knots" in payload:
        return dict(payload)
    raise ValueError(f"Could not load controller from {path}")


def export_states(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    seconds: float,
    zero_noise: bool,
    max_angle: float,
    min_time: float,
    max_time: float | None,
    max_hinge_rms: float | None,
    max_cart_abs: float | None,
    max_cart_velocity: float | None,
    stride: int,
    keep_best: int | None,
    controller: dict[str, Any],
) -> dict[str, Any]:
    cfg = {**cfg, "env": {**cfg["env"]}}
    if zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0

    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    _, reset_info = env.reset()
    max_time = seconds if max_time is None else max_time
    steps = min(env.max_steps, int(seconds / env.dt))
    knots = np.asarray(controller["knots"], dtype=np.float64)
    trajectory_seconds = float(controller["trajectory_seconds"])
    kp = float(controller["kp"])
    kd = float(controller["kd"])

    states: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    max_cart_abs_filter = max_cart_abs
    max_cart_abs_seen = abs(float(reset_info["x"]))
    for step in range(steps):
        t = step * env.dt
        action = trajectory_action(env, t, knots, trajectory_seconds, kp, kd)
        _, reward, terminated, truncated, info = env.step([action])
        rel, abs_angles = env._angles()
        hinge_rms = float(np.sqrt(np.mean(env.data.qvel[1 : 1 + env.n] ** 2)))
        max_cart_abs_seen = max(max_cart_abs_seen, abs(float(info["x"])))
        row = {
            "source": "swingup_cart_position_pd",
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "reward": float(reward),
            "action": float(action),
            "qpos": np.array(env.data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.array(env.data.qvel, dtype=np.float64).astype(float).tolist(),
            "x": float(info["x"]),
            "cart_velocity": float(env.data.qvel[0]),
            "max_abs_angle": float(info["max_abs_angle"]),
            "mean_abs_angle": float(info["mean_abs_angle"]),
            "hinge_velocity_rms": hinge_rms,
            "relative_angles": rel.astype(float).tolist(),
            "absolute_angles": abs_angles.astype(float).tolist(),
            "is_upright": bool(info["is_upright"]),
            "upright_streak_seconds": float(info["upright_streak_seconds"]),
            "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
            "time_to_first_upright": info["time_to_first_upright"],
        }
        if best_state is None or row["max_abs_angle"] < best_state["max_abs_angle"]:
            best_state = row
        if (
            row["time_seconds"] >= min_time
            and row["time_seconds"] <= max_time
            and row["max_abs_angle"] <= max_angle
            and (max_hinge_rms is None or row["hinge_velocity_rms"] <= max_hinge_rms)
            and (max_cart_abs_filter is None or abs(row["x"]) <= max_cart_abs_filter)
            and (max_cart_velocity is None or abs(row["cart_velocity"]) <= max_cart_velocity)
            and int(step + 1) % max(1, stride) == 0
        ):
            states.append(row)
        if terminated or truncated:
            break

    env.close()
    if keep_best is not None:
        states = sorted(
            states,
            key=lambda item: (
                float(item["max_abs_angle"]),
                0.15 * float(item["hinge_velocity_rms"]),
                0.10 * abs(float(item["x"])),
                0.05 * abs(float(item["cart_velocity"])),
            ),
        )[:keep_best]
        states = sorted(states, key=lambda item: int(item["step"]))
    return {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Replayable swing expert states for training chained capture experts; not final benchmark evidence.",
        "seed": int(seed),
        "progress": float(progress),
        "zero_noise": bool(zero_noise),
        "selection": {
            "max_angle": float(max_angle),
            "min_time": float(min_time),
            "max_time": float(max_time),
            "max_hinge_rms": None if max_hinge_rms is None else float(max_hinge_rms),
            "max_cart_abs": None if max_cart_abs_filter is None else float(max_cart_abs_filter),
            "max_cart_velocity": None if max_cart_velocity is None else float(max_cart_velocity),
            "stride": int(stride),
            "keep_best": keep_best,
        },
        "state_count": int(len(states)),
        "states": states,
        "best_state": best_state,
        "max_cart_abs": float(max_cart_abs_seen),
        "controller": {
            "type": str(controller.get("type", "cart_position_pd_fixed_knots")),
            "trajectory_seconds": float(trajectory_seconds),
            "kp": float(kp),
            "kd": float(kd),
            "knots": knots.astype(float).tolist(),
        },
        "environment": {
            "n_links": int(cfg["env"]["n_links"]),
            "init_mode": str(cfg["env"].get("init_mode", "upright")),
            "episode_seconds": float(cfg["env"]["episode_seconds"]),
            "force_limit": float(cfg["env"]["force_limit"]),
            "rail_limit": float(cfg["env"]["rail_limit"]),
            "success_upright_threshold": float(
                cfg["env"].get("success_upright_threshold", cfg["env"].get("reward", {}).get("upright_threshold", 0.10))
            ),
            "success_sustain_seconds": float(cfg["env"].get("success_sustain_seconds", 0.0)),
        },
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export replayable states from the six-link hanging-start swing expert")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default="runs/swingup6_expert_chain/swing_states.json")
    parser.add_argument("--zero-noise", action="store_true", help="Export exact hanging trajectory states by overriding init noise to zero")
    parser.add_argument("--controller-json", default=None)
    parser.add_argument("--max-angle", type=float, default=0.70)
    parser.add_argument("--min-time", type=float, default=1.0)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--max-hinge-rms", type=float, default=4.0)
    parser.add_argument("--max-cart-abs", type=float, default=2.75)
    parser.add_argument("--max-cart-velocity", type=float, default=1.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--keep-best", type=int, default=64)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    controller = load_controller(args.controller_json)
    result = export_states(
        cfg,
        progress=args.progress,
        seed=args.seed,
        seconds=args.seconds,
        zero_noise=args.zero_noise,
        max_angle=args.max_angle,
        min_time=args.min_time,
        max_time=args.max_time,
        max_hinge_rms=args.max_hinge_rms,
        max_cart_abs=args.max_cart_abs,
        max_cart_velocity=args.max_cart_velocity,
        stride=args.stride,
        keep_best=args.keep_best,
        controller=controller,
    )
    dump_json(result, Path(args.out))
    best = result.get("best_state") or {}
    print(
        f"states={result['state_count']} "
        f"best_angle={best.get('max_abs_angle', float('nan')):.6f} "
        f"best_time={best.get('time_seconds', float('nan')):.3f}s"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
