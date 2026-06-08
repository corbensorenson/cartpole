#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv


# Rail-safe cart-position trajectory found by direct CEM probing. It reaches
# the upright threshold from the exact hanging start at much lower hinge speed
# than the first probe, but it is still a starting point for capture work, not
# a solved swing-up controller.
DEFAULT_KNOTS = np.asarray(
    [
        0.0,
        1.3477824329524968,
        2.6,
        -1.8797531264904297,
        -0.7263162704385387,
        -0.48498085401892155,
        0.2274783383974651,
        0.9721886190708176,
        1.1991037737107664,
        1.4681396813967078,
        1.6529353266254345,
        2.6,
        -0.19672935448735163,
        -1.43587504087747,
        0.04151199607926792,
        -2.6,
        0.5163352440729813,
    ],
    dtype=np.float64,
)
DEFAULT_TRAJECTORY_SECONDS = 10.57367634226263
DEFAULT_KP = 0.9414756587106877
DEFAULT_KD = 0.7572946397505047


def trajectory_action(env: NLinkCartPoleEnv, t: float, knots: np.ndarray, trajectory_seconds: float, kp: float, kd: float) -> float:
    times = np.linspace(0.0, trajectory_seconds, len(knots), dtype=np.float64)
    tt = min(t, trajectory_seconds)
    x_target = float(np.interp(tt, times, knots))
    x_next = float(np.interp(min(trajectory_seconds, tt + env.dt), times, knots))
    v_target = (x_next - x_target) / env.dt if tt < trajectory_seconds else 0.0
    action = kp * (x_target - float(env.data.qpos[0])) + kd * (v_target - float(env.data.qvel[0]))
    return float(np.clip(action, -1.0, 1.0))


def run_probe(cfg: dict[str, Any], *, progress: float, seed: int, seconds: float, zero_noise: bool) -> dict[str, Any]:
    cfg = {**cfg, "env": {**cfg["env"]}}
    if zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0

    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    obs, reset_info = env.reset()
    del obs

    steps = min(env.max_steps, int(seconds / env.dt))
    best: dict[str, Any] | None = None
    upright_events: list[dict[str, Any]] = []
    done_events: list[dict[str, Any]] = []
    action_abs_max = 0.0
    max_cart_abs = abs(float(reset_info["x"]))

    for step in range(steps):
        t = step * env.dt
        action = trajectory_action(env, t, DEFAULT_KNOTS, DEFAULT_TRAJECTORY_SECONDS, DEFAULT_KP, DEFAULT_KD)
        action_abs_max = max(action_abs_max, abs(action))
        _, reward, terminated, truncated, info = env.step([action])
        rel, abs_angles = env._angles()
        hinge_rms = float(np.sqrt(np.mean(env.data.qvel[1 : 1 + env.n] ** 2)))
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        row = {
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "reward": float(reward),
            "action": float(action),
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
        if best is None or row["max_abs_angle"] < best["max_abs_angle"]:
            best = row
        if info["is_upright"]:
            upright_events.append(row)
        if terminated or truncated:
            done_events.append(
                {
                    "step": int(step + 1),
                    "time_seconds": float((step + 1) * env.dt),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success": bool(info.get("success", False)),
                    "x": float(info["x"]),
                    "max_abs_angle": float(info["max_abs_angle"]),
                    "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
                }
            )
            break

    final_info = info
    env.close()
    return {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Fixed trajectory reaches the upright angle threshold at low hinge speed but does not stabilize.",
        "success": bool(final_info.get("success", False)),
        "zero_noise": bool(zero_noise),
        "seed": int(seed),
        "progress": float(progress),
        "simulated_steps": int(step + 1),
        "simulated_seconds": float((step + 1) * env.dt),
        "max_cart_abs": float(max_cart_abs),
        "action_abs_max": float(action_abs_max),
        "best_upright_pass": best,
        "upright_event_count": int(len(upright_events)),
        "upright_events": upright_events[:10],
        "done_events": done_events,
        "final_info": dict(final_info),
        "controller": {
            "type": "cart_position_pd_fixed_knots",
            "trajectory_seconds": float(DEFAULT_TRAJECTORY_SECONDS),
            "kp": float(DEFAULT_KP),
            "kd": float(DEFAULT_KD),
            "knots": DEFAULT_KNOTS.astype(float).tolist(),
        },
        "environment": {
            "n_links": int(cfg["env"]["n_links"]),
            "init_mode": str(cfg["env"].get("init_mode", "upright")),
            "init_angle_noise": float(cfg["env"].get("init_angle_noise", 0.02)),
            "init_vel_noise": float(cfg["env"].get("init_vel_noise", 0.01)),
            "episode_seconds": float(cfg["env"]["episode_seconds"]),
            "force_limit": float(cfg["env"]["force_limit"]),
            "rail_limit": float(cfg["env"]["rail_limit"]),
            "terminate_abs_angle": cfg["env"].get("terminate_abs_angle", 1.25),
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
    parser = argparse.ArgumentParser(description="Probe a fixed six-link hanging-start swing-up trajectory")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default="runs/swingup6_trajectory_probe/probe.json")
    parser.add_argument("--zero-noise", action="store_true", help="Probe exact hanging state by overriding init noise to zero")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    result = run_probe(cfg, progress=args.progress, seed=args.seed, seconds=args.seconds, zero_noise=args.zero_noise)
    dump_json(result, Path(args.out))
    best = result["best_upright_pass"] or {}
    print(
        "best_max_abs_angle="
        f"{best.get('max_abs_angle', float('nan')):.6f} "
        f"best_time={best.get('time_seconds', float('nan')):.3f}s "
        f"max_upright_streak={result['final_info'].get('max_upright_streak_seconds', 0.0):.3f}s "
        f"success={result['success']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
