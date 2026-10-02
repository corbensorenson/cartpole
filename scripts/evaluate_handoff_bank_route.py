#!/usr/bin/env python
"""Replay a handoff-bank CEM waveform on the canonical plant with LQR tail."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp

try:
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from search_swingup_capture import lqr_action, lqr_gain


ROOT = Path(__file__).resolve().parents[1]


def load_controls(path: str) -> tuple[np.ndarray, dict[str, Any]]:
    artifact = Path(path)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    record = payload.get("best") if isinstance(payload, dict) else None
    if not isinstance(record, dict) or not isinstance(record.get("controls"), list):
        raise ValueError("route artifact must contain best.controls")
    controls = np.asarray(record["controls"], dtype=np.float64)
    if controls.ndim != 1 or controls.size == 0 or not np.all(np.isfinite(controls)):
        raise ValueError("route controls must be a finite nonempty vector")
    return controls, {
        "path": str(artifact),
        "file": file_metadata(artifact),
        "iteration": record.get("iteration"),
        "search_cost": record.get("cost"),
        "best_distance": record.get("best_distance"),
        "terminal_distance": record.get("terminal_distance"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup11_uniform.yaml")
    parser.add_argument("--route", required=True)
    parser.add_argument("--seconds-after-route", type=float, default=8.0)
    parser.add_argument("--lqr-scale", type=float, default=1.0)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.seconds_after_route, args.lqr_scale, args.lqr_control_cost) <= 0.0:
        raise ValueError("tail seconds and LQR parameters must be positive")

    cfg = load_config(args.config)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "episode_seconds": max(
            float(cfg["env"].get("episode_seconds", 30.0)),
            30.0,
        ),
    }
    controls, route_metadata = load_controls(args.route)
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    gain = lqr_gain(
        cfg,
        progress=1.0,
        fd_eps=1e-7,
        control_cost=args.lqr_control_cost,
    )
    env.reset(seed=0)
    trace: list[dict[str, Any]] = []
    max_cart = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False
    for route_index, action in enumerate(controls):
        _, _, terminated, truncated, info = env.step([float(np.clip(action, -1.0, 1.0))])
        final_info = dict(info)
        max_cart = max(max_cart, abs(float(env.data.qpos[0])))
        trace.append(
            {
                "step": int(env.step_count),
                "time_seconds": float(env.step_count * env.dt),
                "phase": "swing_route",
                "route_index": int(route_index),
                "action": float(np.clip(action, -1.0, 1.0)),
                "x": float(env.data.qpos[0]),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "is_upright": bool(info["is_upright"]),
                "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
                "termination_reason": info.get("termination_reason"),
            }
        )
        if terminated or truncated:
            break

    route_steps = int(env.step_count)
    route_end_state = {
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "time_seconds": float(env.step_count * env.dt),
    }
    tail_steps = min(env.max_steps - env.step_count, round(args.seconds_after_route / env.dt))
    if not (terminated or truncated):
        for _ in range(tail_steps):
            action = lqr_action(env, gain, scale=args.lqr_scale, cart_target=0.0)
            _, _, terminated, truncated, info = env.step([action])
            final_info = dict(info)
            max_cart = max(max_cart, abs(float(env.data.qpos[0])))
            trace.append(
                {
                    "step": int(env.step_count),
                    "time_seconds": float(env.step_count * env.dt),
                    "phase": "lqr_capture",
                    "route_index": None,
                    "action": float(action),
                    "x": float(env.data.qpos[0]),
                    "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                    "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
                    "max_abs_angle": float(info["max_abs_angle"]),
                    "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                    "is_upright": bool(info["is_upright"]),
                    "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
                    "termination_reason": info.get("termination_reason"),
                }
            )
            if terminated or truncated:
                break
    env.close()
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "canonical_11_route_replay_not_solution",
        "not_solution": True,
        "summary": "Uninterrupted canonical-rail replay of a handoff-bank CEM waveform followed by exact upright LQR.",
        "config": file_metadata(Path(args.config)),
        "route": route_metadata,
        "capture": {
            "seconds_after_route": float(args.seconds_after_route),
            "lqr_scale": float(args.lqr_scale),
            "lqr_control_cost": float(args.lqr_control_cost),
            "route_steps": route_steps,
            "tail_steps": int(len(trace) - route_steps),
        },
        "result": {
            "success": bool(final_info.get("success", False)),
            "termination_reason": final_info.get("termination_reason"),
            "max_cart_excursion": float(max_cart),
            "max_upright_streak_seconds": float(
                final_info.get("max_upright_streak_seconds", 0.0)
            ),
            "final_max_abs_angle": float(final_info.get("max_abs_angle", np.inf)),
            "final_hinge_velocity_rms": float(
                final_info.get("hinge_velocity_rms", np.inf)
            ),
            "route_end_state": route_end_state,
        },
        "trace": trace,
        "runtime": runtime_metadata(),
        "git": git_metadata(ROOT),
    }
    dump_json(payload, Path(args.out))
    print(
        f"success={payload['result']['success']} "
        f"hold={payload['result']['max_upright_streak_seconds']:.3f}s "
        f"max_cart={payload['result']['max_cart_excursion']:.3f} "
        f"termination={payload['result']['termination_reason']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
