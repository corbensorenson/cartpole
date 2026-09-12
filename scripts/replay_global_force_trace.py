#!/usr/bin/env python
"""Replay a batched global-force candidate through serial MuJoCo.

Global force CEM artifacts are proposal generators.  This utility replays the
saved normalized-force knots through :class:`NLinkCartPoleEnv`, records the
post-action state at every 50 Hz step, and optionally exports measured states
for downstream capture searches.  It is deliberately separate from final
benchmark evaluation: a candidate still needs a feedback controller and
held-out reset-free evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp


def load_record(path: str, key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    record = payload.get(key)
    if not isinstance(record, dict) or not isinstance(record.get("knots"), list):
        raise ValueError(f"{path} does not contain a normalized-force record at {key!r}")
    return payload, record


def normalized_force(knots: np.ndarray, *, step: int, action_count: int) -> float:
    phase = float(np.clip(step, 0, action_count - 1)) / float(max(1, action_count - 1))
    knot_phase = np.linspace(0.0, 1.0, len(knots), dtype=np.float64)
    return float(np.clip(np.interp(phase, knot_phase, knots), -1.0, 1.0))


def row_from_env(
    env: NLinkCartPoleEnv,
    *,
    step: int,
    action: float,
    reward: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    relative_angles, absolute_angles = env._angles()
    hinge_rates = np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64)
    absolute_rates = np.cumsum(hinge_rates)
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "action": float(action),
        "force_newtons": float(action * env.force_limit),
        "reward": float(reward),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "x": float(info["x"]),
        "cart_velocity": float(env.data.qvel[0]),
        "relative_angles": relative_angles.astype(float).tolist(),
        "absolute_angles": absolute_angles.astype(float).tolist(),
        "max_abs_angle": float(np.max(np.abs(absolute_angles))),
        "mean_abs_angle": float(np.mean(np.abs(absolute_angles))),
        "hinge_velocity_rms": float(np.sqrt(np.mean(hinge_rates**2))),
        "absolute_angular_velocity_rms": float(np.sqrt(np.mean(absolute_rates**2))),
        "is_upright": bool(info["is_upright"]),
        "upright_streak_seconds": float(info["upright_streak_seconds"]),
        "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
        "time_to_first_upright": info.get("time_to_first_upright"),
        "termination_reason": info.get("termination_reason"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a global force-CEM candidate through serial MuJoCo")
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller-json", required=True)
    parser.add_argument("--record-key", default="best")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--state-out", default=None)
    parser.add_argument("--state-min-time", type=float, default=0.0)
    parser.add_argument("--state-max-time", type=float, default=None)
    parser.add_argument("--state-max-angle", type=float, default=np.inf)
    parser.add_argument("--state-max-absolute-rate", type=float, default=np.inf)
    parser.add_argument("--state-stride", type=int, default=1)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    payload, record = load_record(args.controller_json, args.record_key)
    search = payload.get("search", {})
    seconds = float(args.seconds if args.seconds is not None else search.get("seconds", 0.0))
    if seconds <= 0.0:
        raise ValueError("the force candidate must have a positive duration")
    knots = np.asarray(record["knots"], dtype=np.float64)
    if knots.ndim != 1 or len(knots) < 2:
        raise ValueError("force candidate must contain at least two knots")

    cfg = apply_overrides(load_config(args.config), args.override)
    # Match search_swingup_global_cem.py: proposal artifacts are generated
    # from one deterministic hanging state, even when the source config is a
    # noisy benchmark config.
    cfg["env"] = {**cfg["env"], "init_mode": "hanging"}
    for noise_key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        cfg["env"][noise_key] = 0.0
        cfg["env"][f"{noise_key}_start"] = 0.0
        cfg["env"][f"{noise_key}_end"] = 0.0

    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    _, reset_info = env.reset(seed=args.seed)
    action_count = max(2, int(round(seconds / env.dt)))
    rows: list[dict[str, Any]] = []
    done_events: list[dict[str, Any]] = []
    max_cart_abs = abs(float(reset_info["x"]))
    total_return = 0.0
    for step in range(action_count):
        action = normalized_force(knots, step=step, action_count=action_count)
        _, reward, terminated, truncated, info = env.step([action])
        total_return += float(reward)
        row = row_from_env(
            env,
            step=step + 1,
            action=action,
            reward=reward,
            info=info,
        )
        rows.append(row)
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        if terminated or truncated:
            done_events.append(
                {
                    "step": int(step + 1),
                    "time_seconds": float((step + 1) * env.dt),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "termination_reason": info.get("termination_reason"),
                }
            )
            break

    if not rows:
        raise RuntimeError("serial replay produced no post-action states")
    best_angle = min(rows, key=lambda row: row["max_abs_angle"])
    source_best_time = record.get("best_time_seconds")
    source_replay_row = None
    source_replay_error = None
    if source_best_time is not None:
        source_replay_row = min(
            rows,
            key=lambda row: abs(float(row["time_seconds"]) - float(source_best_time)),
        )
        source_qpos = np.asarray(record.get("best_state", {}).get("qpos", []), dtype=np.float64)
        source_qvel = np.asarray(record.get("best_state", {}).get("qvel", []), dtype=np.float64)
        replay_qpos = np.asarray(source_replay_row["qpos"], dtype=np.float64)
        replay_qvel = np.asarray(source_replay_row["qvel"], dtype=np.float64)
        if source_qpos.shape == replay_qpos.shape and source_qvel.shape == replay_qvel.shape:
            source_replay_error = {
                "time_seconds": float(source_replay_row["time_seconds"]),
                "qpos_max_abs_error": float(np.max(np.abs(source_qpos - replay_qpos))),
                "qvel_max_abs_error": float(np.max(np.abs(source_qvel - replay_qvel))),
            }

    selected_states = [
        row
        for row in rows
        if float(row["time_seconds"]) >= float(args.state_min_time)
        and (args.state_max_time is None or float(row["time_seconds"]) <= float(args.state_max_time))
        and float(row["max_abs_angle"]) <= float(args.state_max_angle)
        and float(row["absolute_angular_velocity_rms"]) <= float(args.state_max_absolute_rate)
        and int(row["step"]) % max(1, int(args.state_stride)) == 0
    ]
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Serial exact-MuJoCo replay of a batched global-force proposal; requires downstream feedback capture and held-out verification.",
        "source_controller_json": str(Path(args.controller_json)),
        "source_record_key": args.record_key,
        "source_record": {
            "cost": record.get("cost"),
            "best_time_seconds": record.get("best_time_seconds"),
            "max_angle": record.get("max_angle"),
            "hinge_rms": record.get("hinge_rms"),
            "absolute_rate_rms": record.get("absolute_rate_rms"),
            "cart": record.get("cart"),
            "cart_velocity": record.get("cart_velocity"),
            "rail": record.get("rail"),
        },
        "resolved_config_sha256": data_sha256(cfg),
        "replay": {
            "seed": int(args.seed),
            "progress": float(args.progress),
            "seconds": seconds,
            "action_count": int(action_count),
            "knot_count": int(len(knots)),
            "dt_seconds": float(env.dt),
            "frame_skip": int(env.frame_skip),
            "force_limit": float(env.force_limit),
        },
        "initial_state": {
            "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        },
        "simulated_seconds": float(len(rows) * env.dt),
        "simulated_steps": int(len(rows)),
        "episode_return": float(total_return),
        "max_cart_abs": float(max_cart_abs),
        "best_angle": best_angle,
        "source_best_replay_comparison": source_replay_error,
        "done_events": done_events,
        "trace": rows,
        "trajectory_sha256": data_sha256(rows),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(result, Path(args.out))
    if args.state_out:
        state_payload = {
            "schema_version": 1,
            "generated_at": utc_timestamp(),
            "not_solution": True,
            "summary": "Measured post-action states from a serial force-trace replay; component training data, not final evidence.",
            "source_trace": str(Path(args.out)),
            "state_count": int(len(selected_states)),
            "selection": {
                "state_min_time": float(args.state_min_time),
                "state_max_time": args.state_max_time,
                "state_max_angle": float(args.state_max_angle),
                "state_max_absolute_rate": float(args.state_max_absolute_rate),
                "state_stride": int(args.state_stride),
            },
            "states": selected_states,
            "resolved_config_sha256": data_sha256(cfg),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        }
        dump_json(state_payload, Path(args.state_out))
        print(f"Wrote {args.state_out} states={len(selected_states)}")
    print(
        f"Wrote {args.out} steps={len(rows)} best_angle={best_angle['max_abs_angle']:.6f} "
        f"best_time={best_angle['time_seconds']:.3f}s max_rail={max_cart_abs:.3f} "
        f"replay_error={source_replay_error}"
    )


if __name__ == "__main__":
    main()
