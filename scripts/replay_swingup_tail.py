#!/usr/bin/env python
"""Replay a tail proposal from an exact recorded swing state.

Tail CEM artifacts are proposal generators.  This utility restores the exact
state emitted by a prior serial trajectory and applies the saved normalized
force knots through the ordinary MuJoCo environment step loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp


def load_trace_row(path: str, target_time: float) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("trace")
    if not isinstance(rows, list) or not rows:
        rows = payload.get("trajectory")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not contain trace or trajectory rows")
    candidates = [row for row in rows if isinstance(row, dict)]
    row = min(candidates, key=lambda candidate: abs(float(candidate["time_seconds"]) - target_time))
    if not isinstance(row.get("qpos"), list) or not isinstance(row.get("qvel"), list):
        raise ValueError("trace row must contain qpos and qvel lists")
    return row


def load_tail_record(path: str, key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    record = payload.get(key)
    if not isinstance(record, dict) or not isinstance(record.get("knots"), list):
        raise ValueError(f"{path} does not contain normalized-force knots at {key!r}")
    return payload, record


def normalized_force(knots: np.ndarray, step: int, step_count: int) -> float:
    phase = float(np.clip(step, 0, step_count - 1)) / float(max(1, step_count - 1))
    return float(np.clip(np.interp(phase, np.linspace(0.0, 1.0, len(knots)), knots), -1.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay an exact MuJoCo tail proposal")
    parser.add_argument("--config", required=True)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--initial-trace-json", required=True)
    parser.add_argument("--initial-trace-time", type=float, required=True)
    parser.add_argument("--tail-json", required=True)
    parser.add_argument("--tail-key", default="best")
    parser.add_argument("--tail-seconds", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("--progress must be in [0, 1]")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"], "init_mode": "hanging"}
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][key] = 0.0
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0

    tail_payload, tail_record = load_tail_record(args.tail_json, args.tail_key)
    source_row = load_trace_row(args.initial_trace_json, args.initial_trace_time)
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=0)
    env.reset(seed=0)
    qpos = np.asarray(source_row["qpos"], dtype=np.float64)
    qvel = np.asarray(source_row["qvel"], dtype=np.float64)
    if qpos.shape != (env.model.nq,) or qvel.shape != (env.model.nv,):
        raise ValueError(f"state shape mismatch: qpos={qpos.shape} qvel={qvel.shape}")
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.time = float(source_row["time_seconds"])
    mujoco.mj_forward(env.model, env.data)
    env._last_potential_energy = env._potential_energy()
    env.max_cart_excursion = abs(float(env.data.qpos[0]))
    env._update_upright_tracking()

    search = tail_payload.get("search", {})
    default_seconds = float(tail_payload.get("tail_horizon_seconds", search.get("tail_seconds", 0.0)))
    seconds = float(args.tail_seconds if args.tail_seconds is not None else default_seconds)
    if seconds <= 0.0:
        raise ValueError("tail duration must be positive")
    steps = max(2, int(round(seconds / env.dt)))
    knots = np.asarray(tail_record["knots"], dtype=np.float64)
    rows: list[dict[str, Any]] = []
    done_events: list[dict[str, Any]] = []
    max_cart_abs = abs(float(env.data.qpos[0]))
    total_return = 0.0
    for step in range(steps):
        action = normalized_force(knots, step, steps)
        _, reward, terminated, truncated, info = env.step([action])
        total_return += float(reward)
        _, absolute = env._angles()
        absolute_rates = env._absolute_angular_velocity()
        row = {
            "step": int(step + 1),
            "time_seconds": float(source_row["time_seconds"] + (step + 1) * env.dt),
            "tail_time_seconds": float((step + 1) * env.dt),
            "action": float(action),
            "force_newtons": float(action * env.force_limit),
            "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
            "x": float(env.data.qpos[0]),
            "cart_velocity": float(env.data.qvel[0]),
            "absolute_angles": absolute.astype(float).tolist(),
            "max_abs_angle": float(np.max(np.abs(absolute))),
            "hinge_velocity_rms": float(np.sqrt(np.mean(np.asarray(env.data.qvel[1 : 1 + env.n]) ** 2))),
            "absolute_angular_velocity_rms": float(np.sqrt(np.mean(absolute_rates**2))),
            "is_upright": bool(info["is_upright"]),
            "upright_streak_seconds": float(info["upright_streak_seconds"]),
            "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
            "termination_reason": info.get("termination_reason"),
        }
        rows.append(row)
        max_cart_abs = max(max_cart_abs, abs(float(row["x"])))
        if terminated or truncated:
            done_events.append({
                "step": int(step + 1),
                "time_seconds": float(row["time_seconds"]),
                "termination_reason": info.get("termination_reason"),
            })
            break

    best = min(rows, key=lambda row: float(row["max_abs_angle"]))
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact serial replay of a tail proposal from an actual recorded swing state.",
        "source_trace": str(Path(args.initial_trace_json)),
        "source_trace_time_requested": float(args.initial_trace_time),
        "source_trace_time_used": float(source_row["time_seconds"]),
        "tail_json": str(Path(args.tail_json)),
        "tail_key": args.tail_key,
        "progress": float(args.progress),
        "tail_seconds": seconds,
        "initial_state": {"qpos": qpos.astype(float).tolist(), "qvel": qvel.astype(float).tolist()},
        "simulated_steps": len(rows),
        "simulated_seconds": float(len(rows) * env.dt),
        "episode_return": float(total_return),
        "success": bool(rows[-1].get("max_upright_streak_seconds", 0.0) >= float(cfg["env"].get("success_sustain_seconds", 5.0))),
        "max_upright_streak_seconds": float(rows[-1].get("max_upright_streak_seconds", 0.0)),
        "max_cart_abs": float(max_cart_abs),
        "best_angle": best,
        "done_events": done_events,
        "trace": rows,
        "trajectory_sha256": data_sha256(rows),
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(result, Path(args.out))
    print(
        f"Wrote {args.out} steps={len(rows)} best_angle={best['max_abs_angle']:.6f} "
        f"best_hinge={best['hinge_velocity_rms']:.6f} max_rail={max_cart_abs:.3f}"
    )


if __name__ == "__main__":
    main()
