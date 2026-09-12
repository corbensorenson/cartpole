#!/usr/bin/env python
"""Replay a searched swing/capture chain and save its exact state trace.

The chain search stores only summary states.  This utility replays the same
state-gated hybrid controller against exact MuJoCo transitions and records the
pre-action state, action, post-action state, and active expert at every step.
That makes a discovered handoff usable by downstream tail optimizers without
silently replacing the hybrid prefix with a pure swing trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from evaluate_expert_chain import hinge_velocity_rms
from probe_swingup_trajectory import trajectory_action
from search_swingup_capture import lqr_action, lqr_gain


def load_controller(path: str, key: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload.get("best_by"), dict) and isinstance(payload["best_by"].get(key), dict):
        record = payload["best_by"][key]
        if isinstance(record.get("controller"), dict):
            return dict(record["controller"])
    if isinstance(payload.get("best"), dict) and isinstance(payload["best"].get("controller"), dict):
        return dict(payload["best"]["controller"])
    if isinstance(payload.get("controller"), dict):
        return dict(payload["controller"])
    if isinstance(payload.get("knots"), list):
        return dict(payload)
    raise ValueError(f"could not load a swing controller from {path}")


def row_from_env(
    env: NLinkCartPoleEnv,
    *,
    step: int,
    stage: str,
    action: float,
    reward: float,
    info: dict[str, Any],
    pre_qpos: np.ndarray,
    pre_qvel: np.ndarray,
) -> dict[str, Any]:
    relative, absolute = env._angles()
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "stage": stage,
        "action": float(action),
        "reward": float(reward),
        "pre_qpos": pre_qpos.astype(float).tolist(),
        "pre_qvel": pre_qvel.astype(float).tolist(),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "x": float(info["x"]),
        "cart_velocity": float(env.data.qvel[0]),
        "relative_angles": relative.astype(float).tolist(),
        "absolute_angles": absolute.astype(float).tolist(),
        "max_abs_angle": float(info["max_abs_angle"]),
        "mean_abs_angle": float(info["mean_abs_angle"]),
        "hinge_velocity_rms": float(info.get("hinge_velocity_rms", hinge_velocity_rms(env))),
        "absolute_angular_velocity_rms": float(
            np.sqrt(
                np.mean(
                    np.cumsum(np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64)) ** 2
                )
            )
        ),
        "capture_quality": float(info.get("capture_quality", 0.0)),
        "is_upright": bool(info["is_upright"]),
        "upright_streak_seconds": float(info["upright_streak_seconds"]),
        "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
        "time_to_first_upright": info.get("time_to_first_upright"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a hybrid swing/capture controller and save its exact trace")
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller-json", required=True)
    parser.add_argument("--controller-key", default="score")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--lqr-capture-scale", type=float, default=1.0)
    parser.add_argument("--lqr-stabilize-scale", type=float, default=1.0)
    parser.add_argument("--min-capture-seconds", type=float, default=0.50)
    parser.add_argument("--stabilize-enter-angle", type=float, default=0.15)
    parser.add_argument("--stabilize-enter-streak", type=float, default=0.02)
    parser.add_argument("--stabilize-hinge-rms", type=float, default=1.0)
    parser.add_argument(
        "--disable-capture",
        action="store_true",
        help="Replay only the swing expert so pre-crossing states can be exported without an LQR switch.",
    )
    parser.add_argument("--state-out", default=None)
    parser.add_argument("--state-min-time", type=float, default=0.0)
    parser.add_argument("--state-max-time", type=float, default=None)
    parser.add_argument("--state-max-angle", type=float, default=np.inf)
    parser.add_argument("--state-max-absolute-rate", type=float, default=np.inf)
    parser.add_argument("--state-stride", type=int, default=1)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"]}
    if args.zero_noise:
        # Match search_swingup_chain.py exactly: its historical --zero-noise
        # flag removes angle and joint-velocity noise but intentionally leaves
        # the configured cart-velocity perturbation in place.
        for name in ("init_angle_noise", "init_vel_noise"):
            if name in cfg["env"]:
                cfg["env"][name] = 0.0

    controller = load_controller(args.controller_json, args.controller_key)
    required = ("knots", "kp", "kd", "trajectory_seconds", "capture_min_time", "capture_enter_angle")
    missing = [name for name in required if name not in controller]
    if missing:
        raise ValueError(f"controller is missing fields: {missing}")

    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    # The source evaluator seeds the environment at construction and calls
    # reset without a second seed. Preserve that RNG sequence for replay.
    _, reset_info = env.reset()
    capture_gain = lqr_gain(
        cfg,
        progress=args.progress,
        fd_eps=args.fd_eps,
        control_cost=args.lqr_control_cost,
    )
    stabilize_gain = capture_gain
    knots = np.asarray(controller["knots"], dtype=np.float64)
    trajectory_seconds = float(controller["trajectory_seconds"])
    kp = float(controller["kp"])
    kd = float(controller["kd"])
    capture_min_time = float("inf") if args.disable_capture else float(controller["capture_min_time"])
    capture_enter_angle = float(controller["capture_enter_angle"])
    steps = min(env.max_steps, int(args.seconds / env.dt))

    stage = "swing"
    stage_enter_time = 0.0
    stage_counts = {"swing": 0, "capture": 0, "stabilize": 0}
    stage_events: list[dict[str, Any]] = [
        {"time_seconds": 0.0, "stage": stage, "reason": "reset"}
    ]
    trace: list[dict[str, Any]] = []
    selected_states: list[dict[str, Any]] = []
    done_events: list[dict[str, Any]] = []
    max_cart_abs = abs(float(reset_info["x"]))
    final_info: dict[str, Any] = {}
    best_angle: dict[str, Any] | None = None

    for step in range(steps):
        t = step * env.dt
        _, absolute_angles = env._angles()
        max_abs_angle = float(np.max(np.abs(absolute_angles)))
        hinge_rms = hinge_velocity_rms(env)
        if stage == "swing" and t >= capture_min_time and max_abs_angle <= capture_enter_angle:
            stage = "capture"
            stage_enter_time = t
            stage_events.append(
                {
                    "time_seconds": float(t),
                    "stage": stage,
                    "reason": "capture_enter_angle",
                    "max_abs_angle": max_abs_angle,
                    "hinge_velocity_rms": hinge_rms,
                    "x": float(env.data.qpos[0]),
                }
            )
        if (
            stage == "capture"
            and t - stage_enter_time >= args.min_capture_seconds
            and max_abs_angle <= args.stabilize_enter_angle
            and float(env._info()["upright_streak_seconds"]) >= args.stabilize_enter_streak
            and hinge_rms <= args.stabilize_hinge_rms
        ):
            stage = "stabilize"
            stage_enter_time = t
            stage_events.append(
                {
                    "time_seconds": float(t),
                    "stage": stage,
                    "reason": "stabilize_gate",
                    "max_abs_angle": max_abs_angle,
                    "hinge_velocity_rms": hinge_rms,
                    "x": float(env.data.qpos[0]),
                }
            )

        pre_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
        pre_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
        if stage == "swing":
            action = trajectory_action(env, t, knots, trajectory_seconds, kp, kd)
        elif stage == "capture":
            action = lqr_action(env, capture_gain, scale=args.lqr_capture_scale, cart_target=0.0)
        else:
            action = lqr_action(env, stabilize_gain, scale=args.lqr_stabilize_scale, cart_target=0.0)
        stage_counts[stage] += 1
        _, reward, terminated, truncated, info = env.step([action])
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        row = row_from_env(
            env,
            step=step + 1,
            stage=stage,
            action=action,
            reward=reward,
            info=info,
            pre_qpos=pre_qpos,
            pre_qvel=pre_qvel,
        )
        trace.append(row)
        if (
            args.state_out
            and row["time_seconds"] >= float(args.state_min_time)
            and (args.state_max_time is None or row["time_seconds"] <= float(args.state_max_time))
            and row["max_abs_angle"] <= float(args.state_max_angle)
            and row["absolute_angular_velocity_rms"] <= float(args.state_max_absolute_rate)
            and (step + 1) % max(1, int(args.state_stride)) == 0
        ):
            selected_states.append(dict(row))
        if best_angle is None or row["max_abs_angle"] < best_angle["max_abs_angle"]:
            best_angle = row
        final_info = dict(info)
        if terminated or truncated:
            done_events.append(
                {
                    "step": int(step + 1),
                    "time_seconds": float((step + 1) * env.dt),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success": bool(info.get("success", False)),
                    "termination_reason": info.get("termination_reason"),
                    "x": float(info["x"]),
                    "max_abs_angle": float(info["max_abs_angle"]),
                    "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
                }
            )
            break

    env.close()
    assert best_angle is not None
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact replay trace for a state-gated swing/capture controller; this is a discovery artifact, not a held-out solution.",
        "controller_json": str(Path(args.controller_json)),
        "controller_key": args.controller_key,
        "controller": controller,
        "progress": float(args.progress),
        "seed": int(args.seed),
        "zero_noise": bool(args.zero_noise),
        "search_config": {
            "seconds": float(args.seconds),
            "lqr_control_cost": float(args.lqr_control_cost),
            "lqr_capture_scale": float(args.lqr_capture_scale),
            "lqr_stabilize_scale": float(args.lqr_stabilize_scale),
            "min_capture_seconds": float(args.min_capture_seconds),
            "stabilize_enter_angle": float(args.stabilize_enter_angle),
            "stabilize_enter_streak": float(args.stabilize_enter_streak),
            "stabilize_hinge_rms": float(args.stabilize_hinge_rms),
        },
        "stage_events": stage_events,
        "stage_counts": stage_counts,
        "best_angle": best_angle,
        "success": bool(final_info.get("success", False)),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "max_cart_excursion": float(max_cart_abs),
        "simulated_seconds": float(len(trace) * env.dt),
        "done_events": done_events,
        "final_info": final_info,
        "trace": trace,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    if args.state_out:
        state_payload = {
            "schema_version": 1,
            "generated_at": utc_timestamp(),
            "not_solution": True,
            "summary": "Measured states sampled from an uninterrupted exact swing-only replay; component training data, not final evidence.",
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
            "config_sha256": data_sha256(cfg),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        }
        dump_json(state_payload, Path(args.state_out))
        print(f"Wrote {args.state_out} states={len(selected_states)}")
    print(
        f"Wrote {args.out} steps={len(trace)} success={payload['success']} "
        f"best_angle={best_angle['max_abs_angle']:.6f} "
        f"best_hinge={best_angle['hinge_velocity_rms']:.6f} "
        f"max_streak={payload['max_upright_streak_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
