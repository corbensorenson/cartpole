#!/usr/bin/env python
"""Evaluate a transferred route with the source-count LQR as its tail.

This is a continuation diagnostic.  The route and feedback law run in the
target-count coordinates until their saved horizon; afterward the physical
source-count LQR acts on the proximal source joints only.  It tests whether a
new distal degree of freedom is the cause of a native target-count handoff
failure without claiming a final target-count solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.modal import StateScales, dimensionless_absolute_transform, dimensionless_wrapped_state
from gcartpole.ilqr import data_state

try:
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from search_swingup_capture import lqr_gain


def source_lqr_action(env: NLinkCartPoleEnv, gain: np.ndarray, source_links: int, scale: float) -> float:
    qpos = np.asarray(env.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    state = np.r_[qpos[0], wrap_angle(qpos[1 : source_links + 1]), qvel[0], qvel[1 : source_links + 1]]
    return float(np.clip(-scale * float(gain @ state), -1.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--route-feedback-scale", type=float, default=1.0)
    parser.add_argument("--lqr-scale", type=float, default=1.0)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument(
        "--tail-action-json",
        default=None,
        help="optional solved-source replay artifact whose actions continue the route",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.seconds, args.route_feedback_scale, args.lqr_scale, args.lqr_control_cost) <= 0.0:
        raise ValueError("seconds, gains, and control cost must be positive")

    source_cfg = load_config(args.source_config)
    target_cfg = load_config(args.target_config)
    source_links = int(source_cfg["env"]["n_links"])
    target_links = int(target_cfg["env"]["n_links"])
    if target_links <= source_links:
        raise ValueError("target must contain more links than source")
    payload = json.loads(Path(args.controller).read_text(encoding="utf-8"))
    controller = payload.get("controller")
    search = payload.get("search")
    if not isinstance(controller, dict) or not isinstance(search, dict):
        raise ValueError("controller artifact must contain controller and search objects")
    controls = np.asarray(controller.get("controls"), dtype=np.float64)
    feedback = np.asarray(controller.get("feedback_gains"), dtype=np.float64)
    nominal = np.asarray(search.get("nominal_coordinate_states"), dtype=np.float64)
    expected_dim = 2 * (target_links + 1)
    if controls.ndim != 1 or feedback.shape != (controls.size, expected_dim):
        raise ValueError("route controls and feedback dimensions do not match target count")
    if nominal.shape != (controls.size + 1, expected_dim):
        raise ValueError("route nominal states do not match controls")

    spec = load_config("benchmarks/p1_capture_envelope.yaml")
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        target_links,
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    source_gain = lqr_gain(
        source_cfg,
        progress=1.0,
        fd_eps=1.0e-7,
        control_cost=float(args.lqr_control_cost),
    )
    tail_actions: np.ndarray | None = None
    if args.tail_action_json is not None:
        tail_payload = json.loads(Path(args.tail_action_json).read_text(encoding="utf-8"))
        tail_rows = tail_payload.get("result", {}).get("trajectory")
        if tail_rows is None:
            tail_rows = tail_payload.get("trajectory")
        if not isinstance(tail_rows, list) or not tail_rows:
            raise ValueError("tail-action-json must contain a trajectory")
        tail_actions = np.asarray([row["action"] for row in tail_rows], dtype=np.float64)
    cfg = {**target_cfg, "env": {**target_cfg["env"]}}
    cfg["env"]["init_mode"] = "hanging"
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][key] = 0.0
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    env.reset(seed=args.seed)
    rows: list[dict[str, Any]] = []
    episode_return = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    steps = min(env.max_steps, max(1, round(float(args.seconds) / env.dt)))
    for step in range(steps):
        coordinate_state = dimensionless_wrapped_state(env.data.qpos, env.data.qvel, transform)
        if step < controls.size:
            error = coordinate_state - nominal[step]
            action = float(
                np.clip(
                    controls[step] + float(args.route_feedback_scale) * feedback[step] @ error,
                    -1.0,
                    1.0,
                )
            )
            mode = "transferred_route"
        else:
            tail_step = step - controls.size
            source_tail_index = controls.size + tail_step
            if tail_actions is not None and source_tail_index < tail_actions.size:
                action = float(np.clip(tail_actions[source_tail_index], -1.0, 1.0))
                mode = "source_replay_action_tail"
            else:
                action = source_lqr_action(env, source_gain, source_links, float(args.lqr_scale))
                mode = "source_count_lqr_tail"
        _, reward, terminated, truncated, info = env.step([action])
        episode_return += float(reward)
        relative, absolute = env._angles()
        rows.append(
            {
                "step": int(step + 1),
                "time_seconds": float((step + 1) * env.dt),
                "action": float(action),
                "controller_mode": mode,
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
                "relative_angles": relative.astype(float).tolist(),
                "absolute_angles": absolute.astype(float).tolist(),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "cart_velocity": float(env.data.qvel[0]),
                "is_upright": bool(info["is_upright"]),
                "upright_streak_seconds": float(info["upright_streak_seconds"]),
            }
        )
        if terminated or truncated:
            break
    env.close()
    policy_dt = float(target_cfg["env"]["timestep"] * target_cfg["env"].get("frame_skip", 1))
    result = {
        "success": bool(info.get("success", False)),
        "latched": bool(info.get("success", False)),
        "termination_reason": info.get("termination_reason"),
        "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(info.get("max_low_momentum_upright_streak_seconds", 0.0)),
        "max_cart_excursion": float(info.get("max_cart_excursion", 0.0)),
        "length": len(rows),
        "final_info": info,
        "return": float(episode_return),
    }
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "source_lqr_tail_diagnostic_not_solution_evidence",
        "not_solution": True,
        "summary": "Transferred route with source-count LQR tail; continuation diagnostic only.",
        "source": {"config": file_metadata(Path(args.source_config)), "links": source_links},
        "target": {"config": file_metadata(Path(args.target_config)), "links": target_links},
        "controller": {
            "route": file_metadata(Path(args.controller)),
            "route_feedback_scale": float(args.route_feedback_scale),
            "source_lqr_scale": float(args.lqr_scale),
            "source_lqr_control_cost": float(args.lqr_control_cost),
            "tail_action_source": (
                None if args.tail_action_json is None else file_metadata(Path(args.tail_action_json))
            ),
            "route_horizon_steps": int(controls.size),
            "policy_dt": policy_dt,
        },
        "result": result,
        "trajectory": rows,
        "evidence": {
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(output, Path(args.out))
    print(
        f"success={result['success']} hold={result['max_upright_streak_seconds']:.3f}s "
        f"cart={result['max_cart_excursion']:.3f} steps={len(rows)}"
    )


if __name__ == "__main__":
    main()
