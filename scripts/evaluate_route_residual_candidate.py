#!/usr/bin/env python
"""Replay a saved closed-loop route-residual candidate on an exact plant.

This is a development diagnostic.  The route feedback and residual knots are
reconstructed from their saved artifacts, and the replay starts once from the
hanging state with reset noise disabled.  It does not constitute a canonical
promotion or a public claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    text_sha256,
    utc_timestamp,
)
from search_swingup_handoff_bank_cem import (
    feedback_route_action,
    interpolation_matrix,
    load_feedback_route,
    reset_feedback_route,
)


def replay_candidate(
    cfg: dict[str, Any],
    candidate: dict[str, Any],
    route: dict[str, Any],
    *,
    tracking_gain_scale: float,
    phase_window: int,
    include_trace: bool,
) -> dict[str, Any]:
    search = candidate.get("search", {})
    best = candidate.get("best", {})
    seconds = float(search.get("seconds", 16.0))
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    steps = min(env.max_steps, max(1, round(seconds / env.dt)))
    knots = np.asarray(best.get("residual_knots", []), dtype=np.float64)
    if knots.ndim != 1 or knots.size < 2:
        raise ValueError("candidate best record must contain residual_knots")
    residual = knots @ interpolation_matrix(knots.size, steps).T
    start_step = min(
        steps,
        max(0, round(float(search.get("residual_start_time", 0.0)) / env.dt)),
    )
    residual[:start_step] = 0.0
    reset_feedback_route(route)
    rows: list[dict[str, Any]] = []
    max_cart = 0.0
    final_info: dict[str, Any] = {}
    for index in range(steps):
        route_action, route_index = feedback_route_action(
            env,
            route,
            tracking_gain_scale=tracking_gain_scale,
            phase_window=phase_window,
        )
        action = float(np.clip(route_action + residual[index], -1.0, 1.0))
        _, _, terminated, truncated, info = env.step([action])
        max_cart = max(max_cart, abs(float(env.data.qpos[0])))
        final_info = dict(info)
        if include_trace:
            rows.append(
                {
                    "step": int(env.step_count),
                    "time_seconds": float(env.step_count * env.dt),
                    "route_index": None if route_index is None else int(route_index),
                    "route_action": float(route_action),
                    "residual_action": float(residual[index]),
                    "action": float(action),
                    "qpos": np.asarray(env.data.qpos, dtype=np.float64).tolist(),
                    "qvel": np.asarray(env.data.qvel, dtype=np.float64).tolist(),
                    "max_abs_angle": float(info.get("max_abs_angle", 0.0)),
                    "hinge_velocity_rms": float(info.get("hinge_velocity_rms", 0.0)),
                    "x": float(env.data.qpos[0]),
                    "is_upright": bool(info.get("is_upright", False)),
                    "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
                }
            )
        if terminated or truncated:
            break
    completed_steps = int(env.step_count)
    env.close()
    termination_reason = final_info.get("termination_reason")
    if termination_reason is None and completed_steps >= steps:
        termination_reason = "replay_horizon"
    return {
        "success": bool(final_info.get("success", False)),
        "steps": completed_steps,
        "termination_reason": termination_reason,
        "max_cart_excursion": float(max_cart),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "time_to_capture": final_info.get("time_to_capture"),
        "final_info": final_info,
        "trace": rows if include_trace else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--feedback-route-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tracking-gain-scale", type=float, default=None)
    parser.add_argument("--phase-window", type=int, default=None)
    parser.add_argument("--include-trace", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    search = candidate.get("search", {})
    route_settings = search.get("feedback_route", {})
    tracking_gain_scale = float(
        route_settings.get("tracking_gain_scale", 1.0)
        if args.tracking_gain_scale is None
        else args.tracking_gain_scale
    )
    phase_window = int(
        route_settings.get("phase_window", 12)
        if args.phase_window is None
        else args.phase_window
    )
    if tracking_gain_scale < 0.0 or phase_window < 0:
        raise ValueError("tracking gain scale and phase window must be nonnegative")
    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    route = load_feedback_route(args.feedback_route_json, int(cfg["env"]["n_links"]))
    result = replay_candidate(
        cfg,
        candidate,
        route,
        tracking_gain_scale=tracking_gain_scale,
        phase_window=phase_window,
        include_trace=args.include_trace,
    )
    probe = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    xml_hash = text_sha256(probe.xml)
    probe.close()
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_route_residual_canonical_replay_not_solution",
        "not_solution": True,
        "summary": "Exact hanging-start replay of a closed-loop route-residual CEM candidate.",
        "config": {"path": str(Path(args.config)), "sha256": data_sha256(cfg)},
        "candidate": file_metadata(Path(args.candidate)),
        "feedback_route": file_metadata(Path(args.feedback_route_json)),
        "route_settings": {
            "tracking_gain_scale": tracking_gain_scale,
            "phase_window": phase_window,
        },
        "generated_xml_sha256": xml_hash,
        "result": result,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, Path(args.out))
    print(
        f"success={result['success']} steps={result['steps']} "
        f"termination={result['termination_reason']} max_cart={result['max_cart_excursion']:.3f}"
    )


if __name__ == "__main__":
    main()
