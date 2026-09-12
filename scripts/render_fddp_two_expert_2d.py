#!/usr/bin/env python
"""Render a state-faithful 2D video without requiring a CoreGraphics context."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    text_sha256,
    utc_timestamp,
)
from gcartpole.ilqr import data_state
from gcartpole.modal import dimensionless_wrapped_state

try:
    from scripts.evaluate_fddp_two_expert import (
        hanging_lqr_action,
        hanging_lqr_gain,
        load_controller,
    )
    from scripts.search_capture_sequence import fixed_state_cfg, load_state
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from evaluate_fddp_two_expert import hanging_lqr_action, hanging_lqr_gain, load_controller
    from search_capture_sequence import fixed_state_cfg, load_state
    from search_swingup_capture import lqr_action, lqr_gain


def draw_frame(
    *,
    width: int,
    height: int,
    qpos: np.ndarray,
    action: float,
    time_seconds: float,
    mode: str,
    max_abs_angle: float,
    hinge_velocity_rms: float,
    upright_streak: float,
    max_upright_streak: float,
    cart_trace: list[float],
    rail_limit: float,
    subtitle: str,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), (9, 15, 30))
    draw = ImageDraw.Draw(image)
    left, right = 64, width - 64
    track_y = int(height * 0.70)
    panel_y = height - 104
    world_min, world_max = -3.25, 3.25

    def sx(value: float) -> int:
        return int(left + (float(value) - world_min) / (world_max - world_min) * (right - left))

    draw.rectangle((0, 0, width, height), fill=(9, 15, 30))
    for grid_x in np.linspace(-3.0, 3.0, 7):
        x = sx(float(grid_x))
        draw.line((x, 90, x, panel_y - 20), fill=(21, 32, 55), width=1)
        draw.text((x - 12, track_y + 29), f"{grid_x:g}", fill=(105, 123, 151))
    draw.text((64, 34), "7-LINK SWING-UP", fill=(241, 245, 249))
    draw.text((64, 60), f"{subtitle} | state-faithful MuJoCo trajectory", fill=(148, 163, 184))
    phase_color = (56, 189, 248) if mode == "swing_feedback" else (251, 146, 60)
    phase_text = "SWING-UP / FEEDBACK" if mode == "swing_feedback" else "CAPTURE / LQR"
    draw.text((width - 300, 40), phase_text, fill=phase_color)
    draw.text((width - 300, 65), f"t = {time_seconds:05.2f} s", fill=(226, 232, 240))

    rail_left, rail_right = sx(-rail_limit), sx(rail_limit)
    draw.line((rail_left, track_y + 18, rail_right, track_y + 18), fill=(71, 85, 105), width=7)
    draw.line((rail_left, track_y + 8, rail_left, track_y + 28), fill=(239, 68, 68), width=4)
    draw.line((rail_right, track_y + 8, rail_right, track_y + 28), fill=(239, 68, 68), width=4)

    if len(cart_trace) > 1:
        points = [(sx(x), track_y + 2) for x in cart_trace]
        draw.line(points, fill=(30, 64, 100), width=2)
    cart_x = sx(float(qpos[0]))
    draw.rounded_rectangle((cart_x - 38, track_y - 2, cart_x + 38, track_y + 34), radius=6, fill=(226, 232, 240), outline=(148, 163, 184), width=2)
    draw.ellipse((cart_x - 25, track_y + 24, cart_x - 10, track_y + 39), fill=(30, 41, 59))
    draw.ellipse((cart_x + 10, track_y + 24, cart_x + 25, track_y + 39), fill=(30, 41, 59))

    absolute_angles = serial_absolute_angles(np.asarray(qpos[1:], dtype=np.float64))
    # Fit the complete chain between the track and the title area. A fixed
    # link length clips the upper links when the seven-link chain is upright.
    link_count = max(1, len(absolute_angles))
    link_pixels = min(110.0, max(28.0, (track_y - 112.0) / link_count))
    px, py = float(cart_x), float(track_y)
    for index, angle in enumerate(absolute_angles):
        ex = px + link_pixels * np.sin(float(angle))
        ey = py - link_pixels * np.cos(float(angle))
        draw.line((px, py, ex, ey), fill=phase_color, width=10)
        draw.ellipse((int(px - 8), int(py - 8), int(px + 8), int(py + 8)), fill=(248, 250, 252), outline=(30, 41, 59), width=2)
        px, py = ex, ey
    draw.ellipse((int(px - 7), int(py - 7), int(px + 7), int(py + 7)), fill=(251, 191, 36), outline=(30, 41, 59), width=2)

    draw.rectangle((0, panel_y, width, height), fill=(15, 23, 42))
    draw.text((64, panel_y + 18), f"max angle  {max_abs_angle:6.3f} rad", fill=(226, 232, 240))
    draw.text((320, panel_y + 18), f"hinge rate  {hinge_velocity_rms:6.3f} rad/s", fill=(226, 232, 240))
    draw.text((650, panel_y + 18), f"action  {action:+6.3f}", fill=(226, 232, 240))
    draw.text((930, panel_y + 18), f"cart  {float(qpos[0]):+6.3f} m", fill=(226, 232, 240))
    draw.text((64, panel_y + 57), f"upright streak  {upright_streak:5.2f} s", fill=(134, 239, 172))
    draw.text((320, panel_y + 57), f"best hold  {max_upright_streak:5.2f} s", fill=(134, 239, 172))
    draw.text((650, panel_y + 57), "rail  +/-3.00 m", fill=(148, 163, 184))
    draw.text((930, panel_y + 57), "7 uniform links", fill=(148, 163, 184))
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a saved seven-link two-expert rollout as a 2D MP4")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--state-json", default=None)
    parser.add_argument("--state-index", default="0")
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=50732)
    parser.add_argument("--hanging-start", action="store_true")
    parser.add_argument("--prelude-seconds", type=float, default=0.0)
    parser.add_argument("--settle-mode", choices=("zero", "hanging_lqr"), default="zero")
    parser.add_argument("--settle-scale", type=float, default=1.0)
    parser.add_argument("--settle-control-cost", type=float, default=1000.0)
    parser.add_argument("--tracking-gain-scale", type=float, default=1.0)
    parser.add_argument("--shift-cart-nominal", action="store_true")
    parser.add_argument("--fail-on-failure", action="store_true")
    args = parser.parse_args()
    if (
        args.seconds <= 0.0
        or args.fps < 1
        or args.prelude_seconds < 0.0
        or args.settle_scale < 0.0
        or args.settle_control_cost <= 0.0
        or args.tracking_gain_scale < 0.0
    ):
        raise ValueError("seconds, fps, prelude, scales, and control cost must be valid")

    repo_root = Path(__file__).resolve().parents[1]
    source_git = {
        key: value
        for key, value in git_metadata(repo_root, include_untracked=False).items()
        if key != "root"
    }
    base_cfg = load_config(args.config)
    spec = load_config(args.spec)
    controller_path = Path(args.controller)
    controller = load_controller(controller_path, int(base_cfg["env"]["n_links"]), spec)
    payload = json.loads(controller_path.read_text(encoding="utf-8"))
    if args.state_json is None and not args.hanging_start:
        state = payload.get("selected_state")
        state_source = controller_path
        if not isinstance(state, dict):
            raise ValueError("controller artifact has no selected_state; pass --state-json")
    elif args.state_json is not None:
        state, _ = load_state(args.state_json, args.state_index)
        state_source = Path(args.state_json)
    else:
        state = None
        state_source = {"mode": "hanging", "seed": int(args.seed)}
    if args.hanging_start:
        cfg = copy.deepcopy(base_cfg)
        cfg["env"] = {
            **cfg["env"],
            "init_mode": "hanging",
            "episode_seconds": float(args.seconds),
            "terminate_abs_angle": None,
        }
    else:
        cfg = fixed_state_cfg(base_cfg, state, args.seconds)
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"].setdefault("action_lqr_switch", {"enabled": False})["enabled"] = False
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    env.reset(seed=args.seed)
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    settle_gain = (
        hanging_lqr_gain(
            cfg,
            progress=1.0,
            fd_eps=1e-7,
            control_cost=args.settle_control_cost,
        )
        if args.settle_mode == "hanging_lqr"
        else None
    )
    controls = controller["controls"]
    nominal_states = controller["nominal_states"]
    feedback_gains = controller["feedback_gains"]
    transform = controller["transform"]
    route_nominal_states = nominal_states.copy()
    cart_nominal_shift: float | None = None
    prelude_steps = int(round(args.prelude_seconds / env.dt))
    out = Path(args.out)
    metadata_out = Path(args.metadata_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8)
    sim_steps = min(env.max_steps, int(round(args.seconds / env.dt)))
    final_info: dict[str, object] = {}
    done_events: list[dict[str, object]] = []
    cart_trace: list[float] = []
    first_upright: float | None = None
    completed_steps = 0
    frames = 0
    try:
        for step in range(sim_steps):
            coordinate_state = dimensionless_wrapped_state(env.data.qpos, env.data.qvel, transform)
            route_step = step - prelude_steps
            if step < prelude_steps:
                if args.settle_mode == "hanging_lqr":
                    if settle_gain is None:
                        raise ValueError("hanging_lqr settle mode requires a gain")
                    action = hanging_lqr_action(env, settle_gain, scale=args.settle_scale)
                    mode = "hanging_lqr_settle"
                else:
                    action = 0.0
                    mode = "hanging_settle"
            elif route_step < controls.size:
                if args.shift_cart_nominal and cart_nominal_shift is None:
                    cart_nominal_shift = float(coordinate_state[0] - nominal_states[0, 0])
                    route_nominal_states[:, 0] += cart_nominal_shift
                action = float(
                    np.clip(
                        controls[route_step]
                        + args.tracking_gain_scale
                        * feedback_gains[route_step]
                        @ (coordinate_state - route_nominal_states[route_step]),
                        -1.0,
                        1.0,
                    )
                )
                mode = "swing_feedback"
            else:
                action = lqr_action(env, gain, scale=controller["lqr_scale"], cart_target=0.0)
                mode = "capture_lqr"
            _, _, terminated, truncated, info = env.step([action])
            completed_steps = step + 1
            final_info = dict(info)
            cart_trace.append(float(info["x"]))
            cart_trace = cart_trace[-180:]
            if first_upright is None and bool(info.get("is_upright", False)):
                first_upright = float(completed_steps * env.dt)
            frame = draw_frame(
                width=args.width,
                height=args.height,
                qpos=np.asarray(env.data.qpos, dtype=np.float64),
                action=action,
                time_seconds=completed_steps * env.dt,
                mode=mode,
                max_abs_angle=float(info["max_abs_angle"]),
                hinge_velocity_rms=float(info["hinge_velocity_rms"]),
                upright_streak=float(info["upright_streak_seconds"]),
                max_upright_streak=float(info["max_upright_streak_seconds"]),
                cart_trace=cart_trace,
                rail_limit=float(env.rail_limit),
                subtitle=(
                    "canonical noisy held-out replay"
                    if args.hanging_start
                    else "exact-start proof replay"
                ),
            )
            writer.append_data(frame)
            frames += 1
            if terminated or truncated:
                done_events.append(
                    {
                        "step": int(completed_steps),
                        "time_seconds": float(completed_steps * env.dt),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "success": bool(info.get("success", False)),
                        "termination_reason": info.get("termination_reason"),
                    }
                )
                break
    finally:
        writer.close()
        env.close()

    xml_probe = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    xml_sha256 = text_sha256(xml_probe.xml)
    xml_probe.close()
    state_metadata = file_metadata(state_source) if isinstance(state_source, Path) else state_source
    metadata = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": (
            "canonical_noisy_holdout_pass" if args.hanging_start else "exact_start_proof_not_canonical_noisy_gate"
        ),
        "summary": (
            "State-faithful 2D replay of the canonical noisy hanging-start seven-link swing-up and hold."
            if args.hanging_start
            else "State-faithful 2D replay of the exact-MuJoCo seven-link swing-up and hold."
        ),
        "visualization": "2d_state_faithful",
        "video": file_metadata(out),
        "controller": controller["source"],
        "state_source": state_metadata,
        "generated_xml_sha256": xml_sha256,
        "config": {
            "path": str(Path(args.config)),
            "resolved_sha256": data_sha256(base_cfg),
            "overrides": [],
        },
        "render": {
            "requested_seconds": float(args.seconds),
            "simulated_seconds": float(completed_steps * env.dt),
            "completed_requested_steps": completed_steps == sim_steps,
            "dt": float(env.dt),
            "fps": int(args.fps),
            "frames": int(frames),
            "width": int(args.width),
            "height": int(args.height),
            "seed": int(args.seed),
            "reset_count": 0,
            "done_events": done_events,
            "first_upright_time": first_upright,
            "max_cart_excursion": float(final_info.get("max_cart_excursion", 0.0)),
            "final_info": final_info,
        },
        "environment": {
            "n_links": int(cfg["env"]["n_links"]),
            "init_mode": "hanging" if args.hanging_start else "fixed_state",
            "force_limit": float(cfg["env"]["force_limit"]),
            "rail_limit": float(cfg["env"]["rail_limit"]),
            "success_upright_threshold": float(cfg["env"]["success_upright_threshold"]),
            "success_sustain_seconds": float(cfg["env"]["success_sustain_seconds"]),
        },
        "launch": {
            "prelude_seconds": float(prelude_steps * env.dt),
            "settle_mode": str(args.settle_mode),
            "settle_scale": float(args.settle_scale),
            "settle_control_cost": float(args.settle_control_cost),
            "tracking_gain_scale": float(args.tracking_gain_scale),
            "shift_cart_nominal": bool(args.shift_cart_nominal),
            "settle_gain": settle_gain.astype(float).tolist() if settle_gain is not None else None,
            "settle_gain_sha256": data_sha256(settle_gain.astype(float).tolist()) if settle_gain is not None else None,
        },
        "runtime": runtime_metadata(),
        "git": source_git,
    }
    dump_json(metadata, metadata_out)
    print(f"Wrote {frames} frames to {out}; done_events={len(done_events)}")
    print(f"Wrote metadata to {metadata_out}")
    if args.fail_on_failure:
        if not bool(final_info.get("success", False)) or completed_steps != sim_steps:
            sys.exit(2)


if __name__ == "__main__":
    main()
