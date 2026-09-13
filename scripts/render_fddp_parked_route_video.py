#!/usr/bin/env python
"""Render a zoomed-out, state-faithful parked-route swing-up video."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles, wrap_angle
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    text_sha256,
    utc_timestamp,
)
from gcartpole.generalized_energy import hanging_lqr_gain
from gcartpole.ilqr import data_state
from gcartpole.modal import dimensionless_wrapped_state

try:
    from scripts.evaluate_fddp_two_expert import load_controller
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from evaluate_fddp_two_expert import load_controller
    from search_swingup_capture import lqr_action, lqr_gain


def hanging_target_action(
    env: NLinkCartPoleEnv, gain: np.ndarray, cart_target: float
) -> float:
    state = data_state(env.data)
    state[0] = float(env.data.qpos[0]) - float(cart_target)
    state[1] = wrap_angle(float(env.data.qpos[1]) - np.pi)
    if env.n > 1:
        state[2 : env.n + 1] = wrap_angle(
            np.asarray(env.data.qpos[2 : env.n + 1], dtype=np.float64)
        )
    return float(np.clip(-np.asarray(gain, dtype=np.float64) @ state, -1.0, 1.0))


def draw_frame(
    *,
    width: int,
    height: int,
    qpos: np.ndarray,
    action: float,
    time_seconds: float,
    phase: str,
    max_abs_angle: float,
    hinge_velocity_rms: float,
    upright_streak: float,
    max_upright_streak: float,
    cart_trace: list[float],
    rail_limit: float,
    n_links: int,
) -> np.ndarray:
    image = Image.new("RGB", (width, height), (9, 15, 30))
    draw = ImageDraw.Draw(image)
    left, right = 64, width - 64
    # Center the cart vertically so the full chain fits in both the hanging
    # and upright poses.  A fixed camera is important for a presentation
    # replay: the viewer should never lose the top or bottom links.
    track_y = int(height * 0.50)
    panel_y = height - 106
    world_half = max(rail_limit + 0.25, 3.25)
    world_min, world_max = -world_half, world_half

    def sx(value: float) -> int:
        return int(
            left
            + (float(value) - world_min) / (world_max - world_min) * (right - left)
        )

    draw.rectangle((0, 0, width, height), fill=(9, 15, 30))
    for grid_x in np.linspace(-rail_limit, rail_limit, 7):
        x = sx(float(grid_x))
        draw.line((x, 88, x, panel_y - 22), fill=(21, 32, 55), width=1)
        draw.text((x - 14, track_y + 30), f"{grid_x:g}", fill=(105, 123, 151))
    draw.text((64, 30), f"{n_links}-LINK SWING-UP", fill=(241, 245, 249))
    draw.text(
        (64, 57),
        "canonical noisy hanging start | state-faithful MuJoCo trajectory",
        fill=(148, 163, 184),
    )
    phase_color = {
        "park_hanging": (251, 146, 60),
        "swing_route_feedback": (56, 189, 248),
        "capture_lqr": (134, 239, 172),
    }[phase]
    phase_text = {
        "park_hanging": "SETTLE / PARK",
        "swing_route_feedback": "SWING-UP / FEEDBACK",
        "capture_lqr": "CAPTURE / LQR",
    }[phase]
    draw.text((width - 330, 34), phase_text, fill=phase_color)
    draw.text((width - 330, 59), f"t = {time_seconds:05.2f} s", fill=(226, 232, 240))

    rail_left, rail_right = sx(-rail_limit), sx(rail_limit)
    draw.line(
        (rail_left, track_y + 18, rail_right, track_y + 18),
        fill=(71, 85, 105),
        width=7,
    )
    draw.line(
        (rail_left, track_y + 8, rail_left, track_y + 28),
        fill=(239, 68, 68),
        width=4,
    )
    draw.line(
        (rail_right, track_y + 8, rail_right, track_y + 28),
        fill=(239, 68, 68),
        width=4,
    )
    if len(cart_trace) > 1:
        draw.line(
            [(sx(value), track_y + 2) for value in cart_trace],
            fill=(30, 64, 100),
            width=2,
        )

    cart_x = sx(float(qpos[0]))
    draw.rounded_rectangle(
        (cart_x - 38, track_y - 2, cart_x + 38, track_y + 34),
        radius=6,
        fill=(226, 232, 240),
        outline=(148, 163, 184),
        width=2,
    )
    draw.ellipse(
        (cart_x - 25, track_y + 24, cart_x - 10, track_y + 39),
        fill=(30, 41, 59),
    )
    draw.ellipse(
        (cart_x + 10, track_y + 24, cart_x + 25, track_y + 39),
        fill=(30, 41, 59),
    )

    absolute_angles = serial_absolute_angles(np.asarray(qpos[1:], dtype=np.float64))
    vertical_clearance = min(track_y - 116.0, panel_y - track_y - 30.0)
    link_pixels = max(22.0, vertical_clearance / max(1, n_links))
    px, py = float(cart_x), float(track_y)
    for angle in absolute_angles:
        ex = px + link_pixels * np.sin(float(angle))
        ey = py - link_pixels * np.cos(float(angle))
        draw.line((px, py, ex, ey), fill=phase_color, width=10)
        draw.ellipse(
            (int(px - 8), int(py - 8), int(px + 8), int(py + 8)),
            fill=(248, 250, 252),
            outline=(30, 41, 59),
            width=2,
        )
        px, py = ex, ey
    draw.ellipse(
        (int(px - 7), int(py - 7), int(px + 7), int(py + 7)),
        fill=(251, 191, 36),
        outline=(30, 41, 59),
        width=2,
    )

    draw.rectangle((0, panel_y, width, height), fill=(15, 23, 42))
    draw.text(
        (64, panel_y + 18),
        f"max angle  {max_abs_angle:6.3f} rad",
        fill=(226, 232, 240),
    )
    draw.text(
        (320, panel_y + 18),
        f"hinge rate  {hinge_velocity_rms:6.3f} rad/s",
        fill=(226, 232, 240),
    )
    draw.text(
        (650, panel_y + 18),
        f"action  {action:+6.3f}",
        fill=(226, 232, 240),
    )
    draw.text(
        (930, panel_y + 18),
        f"cart  {float(qpos[0]):+6.3f} m",
        fill=(226, 232, 240),
    )
    draw.text(
        (64, panel_y + 57),
        f"upright streak  {upright_streak:5.2f} s",
        fill=(134, 239, 172),
    )
    draw.text(
        (320, panel_y + 57),
        f"best hold  {max_upright_streak:5.2f} s",
        fill=(134, 239, 172),
    )
    draw.text(
        (650, panel_y + 57),
        f"rail  +/-{rail_limit:.2f} m",
        fill=(148, 163, 184),
    )
    draw.text(
        (930, panel_y + 57),
        f"{n_links} uniform links",
        fill=(148, 163, 184),
    )
    return np.asarray(image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup8_uniform.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--controller", default="runs/generalized_solver/n8_capture_fddp_feedback120.json"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=90901)
    parser.add_argument("--park-seconds", type=float, default=14.0)
    parser.add_argument("--cart-target", type=float, default=-0.15)
    parser.add_argument("--tracking-gain-scale", type=float, default=0.75)
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--fail-on-failure", action="store_true")
    args = parser.parse_args()
    if min(args.seconds, args.fps, args.width, args.height) <= 0:
        raise ValueError("seconds, fps, width, and height must be positive")
    if args.park_seconds < 0.0 or args.tracking_gain_scale < 0.0:
        raise ValueError("park duration and tracking gain must be nonnegative")
    if args.cart_target >= 0.0:
        raise ValueError("cart target must be negative")

    source_cfg = load_config(args.config)
    cfg = copy.deepcopy(source_cfg)
    cfg["env"]["init_mode"] = "hanging"
    cfg["env"]["episode_seconds"] = float(args.seconds)
    cfg["env"]["terminate_abs_angle"] = None
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"].setdefault("action_lqr_switch", {"enabled": False})["enabled"] = False
    if args.zero_noise:
        for key in (
            "init_angle_noise",
            "init_vel_noise",
            "init_cart_noise",
            "init_cart_vel_noise",
        ):
            cfg["env"][key] = 0.0
            cfg["env"][f"{key}_start"] = 0.0
            cfg["env"][f"{key}_end"] = 0.0

    repo_root = Path(__file__).resolve().parents[1]
    source_git = {
        key: value
        for key, value in git_metadata(repo_root, include_untracked=False).items()
        if key != "root"
    }
    spec = load_config(args.spec)
    controller_path = Path(args.controller)
    controller = load_controller(controller_path, int(cfg["env"]["n_links"]), spec)
    capture_gain = lqr_gain(
        cfg,
        progress=1.0,
        fd_eps=1.0e-7,
        control_cost=controller["lqr_control_cost"],
        q_weights=controller["lqr_weights"],
    )
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    _, reset_info = env.reset(seed=args.seed)
    settle_gain = hanging_lqr_gain(env, control_cost=1000.0)
    controls = controller["controls"]
    nominal_states = controller["nominal_states"]
    feedback_gains = controller["feedback_gains"]
    transform = controller["transform"]
    translated_nominal_states = nominal_states.copy()
    cart_nominal_shift: float | None = None
    park_steps = round(args.park_seconds / env.dt)
    sim_steps = min(env.max_steps, round(args.seconds / env.dt))
    output_path = Path(args.out)
    metadata_path = Path(args.metadata_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=args.fps, codec="libx264", quality=8)
    cart_trace: list[float] = []
    done_events: list[dict[str, object]] = []
    final_info: dict[str, object] = dict(reset_info)
    first_upright: float | None = None
    max_cart = abs(float(reset_info.get("x", env.data.qpos[0])))
    frames = 0
    completed_steps = 0
    try:
        for step in range(sim_steps):
            if step < park_steps:
                phase = "park_hanging"
                action = hanging_target_action(env, settle_gain, args.cart_target)
            else:
                route_step = step - park_steps
                coordinate_state = dimensionless_wrapped_state(
                    np.asarray(env.data.qpos, dtype=np.float64),
                    np.asarray(env.data.qvel, dtype=np.float64),
                    transform,
                )
                if cart_nominal_shift is None:
                    cart_nominal_shift = float(
                        coordinate_state[0] - nominal_states[0, 0]
                    )
                    translated_nominal_states[:, 0] += cart_nominal_shift
                if route_step < controls.size:
                    phase = "swing_route_feedback"
                    action = float(
                        np.clip(
                            controls[route_step]
                            + args.tracking_gain_scale
                            * feedback_gains[route_step]
                            @ (coordinate_state - translated_nominal_states[route_step]),
                            -1.0,
                            1.0,
                        )
                    )
                else:
                    phase = "capture_lqr"
                    action = lqr_action(
                        env,
                        capture_gain,
                        scale=controller["lqr_scale"],
                        cart_target=args.cart_target,
                    )
            _, _, terminated, truncated, info = env.step([action])
            completed_steps = step + 1
            final_info = dict(info)
            max_cart = max(max_cart, abs(float(info["x"])))
            cart_trace.append(float(info["x"]))
            cart_trace = cart_trace[-180:]
            if first_upright is None and bool(info.get("is_upright", False)):
                first_upright = float(completed_steps * env.dt)
            writer.append_data(
                draw_frame(
                    width=args.width,
                    height=args.height,
                    qpos=np.asarray(env.data.qpos, dtype=np.float64),
                    action=action,
                    time_seconds=completed_steps * env.dt,
                    phase=phase,
                    max_abs_angle=float(info["max_abs_angle"]),
                    hinge_velocity_rms=float(info["hinge_velocity_rms"]),
                    upright_streak=float(info["upright_streak_seconds"]),
                    max_upright_streak=float(info["max_upright_streak_seconds"]),
                    cart_trace=cart_trace,
                    rail_limit=float(env.rail_limit),
                    n_links=env.n,
                )
            )
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
    metadata = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_eight_link_video_candidate",
        "summary": "Reset-free state-faithful eight-link noisy hanging-start parked-route replay.",
        "visualization": "2d_state_faithful_zoomed_out",
        "video": file_metadata(output_path),
        "controller": controller["source"],
        "generated_xml_sha256": xml_sha256,
        "config": {
            "path": str(Path(args.config)),
            "resolved_sha256": data_sha256(source_cfg),
            "runtime_resolved_sha256": data_sha256(cfg),
            "overrides": [
                "env.episode_seconds",
                "env.action_lqr_residual.enabled",
                "env.action_lqr_switch.enabled",
            ],
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
            "max_cart_excursion": float(max_cart),
            "final_info": final_info,
        },
        "environment": {
            "n_links": int(cfg["env"]["n_links"]),
            "init_mode": "hanging",
            "force_limit": float(cfg["env"]["force_limit"]),
            "rail_limit": float(cfg["env"]["rail_limit"]),
            "success_upright_threshold": float(cfg["env"]["success_upright_threshold"]),
            "success_sustain_seconds": float(cfg["env"]["success_sustain_seconds"]),
            "initial_angle_noise": float(cfg["env"]["init_angle_noise"]),
            "initial_velocity_noise": float(cfg["env"]["init_vel_noise"]),
        },
        "launch": {
            "park_seconds": float(park_steps * env.dt),
            "cart_target_m": float(args.cart_target),
            "route_steps": int(controls.size),
            "route_seconds": float(controls.size * env.dt),
            "tracking_gain_scale": float(args.tracking_gain_scale),
            "capture_lqr_scale": float(controller["lqr_scale"]),
            "capture_cart_target_m": float(args.cart_target),
            "settle_control_cost": 1000.0,
        },
        "runtime": runtime_metadata(),
        "git": source_git,
    }
    dump_json(metadata, metadata_path)
    print(f"Wrote {frames} frames to {output_path}; success={final_info.get('success', False)}")
    print(f"Wrote metadata to {metadata_path}")
    if args.fail_on_failure:
        if not bool(final_info.get("success", False)) or completed_steps != sim_steps:
            sys.exit(2)


if __name__ == "__main__":
    main()
