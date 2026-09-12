#!/usr/bin/env python
"""Render a saved exact-MuJoCo FDDP swing plus LQR capture rollout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
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
    from scripts.evaluate_fddp_two_expert import load_controller
    from scripts.search_capture_sequence import fixed_state_cfg, load_state
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from evaluate_fddp_two_expert import load_controller
    from search_capture_sequence import fixed_state_cfg, load_state
    from search_swingup_capture import lqr_action, lqr_gain


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a saved two-expert seven-link rollout")
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
    parser.add_argument("--seed", type=int, default=20732)
    parser.add_argument("--fail-on-failure", action="store_true")
    args = parser.parse_args()
    if args.seconds <= 0.0 or args.fps < 1:
        raise ValueError("seconds and fps must be positive")

    base_cfg = load_config(args.config)
    spec = load_config(args.spec)
    controller_path = Path(args.controller)
    controller = load_controller(controller_path, int(base_cfg["env"]["n_links"]), spec)
    controller_payload = json.loads(controller_path.read_text(encoding="utf-8"))
    if args.state_json is None:
        selected_state = controller_payload.get("selected_state")
        if not isinstance(selected_state, dict):
            raise ValueError("controller artifact has no selected_state; pass --state-json")
        state_source = controller_path
    else:
        selected_state, _ = load_state(args.state_json, args.state_index)
        state_source = Path(args.state_json)
    cfg = fixed_state_cfg(base_cfg, selected_state, args.seconds)
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"].setdefault("action_lqr_switch", {"enabled": False})["enabled"] = False
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    env.reset(seed=args.seed)
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    controls = controller["controls"]
    nominal_states = controller["nominal_states"]
    feedback_gains = controller["feedback_gains"]
    transform = controller["transform"]
    output_path = Path(args.out)
    metadata_path = Path(args.metadata_out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(output_path), fps=args.fps, codec="libx264", quality=8)
    sim_steps = min(env.max_steps, int(round(args.seconds / env.dt)))
    render_every = max(1, int(round(1.0 / (args.fps * env.dt))))
    frames = 0
    completed_steps = 0
    final_info: dict[str, object] = {}
    done_events: list[dict[str, object]] = []
    first_upright: float | None = None
    max_cart = 0.0
    try:
        for step in range(sim_steps):
            coordinate_state = dimensionless_wrapped_state(
                env.data.qpos, env.data.qvel, transform
            )
            if step < controls.size:
                action = float(
                    np.clip(
                        controls[step]
                        + feedback_gains[step] @ (coordinate_state - nominal_states[step]),
                        -1.0,
                        1.0,
                    )
                )
                mode = "swing_feedback"
            else:
                action = lqr_action(
                    env,
                    gain,
                    scale=controller["lqr_scale"],
                    cart_target=0.0,
                )
                mode = "capture_lqr"
            _, _, terminated, truncated, info = env.step([action])
            completed_steps = step + 1
            final_info = dict(info)
            max_cart = max(max_cart, abs(float(info["x"])))
            if first_upright is None and bool(info.get("is_upright", False)):
                first_upright = float(completed_steps * env.dt)
            if step % render_every == 0 or step == sim_steps - 1:
                frame = env.render_rgb(width=args.width, height=args.height)
                writer.append_data(np.asarray(frame))
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
                        "controller_mode": mode,
                        "x": float(info.get("x", np.nan)),
                        "max_abs_angle": float(info.get("max_abs_angle", np.nan)),
                        "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
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
        "claim_status": "exact_start_proof_not_canonical_noisy_gate",
        "summary": "Reset-free exact-MuJoCo seven-link swing-up and hold rendered from the saved two-expert controller.",
        "video": file_metadata(output_path),
        "controller": controller["source"],
        "state_source": file_metadata(state_source),
        "generated_xml_sha256": xml_sha256,
        "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(cfg)},
        "render": {
            "requested_seconds": float(args.seconds),
            "simulated_seconds": float(completed_steps * 0.02),
            "completed_requested_steps": completed_steps == sim_steps,
            "dt": 0.02,
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
            "init_mode": "fixed_state",
            "force_limit": float(cfg["env"]["force_limit"]),
            "rail_limit": float(cfg["env"]["rail_limit"]),
            "success_upright_threshold": float(cfg["env"]["success_upright_threshold"]),
            "success_sustain_seconds": float(cfg["env"]["success_sustain_seconds"]),
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(metadata, metadata_path)
    print(f"Wrote {frames} frames to {output_path}; done_events={len(done_events)}")
    print(f"Wrote metadata to {metadata_path}")
    if args.fail_on_failure:
        good = bool(final_info.get("success", False)) and completed_steps == sim_steps
        if not good or any(bool(event["terminated"]) for event in done_events):
            sys.exit(2)


if __name__ == "__main__":
    main()
