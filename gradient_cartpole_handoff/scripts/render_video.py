#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v2 as imageio
import mlx.core as mx
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, text_sha256, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, load_model, sample_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Render MP4 video from trained checkpoint")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--progress", type=float, default=1.0, help="1.0 = final uniform morphology")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--metadata-out", default=None, help="Optional JSON path for video evidence metadata")
    parser.add_argument("--no-reset-on-done", action="store_true", help="Stop recording at the first termination/truncation")
    parser.add_argument("--fail-on-reset", action="store_true", help="Exit nonzero if the video required a reset or terminated")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    obs, _ = env.reset()
    xml_sha256 = text_sha256(env.xml)

    ppo = cfg["ppo"]
    model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0], list(ppo.get("hidden_sizes", [256, 256])), float(ppo.get("action_std_init", 0.7)))
    mx.eval(model.parameters())
    load_model(model, args.checkpoint)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264", quality=8)
    sim_steps = int(args.seconds / env.dt)
    render_every = max(1, int(round(1.0 / (args.fps * env.dt))))
    frames = 0
    reset_count = 0
    done_events = []
    final_info = {}
    completed_steps = 0
    stopped_early = False
    try:
        for step in range(sim_steps):
            action, _, _ = sample_action(model, obs[None, :], deterministic=True)
            obs, reward, term, trunc, info = env.step(action[0])
            completed_steps = step + 1
            final_info = dict(info)
            if step % render_every == 0:
                frame = env.render_rgb(width=args.width, height=args.height)
                writer.append_data(np.asarray(frame))
                frames += 1
            if term or trunc:
                event = {
                    "step": int(step + 1),
                    "time_seconds": float((step + 1) * env.dt),
                    "terminated": bool(term),
                    "truncated": bool(trunc),
                    "success": bool(info.get("success", False)),
                    "termination_reason": info.get("termination_reason"),
                    "x": float(info.get("x", np.nan)),
                    "max_abs_angle": float(info.get("max_abs_angle", np.nan)),
                    "time_to_first_upright": info.get("time_to_first_upright"),
                    "time_to_capture": info.get("time_to_capture"),
                    "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
                    "final_upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
                    "max_cart_excursion": float(info.get("max_cart_excursion", 0.0)),
                }
                done_events.append(event)
                if step < sim_steps - 1:
                    if args.no_reset_on_done:
                        stopped_early = True
                        break
                    reset_count += 1
                    obs, _ = env.reset()
    finally:
        writer.close()
        env.close()

    metadata = {
        "generated_at": utc_timestamp(),
        "video": file_metadata(out),
        "checkpoint": file_metadata(args.checkpoint),
        "generated_xml_sha256": xml_sha256,
        "config": {
            "path": str(Path(args.config)),
            "resolved_sha256": data_sha256(cfg),
            "overrides": list(args.override),
        },
        "render": {
            "requested_seconds": float(args.seconds),
            "simulated_seconds": float(completed_steps * env.dt),
            "completed_requested_steps": completed_steps == sim_steps,
            "dt": float(env.dt),
            "action_frequency_hz": float(1.0 / env.dt),
            "fps": int(args.fps),
            "frames": int(frames),
            "width": int(args.width),
            "height": int(args.height),
            "progress": float(args.progress),
            "seed": int(args.seed),
            "reset_count": int(reset_count),
            "stopped_early": bool(stopped_early),
            "done_events": done_events,
            "final_info": final_info,
        },
        "environment": {
            "n_links": int(cfg["env"]["n_links"]),
            "observation_dim": int(env.observation_space.shape[0]),
            "action_dim": int(env.action_space.shape[0]),
            "init_mode": str(cfg["env"].get("init_mode", "upright")),
            "init_angle_noise": float(cfg["env"].get("init_angle_noise", 0.02)),
            "init_vel_noise": float(cfg["env"].get("init_vel_noise", 0.01)),
            "episode_seconds": float(cfg["env"]["episode_seconds"]),
            "force_limit": float(cfg["env"]["force_limit"]),
            "rail_limit": float(cfg["env"]["rail_limit"]),
            "terminate_abs_angle": cfg["env"].get("terminate_abs_angle", 1.25),
            "success_upright_threshold": float(cfg["env"].get("success_upright_threshold", cfg["env"].get("reward", {}).get("upright_threshold", 0.10))),
            "success_sustain_seconds": float(cfg["env"].get("success_sustain_seconds", 0.0)),
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    if args.metadata_out:
        dump_json(metadata, Path(args.metadata_out))

    print(f"Wrote {frames} frames to {out}; resets={reset_count}; done_events={len(done_events)}")
    if args.metadata_out:
        print(f"Wrote metadata to {args.metadata_out}")

    had_termination = any(event["terminated"] for event in done_events)
    had_unsuccessful_done = any(not event["success"] for event in done_events)
    if args.fail_on_reset and (reset_count > 0 or had_termination or had_unsuccessful_done or stopped_early):
        sys.exit(2)


if __name__ == "__main__":
    main()
