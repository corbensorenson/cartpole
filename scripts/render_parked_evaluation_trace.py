#!/usr/bin/env python
"""Render the actual parked-evaluator trajectory, preserving its executed policy."""
import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import file_metadata, runtime_metadata, text_sha256, utc_timestamp
try:
    from scripts.render_fddp_parked_route_video import draw_frame
except ModuleNotFoundError:
    from render_fddp_parked_route_video import draw_frame


def validate_trace(payload, episode_index, *, success_threshold=.15, required_hold=5.):
    episode = payload["episode_results"][episode_index]
    rows = episode["trajectory"]
    n = int(payload["n_links"])
    environment = payload["environment"]
    dt = 1. / environment["action_frequency_hz"]
    expected = round(environment["episode_seconds"] / dt)
    if (not episode["success"] or episode["termination_reason"] != "time_limit"
            or episode["final_info"].get("simulation_error") is not None
            or len(rows) != expected):
        raise ValueError("video requires a successful complete clean evaluator episode")
    if not np.allclose([r["time_seconds"] for r in rows], np.arange(1, expected+1)*dt,
                       rtol=0, atol=1e-9):
        raise ValueError("trace must preserve every canonical control step")
    if [r["step"] for r in rows] != list(range(1, expected+1)):
        raise ValueError("trace step indices must be consecutive")
    streak = maximum_streak = 0
    for row in rows:
        q = np.asarray(row["qpos"], dtype=float)
        v = np.asarray(row["qvel"], dtype=float)
        if (q.shape != (n+1,) or v.shape != (n+1,)
                or not np.all(np.isfinite(np.r_[q, v, row["action"]]))
                or abs(row["action"]) > 1.
                or abs(q[0]) > environment["rail_limit"]
                or not np.isclose(q[0], row["x"], rtol=0, atol=1e-10)):
            raise ValueError("trace has invalid physical states or actuator/rail values")
        if row["phase"] not in ("park_hanging", "swing_route_feedback", "capture_lqr"):
            raise ValueError("unknown executed phase")
        angle = float(np.max(abs(serial_absolute_angles(q[1:]))))
        upright = angle < success_threshold
        if (not np.isclose(angle, row["max_abs_angle"], rtol=0, atol=1e-10)
                or upright != row["is_upright"]):
            raise ValueError("reported upright status differs from the physical pose")
        streak = streak+1 if upright else 0
        maximum_streak = max(maximum_streak, streak)
    if (maximum_streak*dt+1e-12 < required_hold
            or not np.isclose(maximum_streak*dt, episode["max_upright_streak_seconds"],
                              rtol=0, atol=1e-9)):
        raise ValueError("physical trace does not reproduce the claimed upright hold")
    return episode, rows, dt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--metadata-out", required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    output, metadata = Path(args.out), Path(args.metadata_out)
    if output.exists() or metadata.exists():
        raise FileExistsError("refusing to overwrite video evidence")
    payload = json.loads(Path(args.evaluation).read_text())
    cfg = load_config(payload["config"]["path"])
    episode, rows, dt = validate_trace(payload, args.episode_index,
        success_threshold=cfg["env"]["success_upright_threshold"],
        required_hold=cfg["env"]["success_sustain_seconds"])
    if file_metadata(payload["config"]["path"])["sha256"] != payload["config"]["sha256"]:
        raise ValueError("source configuration has changed since evaluation")
    probe = NLinkCartPoleEnv(cfg, progress=1., seed=0)
    try:
        lengths = probe.morphology.lengths.copy()
        if text_sha256(probe.xml) != payload["generated_xml_sha256"]:
            raise ValueError("source geometry differs from the evaluated plant")
        if not np.allclose(lengths, lengths[0], rtol=0, atol=1e-12):
            raise ValueError("this renderer requires the uniform benchmark chain")
        total_length = float(np.sum(lengths))
    finally:
        probe.close()
    fps = round(1./dt)
    if not np.isclose(fps*dt, 1., rtol=0, atol=1e-12):
        raise ValueError("video requires an integer canonical action frequency")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    streak = 0.
    cart_trace = []
    with imageio.get_writer(str(output), fps=fps, codec="libx264", quality=8,
                           ffmpeg_params=["-threads", "1"]) as writer:
        for row in rows:
            streak = streak+dt if row["is_upright"] else 0.
            cart_trace = (cart_trace+[row["x"]])[-180:]
            writer.append_data(draw_frame(
                width=args.width, height=args.height, qpos=np.asarray(row["qpos"]),
                action=row["action"], time_seconds=row["time_seconds"], phase=row["phase"],
                max_abs_angle=row["max_abs_angle"], hinge_velocity_rms=row["hinge_velocity_rms"],
                upright_streak=streak, max_upright_streak=row["max_upright_streak_seconds"],
                cart_trace=cart_trace, rail_limit=payload["environment"]["rail_limit"],
                n_links=payload["n_links"],
                total_length=total_length,
            ))
    dump_json(dict(schema_version=1, generated_at=utc_timestamp(),
        not_solution=True, claim_status="state_faithful_evaluator_trace_video",
        source_evaluation=file_metadata(args.evaluation), source_controller=payload["controller"],
        video=file_metadata(output), runtime=runtime_metadata(),
        generated_xml_sha256=payload["generated_xml_sha256"],
        resolved_config_sha256=payload["resolved_config_sha256"],
        render=dict(seed=episode["seed"], episode_index=args.episode_index,
                    frames=len(rows), fps=fps, simulated_seconds=len(rows)*dt,
                    width=args.width, height=args.height, resets_after_initialization=0,
                    reset_count=0, completed_requested_steps=True,
                    final_info=episode["final_info"],
                    done_events=[dict(step=len(rows), termination_reason=episode["termination_reason"])],
                    total_length=total_length, camera_scale="isotropic_fixed_full_chain",
                    dynamics_reexecuted=False, success=episode["success"]),
        executed_policy=dict(park_seconds=payload["park_seconds"], cart_target=payload["cart_target"],
            settle_control_cost=payload.get("settle_control_cost", 1000.0),
            settle_cart_position_cost=payload.get("settle_cart_position_cost", 0.1),
            settle_cart_velocity_cost=payload.get("settle_cart_velocity_cost", 0.1),
            settle_gain_sha256=payload.get("settle_gain_sha256"),
            tracking_gain_scale=payload["tracking_gain_scale"], phase_adaptive=payload["phase_adaptive"],
            phase_window=payload["phase_window"], capture_gate=payload.get("capture_gate"),
            capture_metric_sha256=payload.get("capture_metric_sha256"),
            first_handoff_time=episode["first_handoff_time"], handoff_reason=episode["handoff_reason"]),
        note="Every frame depicts the evaluator's recorded physical post-step state. No second controller, resimulation, state interpolation or runtime projection is used. This development video alone does not promote a count."), metadata)
    print(f"Rendered {len(rows)} physical-state frames at {fps} fps to {output}")


if __name__ == "__main__":
    main()
