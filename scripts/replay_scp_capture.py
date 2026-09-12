#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import (
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_ilqr_capture import execute_controller
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_capture_sequence import fixed_state_cfg
    from search_ilqr_capture import execute_controller
    from search_swingup_capture import lqr_gain


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Strictly replay a saved local-SCP capture controller"
    )
    parser.add_argument("--controller", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--expect-success", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    controller_path = Path(args.controller)
    payload = json.loads(controller_path.read_text(encoding="utf-8"))
    controller = payload["controller"]
    if controller.get("type") != "local_scp_exact_mujoco_then_lqr":
        raise ValueError("controller is not a local-SCP capture artifact")
    config_path = payload["evidence"]["config"]["path"]
    spec_path = payload["lyapunov"]["coordinate_source"]["path"]
    base_cfg = load_config(config_path)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    cfg = fixed_state_cfg(
        base_cfg,
        payload["selected_state"],
        float(base_cfg["env"]["episode_seconds"]),
    )
    distribution = load_config(spec_path)["distribution"]
    scales = StateScales(
        float(distribution["cart_position_abs_max"]),
        float(distribution["absolute_link_angle_abs_max"]),
        float(distribution["cart_velocity_abs_max"]),
        float(distribution["hinge_velocity_rms_max"]),
    )
    n_links = int(cfg["env"]["n_links"])
    transform = dimensionless_absolute_transform(n_links, scales)
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix,
        input_matrix,
        gain,
        transform,
        feedback_scale=float(controller["lqr_scale"]),
    )
    controls = np.asarray(controller["controls"], dtype=np.float64)
    nominal_states = np.asarray(
        payload["search"]["nominal_coordinate_states"], dtype=np.float64
    )
    feedback_gains = np.asarray(controller["feedback_gains"], dtype=np.float64)
    nx = transform.shape[0]
    if nominal_states.shape != (controls.size + 1, nx):
        raise ValueError("nominal state horizon does not match controls")
    if feedback_gains.shape != (controls.size, nx):
        raise ValueError("feedback-gain horizon does not match controls")

    result = execute_controller(
        cfg,
        seed=int(payload["seed"]),
        controls=controls,
        nominal_states=nominal_states,
        feedback_gains=feedback_gains,
        gain=gain,
        lqr_scale=float(controller["lqr_scale"]),
        transform=transform,
        lyapunov=lyapunov,
        handoff_lyapunov=float(controller["handoff_lyapunov"]),
        handoff_cart_abs=float(controller["handoff_cart_abs"]),
        handoff_angle_abs=float(controller["handoff_angle_abs"]),
        handoff_cart_velocity_abs=float(controller["handoff_cart_velocity_abs"]),
        handoff_hinge_velocity_rms=float(controller["handoff_hinge_velocity_abs"]),
        tracking_mode="local_scp_tracking_replay",
    )
    replay = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_p1_evidence": True,
        "controller": file_metadata(controller_path),
        "state_index": int(payload["state_index"]),
        "seed": int(payload["seed"]),
        "closed_loop_spectral_radius": float(spectral_radius),
        "result": result,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(replay, args.out)
    print(
        f"success={result['success']} latched={result['latched']} "
        f"handoff={result.get('first_handoff_step')} "
        f"min_v={result['minimum_lyapunov']:.3f} "
        f"hold={result['max_upright_streak_seconds']:.3f}s "
        f"cart={result['max_cart_excursion']:.3f}"
    )
    print(f"Wrote {args.out}")
    if bool(result["success"]) != bool(args.expect_success):
        raise SystemExit("strict replay outcome did not match expectation")


if __name__ == "__main__":
    main()
