#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import (
    data_sha256,
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
    from scripts.evaluate_ilqr_chain_basin import (
        select_feedback_gains,
        state_coordinates,
    )
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.refine_ilqr_capture_chain import source_trajectory
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_ilqr_capture import execute_controller
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from evaluate_ilqr_chain_basin import select_feedback_gains, state_coordinates
    from make_lqr_checkpoint import finite_difference_dynamics
    from refine_ilqr_capture_chain import source_trajectory
    from search_capture_sequence import fixed_state_cfg
    from search_ilqr_capture import execute_controller
    from search_swingup_capture import lqr_gain


def realized_trajectory(
    initial_state: dict[str, Any],
    trajectory: list[dict[str, Any]],
    transform: np.ndarray,
    horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(trajectory) < horizon_steps:
        raise ValueError("successful rollout ended before the controller horizon")
    controls = np.asarray(
        [row["action"] for row in trajectory[:horizon_steps]], dtype=np.float64
    )
    states = [state_coordinates(initial_state, transform)]
    for row in trajectory[:horizon_steps]:
        states.append(state_coordinates(row, transform))
    return controls, np.asarray(states, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recenter a successful time-varying controller on its realized rollout"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument(
        "--feedback-gain-source", choices=("applied", "solver"), default="solver"
    )
    parser.add_argument("--feedback-gain-scale", type=float, default=1.0)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--handoff-angle-abs", type=float, default=0.15)
    parser.add_argument("--handoff-cart-velocity-abs", type=float, default=0.5)
    parser.add_argument("--handoff-hinge-velocity-rms", type=float, default=0.75)
    parser.add_argument("--minimum-upright-hold", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if (
        min(
            args.feedback_gain_scale,
            args.lqr_scale,
            args.handoff_lyapunov,
            args.handoff_cart_abs,
            args.handoff_angle_abs,
            args.handoff_cart_velocity_abs,
            args.handoff_hinge_velocity_rms,
            args.minimum_upright_hold,
        )
        <= 0.0
    ):
        raise ValueError("controller scales and thresholds must be positive")

    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    states = dataset["states"]
    if not 0 <= args.state_index < len(states):
        raise IndexError("state index is outside the dataset")
    selected_state = states[args.state_index]

    controller_path = Path(args.controller)
    source_payload = json.loads(controller_path.read_text(encoding="utf-8"))
    source_controls, source_states, applied_gains = source_trajectory(source_payload)
    feedback_gains = select_feedback_gains(
        source_payload["controller"],
        applied_gains,
        args.feedback_gain_source,
        args.feedback_gain_scale,
    )
    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    cfg = fixed_state_cfg(
        base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"])
    )
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    distribution = load_config(args.spec)["distribution"]
    transform = dimensionless_absolute_transform(
        int(cfg["env"]["n_links"]),
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix,
        input_matrix,
        gain,
        transform,
        feedback_scale=args.lqr_scale,
    )
    result = execute_controller(
        cfg,
        seed=args.seed,
        controls=source_controls,
        nominal_states=source_states,
        feedback_gains=feedback_gains,
        gain=gain,
        lqr_scale=args.lqr_scale,
        transform=transform,
        lyapunov=lyapunov,
        handoff_lyapunov=args.handoff_lyapunov,
        handoff_cart_abs=args.handoff_cart_abs,
        handoff_angle_abs=args.handoff_angle_abs,
        handoff_cart_velocity_abs=args.handoff_cart_velocity_abs,
        handoff_hinge_velocity_rms=args.handoff_hinge_velocity_rms,
        tracking_mode="recenter_source_tracking",
    )
    if not result["success"]:
        raise RuntimeError(
            "source controller does not succeed from the requested state"
        )
    if result["max_upright_streak_seconds"] < args.minimum_upright_hold:
        raise RuntimeError("source controller misses the minimum upright hold")
    controls, nominal_states = realized_trajectory(
        selected_state, result["trajectory"], transform, source_controls.size
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Realized trajectory recentered from one successful feedback rollout; not P1 evidence.",
        "state_index": int(args.state_index),
        "selected_state": selected_state,
        "seed": int(args.seed),
        "controller": {
            "type": "recentered_time_varying_feedback_then_lqr",
            "source_controller": file_metadata(controller_path),
            "feedback_gain_source": args.feedback_gain_source,
            "feedback_gain_scale": float(args.feedback_gain_scale),
            "horizon_steps": int(controls.size),
            "controls": controls.astype(float).tolist(),
            "feedback_gains": feedback_gains.astype(float).tolist(),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "handoff_angle_abs": float(args.handoff_angle_abs),
            "handoff_cart_velocity_abs": float(args.handoff_cart_velocity_abs),
            "handoff_hinge_velocity_rms": float(args.handoff_hinge_velocity_rms),
        },
        "search": {
            "method": "successful_feedback_rollout_recenter",
            "nominal_coordinate_states": nominal_states.astype(float).tolist(),
        },
        "result": result,
        "lyapunov": {
            "coordinate_source": file_metadata(args.spec),
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "evidence": {
            "config": {"path": args.config, "resolved_sha256": data_sha256(cfg)},
            "dataset": file_metadata(dataset_path),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"state={args.state_index} success={result['success']} "
        f"hold={result['max_upright_streak_seconds']:.3f}s "
        f"cart={result['max_cart_excursion']:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
