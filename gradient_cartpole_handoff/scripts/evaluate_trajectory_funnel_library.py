#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.capture_envelope import validate_capture_config, validate_capture_states
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
    from scripts.evaluate_ilqr_chain_basin import result_summary
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.refine_ilqr_capture_chain import source_trajectory
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_ilqr_capture import execute_controller
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from evaluate_ilqr_chain_basin import result_summary
    from make_lqr_checkpoint import finite_difference_dynamics
    from refine_ilqr_capture_chain import source_trajectory
    from search_capture_sequence import fixed_state_cfg
    from search_ilqr_capture import execute_controller
    from search_swingup_capture import lqr_gain


def selection_key(row: dict[str, Any]) -> tuple[Any, ...]:
    result = row["result"]
    return (
        not bool(result["success"]),
        -float(result["max_upright_streak_seconds"]),
        float(result["minimum_lyapunov"]),
        float(result["max_cart_excursion"]),
        int(row["controller_index"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure fixed-cohort oracle coverage of a trajectory-funnel library"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--controller", action="append", required=True)
    parser.add_argument("--gain-scale", action="append", type=float, default=[])
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--seed", type=int, default=72001)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--handoff-angle-abs", type=float, default=0.15)
    parser.add_argument("--handoff-cart-velocity-abs", type=float, default=0.5)
    parser.add_argument("--handoff-hinge-velocity-rms", type=float, default=0.75)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    gain_scales = args.gain_scale or [1.0] * len(args.controller)
    if len(gain_scales) != len(args.controller):
        raise ValueError("provide one gain scale per controller")
    if (
        args.limit <= 0
        or min(
            gain_scales
            + [
                args.lqr_scale,
                args.handoff_lyapunov,
                args.handoff_cart_abs,
                args.handoff_angle_abs,
                args.handoff_cart_velocity_abs,
                args.handoff_hinge_velocity_rms,
            ]
        )
        <= 0.0
    ):
        raise ValueError("counts, scales, and thresholds must be positive")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    spec = load_config(args.spec)
    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_errors = validate_capture_states(dataset, spec, args.split)
    if dataset_errors:
        raise ValueError(f"invalid capture envelope: {dataset_errors[:20]}")
    config_errors = validate_capture_config(cfg, spec)
    if config_errors:
        raise ValueError(
            f"config violates frozen capture benchmark: {config_errors[:20]}"
        )
    states = dataset["states"][: min(args.limit, len(dataset["states"]))]
    probe_cfg = fixed_state_cfg(cfg, states[0], float(cfg["env"]["episode_seconds"]))
    gain = lqr_gain(probe_cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        int(cfg["env"]["n_links"]),
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    state_matrix, input_matrix = finite_difference_dynamics(probe_cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix,
        input_matrix,
        gain,
        transform,
        feedback_scale=args.lqr_scale,
    )
    controllers = []
    for index, (path_text, gain_scale) in enumerate(
        zip(args.controller, gain_scales, strict=True)
    ):
        path = Path(path_text)
        payload = json.loads(path.read_text(encoding="utf-8"))
        controls, nominal_states, feedback_gains = source_trajectory(payload)
        controllers.append(
            {
                "index": index,
                "path": path,
                "metadata": file_metadata(path),
                "gain_scale": float(gain_scale),
                "controls": controls,
                "nominal_states": nominal_states,
                "feedback_gains": gain_scale * feedback_gains,
            }
        )

    episodes = []
    controller_selection_counts = np.zeros(len(controllers), dtype=np.int64)
    oracle_coverage_counts = np.zeros(len(controllers), dtype=np.int64)
    for state_index, state in enumerate(states):
        state_cfg = fixed_state_cfg(cfg, state, float(cfg["env"]["episode_seconds"]))
        candidates = []
        for controller in controllers:
            result = result_summary(
                execute_controller(
                    state_cfg,
                    seed=args.seed + state_index,
                    controls=controller["controls"],
                    nominal_states=controller["nominal_states"],
                    feedback_gains=controller["feedback_gains"],
                    gain=gain,
                    lqr_scale=args.lqr_scale,
                    transform=transform,
                    lyapunov=lyapunov,
                    handoff_lyapunov=args.handoff_lyapunov,
                    handoff_cart_abs=args.handoff_cart_abs,
                    handoff_angle_abs=args.handoff_angle_abs,
                    handoff_cart_velocity_abs=args.handoff_cart_velocity_abs,
                    handoff_hinge_velocity_rms=args.handoff_hinge_velocity_rms,
                    tracking_mode="trajectory_funnel_library",
                )
            )
            if result["success"]:
                oracle_coverage_counts[controller["index"]] += 1
            candidates.append(
                {"controller_index": controller["index"], "result": result}
            )
        selected = min(candidates, key=selection_key)
        controller_selection_counts[selected["controller_index"]] += 1
        episodes.append(
            {
                "state_index": state_index,
                "state_id": state.get("state_id"),
                "seed": args.seed + state_index,
                "selected_controller_index": selected["controller_index"],
                "result": selected["result"],
                "candidates": candidates,
            }
        )

    successes = [row["result"]["success"] for row in episodes]
    holds = [row["result"]["max_upright_streak_seconds"] for row in episodes]
    successful_rail_hits = sum(
        row["result"]["success"]
        and row["result"]["termination_reason"] == "rail_violation"
        for row in episodes
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Fixed-cohort exhaustive model-based trajectory-funnel coverage; not a learned selector or P1 evidence.",
        "episodes": episodes,
        "metrics": {
            "episode_count": len(episodes),
            "success_count": int(sum(successes)),
            "success_rate": float(np.mean(successes)),
            "median_max_upright_streak_seconds": float(np.median(holds)),
            "successful_rail_hit_count": int(successful_rail_hits),
            "strict_ten_second_count": int(sum(hold >= 10.0 for hold in holds)),
        },
        "controllers": [
            {
                "index": controller["index"],
                "artifact": controller["metadata"],
                "gain_scale": controller["gain_scale"],
                "selected_count": int(controller_selection_counts[controller["index"]]),
                "oracle_success_count": int(
                    oracle_coverage_counts[controller["index"]]
                ),
            }
            for controller in controllers
        ],
        "settings": {
            "selection_rule": "simulate_all_then_rank_success_hold_lyapunov_cart",
            "limit": int(args.limit),
            "seed": int(args.seed),
            "lqr_scale": float(args.lqr_scale),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "handoff_angle_abs": float(args.handoff_angle_abs),
            "handoff_cart_velocity_abs": float(args.handoff_cart_velocity_abs),
            "handoff_hinge_velocity_rms": float(args.handoff_hinge_velocity_rms),
        },
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
    metrics = payload["metrics"]
    print(
        f"success={metrics['success_count']}/{metrics['episode_count']} "
        f"rate={metrics['success_rate']:.4f} "
        f"strict10={metrics['strict_ten_second_count']} "
        f"median_hold={metrics['median_max_upright_streak_seconds']:.3f}s"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
