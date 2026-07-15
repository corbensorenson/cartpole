#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
    dimensionless_wrapped_state,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.refine_ilqr_capture_chain import source_trajectory
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_ilqr_capture import execute_controller
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from refine_ilqr_capture_chain import source_trajectory
    from search_capture_sequence import fixed_state_cfg
    from search_ilqr_capture import execute_controller
    from search_swingup_capture import lqr_gain


def parse_radii(value: str) -> list[float]:
    radii = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not radii or any(radius <= 0.0 for radius in radii):
        raise ValueError("perturbation radii must be a nonempty list of positive values")
    return radii


def state_coordinates(state: dict[str, Any], transform: np.ndarray) -> np.ndarray:
    return dimensionless_wrapped_state(
        np.asarray(state["qpos"], dtype=np.float64),
        np.asarray(state["qvel"], dtype=np.float64),
        transform,
    )


def result_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: result.get(key)
        for key in (
            "success",
            "termination_reason",
            "max_upright_streak_seconds",
            "max_low_momentum_upright_streak_seconds",
            "max_cart_excursion",
            "minimum_lyapunov",
            "latched",
            "first_handoff_step",
            "first_handoff_time",
        )
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0, "success_count": 0, "success_rate": 0.0}
    success_count = sum(bool(row["result"]["success"]) for row in rows)
    latch_count = sum(bool(row["result"]["latched"]) for row in rows)
    return {
        "count": len(rows),
        "success_count": int(success_count),
        "success_rate": float(success_count / len(rows)),
        "latch_count": int(latch_count),
        "latch_rate": float(latch_count / len(rows)),
        "median_minimum_lyapunov": float(
            np.median([row["result"]["minimum_lyapunov"] for row in rows])
        ),
        "maximum_cart_excursion": float(
            max(row["result"]["max_cart_excursion"] for row in rows)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure the local feedback basin of a saved reset-free iLQR chain"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--nearest-count", type=int, default=23)
    parser.add_argument("--perturbation-radii", default="0.005,0.01,0.02,0.05,0.1,0.2,0.4,0.8")
    parser.add_argument("--samples-per-radius", type=int, default=32)
    parser.add_argument("--seed", type=int, default=77001)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    radii = parse_radii(args.perturbation_radii)
    if args.nearest_count < 0 or args.samples_per_radius < 1:
        raise ValueError("nearest count must be nonnegative and samples per radius positive")
    if min(args.lqr_scale, args.handoff_lyapunov, args.handoff_cart_abs) <= 0.0:
        raise ValueError("controller scales and handoff thresholds must be positive")

    controller_path = Path(args.controller)
    controller_payload = json.loads(controller_path.read_text(encoding="utf-8"))
    controls, nominal_states, feedback_gains = source_trajectory(controller_payload)
    selected_state = controller_payload["selected_state"]
    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    source_cfg = fixed_state_cfg(
        base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"])
    )
    gain = lqr_gain(source_cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    spec = load_config(args.spec)
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        int(source_cfg["env"]["n_links"]),
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    inverse_transform = np.linalg.inv(transform)
    state_matrix, input_matrix = finite_difference_dynamics(source_cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix,
        input_matrix,
        gain,
        transform,
        feedback_scale=args.lqr_scale,
    )

    def evaluate(state: dict[str, Any], seed: int) -> dict[str, Any]:
        cfg = fixed_state_cfg(base_cfg, state, float(base_cfg["env"]["episode_seconds"]))
        return result_summary(
            execute_controller(
                cfg,
                seed=seed,
                controls=controls,
                nominal_states=nominal_states,
                feedback_gains=feedback_gains,
                gain=gain,
                lqr_scale=args.lqr_scale,
                transform=transform,
                lyapunov=lyapunov,
                handoff_lyapunov=args.handoff_lyapunov,
                handoff_cart_abs=args.handoff_cart_abs,
                tracking_mode="ilqr_chain_basin_replay",
            )
        )

    source_result = evaluate(selected_state, args.seed)
    source_coordinate = state_coordinates(selected_state, transform)
    dataset_payload = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    neighbors: list[tuple[float, int, dict[str, Any]]] = []
    for index, state in enumerate(dataset_payload["states"]):
        distance = float(np.linalg.norm(state_coordinates(state, transform) - source_coordinate))
        if distance > 1e-12:
            neighbors.append((distance, index, state))
    neighbors.sort(key=lambda row: row[0])
    nearest_rows: list[dict[str, Any]] = []
    for rank, (distance, index, state) in enumerate(neighbors[: args.nearest_count], start=1):
        nearest_rows.append(
            {
                "rank": rank,
                "dataset_index": index,
                "state_id": state.get("state_id"),
                "normalized_distance": distance,
                "result": evaluate(state, args.seed + index + 1),
            }
        )

    rng = np.random.default_rng(args.seed)
    directions = rng.normal(size=(args.samples_per_radius, source_coordinate.size))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    perturbation_groups: list[dict[str, Any]] = []
    nq = len(selected_state["qpos"])
    for radius_index, radius in enumerate(radii):
        rows: list[dict[str, Any]] = []
        for sample_index, direction in enumerate(directions):
            physical = inverse_transform @ (source_coordinate + radius * direction)
            state = dict(selected_state)
            state["qpos"] = physical[:nq].astype(float).tolist()
            state["qvel"] = physical[nq:].astype(float).tolist()
            rows.append(
                {
                    "sample_index": sample_index,
                    "result": evaluate(
                        state,
                        args.seed + 100_000 * (radius_index + 1) + sample_index,
                    ),
                }
            )
        perturbation_groups.append(
            {
                "normalized_radius": radius,
                "summary": aggregate(rows),
                "episodes": rows,
            }
        )

    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Local feedback-basin diagnostic for one saved iLQR chain; not P1 evidence.",
        "controller": file_metadata(controller_path),
        "source_state_id": selected_state.get("state_id"),
        "source_result": source_result,
        "nearest_dataset_states": {
            "summary": aggregate(nearest_rows),
            "episodes": nearest_rows,
        },
        "radial_perturbations": perturbation_groups,
        "settings": {
            "nearest_count": int(args.nearest_count),
            "perturbation_radii": radii,
            "samples_per_radius": int(args.samples_per_radius),
            "seed": int(args.seed),
            "lqr_scale": float(args.lqr_scale),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
        },
        "lyapunov": {
            "coordinate_source": file_metadata(args.spec),
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "evidence": {
            "config": {"path": args.config, "resolved_sha256": data_sha256(source_cfg)},
            "dataset": file_metadata(args.dataset),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    nearest_summary = payload["nearest_dataset_states"]["summary"]
    print(
        f"source_success={source_result['success']} nearest_success="
        f"{nearest_summary['success_count']}/{nearest_summary['count']}"
    )
    for group in perturbation_groups:
        group_summary = group["summary"]
        print(
            f"radius={group['normalized_radius']:.4f} success="
            f"{group_summary['success_count']}/{group_summary['count']} "
            f"latch={group_summary['latch_count']}/{group_summary['count']}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
