#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.capture_envelope import validate_capture_config, validate_capture_states
from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
)

try:
    from scripts.evaluate_feedback_mpc_capture import evaluate_feedback_mpc
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from evaluate_feedback_mpc_capture import evaluate_feedback_mpc
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_capture_sequence import fixed_state_cfg
    from search_swingup_capture import lqr_gain


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in result.items() if key not in {"trajectory", "final_info"}}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refine unresolved adaptive capture states with exact-model feedback MPC"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--initial-evaluation",
        default="runs/p1_capture_target_teachers/eval_adaptive_refined_validation256_p0065.json",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--out", required=True)
    parser.add_argument("--horizon-steps", type=int, default=75)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--mpc-seconds", type=float, default=2.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=2.4)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--target-count", type=int, default=6)
    parser.add_argument("--residual-count", type=int, default=8)
    parser.add_argument("--target-limit", type=float, default=1.5)
    parser.add_argument("--residual-limit", type=float, default=0.4)
    parser.add_argument("--target-sigma", type=float, default=0.5)
    parser.add_argument("--residual-sigma", type=float, default=0.15)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.horizon_steps < 2 or args.replan_steps < 1:
        raise ValueError("horizon steps must be >= 2 and replan steps must be positive")
    if args.iterations < 1 or args.population < 2 or not 1 <= args.elites <= args.population:
        raise ValueError("invalid CEM iterations, population, or elite count")
    if min(args.target_count, args.residual_count) < 2:
        raise ValueError("target and residual schedules require at least two knots")
    if min(
        args.mpc_seconds,
        args.handoff_lyapunov,
        args.handoff_cart_abs,
        args.target_limit,
        args.residual_limit,
        args.target_sigma,
        args.residual_sigma,
    ) <= 0.0:
        raise ValueError("MPC durations, thresholds, limits, and sigmas must be positive")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    spec = load_config(args.spec)
    dataset_path = Path(args.dataset)
    initial_path = Path(args.initial_evaluation)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    errors = validate_capture_config(cfg, spec)
    if args.split not in spec.get("splits", {}):
        errors.append(f"split {args.split!r} is not declared by the capture specification")
    else:
        errors.extend(validate_capture_states(dataset, spec, args.split))
    if initial.get("split") != args.split or initial.get("benchmark") != dataset.get("benchmark"):
        errors.append("initial evaluation split or benchmark does not match the dataset")
    if initial.get("evidence", {}).get("config", {}).get("resolved_sha256") != data_sha256(cfg):
        errors.append("initial evaluation config hash does not match the current config")
    if initial.get("evidence", {}).get("dataset", {}).get("sha256") != file_metadata(dataset_path).get(
        "sha256"
    ):
        errors.append("initial evaluation dataset hash does not match the current dataset")
    rows = initial.get("episode_results")
    state_indices = initial.get("evidence", {}).get("state_indices")
    if not isinstance(rows, list) or not rows:
        errors.append("initial evaluation must contain episode_results")
    if not isinstance(state_indices, list) or len(state_indices) != len(rows or []):
        errors.append("initial evaluation state_indices must match episode_results")
    elif any(int(row.get("state_index", -1)) != int(index) for row, index in zip(rows, state_indices)):
        errors.append("initial evaluation rows do not match its state_indices")
    if errors:
        raise ValueError("Feedback MPC refinement validation failed:\n- " + "\n- ".join(errors))

    progress = float(initial["progress"])
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
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
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix,
        input_matrix,
        gain,
        transform,
        feedback_scale=args.lqr_scale,
    )
    policy_dt = float(cfg["env"]["timestep"]) * int(cfg["env"].get("frame_skip", 1))
    mpc_steps = max(1, int(round(args.mpc_seconds / policy_dt)))
    states = dataset["states"]
    started = time.time()
    refined_rows: list[dict[str, Any]] = []

    for episode, initial_row in enumerate(rows):
        state_index = int(initial_row["state_index"])
        if bool(initial_row["success"]):
            row = dict(initial_row)
            row["feedback_mpc_invoked"] = False
            row["feedback_mpc"] = None
        else:
            fixed_cfg = fixed_state_cfg(cfg, states[state_index], float(cfg["env"]["episode_seconds"]))
            result = evaluate_feedback_mpc(
                fixed_cfg,
                progress=progress,
                seed=int(initial_row["seed"]),
                gain=gain,
                lqr_scale=args.lqr_scale,
                transform=transform,
                lyapunov=lyapunov,
                horizon_steps=args.horizon_steps,
                replan_steps=args.replan_steps,
                mpc_steps=mpc_steps,
                handoff_lyapunov=args.handoff_lyapunov,
                handoff_cart_abs=args.handoff_cart_abs,
                iterations=args.iterations,
                population=args.population,
                elites=args.elites,
                target_count=args.target_count,
                residual_count=args.residual_count,
                target_limit=args.target_limit,
                residual_limit=args.residual_limit,
                target_sigma=args.target_sigma,
                residual_sigma=args.residual_sigma,
            )
            row = {
                "episode": int(episode),
                "state_index": state_index,
                "state_id": states[state_index].get("state_id"),
                "seed": int(initial_row["seed"]),
                "baseline": initial_row.get("baseline"),
                "planner_invoked": bool(initial_row.get("planner_invoked", False)),
                "planning_history": initial_row.get("planning_history", []),
                "selected_controller": initial_row.get("selected_controller"),
                "source_success": False,
                "feedback_mpc_invoked": True,
                "feedback_mpc": compact_result(result),
                **{
                    key: value
                    for key, value in compact_result(result).items()
                    if key not in {"plan_events"}
                },
            }
        refined_rows.append(row)
        print(
            f"episode={episode + 1}/{len(rows)} state={state_index} "
            f"mpc={row['feedback_mpc_invoked']} success={row['success']} "
            f"hold={row['max_upright_streak_seconds']:.3f}s"
        )

    successes = np.asarray([row["success"] for row in refined_rows], dtype=np.float64)
    holds = np.asarray([row["max_upright_streak_seconds"] for row in refined_rows], dtype=np.float64)
    gate_spec = spec["capture_gate"]
    count = len(refined_rows)
    success_rate = float(np.mean(successes))
    median_hold = float(np.median(holds))
    gate = {
        "required_episodes": int(gate_spec["required_episodes"]),
        "required_success_rate": float(gate_spec["required_success_rate"]),
        "required_median_upright_hold_seconds": float(
            gate_spec["required_median_upright_hold_seconds"]
        ),
        "successful_episode_rail_hits_required": int(
            gate_spec["successful_episode_rail_hits_required"]
        ),
        "passed_episode_count": count == int(gate_spec["required_episodes"]),
        "passed_final_progress": progress == 1.0,
        "passed_success_rate": success_rate >= float(gate_spec["required_success_rate"]),
        "passed_median_hold": median_hold
        >= float(gate_spec["required_median_upright_hold_seconds"]),
        "passed_successful_rail_safety": not any(
            row["success"] and row["rail_hit"] for row in refined_rows
        ),
    }
    gate["passed"] = all(value for key, value in gate.items() if key.startswith("passed_"))
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Feedback MPC refinement of unresolved partial P1 validation states.",
        "benchmark": dataset["benchmark"],
        "split": args.split,
        "progress": progress,
        "episodes": count,
        "success_rate": success_rate,
        "capture_success_count": int(np.sum(successes)),
        "initial_success_count": int(sum(bool(row["success"]) for row in rows)),
        "feedback_mpc_invocation_count": int(
            sum(bool(row["feedback_mpc_invoked"]) for row in refined_rows)
        ),
        "feedback_mpc_recovery_count": int(
            sum(bool(row["feedback_mpc_invoked"] and row["success"]) for row in refined_rows)
        ),
        "max_upright_streak_median": median_hold,
        "max_upright_streak_mean": float(np.mean(holds)),
        "rail_hit_count": int(sum(bool(row["rail_hit"]) for row in refined_rows)),
        "wall_time_seconds": float(time.time() - started),
        "gate": gate,
        "episode_results": refined_rows,
        "controller": {
            "type": "adaptive_target_planning_then_feedback_mpc_then_lqr",
            "feedback_mpc": {
                "horizon_steps": int(args.horizon_steps),
                "horizon_seconds": float(args.horizon_steps * policy_dt),
                "replan_steps": int(args.replan_steps),
                "mpc_seconds": float(args.mpc_seconds),
                "handoff_lyapunov": float(args.handoff_lyapunov),
                "handoff_cart_abs": float(args.handoff_cart_abs),
                "iterations": int(args.iterations),
                "population": int(args.population),
                "elites": int(args.elites),
                "target_count": int(args.target_count),
                "residual_count": int(args.residual_count),
                "target_limit": float(args.target_limit),
                "residual_limit": float(args.residual_limit),
                "target_sigma": float(args.target_sigma),
                "residual_sigma": float(args.residual_sigma),
                "lqr_scale": float(args.lqr_scale),
                "lqr_gain": gain.astype(float).tolist(),
            },
        },
        "lyapunov": {
            "coordinate_source": file_metadata(args.spec),
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "evidence": {
            "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(cfg)},
            "overrides": list(args.override),
            "spec": file_metadata(args.spec),
            "dataset": file_metadata(dataset_path),
            "initial_evaluation": file_metadata(initial_path),
            "state_indices": [int(index) for index in state_indices],
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"success={payload['capture_success_count']}/{count} "
        f"recoveries={payload['feedback_mpc_recovery_count']} "
        f"median_hold={median_hold:.3f}s gate={gate['passed']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
