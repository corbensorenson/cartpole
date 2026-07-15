#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

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


def failed_state_indices(payload: dict[str, Any]) -> list[int]:
    evaluation = payload.get("evaluation", payload)
    rows = evaluation.get("episode_results") if isinstance(evaluation, dict) else None
    if not isinstance(rows, list):
        raise ValueError("rollout evidence must contain episode_results")
    return sorted(
        int(row["state_index"])
        for row in rows
        if not bool(row.get("success", False))
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build training-only action teachers with exact-model feedback MPC"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/train.json")
    parser.add_argument("--rollouts", default="runs/p1_capture_envelope/train_hard_007.json")
    parser.add_argument("--progress", type=float, default=0.07)
    parser.add_argument("--failure-limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=78401)
    parser.add_argument("--horizon-steps", type=int, default=75)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--mpc-seconds", type=float, default=3.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--planning-lqr-scale", type=float, default=1.30)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.failure_limit < 1 or args.horizon_steps < 2 or args.replan_steps < 1:
        raise ValueError("failure limit and planning horizons must be positive")
    if args.population < 2 or not 1 <= args.elites <= args.population:
        raise ValueError("invalid feedback-MPC population or elite count")

    dataset_path = Path(args.dataset)
    rollout_path = Path(args.rollouts)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rollouts = json.loads(rollout_path.read_text(encoding="utf-8"))
    if dataset.get("split") != "train":
        raise ValueError("feedback-MPC teachers are restricted to the frozen training split")
    states = dataset.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("training dataset must contain states")
    indices = failed_state_indices(rollouts)[: args.failure_limit]

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    cfg["env"].get("action_lqr_switch", {})["enabled"] = False
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

    started = time.time()
    teachers: list[dict[str, Any]] = []
    policy_dt = float(cfg["env"]["timestep"]) * int(cfg["env"].get("frame_skip", 1))
    for ordinal, state_index in enumerate(indices):
        fixed_cfg = fixed_state_cfg(cfg, states[state_index], float(cfg["env"]["episode_seconds"]))
        result = evaluate_feedback_mpc(
            fixed_cfg,
            progress=args.progress,
            seed=args.seed + ordinal,
            gain=gain,
            lqr_scale=args.lqr_scale,
            planning_lqr_scale=args.planning_lqr_scale,
            transform=transform,
            lyapunov=lyapunov,
            horizon_steps=args.horizon_steps,
            replan_steps=args.replan_steps,
            mpc_steps=max(1, int(round(args.mpc_seconds / policy_dt))),
            handoff_lyapunov=args.handoff_lyapunov,
            handoff_cart_abs=args.handoff_cart_abs,
            iterations=args.iterations,
            population=args.population,
            elites=args.elites,
            target_count=6,
            residual_count=8,
            target_limit=1.5,
            residual_limit=0.4,
            target_sigma=0.5,
            residual_sigma=0.15,
        )
        teachers.append(
            {
                "ordinal": ordinal,
                "source_index": state_index,
                "state_id": states[state_index].get("state_id"),
                "success": bool(result["success"]),
                "label_count": (
                    sum(row["controller_mode"] == "feedback_mpc" for row in result["trajectory"])
                    if result["success"]
                    else 0
                ),
                "result": result,
            }
        )
        print(
            f"teacher={ordinal + 1}/{len(indices)} state={state_index} "
            f"success={result['success']} hold={result['max_upright_streak_seconds']:.3f}s "
            f"plans={result['plan_event_count']}",
            flush=True,
        )

    success_count = sum(bool(row["success"]) for row in teachers)
    label_count = sum(int(row["label_count"]) for row in teachers)
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Training-only successful feedback-MPC action teachers; not P1 evidence.",
        "progress": float(args.progress),
        "teacher_count": len(teachers),
        "teacher_success_count": int(success_count),
        "teacher_success_rate": float(success_count / max(1, len(teachers))),
        "successful_label_count": int(label_count),
        "wall_time_seconds": float(time.time() - started),
        "teachers_sha256": data_sha256(teachers),
        "teachers": teachers,
        "controller": {
            "type": "receding_horizon_cem_target_residual_plus_lqr_feedback",
            "horizon_steps": int(args.horizon_steps),
            "replan_steps": int(args.replan_steps),
            "mpc_seconds": float(args.mpc_seconds),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "planning_lqr_scale": float(args.planning_lqr_scale),
            "lqr_scale": float(args.lqr_scale),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
        },
        "lyapunov": {
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "evidence": {
            "config": {"path": args.config, "resolved_sha256": data_sha256(cfg)},
            "dataset": file_metadata(dataset_path),
            "rollouts": file_metadata(rollout_path),
            "state_indices": indices,
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(f"success={success_count}/{len(teachers)} labels={label_count} Wrote {args.out}")


if __name__ == "__main__":
    main()
