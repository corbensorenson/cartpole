#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

from gcartpole.config import dump_json, load_config
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


def physical_state_cfg(cfg: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    result = fixed_state_cfg(cfg, query, float(cfg["env"]["episode_seconds"]))
    result["env"].update(
        {
            "init_qpos_scale_start": 1.0,
            "init_qpos_scale_end": 1.0,
            "init_qvel_scale_start": 1.0,
            "init_qvel_scale_end": 1.0,
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run feedback-MPC supervision from learner-visited training states"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_hybrid_policy.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--queries", default="runs/p1_capture_dagger/round1_dense_train_queries.json")
    parser.add_argument("--source-step", type=int, default=1)
    parser.add_argument("--source-indices", default="970,1196")
    parser.add_argument("--seed", type=int, default=79701)
    parser.add_argument("--horizon-steps", type=int, default=75)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--mpc-seconds", type=float, default=3.0)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--out", default="runs/p1_capture_dagger/round1_feedback_labels_step1.json")
    args = parser.parse_args()
    source_indices = {int(value) for value in args.source_indices.split(",") if value.strip()}
    if not source_indices:
        raise ValueError("at least one source index is required")

    config_path = Path(args.config)
    spec_path = Path(args.spec)
    query_path = Path(args.queries)
    cfg = copy.deepcopy(load_config(config_path))
    query_payload = json.loads(query_path.read_text(encoding="utf-8"))
    if query_payload.get("split") != "train":
        raise ValueError("DAgger feedback-MPC labeling is restricted to training queries")
    queries = [
        row
        for row in query_payload["queries"]
        if int(row["step"]) == args.source_step
        and int(row["source_index"]) in source_indices
    ]
    queries.sort(key=lambda row: int(row["source_index"]))
    missing = source_indices - {int(row["source_index"]) for row in queries}
    if missing:
        raise ValueError(f"requested source indices are absent at step {args.source_step}: {sorted(missing)}")

    cfg["env"]["action_lqr_switch"]["enabled"] = False
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    distribution = load_config(spec_path)["distribution"]
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
        feedback_scale=1.30,
    )
    policy_dt = float(cfg["env"]["timestep"]) * int(cfg["env"].get("frame_skip", 1))
    started = time.time()
    evaluations: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for ordinal, query in enumerate(queries):
        fixed_cfg = physical_state_cfg(cfg, query)
        result = evaluate_feedback_mpc(
            fixed_cfg,
            progress=1.0,
            seed=args.seed + ordinal,
            gain=gain,
            lqr_scale=1.30,
            planning_lqr_scale=1.30,
            transform=transform,
            lyapunov=lyapunov,
            horizon_steps=args.horizon_steps,
            replan_steps=args.replan_steps,
            mpc_steps=max(1, int(round(args.mpc_seconds / policy_dt))),
            handoff_lyapunov=1800.0,
            handoff_cart_abs=1.5,
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
        teacher_action = float(result["trajectory"][0]["action"])
        evaluations.append(
            {
                "ordinal": ordinal,
                "query_id": query["query_id"],
                "source_index": int(query["source_index"]),
                "source_step": int(query["step"]),
                "success": bool(result["success"]),
                "teacher_action": teacher_action,
                "result": result,
            }
        )
        if result["success"]:
            labels.append(
                {
                    "query_id": query["query_id"],
                    "source_index": int(query["source_index"]),
                    "source_step": int(query["step"]),
                    "observation": query["observation"],
                    "learner_action": float(query["learner_action"]),
                    "teacher_action": teacher_action,
                    "action_disagreement_abs": abs(
                        float(query["learner_action"]) - teacher_action
                    ),
                    "tier": "dagger_feedback_mpc",
                }
            )
        print(
            f"query={ordinal + 1}/{len(queries)} source={query['source_index']} "
            f"success={result['success']} handoff={result['first_handoff_time']} "
            f"hold={result['max_upright_streak_seconds']:.3f}s",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Successful feedback-MPC labels on learner-visited training states.",
        "split": "train",
        "round": int(query_payload["round"]),
        "progress": float(query_payload["progress"]),
        "query_count": len(queries),
        "successful_query_count": len(labels),
        "successful_query_rate": float(len(labels) / len(queries)),
        "successful_source_count": len({int(row["source_index"]) for row in labels}),
        "wall_time_seconds": float(time.time() - started),
        "labels_sha256": data_sha256(labels),
        "labels": labels,
        "evaluations": evaluations,
        "controller": {
            "horizon_steps": int(args.horizon_steps),
            "replan_steps": int(args.replan_steps),
            "mpc_seconds": float(args.mpc_seconds),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "closed_loop_spectral_radius": float(spectral_radius),
        },
        "evidence": {
            "config": file_metadata(config_path),
            "spec": file_metadata(spec_path),
            "queries": file_metadata(query_path),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(output, args.out)
    print(f"success={len(labels)}/{len(queries)} Wrote {args.out}")


if __name__ == "__main__":
    main()
