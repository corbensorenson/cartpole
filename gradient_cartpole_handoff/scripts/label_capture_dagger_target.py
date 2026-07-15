#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp

try:
    from scripts.evaluate_feedback_mpc_capture import feedback_action
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_capture_target_schedule import evaluate_target_schedule, search_target_schedule
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from evaluate_feedback_mpc_capture import feedback_action
    from search_capture_sequence import fixed_state_cfg
    from search_capture_target_schedule import evaluate_target_schedule, search_target_schedule
    from search_swingup_capture import lqr_gain


def selected_queries(
    payload: dict[str, Any],
    *,
    source_step: int,
    max_lyapunov: float | None,
    limit: int | None,
) -> list[dict[str, Any]]:
    rows = [row for row in payload["queries"] if int(row["step"]) == source_step]
    if max_lyapunov is not None:
        rows = [
            row
            for row in rows
            if float(row["dimensionless_lyapunov_value"]) <= max_lyapunov
        ]
    rows.sort(key=lambda row: (float(row["dimensionless_lyapunov_value"]), int(row["source_index"])))
    return rows if limit is None else rows[: max(0, limit)]


def physical_state_cfg(cfg: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    result = fixed_state_cfg(cfg, query, float(cfg["env"]["episode_seconds"]))
    env_cfg = result["env"]
    env_cfg["init_qpos_scale_start"] = 1.0
    env_cfg["init_qpos_scale_end"] = 1.0
    env_cfg["init_qvel_scale_start"] = 1.0
    env_cfg["init_qvel_scale_end"] = 1.0
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run scheduled-target supervision from learner-visited training states"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_hybrid_policy.yaml")
    parser.add_argument("--queries", default="runs/p1_capture_dagger/round1_train_queries.json")
    parser.add_argument("--source-step", type=int, default=5)
    parser.add_argument("--max-lyapunov", type=float, default=5000.0)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--seed", type=int, default=79601)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--out", default="runs/p1_capture_dagger/round1_target_labels_step5.json")
    args = parser.parse_args()
    if args.source_step < 0 or args.iterations < 1 or args.population < 2:
        raise ValueError("source step and search budget are invalid")
    if not 1 <= args.elites <= args.population:
        raise ValueError("elites must be in 1..population")

    config_path = Path(args.config)
    query_path = Path(args.queries)
    cfg = load_config(config_path)
    query_payload = json.loads(query_path.read_text(encoding="utf-8"))
    if query_payload.get("split") != "train":
        raise ValueError("DAgger target labeling is restricted to training queries")
    queries = selected_queries(
        query_payload,
        source_step=args.source_step,
        max_lyapunov=args.max_lyapunov,
        limit=args.limit,
    )
    if not queries:
        raise ValueError("no DAgger queries match the requested target-label cohort")
    cfg = copy.deepcopy(cfg)
    cfg["env"]["action_lqr_switch"]["enabled"] = False
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    started = time.time()
    evaluations: list[dict[str, Any]] = []
    labels: list[dict[str, Any]] = []
    for ordinal, query in enumerate(queries):
        fixed_cfg = physical_state_cfg(cfg, query)
        seed = args.seed + ordinal
        search = search_target_schedule(
            fixed_cfg,
            progress=1.0,
            seed=seed,
            gain=gain,
            iterations=args.iterations,
            population=args.population,
            elites=args.elites,
            knot_count=8,
            schedule_seconds=2.0,
            target_limit=3.0,
            target_sigma=1.0,
            scale_sigma=0.25,
            sigma_decay=0.90,
            target_sigma_floor=0.03,
            scale_sigma_floor=0.01,
            initial_lqr_scale=1.30,
            success_polish_iterations=1,
            verbose=False,
        )
        controller = search["best"]["controller"]
        verified = evaluate_target_schedule(
            fixed_cfg,
            progress=1.0,
            seed=seed,
            gain=gain,
            target_knots=np.asarray(controller["target_knots"], dtype=np.float64),
            schedule_seconds=2.0,
            lqr_scale=float(controller["lqr_scale"]),
        )
        teacher_action = feedback_action(
            np.asarray(query["qpos"], dtype=np.float64),
            np.asarray(query["qvel"], dtype=np.float64),
            gain,
            n_links=int(cfg["env"]["n_links"]),
            scale=float(controller["lqr_scale"]),
            cart_target=float(controller["target_knots"][0]),
        )
        row = {
            "ordinal": ordinal,
            "query_id": query["query_id"],
            "source_index": int(query["source_index"]),
            "source_step": int(query["step"]),
            "initial_lyapunov": float(query["dimensionless_lyapunov_value"]),
            "success": bool(verified["success"]),
            "teacher_action": float(teacher_action),
            "controller": controller,
            "first_success_iteration": search["first_success_iteration"],
            "history": search["history"],
            "metrics": {key: value for key, value in verified.items() if key != "trajectory"},
        }
        evaluations.append(row)
        if row["success"]:
            labels.append(
                {
                    "query_id": query["query_id"],
                    "source_index": int(query["source_index"]),
                    "source_step": int(query["step"]),
                    "observation": query["observation"],
                    "learner_action": float(query["learner_action"]),
                    "teacher_action": float(teacher_action),
                    "action_disagreement_abs": abs(
                        float(query["learner_action"]) - float(teacher_action)
                    ),
                    "tier": "dagger_target",
                }
            )
        print(
            f"query={ordinal + 1}/{len(queries)} source={query['source_index']} "
            f"V={query['dimensionless_lyapunov_value']:.1f} success={row['success']} "
            f"hold={verified['max_upright_streak_seconds']:.3f}s",
            flush=True,
        )

    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Successful scheduled-target labels on learner-visited training states.",
        "split": "train",
        "round": int(query_payload["round"]),
        "progress": float(query_payload["progress"]),
        "query_count": len(queries),
        "successful_query_count": len(labels),
        "successful_query_rate": float(len(labels) / len(queries)),
        "successful_source_count": len({int(row["source_index"]) for row in labels}),
        "wall_time_seconds": float(time.time() - started),
        "selection": {
            "source_step": int(args.source_step),
            "max_lyapunov": float(args.max_lyapunov),
            "limit": int(args.limit),
        },
        "search": {
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "seed": int(args.seed),
        },
        "labels_sha256": data_sha256(labels),
        "labels": labels,
        "evaluations": evaluations,
        "evidence": {
            "config": file_metadata(config_path),
            "queries": file_metadata(query_path),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(result, args.out)
    print(f"success={len(labels)}/{len(queries)} Wrote {args.out}")


if __name__ == "__main__":
    main()
