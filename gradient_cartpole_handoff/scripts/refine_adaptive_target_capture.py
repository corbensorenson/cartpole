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

try:
    from scripts.evaluate_adaptive_target_capture import compact_metrics
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_capture_target_schedule import evaluate_target_schedule, search_target_schedule
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from evaluate_adaptive_target_capture import compact_metrics
    from search_capture_sequence import fixed_state_cfg
    from search_capture_target_schedule import evaluate_target_schedule, search_target_schedule
    from search_swingup_capture import lqr_gain


def planning_rollout_count(planning: Any) -> int:
    if not isinstance(planning, dict):
        return 0
    return int(planning.get("candidate_rollout_count", 0))


def row_planning_history(row: dict[str, Any]) -> list[dict[str, Any]]:
    existing = row.get("planning_history")
    if isinstance(existing, list):
        return [item for item in existing if isinstance(item, dict)]
    return [
        item
        for item in (
            row.get("planning"),
            row.get("initial_planning"),
            row.get("refinement_planning"),
        )
        if isinstance(item, dict)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Escalate deterministic target planning only on unresolved capture states")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--initial-evaluation",
        default="runs/p1_capture_target_teachers/eval_adaptive_validation256_p0065.json",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--planner-seed", type=int, default=62701)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--elites", type=int, default=20)
    parser.add_argument("--knot-count", type=int, default=8)
    parser.add_argument("--schedule-seconds", type=float, default=2.0)
    parser.add_argument("--target-limit", type=float, default=3.0)
    parser.add_argument("--success-polish-iterations", type=int, default=2)
    parser.add_argument(
        "--out",
        default="runs/p1_capture_target_teachers/eval_adaptive_refined_validation256_p0065.json",
    )
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.population < 2 or not 1 <= args.elites <= args.population:
        raise ValueError("--population must be >= 2 and --elites must be in 1..population")
    cfg = apply_overrides(load_config(args.config), args.override)
    spec = load_config(args.spec)
    dataset_path = Path(args.dataset)
    initial_path = Path(args.initial_evaluation)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    initial = json.loads(initial_path.read_text(encoding="utf-8"))
    errors = validate_capture_config(cfg, spec)
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    if args.split not in spec.get("splits", {}):
        errors.append(f"split {args.split!r} is not declared by the capture specification")
    else:
        errors.extend(validate_capture_states(dataset, spec, args.split))
    if initial.get("split") != args.split or initial.get("benchmark") != dataset.get("benchmark"):
        errors.append("initial evaluation split or benchmark does not match the dataset")
    if initial.get("evidence", {}).get("config", {}).get("resolved_sha256") != data_sha256(cfg):
        errors.append("initial evaluation config hash does not match the current config")
    if initial.get("evidence", {}).get("dataset", {}).get("sha256") != file_metadata(dataset_path).get("sha256"):
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
        raise ValueError("Adaptive refinement validation failed:\n- " + "\n- ".join(errors))

    progress = float(initial["progress"])
    initial_planner_config = initial["controller"].get("planner") or initial["controller"].get("refinement_planner")
    if not isinstance(initial_planner_config, dict):
        raise ValueError("initial evaluation controller does not declare a planner configuration")
    initial_schedule_seconds = float(initial_planner_config["schedule_seconds"])
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    states = dataset["states"]
    started = time.time()
    refined_rows: list[dict[str, Any]] = []

    for episode, initial_row in enumerate(rows):
        state_index = int(initial_row["state_index"])
        fixed_cfg = fixed_state_cfg(cfg, states[state_index], float(cfg["env"]["episode_seconds"]))
        selected_controller = initial_row["selected_controller"]
        initial_replay = evaluate_target_schedule(
            fixed_cfg,
            progress=progress,
            seed=int(initial_row["seed"]),
            gain=gain,
            target_knots=np.asarray(selected_controller["target_knots"], dtype=np.float64),
            schedule_seconds=initial_schedule_seconds,
            lqr_scale=float(selected_controller["lqr_scale"]),
        )
        if bool(initial_replay["success"]) != bool(initial_row["success"]):
            raise RuntimeError(f"initial controller replay changed success for state {state_index}")

        refinement: dict[str, Any] | None = None
        final = initial_replay
        if not bool(initial_replay["success"]):
            search = search_target_schedule(
                fixed_cfg,
                progress=progress,
                seed=args.planner_seed + episode,
                gain=gain,
                iterations=args.iterations,
                population=args.population,
                elites=args.elites,
                knot_count=args.knot_count,
                schedule_seconds=args.schedule_seconds,
                target_limit=args.target_limit,
                target_sigma=1.0,
                scale_sigma=0.25,
                sigma_decay=0.90,
                target_sigma_floor=0.03,
                scale_sigma_floor=0.01,
                initial_lqr_scale=1.30,
                success_polish_iterations=args.success_polish_iterations,
                verbose=False,
                initial_controller=selected_controller,
            )
            candidate = search["best"]
            completed = len(search["history"]) - 1
            refinement = {
                "first_success_iteration": search["first_success_iteration"],
                "iterations_completed": completed,
                "candidate_rollout_count": 1 + completed * int(args.population),
                "best_candidate": compact_metrics(candidate["metrics"]),
                "history": search["history"],
            }
            if float(candidate["score"]) < float(initial_replay["score"]):
                selected_controller = candidate["controller"]
                final = evaluate_target_schedule(
                    fixed_cfg,
                    progress=progress,
                    seed=int(initial_row["seed"]),
                    gain=gain,
                    target_knots=np.asarray(selected_controller["target_knots"], dtype=np.float64),
                    schedule_seconds=args.schedule_seconds,
                    lqr_scale=float(selected_controller["lqr_scale"]),
                )

        refined_row = {
            "episode": episode,
            "state_index": state_index,
            "state_id": states[state_index].get("state_id"),
            "seed": int(initial_row["seed"]),
            "baseline": initial_row["baseline"],
            "planner_invoked": bool(initial_row["planner_invoked"] or refinement is not None),
            "planning_history": row_planning_history(initial_row) + ([refinement] if refinement is not None else []),
            "selected_controller": selected_controller,
            **compact_metrics(final),
        }
        refined_rows.append(refined_row)
        print(
            f"episode={episode + 1}/{len(rows)} state={state_index} "
            f"initial={initial_row['success']} refined={refinement is not None} "
            f"success={final['success']} hold={final['max_upright_streak_seconds']:.3f}s"
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
        "required_median_upright_hold_seconds": float(gate_spec["required_median_upright_hold_seconds"]),
        "successful_episode_rail_hits_required": int(gate_spec["successful_episode_rail_hits_required"]),
        "passed_episode_count": count == int(gate_spec["required_episodes"]),
        "passed_final_progress": progress == 1.0,
        "passed_success_rate": success_rate >= float(gate_spec["required_success_rate"]),
        "passed_median_hold": median_hold >= float(gate_spec["required_median_upright_hold_seconds"]),
        "passed_successful_rail_safety": not any(row["success"] and row["rail_hit"] for row in refined_rows),
    }
    gate["passed"] = all(value for key, value in gate.items() if key.startswith("passed_"))
    refinement_invocations = int(sum(len(row["planning_history"]) > len(row_planning_history(source)) for row, source in zip(refined_rows, rows)))
    refinement_recoveries = int(
        sum(
            len(row["planning_history"]) > len(row_planning_history(source)) and row["success"]
            for row, source in zip(refined_rows, rows)
        )
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Escalated deterministic target planning on unresolved partial P1 validation states.",
        "benchmark": dataset["benchmark"],
        "split": args.split,
        "progress": progress,
        "episodes": count,
        "success_rate": success_rate,
        "capture_success_count": int(np.sum(successes)),
        "baseline_success_count": int(sum(bool(row["baseline"]["success"]) for row in refined_rows)),
        "planner_invocation_count": int(sum(bool(row["planner_invoked"]) for row in refined_rows)),
        "planner_recovery_count": int(
            sum(bool(row["planner_invoked"] and row["success"]) for row in refined_rows)
        ),
        "initial_success_count": int(sum(bool(row["success"]) for row in rows)),
        "refinement_invocation_count": refinement_invocations,
        "refinement_recovery_count": refinement_recoveries,
        "max_upright_streak_median": median_hold,
        "max_upright_streak_mean": float(np.mean(holds)),
        "rail_hit_count": int(sum(bool(row["rail_hit"]) for row in refined_rows)),
        "planner_candidate_rollout_count_total": int(
            sum(
                sum(planning_rollout_count(planning) for planning in row["planning_history"])
                for row in refined_rows
            )
        ),
        "wall_time_seconds": float(time.time() - started),
        "gate": gate,
        "episode_results": refined_rows,
        "controller": {
            "type": "deterministic_escalated_cem_target_schedule_plus_lqr_feedback",
            "lqr_gain": gain.astype(float).tolist(),
            "refinement_planner": {
                "iterations": int(args.iterations),
                "population": int(args.population),
                "elites": int(args.elites),
                "knot_count": int(args.knot_count),
                "schedule_seconds": float(args.schedule_seconds),
                "target_limit": float(args.target_limit),
                "seed": int(args.planner_seed),
                "success_polish_iterations": int(args.success_polish_iterations),
            },
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
        f"success={payload['capture_success_count']}/{count} initial={payload['initial_success_count']} "
        f"refinement_recoveries={refinement_recoveries} median_hold={median_hold:.3f}s gate={gate['passed']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
