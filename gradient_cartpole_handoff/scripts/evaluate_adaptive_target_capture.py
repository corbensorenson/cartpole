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
from gcartpole.ppo_mlx import select_evaluation_state_indices

try:
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_capture_target_schedule import evaluate_target_schedule, search_target_schedule
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg
    from search_capture_target_schedule import evaluate_target_schedule, search_target_schedule
    from search_swingup_capture import lqr_gain


def compact_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key not in {"trajectory", "final_info"}}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate deterministic scheduled-target planning plus LQR capture")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--progress", type=float, default=0.0625)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--index-seed", type=int, default=61201)
    parser.add_argument("--episode-seed", type=int, default=62601)
    parser.add_argument("--planner-seed", type=int, default=62701)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=12)
    parser.add_argument("--knot-count", type=int, default=8)
    parser.add_argument("--schedule-seconds", type=float, default=2.0)
    parser.add_argument("--target-limit", type=float, default=3.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    spec = load_config(args.spec)
    dataset_path = Path(args.dataset)
    source = json.loads(dataset_path.read_text(encoding="utf-8"))
    errors = validate_capture_config(cfg, spec)
    if args.split not in spec.get("splits", {}):
        errors.append(f"split {args.split!r} is not declared by the capture specification")
    else:
        errors.extend(validate_capture_states(source, spec, args.split))
    if errors:
        raise ValueError("Capture benchmark validation failed:\n- " + "\n- ".join(errors))
    states = source["states"]
    count = min(len(states), int(args.limit))
    if count < 1:
        raise ValueError("--limit must select at least one state")
    state_indices = select_evaluation_state_indices(dataset_path, count, args.index_seed)
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    started = time.time()
    episode_results: list[dict[str, Any]] = []

    for episode, state_index in enumerate(state_indices):
        state = states[state_index]
        fixed_cfg = fixed_state_cfg(cfg, state, float(cfg["env"]["episode_seconds"]))
        baseline_controller = {
            "target_knots": [0.0] * int(args.knot_count),
            "lqr_scale": 1.30,
        }
        baseline = evaluate_target_schedule(
            fixed_cfg,
            progress=args.progress,
            seed=args.episode_seed + episode,
            gain=gain,
            target_knots=np.zeros(args.knot_count, dtype=np.float64),
            schedule_seconds=args.schedule_seconds,
            lqr_scale=1.30,
        )
        selected_controller = baseline_controller
        planning: dict[str, Any] | None = None
        if not bool(baseline["success"]):
            search = search_target_schedule(
                fixed_cfg,
                progress=args.progress,
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
                success_polish_iterations=1,
                verbose=False,
            )
            selected_controller = search["best"]["controller"]
            completed = len(search["history"]) - 1
            planning = {
                "first_success_iteration": search["first_success_iteration"],
                "iterations_completed": completed,
                "candidate_rollout_count": 1 + completed * int(args.population),
                "best_candidate": compact_metrics(search["best"]["metrics"]),
                "history": search["history"],
            }
        final = evaluate_target_schedule(
            fixed_cfg,
            progress=args.progress,
            seed=args.episode_seed + episode,
            gain=gain,
            target_knots=np.asarray(selected_controller["target_knots"], dtype=np.float64),
            schedule_seconds=args.schedule_seconds,
            lqr_scale=float(selected_controller["lqr_scale"]),
        )
        row = {
            "episode": episode,
            "state_index": int(state_index),
            "state_id": state.get("state_id"),
            "seed": int(args.episode_seed + episode),
            "baseline": compact_metrics(baseline),
            "planner_invoked": planning is not None,
            "planning": planning,
            "selected_controller": selected_controller,
            **compact_metrics(final),
        }
        episode_results.append(row)
        print(
            f"episode={episode + 1}/{count} state={state_index} baseline={baseline['success']} "
            f"planned={planning is not None} success={final['success']} "
            f"hold={final['max_upright_streak_seconds']:.3f}s"
        )

    successes = np.asarray([row["success"] for row in episode_results], dtype=np.float64)
    holds = np.asarray([row["max_upright_streak_seconds"] for row in episode_results], dtype=np.float64)
    gate_spec = spec["capture_gate"]
    success_rate = float(np.mean(successes))
    median_hold = float(np.median(holds))
    gate = {
        "required_episodes": int(gate_spec["required_episodes"]),
        "required_success_rate": float(gate_spec["required_success_rate"]),
        "required_median_upright_hold_seconds": float(gate_spec["required_median_upright_hold_seconds"]),
        "successful_episode_rail_hits_required": int(gate_spec["successful_episode_rail_hits_required"]),
        "passed_episode_count": count == int(gate_spec["required_episodes"]),
        "passed_final_progress": args.progress == 1.0,
        "passed_success_rate": success_rate >= float(gate_spec["required_success_rate"]),
        "passed_median_hold": median_hold >= float(gate_spec["required_median_upright_hold_seconds"]),
        "passed_successful_rail_safety": not any(row["success"] and row["rail_hit"] for row in episode_results),
    }
    gate["passed"] = all(value for key, value in gate.items() if key.startswith("passed_"))
    payload = {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Adaptive target-planning capture diagnostic; partial progress or episode count cannot pass P1.",
        "benchmark": source["benchmark"],
        "split": args.split,
        "progress": float(args.progress),
        "episodes": count,
        "success_rate": success_rate,
        "capture_success_count": int(np.sum(successes)),
        "baseline_success_count": int(sum(bool(row["baseline"]["success"]) for row in episode_results)),
        "planner_invocation_count": int(sum(bool(row["planner_invoked"]) for row in episode_results)),
        "planner_recovery_count": int(sum(row["planner_invoked"] and row["success"] for row in episode_results)),
        "max_upright_streak_median": median_hold,
        "max_upright_streak_mean": float(np.mean(holds)),
        "rail_hit_count": int(sum(bool(row["rail_hit"]) for row in episode_results)),
        "wall_time_seconds": float(time.time() - started),
        "gate": gate,
        "episode_results": episode_results,
        "controller": {
            "type": "deterministic_cem_target_schedule_plus_lqr_feedback",
            "lqr_gain": gain.astype(float).tolist(),
            "baseline_lqr_scale": 1.30,
            "planner": {
                "iterations": int(args.iterations),
                "population": int(args.population),
                "elites": int(args.elites),
                "knot_count": int(args.knot_count),
                "schedule_seconds": float(args.schedule_seconds),
                "target_limit": float(args.target_limit),
                "seed": int(args.planner_seed),
            },
        },
        "evidence": {
            "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(cfg)},
            "overrides": list(args.override),
            "spec": file_metadata(args.spec),
            "dataset": file_metadata(dataset_path),
            "state_indices": state_indices,
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"success={payload['capture_success_count']}/{count} baseline={payload['baseline_success_count']} "
        f"recoveries={payload['planner_recovery_count']} median_hold={median_hold:.3f}s gate={gate['passed']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
