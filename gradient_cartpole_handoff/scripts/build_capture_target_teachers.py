#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.capture_funnel import effective_state
from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp

try:
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_capture_target_schedule import search_target_schedule
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg
    from search_capture_target_schedule import search_target_schedule
    from search_swingup_capture import lqr_gain


def rollout_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation = payload.get("evaluation", payload)
    rows = evaluation.get("episode_results") if isinstance(evaluation, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("rollout evidence must contain episode_results")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build state-specific scheduled-target teachers from training failures")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/train.json")
    parser.add_argument("--rollouts", default="runs/p1_capture_envelope/train_hard_00625.json")
    parser.add_argument("--progress", type=float, default=0.0625)
    parser.add_argument("--failure-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=62301)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--knot-count", type=int, default=8)
    parser.add_argument("--schedule-seconds", type=float, default=2.0)
    parser.add_argument("--target-limit", type=float, default=3.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    dataset_path = Path(args.dataset)
    rollout_path = Path(args.rollouts)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rollouts = json.loads(rollout_path.read_text(encoding="utf-8"))
    if dataset.get("split") != "train":
        raise ValueError("teacher search is restricted to the frozen training split")
    states = dataset.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("training dataset must contain states")
    failures = [row for row in rollout_rows(rollouts) if not bool(row.get("success", False))]
    failures.sort(key=lambda row: int(row["state_index"]))
    failures = failures[: max(0, int(args.failure_limit))]

    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    env_cfg = cfg["env"]
    qpos_power = float(env_cfg.get("init_qpos_scale_power", 1.0))
    qvel_power = float(env_cfg.get("init_qvel_scale_power", 1.0))
    qpos_start = float(env_cfg.get("init_qpos_scale_start", env_cfg.get("init_qpos_scale", 1.0)))
    qpos_end = float(env_cfg.get("init_qpos_scale_end", env_cfg.get("init_qpos_scale", 1.0)))
    qvel_start = float(env_cfg.get("init_qvel_scale_start", env_cfg.get("init_qvel_scale", 1.0)))
    qvel_end = float(env_cfg.get("init_qvel_scale_end", env_cfg.get("init_qvel_scale", 1.0)))
    teachers: list[dict[str, Any]] = []

    for ordinal, failure in enumerate(failures):
        source_index = int(failure["state_index"])
        state = states[source_index]
        fixed_cfg = fixed_state_cfg(cfg, state, float(cfg["env"]["episode_seconds"]))
        search = search_target_schedule(
            fixed_cfg,
            progress=args.progress,
            seed=args.seed + ordinal,
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
        best = search["best"]
        effective_qpos, effective_qvel = effective_state(
            np.asarray(state["qpos"], dtype=np.float64),
            np.asarray(state["qvel"], dtype=np.float64),
            progress=args.progress,
            qpos_scale_power=qpos_power,
            qvel_scale_power=qvel_power,
            qpos_scale_start=qpos_start,
            qpos_scale_end=qpos_end,
            qvel_scale_start=qvel_start,
            qvel_scale_end=qvel_end,
            cart_target=float(env_cfg.get("init_qpos_cart_target", 0.0)),
        )
        teacher = {
            "ordinal": ordinal,
            "source_index": source_index,
            "state_id": state.get("state_id"),
            "qpos": effective_qpos.astype(float).tolist(),
            "qvel": effective_qvel.astype(float).tolist(),
            "success": bool(best["metrics"]["success"]),
            "controller": best["controller"],
            "metrics": best["metrics"],
            "first_success_iteration": search["first_success_iteration"],
            "history": search["history"],
        }
        teachers.append(teacher)
        print(
            f"teacher={ordinal + 1}/{len(failures)} state={source_index} "
            f"success={teacher['success']} hold={teacher['metrics']['max_upright_streak_seconds']:.3f}s"
        )

    success_count = int(sum(bool(row["success"]) for row in teachers))
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Training-only state-specific recovery teachers; not held-out P1 evidence.",
        "progress": float(args.progress),
        "failure_count_available": int(sum(not bool(row.get("success", False)) for row in rollout_rows(rollouts))),
        "teacher_count": len(teachers),
        "teacher_success_count": success_count,
        "teacher_success_rate": float(success_count / max(1, len(teachers))),
        "search": {
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "knot_count": int(args.knot_count),
            "schedule_seconds": float(args.schedule_seconds),
            "target_limit": float(args.target_limit),
            "seed": int(args.seed),
        },
        "lqr_gain": gain.astype(float).tolist(),
        "teachers_sha256": data_sha256(teachers),
        "teachers": teachers,
        "evidence": {
            "config": file_metadata(args.config),
            "overrides": list(args.override),
            "dataset": file_metadata(dataset_path),
            "rollouts": file_metadata(rollout_path),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(f"success={success_count}/{len(teachers)} Wrote {args.out}")


if __name__ == "__main__":
    main()
