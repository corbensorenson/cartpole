#!/usr/bin/env python
"""Release a deterministic locked-split count continuation adaptively.

The input configuration is produced by ``generalized_swingup_solver.py split``.
Every proposed progress value is frozen into a standalone exact MuJoCo plant.
The inherited route is repaired with bounded waypoint solves and Box-FDDP; a
step is accepted only after an uninterrupted swing-up-and-hold replay succeeds.
Failures bisect the continuation step.  All trials are retained in a resumable
ledger and remain development evidence until the unlocked p=1 gate passes.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config, save_config
from gcartpole.evidence import file_metadata, utc_timestamp
from gcartpole.generalized_solver import (
    AdaptiveHomotopy,
    dimensionless_setup,
    setup_from_config,
)
from gcartpole.morphology import build_morphology

try:
    from scripts.run_generalized_homotopy import (
        fddp_command,
        hanging_state,
        result_summary,
        run,
        successful,
        waypoint_command,
        waypoint_lookaheads,
        waypoint_successful,
        waypoint_usable,
    )
except ModuleNotFoundError:
    from run_generalized_homotopy import (
        fddp_command,
        hanging_state,
        result_summary,
        run,
        successful,
        waypoint_command,
        waypoint_lookaheads,
        waypoint_successful,
        waypoint_usable,
    )


def controller_link_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    gains = np.asarray(
        payload.get("controller", {}).get("feedback_gains"), dtype=np.float64
    )
    if gains.ndim != 2 or gains.shape[1] < 4 or gains.shape[1] % 2:
        raise ValueError("controller feedback gains have an invalid state dimension")
    return gains.shape[1] // 2 - 1


def scheduled_rail_limit(env: dict[str, Any], progress: float) -> float:
    target = float(env["rail_limit"])
    start = float(env.get("rail_limit_start", target))
    end = float(env.get("rail_limit_end", target))
    return float(start + (end - start) * np.clip(progress, 0.0, 1.0))


def freeze_progress_config(
    continuation: dict[str, Any], progress: float, name: str
) -> dict[str, Any]:
    """Freeze every scheduled plant quantity at one continuation coordinate."""

    if not np.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be finite and lie in [0, 1]")
    cfg = copy.deepcopy(continuation)
    morphology = build_morphology(
        cfg["env"], cfg["morphology"], progress=float(progress)
    )
    cfg["experiment"]["name"] = name
    cfg["env"]["rail_limit"] = scheduled_rail_limit(cfg["env"], progress)
    cfg["env"].pop("rail_limit_start", None)
    cfg["env"].pop("rail_limit_end", None)
    cfg["env"].pop("plant_progress", None)
    morph = cfg["morphology"]
    morph["schedule_mode"] = "all_linear"
    for profile_name, values in (
        ("lengths", morphology.lengths),
        ("masses", morphology.masses),
        ("damping", morphology.damping),
        ("frictionloss", morphology.frictionloss),
        ("joint_stiffness", morphology.joint_stiffness),
        ("joint_lock", morphology.joint_lock),
    ):
        morph[f"{profile_name}_start"] = values.astype(float).tolist()
        morph[f"{profile_name}_end"] = values.astype(float).tolist()
    for endpoint in ("start", "end"):
        morph.setdefault(endpoint, {})
        morph[endpoint]["alpha_length"] = 0.0
        morph[endpoint]["alpha_mass"] = 0.0
        morph[endpoint]["alpha_damping"] = 0.0
        morph[endpoint]["alpha_frictionloss"] = 0.0
        morph[endpoint]["total_damping"] = float(np.sum(morphology.damping))
        morph[endpoint]["total_frictionloss"] = float(
            np.sum(morphology.frictionloss)
        )
    return cfg


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    schedule: AdaptiveHomotopy,
    current_controller: Path,
    baseline: dict[str, Any] | None,
    trials: list[dict[str, Any]],
    status: str,
) -> None:
    continuation_path = Path(args.continuation_config)
    cfg = load_config(continuation_path)
    dump_json(
        {
            "schema_version": 1,
            "updated_at": utc_timestamp(),
            "claim_status": "locked_split_continuation_not_record_evidence",
            "not_solution": True,
            "status": status,
            "continuation_config": file_metadata(continuation_path),
            "source_controller": file_metadata(Path(args.source_controller)),
            "link_count": int(cfg["env"]["n_links"]),
            "progress": float(schedule.progress),
            "next_step": float(schedule.step),
            "current_controller": file_metadata(current_controller),
            "locked_start_baseline": baseline,
            "endpoint_dimensionless": {
                "locked_start": dimensionless_setup(
                    setup_from_config(cfg, progress=0.0)
                ).to_dict(),
                "unlocked_target": dimensionless_setup(
                    setup_from_config(cfg, progress=1.0)
                ).to_dict(),
            },
            "trials": trials,
        },
        path,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-config", required=True)
    parser.add_argument("--source-controller", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-step", type=float, default=0.001)
    parser.add_argument("--minimum-step", type=float, default=1e-5)
    parser.add_argument("--maximum-step", type=float, default=0.05)
    parser.add_argument("--growth", type=float, default=1.6)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--retry-iterations", type=int, default=100)
    parser.add_argument("--max-trials", type=int, default=200)
    parser.add_argument("--tracking-gain", type=float, default=1.0)
    parser.add_argument("--disable-waypoint-repair", action="store_true")
    parser.add_argument("--waypoint-segment-steps", type=int, default=24)
    parser.add_argument(
        "--waypoint-segment-multipliers", type=int, nargs="+", default=[1, 2, 4]
    )
    parser.add_argument("--waypoint-max-evaluations", type=int, default=120)
    parser.add_argument("--waypoint-endpoint-weight", type=float, default=10_000.0)
    parser.add_argument(
        "--waypoint-control-regularization", type=float, default=1e-6
    )
    parser.add_argument("--waypoint-rail-soft-margin", type=float, default=0.5)
    parser.add_argument("--waypoint-rail-weight", type=float, default=1_000_000.0)
    parser.add_argument("--waypoint-endpoint-tolerance", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate inputs and write a ready ledger without starting optimization",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    positive = (
        args.initial_step,
        args.minimum_step,
        args.maximum_step,
        args.growth,
        args.iterations,
        args.retry_iterations,
        args.max_trials,
        args.tracking_gain,
        args.waypoint_segment_steps,
        args.waypoint_max_evaluations,
        args.waypoint_endpoint_weight,
        args.waypoint_control_regularization,
        args.waypoint_rail_soft_margin,
        args.waypoint_rail_weight,
        args.waypoint_endpoint_tolerance,
    )
    if min(positive) <= 0:
        raise ValueError("steps, budgets, solver weights, and bounds must be positive")
    if args.minimum_step > args.initial_step or args.initial_step > args.maximum_step:
        raise ValueError("require minimum-step <= initial-step <= maximum-step")
    waypoint_lookaheads(
        args.waypoint_segment_steps, args.waypoint_segment_multipliers
    )
    cfg = load_config(args.continuation_config)
    count = int(cfg["env"]["n_links"])
    if controller_link_count(Path(args.source_controller)) != count:
        raise ValueError("source controller and continuation link counts disagree")
    start = build_morphology(cfg["env"], cfg["morphology"], progress=0.0)
    end = build_morphology(cfg["env"], cfg["morphology"], progress=1.0)
    if not np.any(start.joint_lock > 0.0):
        raise ValueError("continuation start must contain at least one locked joint")
    if np.any(end.joint_lock > 0.0):
        raise ValueError("continuation target must release every inserted joint")
    return cfg, count


def main() -> None:
    args = parse_args()
    continuation, count = validate_args(args)
    waypoint_steps = waypoint_lookaheads(
        args.waypoint_segment_steps, args.waypoint_segment_multipliers
    )
    output_dir = Path(args.output_dir)
    configs_dir = output_dir / "configs"
    trials_dir = output_dir / "trials"
    configs_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"hanging_n{count}.json"
    hanging_state(count, state_path)
    manifest_path = output_dir / "continuation.json"

    if args.resume:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        current_meta = file_metadata(Path(args.continuation_config))
        if saved["continuation_config"]["sha256"] != current_meta["sha256"]:
            raise ValueError("saved ledger uses a different continuation config")
        schedule = AdaptiveHomotopy(
            progress=float(saved["progress"]),
            step=float(saved["next_step"]),
            minimum_step=args.minimum_step,
            maximum_step=args.maximum_step,
            growth=args.growth,
        )
        current_controller = Path(saved["current_controller"]["path"])
        baseline = saved.get("locked_start_baseline")
        if not isinstance(baseline, dict) or not baseline.get("accepted"):
            raise ValueError("saved ledger has no accepted locked-start baseline")
        trials = list(saved["trials"])
    else:
        schedule = AdaptiveHomotopy(
            step=args.initial_step,
            minimum_step=args.minimum_step,
            maximum_step=args.maximum_step,
            growth=args.growth,
        )
        current_controller = Path(args.source_controller)
        baseline = None
        trials: list[dict[str, Any]] = []

    write_manifest(
        manifest_path,
        args,
        schedule,
        current_controller,
        baseline,
        trials,
        "ready" if args.dry_run else "running",
    )
    if args.dry_run:
        print(f"validated locked-split continuation; wrote {manifest_path}")
        return

    if baseline is None:
        baseline_cfg = freeze_progress_config(
            continuation, 0.0, "locked_split_baseline"
        )
        baseline_cfg_path = configs_dir / "locked_split_baseline.yaml"
        save_config(baseline_cfg, baseline_cfg_path)
        baseline_first = trials_dir / "locked_split_baseline_pass1.json"
        baseline_path = baseline_first
        baseline_passed = False
        baseline_waypoints: list[dict[str, Any]] = []
        if not args.disable_waypoint_repair:
            for segment_steps in waypoint_steps:
                multiplier = segment_steps // args.waypoint_segment_steps
                suffix = (
                    "waypoint" if multiplier == 1 else f"waypoint_x{multiplier}"
                )
                waypoint_path = trials_dir / f"locked_split_baseline_{suffix}.json"
                refined_path = (
                    trials_dir / f"locked_split_baseline_{suffix}_fddp.json"
                )
                run(
                    waypoint_command(
                        cfg=baseline_cfg_path,
                        controller=current_controller,
                        output=waypoint_path,
                        segment_steps=segment_steps,
                        max_evaluations=args.waypoint_max_evaluations,
                        endpoint_weight=args.waypoint_endpoint_weight,
                        control_regularization=args.waypoint_control_regularization,
                        rail_soft_margin=args.waypoint_rail_soft_margin,
                        rail_weight=args.waypoint_rail_weight,
                        endpoint_tolerance=args.waypoint_endpoint_tolerance,
                    )
                )
                search_passed = waypoint_successful(waypoint_path)
                usable = waypoint_usable(waypoint_path)
                refinement_passed = False
                if usable:
                    run(
                        fddp_command(
                            cfg=baseline_cfg_path,
                            state=state_path,
                            controller=waypoint_path,
                            output=refined_path,
                            iterations=args.iterations,
                            regularization=1e-6,
                            tracking_gain=args.tracking_gain,
                            exact_initial_trajectory=True,
                            rail_soft_margin=args.waypoint_rail_soft_margin,
                        )
                    )
                    baseline_path = refined_path
                    refinement_passed = successful(refined_path)
                    baseline_passed = refinement_passed
                baseline_waypoints.append(
                    {
                        "segment_steps": int(segment_steps),
                        "search_passed": bool(search_passed),
                        "used_for_refinement": bool(usable),
                        "refinement_passed": bool(refinement_passed),
                        "artifact": file_metadata(waypoint_path),
                        "refinement": file_metadata(refined_path) if usable else None,
                    }
                )
                if baseline_passed:
                    break
        if not baseline_passed:
            run(
                fddp_command(
                    cfg=baseline_cfg_path,
                    state=state_path,
                    controller=current_controller,
                    output=baseline_first,
                    iterations=args.iterations,
                    regularization=1e-6,
                    tracking_gain=args.tracking_gain,
                    rail_soft_margin=args.waypoint_rail_soft_margin,
                )
            )
            baseline_path = baseline_first
            baseline_passed = successful(baseline_first)
        if not baseline_passed:
            baseline_retry = trials_dir / "locked_split_baseline_pass2.json"
            run(
                fddp_command(
                    cfg=baseline_cfg_path,
                    state=state_path,
                    controller=baseline_first,
                    output=baseline_retry,
                    iterations=args.retry_iterations,
                    regularization=1e-7,
                    tracking_gain=0.5 * args.tracking_gain,
                    rail_soft_margin=args.waypoint_rail_soft_margin,
                )
            )
            baseline_path = baseline_retry
            baseline_passed = successful(baseline_retry)
        baseline = {
            "accepted": bool(baseline_passed),
            "config": file_metadata(baseline_cfg_path),
            "result": file_metadata(baseline_path),
            "outcome": result_summary(baseline_path, baseline_cfg),
            "waypoint_attempts": baseline_waypoints,
        }
        if baseline_passed:
            current_controller = baseline_path
        write_manifest(
            manifest_path,
            args,
            schedule,
            current_controller,
            baseline,
            trials,
            "running" if baseline_passed else "locked_start_exact_replay_failed",
        )
        if not baseline_passed:
            print(
                "Locked-start exact replay failed; continuation was not started.",
                flush=True,
            )
            raise SystemExit(3)

    start_trial = len(trials) + 1
    for trial_index in range(start_trial, args.max_trials + 1):
        proposed = schedule.proposal()
        label = f"trial_{trial_index:04d}_p{proposed:.9f}"
        trial_cfg = freeze_progress_config(continuation, proposed, label)
        cfg_path = configs_dir / f"{label}.yaml"
        save_config(trial_cfg, cfg_path)
        first_path = trials_dir / f"{label}_pass1.json"
        accepted_path = first_path
        passed = False
        waypoint_attempts: list[dict[str, Any]] = []

        if not args.disable_waypoint_repair:
            for segment_steps in waypoint_steps:
                multiplier = segment_steps // args.waypoint_segment_steps
                suffix = "waypoint" if multiplier == 1 else f"waypoint_x{multiplier}"
                waypoint_path = trials_dir / f"{label}_{suffix}.json"
                refined_path = trials_dir / f"{label}_{suffix}_fddp.json"
                run(
                    waypoint_command(
                        cfg=cfg_path,
                        controller=current_controller,
                        output=waypoint_path,
                        segment_steps=segment_steps,
                        max_evaluations=args.waypoint_max_evaluations,
                        endpoint_weight=args.waypoint_endpoint_weight,
                        control_regularization=args.waypoint_control_regularization,
                        rail_soft_margin=args.waypoint_rail_soft_margin,
                        rail_weight=args.waypoint_rail_weight,
                        endpoint_tolerance=args.waypoint_endpoint_tolerance,
                    )
                )
                search_passed = waypoint_successful(waypoint_path)
                usable = waypoint_usable(waypoint_path)
                refinement_passed = False
                if usable:
                    run(
                        fddp_command(
                            cfg=cfg_path,
                            state=state_path,
                            controller=waypoint_path,
                            output=refined_path,
                            iterations=args.iterations,
                            regularization=1e-6,
                            tracking_gain=args.tracking_gain,
                            exact_initial_trajectory=True,
                            rail_soft_margin=args.waypoint_rail_soft_margin,
                        )
                    )
                    accepted_path = refined_path
                    refinement_passed = successful(refined_path)
                    passed = refinement_passed
                waypoint_attempts.append(
                    {
                        "segment_steps": int(segment_steps),
                        "search_passed": bool(search_passed),
                        "used_for_refinement": bool(usable),
                        "refinement_passed": bool(refinement_passed),
                        "artifact": file_metadata(waypoint_path),
                        "refinement": file_metadata(refined_path) if usable else None,
                    }
                )
                if passed:
                    break

        if not passed:
            run(
                fddp_command(
                    cfg=cfg_path,
                    state=state_path,
                    controller=current_controller,
                    output=first_path,
                    iterations=args.iterations,
                    regularization=1e-6,
                    tracking_gain=args.tracking_gain,
                    rail_soft_margin=args.waypoint_rail_soft_margin,
                )
            )
            accepted_path = first_path
            passed = successful(first_path)
        if not passed:
            retry_path = trials_dir / f"{label}_pass2.json"
            run(
                fddp_command(
                    cfg=cfg_path,
                    state=state_path,
                    controller=first_path,
                    output=retry_path,
                    iterations=args.retry_iterations,
                    regularization=1e-7,
                    tracking_gain=0.5 * args.tracking_gain,
                    rail_soft_margin=args.waypoint_rail_soft_margin,
                )
            )
            accepted_path = retry_path
            passed = successful(retry_path)

        record = {
            "trial": int(trial_index),
            "from_progress": float(schedule.progress),
            "proposed_progress": float(proposed),
            "step": float(proposed - schedule.progress),
            "accepted": bool(passed),
            "lock_strengths": build_morphology(
                trial_cfg["env"], trial_cfg["morphology"], progress=1.0
            ).joint_lock.tolist(),
            "config": file_metadata(cfg_path),
            "result": file_metadata(accepted_path),
            "outcome": result_summary(accepted_path, trial_cfg),
            "dimensionless": dimensionless_setup(
                setup_from_config(trial_cfg)
            ).to_dict(),
            "waypoint_attempts": waypoint_attempts,
        }
        trials.append(record)
        if passed:
            schedule.accept(proposed)
            current_controller = accepted_path
            status = (
                "unlocked_target_exact_replay_passed"
                if np.isclose(schedule.progress, 1.0)
                else "running"
            )
            write_manifest(
                manifest_path,
                args,
                schedule,
                current_controller,
                baseline,
                trials,
                status,
            )
            print(
                f"ACCEPT p={schedule.progress:.9f}; next step={schedule.step:.9f}",
                flush=True,
            )
            if np.isclose(schedule.progress, 1.0):
                print(
                    "Unlocked target passed exact replay. Package its mirror and "
                    "run the independent noisy gates before promotion.",
                    flush=True,
                )
                return
        else:
            try:
                schedule.reject()
            except RuntimeError as error:
                write_manifest(
                    manifest_path,
                    args,
                    schedule,
                    current_controller,
                    baseline,
                    trials,
                    "minimum_step_frontier_reached",
                )
                raise SystemExit(4) from error
            write_manifest(
                manifest_path,
                args,
                schedule,
                current_controller,
                baseline,
                trials,
                "running",
            )
            print(
                f"REJECT p={proposed:.9f}; bisected step={schedule.step:.9f}",
                flush=True,
            )

    write_manifest(
        manifest_path,
        args,
        schedule,
        current_controller,
        baseline,
        trials,
        "trial_budget_exhausted",
    )
    print(
        f"Trial budget exhausted at p={schedule.progress:.9f}; "
        f"frontier recorded in {manifest_path}",
        flush=True,
    )
    raise SystemExit(5)


if __name__ == "__main__":
    main()
