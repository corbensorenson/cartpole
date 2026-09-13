#!/usr/bin/env python
"""Run deterministic adaptive morphology continuation between adjacent chains.

Each accepted step must pass an uninterrupted exact-MuJoCo swing-up-and-hold
replay. Failed steps receive one same-plant refinement, then are bisected. The
driver is resumable and records every accepted and rejected trial.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config, save_config
from gcartpole.evidence import file_metadata, utc_timestamp
from gcartpole.generalized_solver import (
    AdaptiveHomotopy,
    dimensionless_setup,
    homotopy_morphology,
    setup_from_config,
)

ROOT = Path(__file__).resolve().parents[1]


def uniform_target(base: dict[str, Any], n_links: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["env"]["n_links"] = int(n_links)
    cfg["experiment"]["name"] = f"generalized_uniform_n{n_links}"
    for name in ("lengths", "masses", "joint_stiffness", "joint_lock"):
        cfg["morphology"].pop(f"{name}_start", None)
        cfg["morphology"].pop(f"{name}_end", None)
    for endpoint in ("start", "end"):
        cfg["morphology"][endpoint]["alpha_length"] = 0.0
        cfg["morphology"][endpoint]["alpha_mass"] = 0.0
        cfg["morphology"][endpoint]["alpha_damping"] = 0.0
    return cfg


def explicit_config(
    base: dict[str, Any], lengths: np.ndarray, masses: np.ndarray, name: str
) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["env"]["n_links"] = int(lengths.size)
    cfg["env"]["total_length"] = float(np.sum(lengths))
    cfg["env"]["total_mass"] = float(np.sum(masses))
    cfg["experiment"]["name"] = name
    cfg["morphology"]["lengths_start"] = lengths.astype(float).tolist()
    cfg["morphology"]["lengths_end"] = lengths.astype(float).tolist()
    cfg["morphology"]["masses_start"] = masses.astype(float).tolist()
    cfg["morphology"]["masses_end"] = masses.astype(float).tolist()
    for profile_name in ("joint_stiffness", "joint_lock"):
        cfg["morphology"].pop(f"{profile_name}_start", None)
        cfg["morphology"].pop(f"{profile_name}_end", None)
    return cfg


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def fddp_command(
    *,
    cfg: Path,
    state: Path,
    controller: Path,
    output: Path,
    iterations: int,
    regularization: float,
    tracking_gain: float,
    exact_initial_trajectory: bool = False,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/search_fddp_capture.py",
        "--config", str(cfg),
        "--state-json", str(state),
        "--state-index", "selected",
        "--initial-controller", str(controller),
        "--iterations", str(iterations),
        "--initial-regularization", str(regularization),
        "--tracking-gain-scale", str(tracking_gain),
        "--lqr-scale", "1",
        "--lqr-control-cost", "1000",
        "--control-cost", "0.01",
        "--stage-weight", "0.1",
        "--terminal-weight", "10",
        "--terminal-state-weight", "100000",
        "--terminal-cart-weight", "1000",
        "--terminal-cart-velocity-weight", "1000",
        "--rail-soft-limit", str(0.8333333333333333 * float(load_config(cfg)["env"]["rail_limit"])),
        "--rail-weight", "5000000",
        "--handoff-lyapunov", "1800",
        "--handoff-cart-abs", str(0.5 * float(load_config(cfg)["env"]["rail_limit"])),
        "--handoff-angle-abs", "0.15",
        "--handoff-cart-velocity-abs", "0.5",
        "--handoff-hinge-velocity-rms", "0.75",
        "--defer-handoff-until-horizon",
        "--allow-unstable-lyapunov",
        "--out", str(output),
    ]
    warm_flag = "--initial-feasible" if exact_initial_trajectory else "--rebuild-initial-feedback"
    command.insert(command.index("--iterations"), warm_flag)
    return command


def waypoint_command(
    *,
    cfg: Path,
    controller: Path,
    output: Path,
    segment_steps: int,
    max_evaluations: int,
    endpoint_weight: float,
    control_regularization: float,
    rail_soft_margin: float,
    rail_weight: float,
    endpoint_tolerance: float,
) -> list[str]:
    rail_limit = float(load_config(cfg)["env"]["rail_limit"])
    return [
        sys.executable,
        "scripts/adapt_generalized_route_waypoints.py",
        "--config", str(cfg),
        "--controller", str(controller),
        "--segment-steps", str(segment_steps),
        "--max-evaluations", str(max_evaluations),
        "--endpoint-weight", str(endpoint_weight),
        "--control-regularization", str(control_regularization),
        "--rail-soft-limit", str(max(1.0e-3, rail_limit - rail_soft_margin)),
        "--rail-weight", str(rail_weight),
        "--endpoint-tolerance", str(endpoint_tolerance),
        "--out", str(output),
    ]


def successful(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = payload.get("result", {})
    return bool(result.get("success")) and bool(result.get("latched"))


def waypoint_successful(path: Path) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return bool(payload.get("search", {}).get("success"))


def waypoint_usable(path: Path) -> bool:
    """Require an exact, finite, rail-safe trajectory before FDDP refinement."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    controller = payload.get("controller", {})
    controls = np.asarray(controller.get("controls"), dtype=np.float64)
    states = np.asarray(
        controller.get("nominal_coordinate_states"), dtype=np.float64
    )
    search = payload.get("search", {})
    maximum_cart = float(search.get("maximum_cart_excursion", float("inf")))
    rail_soft_limit = float(search.get("rail_soft_limit", 0.0))
    return bool(
        controls.ndim == 1
        and controls.size > 0
        and states.ndim == 2
        and states.shape[0] == controls.size + 1
        and np.all(np.isfinite(controls))
        and np.all(np.isfinite(states))
        and maximum_cart <= rail_soft_limit
    )


def hanging_state(count: int, path: Path) -> None:
    qpos = [0.0, float(np.pi)] + [0.0] * (count - 1)
    dump_json(
        {"selected_state": {"qpos": qpos, "qvel": [0.0] * (count + 1), "state_index": 0}},
        path,
    )


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    schedule: AdaptiveHomotopy,
    current_controller: Path,
    trials: list[dict[str, Any]],
    status: str,
) -> None:
    dump_json(
        {
            "schema_version": 1,
            "updated_at": utc_timestamp(),
            "claim_status": "development_continuation_not_record_evidence",
            "status": status,
            "source_config": file_metadata(Path(args.source_config)),
            "target_links": int(args.target_links),
            "progress": float(schedule.progress),
            "next_step": float(schedule.step),
            "current_controller": file_metadata(current_controller),
            "trials": trials,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--source-controller", required=True)
    parser.add_argument("--target-links", type=int, default=None)
    parser.add_argument("--target-config", default=None)
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
    parser.add_argument("--waypoint-max-evaluations", type=int, default=120)
    parser.add_argument("--waypoint-endpoint-weight", type=float, default=10_000.0)
    parser.add_argument("--waypoint-control-regularization", type=float, default=1e-6)
    parser.add_argument("--waypoint-rail-soft-margin", type=float, default=0.5)
    parser.add_argument("--waypoint-rail-weight", type=float, default=1_000_000.0)
    parser.add_argument("--waypoint-endpoint-tolerance", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    source_cfg = load_config(args.source_config)
    source = setup_from_config(source_cfg)
    if args.target_config is not None:
        target_cfg = load_config(args.target_config)
        target_links = int(target_cfg["env"]["n_links"])
        if args.target_links is not None and args.target_links != target_links:
            raise ValueError("--target-links disagrees with --target-config")
        args.target_links = target_links
    else:
        if args.target_links is None:
            raise ValueError("provide --target-links or --target-config")
        target_cfg = uniform_target(source_cfg, args.target_links)
    if abs(args.target_links - source.n_links) > 1:
        raise ValueError("run link-count continuations one adjacent count at a time")
    target = setup_from_config(target_cfg)
    if not np.isclose(source.chain_length, target.chain_length) or not np.isclose(source.link_mass, target.link_mass):
        raise ValueError("source and target must preserve total chain length and link mass")
    common_count = max(source.n_links, target.n_links)
    if common_count != source.n_links:
        raise ValueError(
            "upward continuation first needs the source controller embedded in the larger count; "
            "use generalized_swingup_solver.py transfer, then resume this driver"
        )

    output_dir = Path(args.output_dir)
    configs_dir = output_dir / "configs"
    trials_dir = output_dir / "trials"
    configs_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    target_cfg_path = output_dir / f"target_n{args.target_links}.yaml"
    save_config(target_cfg, target_cfg_path)
    state_path = output_dir / f"hanging_n{common_count}.json"
    hanging_state(common_count, state_path)
    manifest_path = output_dir / "continuation.json"

    if args.resume:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(saved["target_links"]) != args.target_links:
            raise ValueError("saved continuation target does not match --target-links")
        schedule = AdaptiveHomotopy(
            progress=float(saved["progress"]),
            step=float(saved["next_step"]),
            minimum_step=args.minimum_step,
            maximum_step=args.maximum_step,
            growth=args.growth,
        )
        current_controller = Path(saved["current_controller"]["path"])
        trials = list(saved["trials"])
    else:
        schedule = AdaptiveHomotopy(
            step=args.initial_step,
            minimum_step=args.minimum_step,
            maximum_step=args.maximum_step,
            growth=args.growth,
        )
        current_controller = Path(args.source_controller)
        trials = []
    start_trial = len(trials) + 1
    for trial_index in range(start_trial, args.max_trials + 1):
        proposed = schedule.proposal()
        lengths, masses = homotopy_morphology(
            source.lengths,
            source.masses,
            target.lengths,
            target.masses,
            proposed,
        )
        label = f"trial_{trial_index:04d}_p{proposed:.9f}"
        # Continue only the chain distribution here. Every other physical
        # quantity (rail, cart mass, force authority, timing, damping, body
        # geometry) comes from the measured target plant throughout, so the
        # accepted p=1 controller is already on the exact target contract.
        trial_cfg = explicit_config(target_cfg, lengths, masses, label)
        cfg_path = configs_dir / f"{label}.yaml"
        save_config(trial_cfg, cfg_path)
        waypoint_path = trials_dir / f"{label}_waypoint.json"
        waypoint_fddp_path = trials_dir / f"{label}_waypoint_fddp.json"
        first_path = trials_dir / f"{label}_pass1.json"
        waypoint_attempted = not args.disable_waypoint_repair
        waypoint_passed = False
        waypoint_refined = False
        passed = False
        accepted_path = first_path
        if waypoint_attempted:
            run(
                waypoint_command(
                    cfg=cfg_path,
                    controller=current_controller,
                    output=waypoint_path,
                    segment_steps=args.waypoint_segment_steps,
                    max_evaluations=args.waypoint_max_evaluations,
                    endpoint_weight=args.waypoint_endpoint_weight,
                    control_regularization=args.waypoint_control_regularization,
                    rail_soft_margin=args.waypoint_rail_soft_margin,
                    rail_weight=args.waypoint_rail_weight,
                    endpoint_tolerance=args.waypoint_endpoint_tolerance,
                )
            )
            waypoint_passed = waypoint_successful(waypoint_path)
            waypoint_refined = waypoint_usable(waypoint_path)
            if waypoint_refined:
                run(
                    fddp_command(
                        cfg=cfg_path,
                        state=state_path,
                        controller=waypoint_path,
                        output=waypoint_fddp_path,
                        iterations=args.iterations,
                        regularization=1e-6,
                        tracking_gain=args.tracking_gain,
                        exact_initial_trajectory=True,
                    )
                )
                accepted_path = waypoint_fddp_path
                passed = successful(waypoint_fddp_path)
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
                )
            )
            accepted_path = retry_path
            passed = successful(retry_path)
        record = {
            "trial": trial_index,
            "from_progress": float(schedule.progress),
            "proposed_progress": float(proposed),
            "step": float(proposed - schedule.progress),
            "accepted": bool(passed),
            "config": file_metadata(cfg_path),
            "result": file_metadata(accepted_path),
            "dimensionless": dimensionless_setup(setup_from_config(trial_cfg)).to_dict(),
        }
        if waypoint_attempted:
            record["waypoint"] = {
                "search_passed": waypoint_passed,
                "used_for_refinement": waypoint_refined,
                "artifact": file_metadata(waypoint_path),
                "refinement": (
                    file_metadata(waypoint_fddp_path)
                    if waypoint_passed
                    else None
                ),
            }
        trials.append(record)
        if passed:
            schedule.accept(proposed)
            current_controller = accepted_path
            write_manifest(manifest_path, args, schedule, current_controller, trials, "running")
            print(f"ACCEPT p={schedule.progress:.9f}; next step={schedule.step:.9f}", flush=True)
            if np.isclose(schedule.progress, 1.0):
                write_manifest(manifest_path, args, schedule, current_controller, trials, "embedded_target_solved")
                print(
                    "Embedded target solved. Project the endpoint to the target count with "
                    "generalized_swingup_solver.py transfer, then run one final exact refinement.",
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
                    trials,
                    "minimum_step_frontier_reached",
                )
                print(
                    f"REJECT p={proposed:.9f}; minimum step frontier recorded in "
                    f"{manifest_path}",
                    flush=True,
                )
                raise SystemExit(4) from error
            write_manifest(manifest_path, args, schedule, current_controller, trials, "running")
            print(f"REJECT p={proposed:.9f}; bisected step={schedule.step:.9f}", flush=True)
    write_manifest(manifest_path, args, schedule, current_controller, trials, "trial_budget_exhausted")
    print(
        f"Trial budget exhausted at p={schedule.progress:.9f}; "
        f"frontier recorded in {manifest_path}",
        flush=True,
    )
    raise SystemExit(5)


if __name__ == "__main__":
    main()
