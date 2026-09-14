#!/usr/bin/env python
"""Adaptively continue one accepted route through a physical similarity family.

The driver contains no link-count cases.  It transfers and gates a proposed
dimensionless scale waypoint, materializes passing target-plant feedback replay,
and refines the last accepted anchor only when the next transfer exposes a thin
capture basin.  Failed proposals shrink the scale step and are never promoted.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import (
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.generalized_solver import setup_from_config
from scripts.solve_generalized_morphology import source_link_count

ROOT = Path(__file__).resolve().parents[1]


def scale_at_progress(target_scale: float, progress: float, *, path: str) -> float:
    if target_scale <= 0.0:
        raise ValueError("target scale must be positive")
    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    if path == "linear":
        return 1.0 + progress * (target_scale - 1.0)
    if path == "log":
        return math.exp(progress * math.log(target_scale))
    raise ValueError(f"unsupported scale path: {path}")


def next_step(step: float, *, accepted: bool, growth: float, minimum: float, maximum: float) -> float:
    if accepted:
        return min(maximum, step * growth)
    return max(minimum, 0.5 * step)


def feedback_gain_candidates(values: list[float] | None) -> list[float]:
    """Return a stable, duplicate-free bounded feedback screen."""

    requested = values or [0.75, 1.25, 0.5, 1.5, 2.0]
    result: list[float] = []
    for value in requested:
        candidate = float(value)
        if not math.isfinite(candidate) or candidate <= 0.0:
            raise ValueError("feedback gain candidates must be finite and positive")
        if candidate != 1.0 and candidate not in result:
            result.append(candidate)
    return result


def segment_feedback_candidates(values: list[str] | None) -> list[tuple[float, float]]:
    """Return bounded swing/tail feedback pairs in deterministic priority order."""

    requested = values or ["1,1.25", "1,1.5", "1,0.75", "0.75,1", "1.25,1"]
    result: list[tuple[float, float]] = []
    for value in requested:
        fields = value.split(",")
        if len(fields) != 2:
            raise ValueError("segment feedback candidates must be SWING,TAIL pairs")
        candidate = (float(fields[0]), float(fields[1]))
        if any(not math.isfinite(item) or item <= 0.0 for item in candidate):
            raise ValueError("segment feedback candidates must be finite and positive")
        if candidate != (1.0, 1.0) and candidate not in result:
            result.append(candidate)
    return result


def run(command: list[str], *, dry_run: bool) -> None:
    print("+", shlex.join(command), flush=True)
    if dry_run:
        return
    environment = os.environ.copy()
    python_path = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def load_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def passed_gate(path: Path, episodes: int) -> bool:
    payload = load_payload(path)
    return bool(
        int(payload.get("episodes", 0)) >= episodes
        and float(payload.get("success_rate", 0.0)) == 1.0
        and len(payload.get("episode_results", [])) >= episodes
    )


def successful_replay(path: Path) -> bool:
    result = load_payload(path).get("result")
    return bool(
        isinstance(result, dict)
        and result.get("success") is True
        and result.get("latched") is True
    )


def materialized_route(path: Path) -> bool:
    controller = load_payload(path).get("controller")
    return bool(
        isinstance(controller, dict)
        and int(controller.get("materialized_lqr_tail_steps", 0)) > 0
    )


def controller_settings(path: Path) -> dict[str, float]:
    controller = load_payload(path).get("controller")
    if not isinstance(controller, dict):
        raise TypeError(f"{path} does not contain a controller object")
    weights = {
        key: float(value)
        for key, value in controller.get("lqr_weights", {}).items()
    }
    defaults = {
        "cart_position": 0.1,
        "absolute_angle": 100.0,
        "cart_velocity": 0.1,
        "absolute_angular_velocity": 1.0,
        "relative_angle": 1.0,
        "relative_angular_velocity": 0.01,
    }
    defaults.update(weights)
    return {
        "lqr_scale": float(controller.get("lqr_scale", 1.0)),
        "lqr_control_cost": float(controller.get("lqr_control_cost", 1000.0)),
        "handoff_angle_abs": float(controller.get("handoff_angle_abs", 0.15)),
        "handoff_cart_abs": float(controller.get("handoff_cart_abs", 1.25)),
        "handoff_cart_velocity_abs": float(
            controller.get("handoff_cart_velocity_abs", 0.5)
        ),
        "handoff_hinge_velocity_rms": float(
            controller.get("handoff_hinge_velocity_rms", 0.75)
        ),
        **defaults,
    }


def gate_command(
    python: str,
    config: Path,
    route: Path,
    mirror: Path,
    output: Path,
    *,
    episodes: int,
    seed: int,
    conditioning_seconds: float,
    tracking_gain: float,
    phase_window: int,
) -> list[str]:
    return [
        python,
        "scripts/evaluate_generalized_route_library.py",
        "--config",
        str(config),
        "--controller",
        str(route),
        "--controller",
        str(mirror),
        "--episodes",
        str(episodes),
        "--seed",
        str(seed),
        "--conditioning-seconds",
        str(conditioning_seconds),
        "--tracking-gain-scale",
        str(tracking_gain),
        "--phase-window",
        str(phase_window),
        "--out",
        str(output),
    ]


def replay_command(
    python: str,
    config: Path,
    route: Path,
    output: Path,
    *,
    rail_soft_margin: float,
    iterations: int,
    replay_only: bool,
) -> list[str]:
    settings = controller_settings(route)
    rail = setup_from_config(load_config(config)).rail_half_length
    command = [
        python,
        "scripts/search_fddp_capture.py",
        "--config",
        str(config),
        "--state-json",
        str(route),
        "--state-index",
        "selected",
        "--initial-controller",
        str(route),
        "--rebuild-initial-feedback",
        "--initial-feedback-scale",
        "1",
        "--initial-feasible",
        "--iterations",
        str(iterations),
        "--initial-regularization",
        "1e-4",
        "--tracking-gain-scale",
        "1",
        "--lqr-scale",
        str(settings["lqr_scale"]),
        "--lqr-control-cost",
        str(settings["lqr_control_cost"]),
        "--lqr-cart-position-cost",
        str(settings["cart_position"]),
        "--lqr-absolute-angle-cost",
        str(settings["absolute_angle"]),
        "--lqr-cart-velocity-cost",
        str(settings["cart_velocity"]),
        "--lqr-absolute-angular-velocity-cost",
        str(settings["absolute_angular_velocity"]),
        "--lqr-relative-angle-cost",
        str(settings["relative_angle"]),
        "--lqr-relative-angular-velocity-cost",
        str(settings["relative_angular_velocity"]),
        "--control-cost",
        "0.1",
        "--stage-weight",
        "0.01",
        "--terminal-weight",
        "1000",
        "--terminal-state-weight",
        "10000",
        "--terminal-cart-weight",
        "100",
        "--terminal-cart-velocity-weight",
        "100",
        "--terminal-hinge-velocity-factor",
        "2",
        "--rail-soft-limit",
        str(max(1e-3, rail - rail_soft_margin)),
        "--rail-weight",
        "1000000",
        "--handoff-angle-abs",
        str(settings["handoff_angle_abs"]),
        "--handoff-cart-abs",
        str(settings["handoff_cart_abs"]),
        "--handoff-cart-velocity-abs",
        str(settings["handoff_cart_velocity_abs"]),
        "--handoff-hinge-velocity-rms",
        str(settings["handoff_hinge_velocity_rms"]),
        "--defer-handoff-until-horizon",
        "--allow-unstable-lyapunov",
        "--out",
        str(output),
    ]
    if replay_only:
        command.insert(command.index("--iterations"), "--replay-only")
    return command


def materialize_command(
    python: str,
    config: Path,
    replay: Path,
    output: Path,
) -> list[str]:
    return [
        python,
        "scripts/materialize_generalized_rollout.py",
        "--config",
        str(config),
        "--replay",
        str(replay),
        "--out",
        str(output),
    ]


def scaled_pair(
    python: str,
    source: Path,
    route: Path,
    mirror: Path,
    *,
    scale: float,
    dry_run: bool,
) -> None:
    run(
        [
            python,
            "scripts/package_generalized_route.py",
            "--controller",
            str(source),
            "--feedback-gain-scale",
            str(scale),
            "--out",
            str(route),
        ],
        dry_run=dry_run,
    )
    run(
        [
            python,
            "scripts/mirror_generalized_route.py",
            "--controller",
            str(route),
            "--out",
            str(mirror),
        ],
        dry_run=dry_run,
    )


def segment_scaled_pair(
    python: str,
    source: Path,
    route: Path,
    mirror: Path,
    *,
    swing_scale: float,
    tail_scale: float,
    dry_run: bool,
) -> None:
    run(
        [
            python,
            "scripts/retune_generalized_route.py",
            "--controller",
            str(source),
            "--swing-feedback-scale",
            str(swing_scale),
            "--tail-feedback-scale",
            str(tail_scale),
            "--out",
            str(route),
        ],
        dry_run=dry_run,
    )
    run(mirror_command(python, route, mirror), dry_run=dry_run)


def mirror_command(python: str, route: Path, mirror: Path) -> list[str]:
    return [
        python,
        "scripts/mirror_generalized_route.py",
        "--controller",
        str(route),
        "--out",
        str(mirror),
    ]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-config", required=True)
    result.add_argument("--source-controller", required=True)
    result.add_argument("--target-length-scale", type=float, required=True)
    result.add_argument("--target-mass-scale", type=float, required=True)
    result.add_argument("--path", choices=("linear", "log"), default="linear")
    result.add_argument("--initial-step", type=float, default=0.1)
    result.add_argument("--minimum-step", type=float, default=0.005)
    result.add_argument("--maximum-step", type=float, default=0.2)
    result.add_argument("--step-growth", type=float, default=1.5)
    result.add_argument("--max-trials", type=int, default=80)
    result.add_argument("--probe-episodes", type=int, default=5)
    result.add_argument("--promotion-episodes", type=int, default=20)
    result.add_argument("--seed", type=int, default=76001)
    result.add_argument("--source-conditioning-seconds", type=float, default=15.0)
    result.add_argument("--tracking-gain", type=float, default=1.0)
    result.add_argument(
        "--feedback-gain-candidate",
        action="append",
        type=float,
        help=(
            "bounded multiplicative feedback candidate; repeat to replace the "
            "default deterministic screen (0.75, 1.25, 0.5, 1.5, 2.0)"
        ),
    )
    result.add_argument(
        "--segment-feedback-candidate",
        action="append",
        help=(
            "bounded SWING,TAIL gain pair; repeat to replace the default "
            "segment screen"
        ),
    )
    result.add_argument("--phase-window", type=int, default=0)
    result.add_argument("--refinement-iterations", type=int, default=20)
    result.add_argument("--rail-soft-margin", type=float, default=0.5)
    result.add_argument("--output-dir", required=True)
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def validate_args(args: argparse.Namespace) -> None:
    if min(args.target_length_scale, args.target_mass_scale) <= 0.0:
        raise ValueError("target scales must be positive")
    if not 0.0 < args.minimum_step <= args.initial_step <= args.maximum_step <= 1.0:
        raise ValueError("step sizes must satisfy 0 < minimum <= initial <= maximum <= 1")
    if args.step_growth < 1.0:
        raise ValueError("step growth must be at least one")
    if not math.isfinite(args.tracking_gain) or args.tracking_gain <= 0.0:
        raise ValueError("tracking gain must be finite and positive")
    feedback_gain_candidates(args.feedback_gain_candidate)
    segment_feedback_candidates(args.segment_feedback_candidate)
    if min(
        args.max_trials,
        args.probe_episodes,
        args.promotion_episodes,
        args.source_conditioning_seconds,
        args.refinement_iterations,
        args.rail_soft_margin,
    ) <= 0:
        raise ValueError("trial, episode, duration, iteration, and margin values must be positive")


def write_ledger(path: Path, ledger: dict[str, Any]) -> None:
    ledger["updated_at"] = utc_timestamp()
    ledger["runtime"] = runtime_metadata()
    ledger["git"] = git_metadata(ROOT)
    dump_json(ledger, path)


def main() -> None:
    args = parser().parse_args()
    validate_args(args)
    python = sys.executable
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = output_dir / "continuation.json"
    source_controller = Path(args.source_controller)
    source_config = Path(args.source_config)
    n_links = source_link_count(source_controller)
    gain_candidates = feedback_gain_candidates(args.feedback_gain_candidate)
    segment_candidates = segment_feedback_candidates(args.segment_feedback_candidate)
    if args.resume and ledger_path.exists():
        ledger = load_payload(ledger_path)
        accepted_state = ledger["accepted"]
        accepted_progress = float(accepted_state["progress"])
        accepted_config = Path(accepted_state["config"])
        accepted_route = Path(accepted_state["route"])
        accepted_refined = bool(accepted_state.get("refined", False))
        step = float(ledger["next_step"])
        trial_index = len(ledger.get("trials", []))
    else:
        accepted_progress = 0.0
        accepted_config = source_config
        accepted_route = source_controller
        accepted_refined = True
        step = args.initial_step
        trial_index = 0
        ledger = {
            "schema_version": 1,
            "generated_at": utc_timestamp(),
            "claim_status": "development_similarity_continuation",
            "passed": False,
            "status": "running",
            "source": {
                "config": file_metadata(source_config),
                "controller": file_metadata(source_controller),
                "n_links": n_links,
            },
            "target": {
                "length_scale": args.target_length_scale,
                "mass_scale": args.target_mass_scale,
                "path": args.path,
            },
            "trials": [],
            "accepted": {
                "progress": 0.0,
                "config": str(source_config),
                "route": str(source_controller),
                "refined": True,
            },
            "next_step": step,
        }
        write_ledger(ledger_path, ledger)

    while accepted_progress < 1.0 and trial_index < args.max_trials:
        proposed_progress = min(1.0, accepted_progress + step)
        label = f"trial_{trial_index:04d}_p{proposed_progress:.9f}"
        trial_dir = output_dir / label
        trial_dir.mkdir(parents=True, exist_ok=True)
        config = trial_dir / "target.yaml"
        similarity = trial_dir / "similarity.json"
        transfer = trial_dir / "transfer.json"
        mirror = trial_dir / "transfer_mirror.json"
        gate = trial_dir / "transfer_gate.json"
        replay = trial_dir / "exact_replay.json"
        route = trial_dir / "route.json"
        route_mirror = trial_dir / "route_mirror.json"
        route_gate = trial_dir / "route_gate.json"
        length_scale = scale_at_progress(
            args.target_length_scale, proposed_progress, path=args.path
        )
        mass_scale = scale_at_progress(
            args.target_mass_scale, proposed_progress, path=args.path
        )
        conditioning = args.source_conditioning_seconds * math.sqrt(length_scale)
        run(
            [
                python,
                "scripts/generalized_swingup_solver.py",
                "similarity",
                "--config",
                str(source_config),
                "--override",
                f"env.n_links={n_links}",
                "--length-scale",
                str(length_scale),
                "--mass-scale",
                str(mass_scale),
                "--out-config",
                str(config),
                "--out",
                str(similarity),
            ],
            dry_run=args.dry_run,
        )
        run(
            [
                python,
                "scripts/generalized_swingup_solver.py",
                "transfer",
                "--source-config",
                str(accepted_config),
                "--target-config",
                str(config),
                "--source-links",
                str(n_links),
                "--target-links",
                str(n_links),
                "--controller",
                str(accepted_route),
                "--out",
                str(transfer),
            ],
            dry_run=args.dry_run,
        )
        run(
            [
                python,
                "scripts/mirror_generalized_route.py",
                "--controller",
                str(transfer),
                "--out",
                str(mirror),
            ],
            dry_run=args.dry_run,
        )
        run(
            gate_command(
                python,
                config,
                transfer,
                mirror,
                gate,
                episodes=args.probe_episodes,
                seed=args.seed + 100 * trial_index,
                conditioning_seconds=conditioning,
                tracking_gain=args.tracking_gain,
                phase_window=args.phase_window,
            ),
            dry_run=args.dry_run,
        )
        if args.dry_run:
            return
        transfer_passed = passed_gate(gate, args.probe_episodes)
        transfer_reference = transfer
        transfer_reference_mirror = mirror
        feedback_screen: list[dict[str, Any]] = []
        if not transfer_passed:
            for ordinal, gain_scale in enumerate(gain_candidates):
                candidate = trial_dir / f"transfer_gain_{gain_scale:g}.json"
                candidate_mirror = trial_dir / f"transfer_gain_{gain_scale:g}_mirror.json"
                candidate_gate = trial_dir / f"transfer_gain_{gain_scale:g}_gate.json"
                scaled_pair(
                    python,
                    transfer,
                    candidate,
                    candidate_mirror,
                    scale=gain_scale,
                    dry_run=False,
                )
                run(
                    gate_command(
                        python,
                        config,
                        candidate,
                        candidate_mirror,
                        candidate_gate,
                        episodes=args.probe_episodes,
                        seed=args.seed + 100 * trial_index + 1 + ordinal,
                        conditioning_seconds=conditioning,
                        tracking_gain=1.0,
                        phase_window=args.phase_window,
                    ),
                    dry_run=False,
                )
                candidate_passed = passed_gate(candidate_gate, args.probe_episodes)
                feedback_screen.append(
                    {
                        "gain_scale": gain_scale,
                        "passed": candidate_passed,
                        "route": file_metadata(candidate),
                        "mirror": file_metadata(candidate_mirror),
                        "gate": file_metadata(candidate_gate),
                    }
                )
                if candidate_passed:
                    transfer_passed = True
                    transfer_reference = candidate
                    transfer_reference_mirror = candidate_mirror
                    break
        if not transfer_passed and materialized_route(transfer):
            for ordinal, (swing_scale, tail_scale) in enumerate(segment_candidates):
                label = f"s{swing_scale:g}_t{tail_scale:g}"
                candidate = trial_dir / f"transfer_segment_{label}.json"
                candidate_mirror = trial_dir / f"transfer_segment_{label}_mirror.json"
                candidate_gate = trial_dir / f"transfer_segment_{label}_gate.json"
                segment_scaled_pair(
                    python,
                    transfer,
                    candidate,
                    candidate_mirror,
                    swing_scale=swing_scale,
                    tail_scale=tail_scale,
                    dry_run=False,
                )
                run(
                    gate_command(
                        python,
                        config,
                        candidate,
                        candidate_mirror,
                        candidate_gate,
                        episodes=args.probe_episodes,
                        seed=args.seed + 100 * trial_index + 10 + ordinal,
                        conditioning_seconds=conditioning,
                        tracking_gain=1.0,
                        phase_window=args.phase_window,
                    ),
                    dry_run=False,
                )
                candidate_passed = passed_gate(candidate_gate, args.probe_episodes)
                feedback_screen.append(
                    {
                        "swing_gain_scale": swing_scale,
                        "tail_gain_scale": tail_scale,
                        "passed": candidate_passed,
                        "route": file_metadata(candidate),
                        "mirror": file_metadata(candidate_mirror),
                        "gate": file_metadata(candidate_gate),
                    }
                )
                if candidate_passed:
                    transfer_passed = True
                    transfer_reference = candidate
                    transfer_reference_mirror = candidate_mirror
                    break
        route_passed = False
        if transfer_passed:
            route_ready = False
            if materialized_route(transfer_reference):
                scaled_pair(
                    python,
                    transfer_reference,
                    route,
                    route_mirror,
                    scale=1.0,
                    dry_run=False,
                )
                route_ready = True
            else:
                run(
                    replay_command(
                        python,
                        config,
                        transfer_reference,
                        replay,
                        rail_soft_margin=args.rail_soft_margin,
                        iterations=1,
                        replay_only=True,
                    ),
                    dry_run=False,
                )
                if successful_replay(replay):
                    run(
                        materialize_command(python, config, replay, route),
                        dry_run=False,
                    )
                    run(
                        mirror_command(python, route, route_mirror),
                        dry_run=False,
                    )
                    route_ready = True
            if route_ready:
                run(
                    gate_command(
                        python,
                        config,
                        route,
                        route_mirror,
                        route_gate,
                        episodes=args.probe_episodes,
                        seed=args.seed + 100 * trial_index + 20,
                        conditioning_seconds=conditioning,
                        tracking_gain=1.0,
                        phase_window=args.phase_window,
                    ),
                    dry_run=False,
                )
                route_passed = passed_gate(route_gate, args.probe_episodes)

        trial: dict[str, Any] = {
            "trial": trial_index,
            "proposed_progress": proposed_progress,
            "step": step,
            "scales": {"length": length_scale, "mass": mass_scale},
            "transfer_passed": transfer_passed,
            "materialized_route_passed": route_passed,
            "artifacts": {
                "config": file_metadata(config),
                "similarity": file_metadata(similarity),
                "transfer": file_metadata(transfer),
                "transfer_mirror": file_metadata(mirror),
                "transfer_gate": file_metadata(gate),
            },
        }
        if feedback_screen:
            trial["feedback_gain_screen"] = feedback_screen
            trial["selected_feedback_gain_scale"] = next(
                (
                    row["gain_scale"]
                    for row in feedback_screen
                    if row["passed"] and "gain_scale" in row
                ),
                None,
            )
            trial["selected_segment_feedback_gain_scales"] = next(
                (
                    {
                        "swing": row["swing_gain_scale"],
                        "tail": row["tail_gain_scale"],
                    }
                    for row in feedback_screen
                    if row["passed"] and "swing_gain_scale" in row
                ),
                None,
            )
        if transfer_reference != transfer:
            trial["artifacts"].update(
                {
                    "selected_transfer": file_metadata(transfer_reference),
                    "selected_transfer_mirror": file_metadata(
                        transfer_reference_mirror
                    ),
                }
            )
        if route_passed:
            trial["accepted"] = True
            trial["artifacts"].update(
                {
                    "route": file_metadata(route),
                    "route_mirror": file_metadata(route_mirror),
                    "route_gate": file_metadata(route_gate),
                }
            )
            if replay.exists():
                trial["artifacts"]["exact_replay"] = file_metadata(replay)
            accepted_progress = proposed_progress
            accepted_config = config
            accepted_route = route
            accepted_refined = False
            step = min(1.0 - accepted_progress, next_step(
                step,
                accepted=True,
                growth=args.step_growth,
                minimum=args.minimum_step,
                maximum=args.maximum_step,
            ))
        else:
            trial["accepted"] = False
            if accepted_progress > 0.0 and not accepted_refined:
                refine_dir = output_dir / f"refine_p{accepted_progress:.9f}"
                refine_dir.mkdir(parents=True, exist_ok=True)
                optimizer = refine_dir / "optimizer.json"
                refined_route = refine_dir / "route.json"
                refined_mirror = refine_dir / "route_mirror.json"
                refined_gate = refine_dir / "gate.json"
                run(
                    replay_command(
                        python,
                        accepted_config,
                        accepted_route,
                        optimizer,
                        rail_soft_margin=args.rail_soft_margin,
                        iterations=args.refinement_iterations,
                        replay_only=False,
                    ),
                    dry_run=False,
                )
                optimizer_payload = load_payload(optimizer)
                refinement_passed = bool(
                    optimizer_payload.get("result", {}).get("success") is True
                    and optimizer_payload.get("result", {}).get("latched") is True
                )
                if refinement_passed:
                    run(
                        materialize_command(
                            python,
                            accepted_config,
                            optimizer,
                            refined_route,
                        ),
                        dry_run=False,
                    )
                    run(
                        mirror_command(python, refined_route, refined_mirror),
                        dry_run=False,
                    )
                    accepted_length = scale_at_progress(
                        args.target_length_scale, accepted_progress, path=args.path
                    )
                    run(
                        gate_command(
                            python,
                            accepted_config,
                            refined_route,
                            refined_mirror,
                            refined_gate,
                            episodes=args.probe_episodes,
                            seed=args.seed + 100 * trial_index + 40,
                            conditioning_seconds=(
                                args.source_conditioning_seconds
                                * math.sqrt(accepted_length)
                            ),
                            tracking_gain=1.0,
                            phase_window=args.phase_window,
                        ),
                        dry_run=False,
                    )
                    refinement_passed = passed_gate(
                        refined_gate, args.probe_episodes
                    )
                trial["anchor_refinement"] = {
                    "passed": refinement_passed,
                    "optimizer": file_metadata(optimizer),
                }
                if refinement_passed:
                    trial["anchor_refinement"].update(
                        {
                            "route": file_metadata(refined_route),
                            "mirror": file_metadata(refined_mirror),
                            "gate": file_metadata(refined_gate),
                        }
                    )
                    accepted_route = refined_route
                else:
                    step = next_step(
                        step,
                        accepted=False,
                        growth=args.step_growth,
                        minimum=args.minimum_step,
                        maximum=args.maximum_step,
                    )
                accepted_refined = True
            else:
                step = next_step(
                    step,
                    accepted=False,
                    growth=args.step_growth,
                    minimum=args.minimum_step,
                    maximum=args.maximum_step,
                )

        ledger["trials"].append(trial)
        ledger["accepted"] = {
            "progress": accepted_progress,
            "config": str(accepted_config),
            "route": str(accepted_route),
            "refined": accepted_refined,
        }
        ledger["next_step"] = step
        write_ledger(ledger_path, ledger)
        trial_index += 1
        if not route_passed and step <= args.minimum_step and accepted_refined:
            break

    if accepted_progress == 1.0:
        final_mirror = output_dir / "final_route_mirror.json"
        final_gate = output_dir / "final_gate.json"
        run(
            [
                python,
                "scripts/mirror_generalized_route.py",
                "--controller",
                str(accepted_route),
                "--out",
                str(final_mirror),
            ],
            dry_run=args.dry_run,
        )
        run(
            gate_command(
                python,
                accepted_config,
                accepted_route,
                final_mirror,
                final_gate,
                episodes=args.promotion_episodes,
                seed=args.seed + 90_000,
                conditioning_seconds=(
                    args.source_conditioning_seconds
                    * math.sqrt(args.target_length_scale)
                ),
                tracking_gain=1.0,
                phase_window=args.phase_window,
            ),
            dry_run=args.dry_run,
        )
        passed = args.dry_run or passed_gate(final_gate, args.promotion_episodes)
        ledger["passed"] = passed
        ledger["status"] = "verified_target_gate_passed" if passed else "final_gate_failed"
        if not args.dry_run:
            ledger["final_gate"] = file_metadata(final_gate)
            ledger["final_route"] = file_metadata(accepted_route)
    else:
        ledger["passed"] = False
        ledger["status"] = (
            "max_trials_reached"
            if trial_index >= args.max_trials
            else "minimum_step_frontier"
        )
    write_ledger(ledger_path, ledger)
    print(
        f"passed={ledger['passed']} status={ledger['status']} "
        f"progress={accepted_progress:.9f} ledger={ledger_path}",
        flush=True,
    )
    if not ledger["passed"] and not args.dry_run:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
