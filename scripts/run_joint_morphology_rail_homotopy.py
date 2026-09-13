#!/usr/bin/env python
"""Continue morphology and rail as separate deterministic coordinates.

Morphology is advanced first.  A failed exact replay may expand the rail only
when the recorded termination is a rail violation.  Once the target morphology
is reached, the same driver contracts the rail toward the requested target.
Every proposal is re-optimized and must pass an uninterrupted exact replay.

The expanded-rail phase is development evidence, not target-plant success.
Only ``target_solved`` means both morphology progress and target rail match the
requested configuration.
"""

from __future__ import annotations

import argparse
import json
import subprocess
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

try:
    from scripts.run_generalized_homotopy import (
        explicit_config,
        fddp_command,
        hanging_state,
        result_summary,
        successful,
        waypoint_command,
        waypoint_lookaheads,
        waypoint_successful,
        waypoint_usable,
    )
except ModuleNotFoundError:
    from run_generalized_homotopy import (
        explicit_config,
        fddp_command,
        hanging_state,
        result_summary,
        successful,
        waypoint_command,
        waypoint_lookaheads,
        waypoint_successful,
        waypoint_usable,
    )

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def target_rail_ratio(cfg: dict[str, Any]) -> float:
    setup = setup_from_config(cfg)
    return float(setup.rail_half_length / setup.chain_length)


def config_at(
    target_cfg: dict[str, Any],
    source_lengths: np.ndarray,
    source_masses: np.ndarray,
    target_lengths: np.ndarray,
    target_masses: np.ndarray,
    *,
    progress: float,
    rail_ratio: float,
    name: str,
) -> dict[str, Any]:
    """Materialize one exact plant on the joint continuation surface."""

    lengths, masses = homotopy_morphology(
        source_lengths,
        source_masses,
        target_lengths,
        target_masses,
        progress,
    )
    cfg = explicit_config(target_cfg, lengths, masses, name)
    cfg["env"]["rail_limit"] = float(rail_ratio * np.sum(lengths))
    return cfg


def rail_rescue_ratio(
    outcome: dict[str, Any],
    *,
    current_ratio: float,
    maximum_ratio: float,
    growth: float,
    clearance_ratio: float,
    previous_outcome: dict[str, Any] | None = None,
    minimum_deficit_improvement: float = 0.0,
) -> float | None:
    """Return a larger rail only while measured normalized clearance improves.

    A failed optimizer can ride the soft rail boundary: increasing the rail then
    increases the failed excursion almost one-for-one.  That is an optimizer
    basin failure, not evidence that the plant needs more rail.  After the first
    rescue, require the dimensionless deficit ``rho_required-rho_configured``
    to decrease before spending another rescue.
    """

    if outcome.get("termination_reason") != "rail_violation":
        return None
    requirement = outcome.get("rail_requirement") or {}
    observed = float(requirement.get("required_rail_ratio", float("nan")))
    if not np.isfinite(observed):
        return None
    deficit = observed - current_ratio
    if previous_outcome is not None:
        previous_requirement = previous_outcome.get("rail_requirement") or {}
        previous_required = float(
            previous_requirement.get("required_rail_ratio", float("nan"))
        )
        previous_configured = float(
            previous_requirement.get("configured_rail_ratio", float("nan"))
        )
        previous_deficit = previous_required - previous_configured
        if not np.isfinite(previous_deficit):
            return None
        if previous_deficit - deficit <= minimum_deficit_improvement:
            return None
    proposed = max(current_ratio * growth, observed + clearance_ratio)
    proposed = min(maximum_ratio, proposed)
    if proposed <= current_ratio + 1.0e-12:
        return None
    return float(proposed)


def rail_contraction_proposal(
    current_ratio: float,
    target_ratio: float,
    step: float,
) -> float:
    """Move monotonically toward the target rail without overshooting it."""

    if not target_ratio > 0.0 or current_ratio < target_ratio or step <= 0.0:
        raise ValueError("rail contraction requires current >= target > 0 and step > 0")
    return float(max(target_ratio, current_ratio - step))


def attempt(
    *,
    cfg_path: Path,
    cfg: dict[str, Any],
    state_path: Path,
    controller: Path,
    prefix: Path,
    args: argparse.Namespace,
) -> tuple[bool, Path, dict[str, Any]]:
    """Run bounded waypoint repairs and exact FDDP for one joint proposal."""

    waypoint_attempts: list[dict[str, Any]] = []
    passed = False
    accepted_path = prefix.with_name(f"{prefix.name}_pass1.json")
    for segment_steps in waypoint_lookaheads(
        args.waypoint_segment_steps, args.waypoint_segment_multipliers
    ):
        multiplier = segment_steps // args.waypoint_segment_steps
        suffix = "waypoint" if multiplier == 1 else f"waypoint_x{multiplier}"
        waypoint_path = prefix.with_name(f"{prefix.name}_{suffix}.json")
        refined_path = prefix.with_name(f"{prefix.name}_{suffix}_fddp.json")
        run(
            waypoint_command(
                cfg=cfg_path,
                controller=controller,
                output=waypoint_path,
                segment_steps=segment_steps,
                max_evaluations=args.waypoint_max_evaluations,
                endpoint_weight=args.waypoint_endpoint_weight,
                control_regularization=args.waypoint_control_regularization,
                rail_soft_margin=args.rail_soft_margin,
                rail_weight=args.rail_weight,
                endpoint_tolerance=args.waypoint_endpoint_tolerance,
            )
        )
        usable = waypoint_usable(waypoint_path)
        refined = False
        if usable:
            run(
                fddp_command(
                    cfg=cfg_path,
                    state=state_path,
                    controller=waypoint_path,
                    output=refined_path,
                    iterations=args.iterations,
                    regularization=1.0e-6,
                    tracking_gain=args.tracking_gain,
                    exact_initial_trajectory=True,
                    rail_soft_margin=args.rail_soft_margin,
                )
            )
            accepted_path = refined_path
            refined = successful(refined_path)
            passed = refined
        waypoint_attempts.append(
            {
                "segment_steps": int(segment_steps),
                "search_passed": waypoint_successful(waypoint_path),
                "used_for_refinement": bool(usable),
                "refinement_passed": bool(refined),
                "artifact": file_metadata(waypoint_path),
                "refinement": file_metadata(refined_path) if usable else None,
            }
        )
        if passed:
            break

    if not passed:
        first_path = prefix.with_name(f"{prefix.name}_pass1.json")
        run(
            fddp_command(
                cfg=cfg_path,
                state=state_path,
                controller=controller,
                output=first_path,
                iterations=args.iterations,
                regularization=1.0e-6,
                tracking_gain=args.tracking_gain,
                rail_soft_margin=args.rail_soft_margin,
            )
        )
        accepted_path = first_path
        passed = successful(first_path)
    if not passed:
        retry_path = prefix.with_name(f"{prefix.name}_pass2.json")
        run(
            fddp_command(
                cfg=cfg_path,
                state=state_path,
                controller=accepted_path,
                output=retry_path,
                iterations=args.retry_iterations,
                regularization=1.0e-7,
                tracking_gain=0.5 * args.tracking_gain,
                rail_soft_margin=args.rail_soft_margin,
            )
        )
        accepted_path = retry_path
        passed = successful(retry_path)
    return passed, accepted_path, {
        "attempts": waypoint_attempts,
        "outcome": result_summary(accepted_path, cfg),
    }


def write_manifest(
    path: Path,
    *,
    source_config: Path,
    target_config: Path,
    current_controller: Path,
    morphology: AdaptiveHomotopy,
    rail_ratio: float,
    rail_step: float,
    target_ratio: float,
    trials: list[dict[str, Any]],
    status: str,
) -> None:
    dump_json(
        {
            "schema_version": 1,
            "updated_at": utc_timestamp(),
            "claim_status": (
                "target_exact_replay_solved"
                if status == "target_solved"
                else "development_joint_continuation_not_record_evidence"
            ),
            "status": status,
            "source_config": file_metadata(source_config),
            "target_config": file_metadata(target_config),
            "morphology_progress": float(morphology.progress),
            "next_morphology_step": float(morphology.step),
            "rail_ratio": float(rail_ratio),
            "target_rail_ratio": float(target_ratio),
            "next_rail_step": float(rail_step),
            "current_controller": file_metadata(current_controller),
            "trials": trials,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--source-controller", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-progress", type=float, default=0.0)
    parser.add_argument("--initial-morphology-step", type=float, default=0.001)
    parser.add_argument("--minimum-morphology-step", type=float, default=1.0e-5)
    parser.add_argument("--maximum-morphology-step", type=float, default=0.05)
    parser.add_argument("--morphology-growth", type=float, default=1.6)
    parser.add_argument("--initial-rail-ratio", type=float, default=None)
    parser.add_argument("--maximum-rail-ratio", type=float, default=2.5)
    parser.add_argument("--rail-rescue-growth", type=float, default=1.08)
    parser.add_argument("--rail-clearance-ratio", type=float, default=0.05)
    parser.add_argument("--minimum-rail-deficit-improvement", type=float, default=0.0)
    parser.add_argument("--initial-rail-step", type=float, default=0.05)
    parser.add_argument("--minimum-rail-step", type=float, default=0.0025)
    parser.add_argument("--rail-growth", type=float, default=1.4)
    parser.add_argument("--max-rail-rescues", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--retry-iterations", type=int, default=100)
    parser.add_argument("--max-trials", type=int, default=200)
    parser.add_argument("--tracking-gain", type=float, default=1.0)
    parser.add_argument("--waypoint-segment-steps", type=int, default=24)
    parser.add_argument("--waypoint-segment-multipliers", nargs="+", type=int, default=[1, 2, 4])
    parser.add_argument("--waypoint-max-evaluations", type=int, default=120)
    parser.add_argument("--waypoint-endpoint-weight", type=float, default=10_000.0)
    parser.add_argument("--waypoint-control-regularization", type=float, default=1.0e-6)
    parser.add_argument("--waypoint-endpoint-tolerance", type=float, default=0.5)
    parser.add_argument("--rail-soft-margin", type=float, default=0.5)
    parser.add_argument("--rail-weight", type=float, default=1_000_000.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.max_rail_rescues < 0:
        raise ValueError("--max-rail-rescues must be nonnegative")
    if not 1.0 < args.rail_rescue_growth:
        raise ValueError("--rail-rescue-growth must exceed one")
    if args.minimum_rail_deficit_improvement < 0.0:
        raise ValueError("--minimum-rail-deficit-improvement must be nonnegative")

    source_path = Path(args.source_config)
    target_path = Path(args.target_config)
    source_cfg = load_config(source_path)
    target_cfg = load_config(target_path)
    source = setup_from_config(source_cfg)
    target = setup_from_config(target_cfg)
    if source.n_links != target.n_links:
        raise ValueError("joint continuation currently requires a common link count")
    if not np.isclose(source.chain_length, target.chain_length):
        raise ValueError("source and target must preserve total chain length")
    if not np.isclose(source.link_mass, target.link_mass):
        raise ValueError("source and target must preserve total link mass")

    target_ratio = target_rail_ratio(target_cfg)
    output_dir = Path(args.output_dir)
    configs_dir = output_dir / "configs"
    trials_dir = output_dir / "trials"
    configs_dir.mkdir(parents=True, exist_ok=True)
    trials_dir.mkdir(parents=True, exist_ok=True)
    state_path = output_dir / f"hanging_n{source.n_links}.json"
    hanging_state(source.n_links, state_path)
    manifest_path = output_dir / "continuation.json"

    if args.resume:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if file_metadata(target_path)["sha256"] != saved["target_config"]["sha256"]:
            raise ValueError("saved continuation target does not match --target-config")
        morphology = AdaptiveHomotopy(
            progress=float(saved["morphology_progress"]),
            step=float(saved["next_morphology_step"]),
            minimum_step=args.minimum_morphology_step,
            maximum_step=args.maximum_morphology_step,
            growth=args.morphology_growth,
        )
        rail_ratio = float(saved["rail_ratio"])
        rail_step = float(saved["next_rail_step"])
        current_controller = Path(saved["current_controller"]["path"])
        trials = list(saved["trials"])
    else:
        morphology = AdaptiveHomotopy(
            progress=args.initial_progress,
            step=args.initial_morphology_step,
            minimum_step=args.minimum_morphology_step,
            maximum_step=args.maximum_morphology_step,
            growth=args.morphology_growth,
        )
        rail_ratio = float(args.initial_rail_ratio or target_ratio)
        rail_step = float(args.initial_rail_step)
        current_controller = Path(args.source_controller)
        trials = []
    if rail_ratio < target_ratio or rail_ratio > args.maximum_rail_ratio:
        raise ValueError("initial rail ratio must lie between target and maximum")

    for trial_number in range(len(trials) + 1, args.max_trials + 1):
        stage = "morphology" if not np.isclose(morphology.progress, 1.0) else "rail_contraction"
        proposed_progress = morphology.proposal() if stage == "morphology" else 1.0
        proposed_rail = (
            rail_ratio
            if stage == "morphology"
            else rail_contraction_proposal(rail_ratio, target_ratio, rail_step)
        )
        rescue_index = 0
        passed = False
        proposal_records: list[dict[str, Any]] = []
        while True:
            label = (
                f"trial_{trial_number:04d}_{stage}_p{proposed_progress:.9f}"
                f"_rho{proposed_rail:.6f}_r{rescue_index}"
            )
            cfg = config_at(
                target_cfg,
                source.lengths,
                source.masses,
                target.lengths,
                target.masses,
                progress=proposed_progress,
                rail_ratio=proposed_rail,
                name=label,
            )
            cfg_path = configs_dir / f"{label}.yaml"
            save_config(cfg, cfg_path)
            passed, result_path, details = attempt(
                cfg_path=cfg_path,
                cfg=cfg,
                state_path=state_path,
                controller=current_controller,
                prefix=trials_dir / label,
                args=args,
            )
            proposal_records.append(
                {
                    "rescue_index": rescue_index,
                    "rail_ratio": float(proposed_rail),
                    "config": file_metadata(cfg_path),
                    "result": file_metadata(result_path),
                    "dimensionless": dimensionless_setup(setup_from_config(cfg)).to_dict(),
                    **details,
                }
            )
            if passed or stage != "morphology" or rescue_index >= args.max_rail_rescues:
                break
            rescued = rail_rescue_ratio(
                details["outcome"],
                current_ratio=proposed_rail,
                maximum_ratio=args.maximum_rail_ratio,
                growth=args.rail_rescue_growth,
                clearance_ratio=args.rail_clearance_ratio,
                previous_outcome=(
                    proposal_records[-2]["outcome"]
                    if len(proposal_records) > 1
                    else None
                ),
                minimum_deficit_improvement=args.minimum_rail_deficit_improvement,
            )
            if rescued is None:
                break
            proposed_rail = rescued
            rescue_index += 1

        trials.append(
            {
                "trial": trial_number,
                "stage": stage,
                "from_morphology_progress": float(morphology.progress),
                "proposed_morphology_progress": float(proposed_progress),
                "from_rail_ratio": float(rail_ratio),
                "proposed_rail_ratio": float(proposed_rail),
                "accepted": bool(passed),
                "proposals": proposal_records,
            }
        )
        if passed:
            current_controller = Path(proposal_records[-1]["result"]["path"])
            rail_ratio = float(proposed_rail)
            if stage == "morphology":
                morphology.accept(proposed_progress)
            else:
                rail_step = min(
                    max(args.minimum_rail_step, rail_ratio - target_ratio),
                    rail_step * args.rail_growth,
                )
            status = (
                "target_solved"
                if np.isclose(morphology.progress, 1.0) and np.isclose(rail_ratio, target_ratio)
                else "running"
            )
            write_manifest(
                manifest_path,
                source_config=source_path,
                target_config=target_path,
                current_controller=current_controller,
                morphology=morphology,
                rail_ratio=rail_ratio,
                rail_step=rail_step,
                target_ratio=target_ratio,
                trials=trials,
                status=status,
            )
            print(
                f"ACCEPT stage={stage} p={morphology.progress:.9f} rho={rail_ratio:.6f}",
                flush=True,
            )
            if status == "target_solved":
                return
        else:
            try:
                if stage == "morphology":
                    morphology.reject()
                else:
                    rail_step /= 2.0
                    if rail_step < args.minimum_rail_step:
                        raise RuntimeError("rail step fell below minimum")
            except RuntimeError as error:
                status = f"{stage}_frontier_reached"
                write_manifest(
                    manifest_path,
                    source_config=source_path,
                    target_config=target_path,
                    current_controller=current_controller,
                    morphology=morphology,
                    rail_ratio=rail_ratio,
                    rail_step=rail_step,
                    target_ratio=target_ratio,
                    trials=trials,
                    status=status,
                )
                raise SystemExit(4) from error
            write_manifest(
                manifest_path,
                source_config=source_path,
                target_config=target_path,
                current_controller=current_controller,
                morphology=morphology,
                rail_ratio=rail_ratio,
                rail_step=rail_step,
                target_ratio=target_ratio,
                trials=trials,
                status="running",
            )
            print(
                f"REJECT stage={stage} p={proposed_progress:.9f} rho={proposed_rail:.6f}",
                flush=True,
            )

    write_manifest(
        manifest_path,
        source_config=source_path,
        target_config=target_path,
        current_controller=current_controller,
        morphology=morphology,
        rail_ratio=rail_ratio,
        rail_step=rail_step,
        target_ratio=target_ratio,
        trials=trials,
        status="trial_budget_exhausted",
    )
    raise SystemExit(5)


if __name__ == "__main__":
    main()
