#!/usr/bin/env python
"""Run the generalized deterministic-to-exact pipeline on a target morphology.

This command is intentionally an orchestrator, not a new optimizer.  It binds
the existing count/arc-length transfer, exact target-plant replay, Box-FDDP
refinement, analytic planar mirror, noisy route selection, and evidence
verification into one morphology-parameterized operation.  The target config
is authoritative; no link-count-specific controller constants are introduced.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import (
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.generalized_solver import dimensionless_setup, setup_from_config

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class PipelinePaths:
    output_dir: Path
    prefix: str

    @property
    def transfer(self) -> Path:
        return self.output_dir / f"{self.prefix}_transfer_warm.json"

    @property
    def transfer_mirror(self) -> Path:
        return self.output_dir / f"{self.prefix}_transfer_warm_mirror.json"

    @property
    def negative(self) -> Path:
        return self.output_dir / f"{self.prefix}_transfer_check.json"

    @property
    def fddp_warm(self) -> Path:
        return self.output_dir / f"{self.prefix}_fddp_warm.json"

    @property
    def optimizer(self) -> Path:
        return self.output_dir / f"{self.prefix}_fddp.json"

    @property
    def route(self) -> Path:
        return self.output_dir / f"{self.prefix}_route.json"

    @property
    def mirror(self) -> Path:
        return self.output_dir / f"{self.prefix}_route_mirror.json"

    @property
    def gate(self) -> Path:
        return self.output_dir / f"{self.prefix}_gate.json"

    @property
    def frontier(self) -> Path:
        return self.output_dir / f"{self.prefix}_frontier.json"

    @property
    def manifest(self) -> Path:
        return self.output_dir / f"{self.prefix}_pipeline.json"

    def artifacts(self) -> dict[str, Path]:
        return {
            "transfer": self.transfer,
            "transfer_mirror": self.transfer_mirror,
            "transfer_check": self.negative,
            "fddp_warm": self.fddp_warm,
            "optimizer": self.optimizer,
            "route": self.route,
            "mirror": self.mirror,
            "gate": self.gate,
            "frontier": self.frontier,
        }


def command_steps(
    args: argparse.Namespace,
    paths: PipelinePaths,
    *,
    source_links: int,
    target_links: int,
    rail_half_length: float,
) -> list[tuple[str, list[str], Path]]:
    python = sys.executable
    rail_soft_limit = max(1e-3, rail_half_length - args.rail_soft_margin)
    handoff_cart_abs = min(args.handoff_cart_abs, 0.5 * rail_half_length)
    transfer = [
        python,
        "scripts/generalized_swingup_solver.py",
        "transfer",
        "--source-config",
        str(args.source_config),
        "--target-config",
        str(args.target_config),
        "--source-links",
        str(source_links),
        "--target-links",
        str(target_links),
        "--controller",
        str(args.source_controller),
        "--out",
        str(paths.transfer),
    ]
    mirror_transfer = [
        python,
        "scripts/mirror_generalized_route.py",
        "--controller",
        str(paths.transfer),
        "--out",
        str(paths.transfer_mirror),
    ]
    transfer_check = [
        python,
        "scripts/evaluate_generalized_route_library.py",
        "--config",
        str(args.target_config),
        "--controller",
        str(paths.transfer),
        "--controller",
        str(paths.transfer_mirror),
        "--episodes",
        str(args.transfer_check_episodes),
        "--seed",
        str(args.transfer_check_seed),
        "--conditioning-seconds",
        str(args.conditioning_seconds),
        "--tracking-gain-scale",
        str(args.transfer_tracking_gain),
        "--phase-window",
        str(args.transfer_phase_window),
        "--out",
        str(paths.negative),
    ]
    make_warm = [
        python,
        "scripts/force_proposal_to_fddp.py",
        "--config",
        str(args.target_config),
        "--spec",
        str(args.spec),
        "--proposal",
        str(paths.transfer),
        "--record-key",
        "controller",
        "--progress",
        "1.0",
        "--out",
        str(paths.fddp_warm),
    ]
    optimize = [
        python,
        "scripts/search_fddp_capture.py",
        "--config",
        str(args.target_config),
        "--spec",
        str(args.spec),
        "--state-json",
        str(paths.fddp_warm),
        "--state-index",
        "selected",
        "--initial-controller",
        str(paths.fddp_warm),
        "--initial-feasible",
        "--iterations",
        str(args.iterations),
        "--initial-regularization",
        str(args.initial_regularization),
        "--tracking-gain-scale",
        str(args.tracking_gain),
        "--lqr-scale",
        str(args.lqr_scale),
        "--lqr-control-cost",
        str(args.lqr_control_cost),
        "--control-cost",
        str(args.control_cost),
        "--stage-weight",
        str(args.stage_weight),
        "--terminal-weight",
        str(args.terminal_weight),
        "--terminal-state-weight",
        str(args.terminal_state_weight),
        "--terminal-cart-weight",
        str(args.terminal_cart_weight),
        "--terminal-cart-velocity-weight",
        str(args.terminal_cart_velocity_weight),
        "--terminal-hinge-velocity-factor",
        str(args.terminal_hinge_velocity_factor),
        "--rail-soft-limit",
        str(rail_soft_limit),
        "--rail-weight",
        str(args.rail_weight),
        "--handoff-angle-abs",
        str(args.handoff_angle_abs),
        "--handoff-cart-abs",
        str(handoff_cart_abs),
        "--handoff-cart-velocity-abs",
        str(args.handoff_cart_velocity_abs),
        "--handoff-hinge-velocity-rms",
        str(args.handoff_hinge_velocity_rms),
        "--defer-handoff-until-horizon",
        "--out",
        str(paths.optimizer),
    ]
    if args.allow_unstable_lyapunov:
        optimize.insert(-2, "--allow-unstable-lyapunov")
    package = [
        python,
        "scripts/package_generalized_route.py",
        "--controller",
        str(paths.optimizer),
        "--out",
        str(paths.route),
    ]
    package_mirror = [*package[:-1], str(paths.mirror), "--mirror"]
    gate = [
        python,
        "scripts/evaluate_generalized_route_library.py",
        "--config",
        str(args.target_config),
        "--controller",
        str(paths.route),
        "--controller",
        str(paths.mirror),
        "--episodes",
        str(args.gate_episodes),
        "--seed",
        str(args.gate_seed),
        "--conditioning-seconds",
        str(args.conditioning_seconds),
        "--tracking-gain-scale",
        str(args.tracking_gain),
        "--phase-window",
        str(args.phase_window),
        "--out",
        str(paths.gate),
    ]
    verify = [
        python,
        "scripts/verify_generalized_morphology_gate.py",
        "--config",
        str(args.target_config),
        "--warm",
        str(paths.transfer),
        "--negative",
        str(paths.negative),
        "--optimizer",
        str(paths.optimizer),
        "--route",
        str(paths.route),
        "--mirror",
        str(paths.mirror),
        "--gate",
        str(paths.gate),
        "--minimum-episodes",
        str(args.gate_episodes),
        "--out",
        str(paths.frontier),
    ]
    if args.require_transfer_failure:
        verify.insert(-2, "--require-warm-failure")
    return [
        ("transfer", transfer, paths.transfer),
        ("mirror_transfer", mirror_transfer, paths.transfer_mirror),
        ("transfer_check", transfer_check, paths.negative),
        ("make_fddp_warm", make_warm, paths.fddp_warm),
        ("optimize", optimize, paths.optimizer),
        ("package", package, paths.route),
        ("package_mirror", package_mirror, paths.mirror),
        ("gate", gate, paths.gate),
        ("verify", verify, paths.frontier),
    ]


def run_command(command: list[str]) -> None:
    environment = os.environ.copy()
    python_path = [str(ROOT), str(ROOT / "src"), str(ROOT / "scripts")]
    if environment.get("PYTHONPATH"):
        python_path.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path)
    print("+", shlex.join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def stage_failure_status(label: str, output: Path) -> str | None:
    """Return the first promotion failure represented by a completed stage."""

    payload = load_config(output)
    if label == "optimize" and not (
        payload.get("result", {}).get("success") is True
        and payload.get("result", {}).get("latched") is True
    ):
        return "exact_refinement_failed"
    if label == "gate" and float(payload.get("success_rate", 0.0)) != 1.0:
        return "noisy_gate_failed"
    return None


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-config", required=True)
    result.add_argument("--source-controller", required=True)
    result.add_argument("--target-config", required=True)
    result.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    result.add_argument("--output-dir", default="runs/generalized_solver")
    result.add_argument("--name", default=None)
    result.add_argument("--transfer-check-episodes", type=int, default=5)
    result.add_argument("--transfer-check-seed", type=int, default=84101)
    result.add_argument("--gate-episodes", type=int, default=20)
    result.add_argument("--gate-seed", type=int, default=84201)
    result.add_argument("--conditioning-seconds", type=float, default=15.0)
    result.add_argument("--transfer-tracking-gain", type=float, default=1.5)
    result.add_argument("--transfer-phase-window", type=int, default=6)
    result.add_argument("--tracking-gain", type=float, default=1.0)
    result.add_argument("--phase-window", type=int, default=0)
    result.add_argument("--iterations", type=int, default=60)
    result.add_argument("--initial-regularization", type=float, default=1e-4)
    result.add_argument("--lqr-scale", type=float, default=1.0)
    result.add_argument("--lqr-control-cost", type=float, default=1000.0)
    result.add_argument("--control-cost", type=float, default=0.1)
    result.add_argument("--stage-weight", type=float, default=0.01)
    result.add_argument("--terminal-weight", type=float, default=1000.0)
    result.add_argument("--terminal-state-weight", type=float, default=10000.0)
    result.add_argument("--terminal-cart-weight", type=float, default=100.0)
    result.add_argument("--terminal-cart-velocity-weight", type=float, default=100.0)
    result.add_argument("--terminal-hinge-velocity-factor", type=float, default=2.0)
    result.add_argument("--rail-soft-margin", type=float, default=0.5)
    result.add_argument("--rail-weight", type=float, default=1_000_000.0)
    result.add_argument("--handoff-angle-abs", type=float, default=0.15)
    result.add_argument("--handoff-cart-abs", type=float, default=1.25)
    result.add_argument("--handoff-cart-velocity-abs", type=float, default=0.5)
    result.add_argument("--handoff-hinge-velocity-rms", type=float, default=0.75)
    result.add_argument("--allow-unstable-lyapunov", action="store_true")
    result.add_argument("--require-transfer-failure", action="store_true")
    result.add_argument("--resume", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def source_link_count(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        raise TypeError("source controller must contain a controller object")
    gains = np.asarray(controller.get("feedback_gains"), dtype=np.float64)
    if gains.ndim != 2 or gains.shape[1] < 4 or gains.shape[1] % 2 != 0:
        raise ValueError("source feedback gains have an invalid state dimension")
    return gains.shape[1] // 2 - 1


def source_config_at_count(cfg: dict[str, Any], n_links: int) -> dict[str, Any]:
    """Match the transfer command's uniform fallback when a base count differs."""

    if int(cfg["env"]["n_links"]) == n_links:
        return cfg
    result = copy.deepcopy(cfg)
    result["env"]["n_links"] = n_links
    for name in ("lengths", "masses", "joint_stiffness", "joint_lock"):
        result["morphology"].pop(f"{name}_start", None)
        result["morphology"].pop(f"{name}_end", None)
    for endpoint in ("start", "end"):
        result["morphology"][endpoint]["alpha_length"] = 0.0
        result["morphology"][endpoint]["alpha_mass"] = 0.0
        result["morphology"][endpoint]["alpha_damping"] = 0.0
    return result


def validate_args(args: argparse.Namespace) -> tuple[Any, Any, int]:
    positive = (
        args.transfer_check_episodes,
        args.gate_episodes,
        args.conditioning_seconds,
        args.iterations,
        args.initial_regularization,
        args.lqr_scale,
        args.lqr_control_cost,
        args.control_cost,
        args.stage_weight,
        args.terminal_weight,
        args.terminal_state_weight,
        args.rail_soft_margin,
        args.rail_weight,
        args.handoff_angle_abs,
        args.handoff_cart_abs,
        args.handoff_cart_velocity_abs,
        args.handoff_hinge_velocity_rms,
    )
    if min(positive) <= 0.0:
        raise ValueError("episode counts, solver weights, bounds, and durations must be positive")
    if min(
        args.transfer_tracking_gain,
        args.tracking_gain,
        args.phase_window,
        args.transfer_phase_window,
        args.terminal_cart_weight,
        args.terminal_cart_velocity_weight,
        args.terminal_hinge_velocity_factor,
    ) < 0.0:
        raise ValueError("tracking, phase, and optional terminal weights must be nonnegative")
    inferred_source_links = source_link_count(Path(args.source_controller))
    source_cfg = source_config_at_count(
        load_config(args.source_config), inferred_source_links
    )
    source = setup_from_config(source_cfg)
    target = setup_from_config(load_config(args.target_config))
    return source, target, inferred_source_links


def write_pipeline_manifest(
    args: argparse.Namespace,
    paths: PipelinePaths,
    source: Any,
    target: Any,
    executed: list[dict[str, Any]],
    *,
    passed: bool,
    status: str,
) -> None:
    manifest = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": (
            "verified_development_generalized_morphology_pipeline"
            if passed
            else "development_generalized_morphology_pipeline_failure"
        ),
        "passed": passed,
        "status": status,
        "summary": (
            "One parameterized command executed deterministic transfer, exact "
            "target-plant refinement, mirror construction, noisy gating, and verification."
            if passed
            else "One parameterized command stopped at the first failed promotion stage; "
            "no failed controller was promoted."
        ),
        "source": {
            "config": file_metadata(Path(args.source_config)),
            "controller": file_metadata(Path(args.source_controller)),
            "dimensionless": dimensionless_setup(source).to_dict(),
        },
        "target": {
            "config": file_metadata(Path(args.target_config)),
            "dimensionless": dimensionless_setup(target).to_dict(),
        },
        "stages": executed,
        "next_action": (
            None
            if passed
            else "Use bounded morphology continuation or revise the deterministic seed; do not promote."
        ),
        "runtime": runtime_metadata(),
        "git": git_metadata(ROOT),
    }
    if any(stage["stage"] == "verify" for stage in executed):
        manifest["frontier"] = file_metadata(paths.frontier)
    dump_json(manifest, paths.manifest)


def main() -> None:
    args = parser().parse_args()
    source, target, source_links = validate_args(args)
    prefix = args.name or Path(args.target_config).stem
    paths = PipelinePaths(Path(args.output_dir), prefix)
    steps = command_steps(
        args,
        paths,
        source_links=source_links,
        target_links=target.n_links,
        rail_half_length=target.rail_half_length,
    )
    if args.dry_run:
        for label, command, output in steps:
            print(f"[{label}] -> {output}\n  {shlex.join(command)}")
        return

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    executed: list[dict[str, Any]] = []
    for label, command, output in steps:
        if args.resume and output.exists():
            print(f"= {label}: reusing {output}", flush=True)
            status = "reused"
        else:
            run_command(command)
            status = "executed"
        executed.append(
            {"stage": label, "status": status, "artifact": file_metadata(output)}
        )

        failure_status = stage_failure_status(label, output)
        if failure_status is not None:
            write_pipeline_manifest(
                args,
                paths,
                source,
                target,
                executed,
                passed=False,
                status=failure_status,
            )
            print(
                f"passed=False status={failure_status}; manifest={paths.manifest}",
                flush=True,
            )
            raise SystemExit(2 if label == "optimize" else 3)

    frontier = load_config(paths.frontier)
    if frontier.get("passed") is not True:
        raise RuntimeError("pipeline verifier did not pass")
    write_pipeline_manifest(
        args,
        paths,
        source,
        target,
        executed,
        passed=True,
        status="verified_gate_passed",
    )
    print(f"passed=True manifest={paths.manifest}")


if __name__ == "__main__":
    main()
