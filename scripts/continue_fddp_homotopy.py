#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from gcartpole.config import dump_json
from gcartpole.evidence import file_metadata, utc_timestamp


def artifact_passes(
    payload: dict[str, Any], *, minimum_upright_hold: float = 10.0
) -> bool:
    result = payload["result"]
    return bool(
        payload["search"]["is_feasible"]
        and result["success"]
        and result["max_upright_streak_seconds"] >= minimum_upright_hold
    )


def candidate_alpha(current: float, target: float, step: float) -> float:
    return min(target, current + step)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Adaptively continue a successful Box-FDDP capture trajectory"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--state-json", default="runs/p1_capture_envelope/validation.json"
    )
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--source-state-index", required=True)
    parser.add_argument("--initial-alpha", type=float, required=True)
    parser.add_argument("--target-alpha", type=float, default=1.0)
    parser.add_argument("--initial-controller", required=True)
    parser.add_argument("--initial-step", type=float, default=0.01)
    parser.add_argument("--minimum-step", type=float, default=0.001)
    parser.add_argument("--maximum-step", type=float, default=0.1)
    parser.add_argument("--step-growth", type=float, default=1.5)
    parser.add_argument("--max-attempts", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--initial-regularization", type=float, default=1e-6)
    parser.add_argument("--tracking-gain-scale", type=float, default=0.0)
    parser.add_argument("--rebuild-initial-states", action="store_true")
    parser.add_argument("--minimum-upright-hold", type=float, default=10.0)
    parser.add_argument("--terminal-weight", type=float, default=100_000.0)
    parser.add_argument("--terminal-state-weight", type=float, default=100_000.0)
    parser.add_argument("--seed", type=int, default=69000)
    parser.add_argument("--out-dir", default="runs/p1_capture_fddp/continuation")
    parser.add_argument("--summary-out", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.initial_alpha < args.target_alpha <= 1.0:
        raise ValueError("require 0 <= initial alpha < target alpha <= 1")
    if not 0.0 < args.minimum_step <= args.initial_step <= args.maximum_step:
        raise ValueError("require 0 < minimum step <= initial step <= maximum step")
    if (
        args.step_growth <= 1.0
        or min(args.max_attempts, args.iterations) <= 0
        or args.minimum_upright_hold <= 0.0
    ):
        raise ValueError("step growth must exceed one and counts must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    current = float(args.initial_alpha)
    step = float(args.initial_step)
    controller = Path(args.initial_controller)
    attempts: list[dict[str, Any]] = []
    script = Path(__file__).with_name("search_fddp_capture.py")

    for attempt in range(args.max_attempts):
        candidate = candidate_alpha(current, args.target_alpha, step)
        tag = f"{candidate:.8f}".replace(".", "p")
        output = out_dir / f"attempt_{attempt:03d}_alpha_{tag}.json"
        command = [
            sys.executable,
            str(script),
            "--config",
            args.config,
            "--spec",
            args.spec,
            "--state-json",
            args.state_json,
            "--state-index",
            str(args.state_index),
            "--interpolate-from-state-index",
            str(args.source_state_index),
            "--interpolation-alpha",
            str(candidate),
            "--initial-controller",
            str(controller),
            "--iterations",
            str(args.iterations),
            "--initial-regularization",
            str(args.initial_regularization),
            "--tracking-gain-scale",
            str(args.tracking_gain_scale),
            "--terminal-weight",
            str(args.terminal_weight),
            "--terminal-state-weight",
            str(args.terminal_state_weight),
            "--seed",
            str(args.seed + attempt),
            "--out",
            str(output),
        ]
        if args.rebuild_initial_states:
            command.append("--rebuild-initial-states")
        completed = subprocess.run(command, check=False)
        payload = (
            json.loads(output.read_text(encoding="utf-8"))
            if completed.returncode == 0 and output.exists()
            else None
        )
        passed = payload is not None and artifact_passes(
            payload, minimum_upright_hold=args.minimum_upright_hold
        )
        attempts.append(
            {
                "attempt": attempt,
                "alpha": candidate,
                "step": step,
                "returncode": completed.returncode,
                "passed": passed,
                "artifact": file_metadata(output) if output.exists() else None,
                "is_feasible": None
                if payload is None
                else bool(payload["search"]["is_feasible"]),
                "success": None
                if payload is None
                else bool(payload["result"]["success"]),
                "max_upright_streak_seconds": None
                if payload is None
                else float(payload["result"]["max_upright_streak_seconds"]),
            }
        )
        if passed:
            current = candidate
            controller = output
            if current >= args.target_alpha:
                break
            step = min(args.maximum_step, step * args.step_growth)
        else:
            step *= 0.5
            if step < args.minimum_step:
                break

    completed_target = current >= args.target_alpha
    dump_json(
        {
            "schema_version": 1,
            "generated_at": utc_timestamp(),
            "summary": "Adaptive exact-MuJoCo Box-FDDP state homotopy.",
            "completed_target": completed_target,
            "initial_alpha": float(args.initial_alpha),
            "final_alpha": current,
            "target_alpha": float(args.target_alpha),
            "minimum_upright_hold": float(args.minimum_upright_hold),
            "final_controller": file_metadata(controller),
            "attempts": attempts,
        },
        args.summary_out,
    )
    print(
        f"completed={completed_target} final_alpha={current:.8f} "
        f"attempts={len(attempts)} controller={controller}"
    )
    if not completed_target:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
