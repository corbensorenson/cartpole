#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from gcartpole.config import dump_json
from gcartpole.evidence import (
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)


STAGES: tuple[dict[str, Any], ...] = (
    {
        "name": "loose_energy",
        "iterations": 400,
        "handoff_lyapunov": 1800.0,
        "box_penalty": 10.0,
        "cart_penalty": 30.0,
    },
    {
        "name": "loose_box",
        "iterations": 500,
        "handoff_lyapunov": 2000.0,
        "box_penalty": 100.0,
        "cart_penalty": 30.0,
    },
    {
        "name": "v1000_energy",
        "iterations": 600,
        "handoff_lyapunov": 1000.0,
        "box_penalty": 10.0,
        "cart_penalty": 30.0,
    },
    {
        "name": "v1000_box",
        "iterations": 400,
        "handoff_lyapunov": 1000.0,
        "box_penalty": 100.0,
        "cart_penalty": 30.0,
    },
    {
        "name": "v500_energy",
        "iterations": 800,
        "handoff_lyapunov": 500.0,
        "box_penalty": 10.0,
        "cart_penalty": 30.0,
    },
    {
        "name": "v500_box",
        "iterations": 500,
        "handoff_lyapunov": 600.0,
        "box_penalty": 100.0,
        "cart_penalty": 30.0,
    },
    {
        "name": "v100_energy",
        "iterations": 1500,
        "handoff_lyapunov": 100.0,
        "box_penalty": 10.0,
        "cart_penalty": 30.0,
    },
    {
        "name": "v100_focus_w20",
        "iterations": 800,
        "handoff_lyapunov": 100.0,
        "window_steps": 20,
        "window_offsets": "0,5,10",
        "box_penalty": 1.0,
        "cart_penalty": 10.0,
    },
    {
        "name": "v100_focus_w45",
        "iterations": 800,
        "handoff_lyapunov": 100.0,
        "window_steps": 45,
        "window_offsets": "0",
        "trust_region": 1e-4,
        "box_penalty": 1.0,
        "cart_penalty": 10.0,
    },
    {
        "name": "v10_energy",
        "iterations": 1500,
        "handoff_lyapunov": 10.0,
        "box_penalty": 10.0,
        "cart_penalty": 30.0,
    },
    {
        "name": "v10_focus",
        "iterations": 1000,
        "handoff_lyapunov": 10.0,
        "trust_region": 0.01,
        "box_penalty": 1.0,
        "cart_penalty": 10.0,
    },
    {
        "name": "v20_tight_velocity",
        "iterations": 1500,
        "handoff_lyapunov": 20.0,
        "handoff_angle_abs": 0.075,
        "handoff_cart_velocity_abs": 0.25,
        "handoff_hinge_velocity_abs": 0.375,
        "box_penalty": 100.0,
        "cart_penalty": 30.0,
    },
)


def artifact_matches_stage(
    payload: dict[str, Any],
    *,
    state_index: int,
    seed: int,
    source_sha256: str,
    stage: dict[str, Any],
) -> bool:
    controller = payload.get("controller", {})
    expected_offsets = [
        int(value)
        for value in str(stage.get("window_offsets", "0,5,10,15,20")).split(",")
    ]
    expected = {
        "iterations": int(stage["iterations"]),
        "handoff_lyapunov": float(stage["handoff_lyapunov"]),
        "window_steps": int(stage.get("window_steps", 30)),
        "window_offsets": expected_offsets,
        "trust_region": float(stage.get("trust_region", 0.003)),
        "box_penalty": float(stage["box_penalty"]),
        "cart_penalty": float(stage["cart_penalty"]),
        "handoff_angle_abs": float(stage.get("handoff_angle_abs", 0.15)),
        "handoff_cart_velocity_abs": float(stage.get("handoff_cart_velocity_abs", 0.5)),
        "handoff_hinge_velocity_abs": float(
            stage.get("handoff_hinge_velocity_abs", 0.75)
        ),
    }
    return bool(
        int(payload.get("state_index", -1)) == state_index
        and int(payload.get("seed", -1)) == seed
        and controller.get("initial_controller", {}).get("sha256") == source_sha256
        and all(controller.get(key) == value for key, value in expected.items())
    )


def stage_command(
    python: str,
    *,
    state_index: int,
    seed: int,
    source: Path,
    output: Path,
    stage: dict[str, Any],
) -> list[str]:
    command = [
        python,
        "scripts/search_scp_capture.py",
        "--state-index",
        str(state_index),
        "--seed",
        str(seed),
        "--initial-controller",
        str(source),
        "--iterations",
        str(stage["iterations"]),
        "--handoff-lyapunov",
        str(stage["handoff_lyapunov"]),
        "--window-steps",
        str(stage.get("window_steps", 30)),
        "--window-offsets",
        str(stage.get("window_offsets", "0,5,10,15,20")),
        "--trust-region",
        str(stage.get("trust_region", 0.003)),
        "--box-penalty",
        str(stage["box_penalty"]),
        "--cart-penalty",
        str(stage["cart_penalty"]),
        "--out",
        str(output),
    ]
    optional_arguments = {
        "handoff_angle_abs": "--handoff-angle-abs",
        "handoff_cart_velocity_abs": "--handoff-cart-velocity-abs",
        "handoff_hinge_velocity_abs": "--handoff-hinge-velocity-abs",
    }
    for key, flag in optional_arguments.items():
        if key in stage:
            command.extend((flag, str(stage[key])))
    return command


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the resumable exact local-SCP capture curriculum"
    )
    parser.add_argument("--state-index", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--initial-controller", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--max-stages", type=int, default=None)
    args = parser.parse_args()
    if args.state_index < 0 or (args.max_stages is not None and args.max_stages <= 0):
        raise ValueError("state index and stage limit are invalid")

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    source = Path(args.initial_controller)
    if not source.exists():
        raise FileNotFoundError(source)
    stages = STAGES if args.max_stages is None else STAGES[: args.max_stages]
    records = []
    started = time.time()
    for index, stage in enumerate(stages):
        artifact = run_dir / f"{index:02d}_{stage['name']}.json"
        source_metadata = file_metadata(source)
        reusable_payload = None
        if not args.no_resume and artifact.exists():
            try:
                candidate_payload = json.loads(artifact.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                candidate_payload = None
            if candidate_payload is not None:
                if artifact_matches_stage(
                    candidate_payload,
                    state_index=args.state_index,
                    seed=args.seed,
                    source_sha256=source_metadata["sha256"],
                    stage=stage,
                ):
                    reusable_payload = candidate_payload
        if reusable_payload is None:
            command = stage_command(
                sys.executable,
                state_index=args.state_index,
                seed=args.seed,
                source=source,
                output=artifact,
                stage=stage,
            )
            print(f"Running {' '.join(command)}", flush=True)
            subprocess.run(command, check=True)
            payload = json.loads(artifact.read_text(encoding="utf-8"))
        else:
            payload = reusable_payload
        if int(payload["state_index"]) != args.state_index:
            raise ValueError(f"{artifact} belongs to a different state")
        if int(payload["seed"]) != args.seed:
            raise ValueError(f"{artifact} uses a different seed")
        if not artifact_matches_stage(
            payload,
            state_index=args.state_index,
            seed=args.seed,
            source_sha256=source_metadata["sha256"],
            stage=stage,
        ):
            raise ValueError(f"{artifact} does not continue from {source}")
        result = payload["result"]
        search = payload["search"]
        records.append(
            {
                "stage_index": index,
                "stage": stage,
                "source": source_metadata,
                "artifact": file_metadata(artifact),
                "search_converged": bool(search["converged"]),
                "constraint_score": float(search["final_constraint_score"]),
                "success": bool(result["success"]),
                "latched": bool(result["latched"]),
                "minimum_lyapunov": float(result["minimum_lyapunov"]),
                "max_upright_streak_seconds": float(
                    result["max_upright_streak_seconds"]
                ),
                "max_cart_excursion": float(result["max_cart_excursion"]),
                "termination_reason": result["termination_reason"],
            }
        )
        source = artifact
        print(
            f"stage={stage['name']} success={result['success']} "
            f"score={search['final_constraint_score']:.3e} "
            f"min_v={result['minimum_lyapunov']:.2f}",
            flush=True,
        )
        if result["success"]:
            break

    final = records[-1]
    summary = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_p1_evidence": True,
        "summary": "State-specific exact local-SCP curriculum diagnostic.",
        "state_index": int(args.state_index),
        "seed": int(args.seed),
        "initial_controller": file_metadata(args.initial_controller),
        "final_controller": final["artifact"],
        "success": bool(final["success"]),
        "stages_completed": len(records),
        "wall_time_seconds": float(time.time() - started),
        "records": records,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(summary, args.summary_out)
    print(
        f"success={summary['success']} stages={len(records)} "
        f"final={summary['final_controller']['path']}"
    )
    print(f"Wrote {args.summary_out}")


if __name__ == "__main__":
    main()
