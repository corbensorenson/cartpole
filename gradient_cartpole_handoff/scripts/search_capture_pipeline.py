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


def parse_state_indices(text: str) -> list[int]:
    indices = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not indices or any(index < 0 for index in indices):
        raise ValueError("state indices must be nonnegative comma-separated integers")
    if len(indices) != len(set(indices)):
        raise ValueError("state indices must be unique")
    return indices


def stage_paths(run_dir: Path, state_index: int) -> dict[str, Path]:
    state_dir = run_dir / f"validation_{state_index}"
    return {
        "predictive": state_dir / "predictive.json",
        "approach": state_dir / "approach_ilqr.json",
        "approach_fddp": state_dir / "approach_fddp.json",
        "tail_seed": state_dir / "tail_cem.json",
        "tail_ddp": state_dir / "tail_ddp.json",
    }


def run_stage(command: list[str], artifact: Path, *, resume: bool) -> dict[str, Any]:
    if not (resume and artifact.exists()):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        print(f"Running {' '.join(command)}", flush=True)
        subprocess.run(command, check=True)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    return payload


def result_summary(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload["result"]
    return {
        "success": bool(result["success"]),
        "latched": bool(result["latched"]),
        "minimum_lyapunov": float(result["minimum_lyapunov"]),
        "max_upright_streak_seconds": float(result["max_upright_streak_seconds"]),
        "max_cart_excursion": float(result["max_cart_excursion"]),
        "termination_reason": result["termination_reason"],
    }


def stage_selection_key(item: tuple[str, dict[str, Any]]) -> tuple[Any, ...]:
    """Rank stages by authoritative uninterrupted rollout behavior."""
    name, payload = item
    result = payload["result"]
    success = bool(result["success"])
    latched = bool(result["latched"])
    hold = float(result["max_upright_streak_seconds"])
    value = float(result["minimum_lyapunov"])
    # Once captured, sustained hold time is decisive. Before capture, short
    # threshold crossings are noise; proximity to the verified funnel matters.
    quality = (-hold, value) if success or latched else (value, -hold)
    return (
        not success,
        not latched,
        *quality,
        float(result["max_cart_excursion"]),
        name,
    )


def best_stage(
    stage_payloads: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    return min(stage_payloads.items(), key=stage_selection_key)


def terminal_lyapunov(payload: dict[str, Any]) -> float:
    search = payload.get("search", {})
    value = search.get("terminal_lyapunov")
    if value is None:
        value = search.get("tail_terminal_lyapunov")
    if value is None:
        raise ValueError("feedback-controller stage lacks a terminal Lyapunov value")
    return float(value)


def best_extension_stage(
    stage_payloads: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    compatible = {
        name: payload
        for name, payload in stage_payloads.items()
        if payload.get("controller", {}).get("controls") is not None
    }
    if not compatible:
        raise ValueError("no appendable feedback-controller stage is available")
    return min(
        compatible.items(),
        key=lambda item: (
            terminal_lyapunov(item[1]),
            stage_selection_key(item),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the resumable predictive-iLQR-settling capture pipeline"
    )
    parser.add_argument("--state-indices", required=True)
    parser.add_argument("--run-dir", default="runs/p1_capture_pipeline")
    parser.add_argument("--summary-out", default=None)
    parser.add_argument("--seed", type=int, default=64010)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--no-fddp-refinement", action="store_true")
    parser.add_argument("--no-ddp-fallback", action="store_true")
    args = parser.parse_args()
    indices = parse_state_indices(args.state_indices)
    run_dir = Path(args.run_dir)
    summary_out = (
        Path(args.summary_out)
        if args.summary_out is not None
        else run_dir / "cohort_summary.json"
    )
    python = sys.executable
    resume = not args.no_resume
    records = []
    started = time.time()

    for ordinal, state_index in enumerate(indices):
        paths = stage_paths(run_dir, state_index)
        state_seed = args.seed + state_index
        predictive = run_stage(
            [
                python,
                "scripts/evaluate_predictive_sampling_capture.py",
                "--state-index",
                str(state_index),
                "--seed",
                str(state_seed),
                "--horizon-seconds",
                "3.0",
                "--replan-steps",
                "150",
                "--mpc-seconds",
                "3.0",
                "--knot-count",
                "32",
                "--iterations",
                "6",
                "--population",
                "4096",
                "--elites",
                "128",
                "--action-sigma",
                "0.8",
                "--threads",
                "8",
                "--out",
                str(paths["predictive"]),
            ],
            paths["predictive"],
            resume=resume,
        )
        approach = run_stage(
            [
                python,
                "scripts/search_ilqr_capture.py",
                "--state-index",
                str(state_index),
                "--seed",
                str(state_seed),
                "--horizon-seconds",
                "3.0",
                "--iterations",
                "300",
                "--initial-controller",
                str(paths["predictive"]),
                "--initial-lqr-scale",
                "0",
                "--terminal-weight",
                "10000",
                "--terminal-state-weight",
                "100000",
                "--switch-angle-abs",
                "0.15",
                "--switch-cart-velocity-abs",
                "0.5",
                "--switch-hinge-velocity-rms",
                "0.75",
                "--out",
                str(paths["approach"]),
            ],
            paths["approach"],
            resume=resume,
        )
        stage_payloads = {"predictive": predictive, "approach": approach}
        if not args.no_fddp_refinement:
            approach_fddp = run_stage(
                [
                    python,
                    "scripts/search_fddp_capture.py",
                    "--state-index",
                    str(state_index),
                    "--seed",
                    str(state_seed),
                    "--initial-controller",
                    str(paths["approach"]),
                    "--initial-feasible",
                    "--iterations",
                    "180",
                    "--tracking-gain-scale",
                    "0.85",
                    "--terminal-weight",
                    "100000",
                    "--terminal-state-weight",
                    "100000",
                    "--out",
                    str(paths["approach_fddp"]),
                ],
                paths["approach_fddp"],
                resume=resume,
            )
            stage_payloads["approach_fddp"] = approach_fddp
        final_stage, final_payload = best_stage(stage_payloads)
        if not final_payload["result"]["success"]:
            source_stage, _ = best_extension_stage(stage_payloads)
            tail_seed = run_stage(
                [
                    python,
                    "scripts/refine_ilqr_capture_chain.py",
                    "--source-controller",
                    str(paths[source_stage]),
                    "--seed",
                    str(state_seed),
                    "--tail-seconds",
                    "1.5",
                    "--iterations",
                    "0",
                    "--initial-lqr-scale",
                    "0",
                    "--initial-action-cem",
                    "--action-knots",
                    "24",
                    "--cem-iterations",
                    "20",
                    "--cem-population",
                    "512",
                    "--cem-elites",
                    "40",
                    "--handoff-angle-abs",
                    "0.15",
                    "--handoff-cart-velocity-abs",
                    "0.5",
                    "--handoff-hinge-velocity-rms",
                    "0.75",
                    "--out",
                    str(paths["tail_seed"]),
                ],
                paths["tail_seed"],
                resume=resume,
            )
            stage_payloads["tail_seed"] = tail_seed
            final_stage, final_payload = best_stage(stage_payloads)
            if not final_payload["result"]["success"] and not args.no_ddp_fallback:
                source_stage, _ = best_extension_stage(stage_payloads)
                tail_ddp = run_stage(
                    [
                        python,
                        "scripts/refine_ilqr_capture_chain.py",
                        "--source-controller",
                        str(paths[source_stage]),
                        "--seed",
                        str(state_seed),
                        "--tail-seconds",
                        "1.5",
                        "--iterations",
                        "400",
                        "--initial-lqr-scale",
                        "0",
                        "--handoff-angle-abs",
                        "0.15",
                        "--handoff-cart-velocity-abs",
                        "0.5",
                        "--handoff-hinge-velocity-rms",
                        "0.75",
                        "--out",
                        str(paths["tail_ddp"]),
                    ],
                    paths["tail_ddp"],
                    resume=resume,
                )
                stage_payloads["tail_ddp"] = tail_ddp
                final_stage, final_payload = best_stage(stage_payloads)
        summary = result_summary(final_payload)
        records.append(
            {
                "ordinal": ordinal,
                "state_index": state_index,
                "seed": state_seed,
                "final_stage": final_stage,
                "result": summary,
                "artifacts": {
                    name: file_metadata(path)
                    for name, path in paths.items()
                    if path.exists()
                },
                "stage_results": {
                    name: result_summary(payload)
                    for name, payload in stage_payloads.items()
                },
            }
        )
        print(
            f"state={state_index} success={summary['success']} "
            f"hold={summary['max_upright_streak_seconds']:.3f}s "
            f"min_v={summary['minimum_lyapunov']:.2f}",
            flush=True,
        )

    success_count = sum(record["result"]["success"] for record in records)
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Small-cohort exact-model capture pipeline diagnostic; not P1 evidence.",
        "state_indices": indices,
        "episodes": len(records),
        "success_count": int(success_count),
        "success_rate": float(success_count / len(records)),
        "wall_time_seconds": float(time.time() - started),
        "records": records,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, summary_out)
    print(
        f"success={success_count}/{len(records)} rate={payload['success_rate']:.4f} "
        f"Wrote {summary_out}"
    )


if __name__ == "__main__":
    main()
