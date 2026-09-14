#!/usr/bin/env python
"""Publish compact, hash-bound evidence from support-continuation ledgers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import (
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)


def select_boundary_trials(
    trials: list[dict[str, Any]], progress: float
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    accepted = [trial for trial in trials if trial.get("accepted")]
    if not accepted:
        raise ValueError("ledger has no accepted trials")
    final_accepted = max(
        accepted,
        key=lambda trial: (
            float(trial["proposed_progress"]),
            int(trial["trial"]),
        ),
    )
    rejected_above = [
        trial
        for trial in trials
        if not trial.get("accepted")
        and float(trial["proposed_progress"]) > progress
    ]
    nearest_rejected = (
        min(
            rejected_above,
            key=lambda trial: (
                float(trial["proposed_progress"]) - progress,
                -int(trial["trial"]),
            ),
        )
        if rejected_above
        else None
    )
    return final_accepted, nearest_rejected


def compact_trial(trial: dict[str, Any] | None) -> dict[str, Any] | None:
    if trial is None:
        return None
    return {
        key: trial.get(key)
        for key in (
            "trial",
            "accepted",
            "from_progress",
            "proposed_progress",
            "step",
            "lock_strengths",
            "dimensionless",
            "outcome",
            "config",
            "result",
            "replay_only_attempt",
        )
    }


def summarize(label: str, ledger_path: Path) -> dict[str, Any]:
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    progress = float(ledger["progress"])
    accepted, rejected = select_boundary_trials(ledger["trials"], progress)
    continuation_path = Path(ledger["continuation_config"]["path"])
    continuation = load_config(continuation_path)
    return {
        "label": label,
        "status": ledger["status"],
        "not_solution": bool(ledger["not_solution"]),
        "accepted_progress": progress,
        "next_step": float(ledger["next_step"]),
        "failed_upper_bound": ledger.get("failed_upper_bound"),
        "support_contract": continuation.get("supported_unlock"),
        "source_ledger": file_metadata(ledger_path),
        "continuation_config": file_metadata(continuation_path),
        "final_accepted": compact_trial(accepted),
        "nearest_rejected_above_frontier": compact_trial(rejected),
    }


def parse_ledger(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("ledger must be LABEL=PATH")
    path = Path(raw_path)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"ledger does not exist: {path}")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger",
        action="append",
        type=parse_ledger,
        required=True,
        help="LABEL=PATH; repeat in intended stage order",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "supported_unlock_frontier_not_target_solution",
        "not_solution": True,
        "summary": (
            "Compact exact-replay frontier for the deterministic unequal-link "
            "count continuation. Homotopy progress is not task completion."
        ),
        "stages": [summarize(label, path) for label, path in args.ledger],
        "limitations": [
            "The equality-release endpoint retains declared physical support.",
            "The unsupported unequal three-link target has not passed.",
            "No mirror or independent noisy target-plant gate has passed.",
            "Only boundary artifacts are published; full optimization scratch histories remain local.",
        ],
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(f"wrote {args.out} with {len(payload['stages'])} stages")


if __name__ == "__main__":
    main()
