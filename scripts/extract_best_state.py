#!/usr/bin/env python
"""Materialize a single planner handoff as a state-list capture input."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a planner best_state into a one-state JSON list")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--record", default="best")
    parser.add_argument(
        "--record-path",
        default=None,
        help="dot-separated record path, such as best_by.score",
    )
    args = parser.parse_args()

    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    record: object = payload
    if args.record_path:
        for part in args.record_path.split("."):
            if not isinstance(record, dict) or part not in record:
                raise ValueError(f"{args.input} does not contain record path {args.record_path}")
            record = record[part]
    else:
        record = payload.get(args.record)
    if not isinstance(record, dict):
        raise ValueError(f"{args.input} does not contain a dictionary record")
    state = record.get("best_state")
    if not isinstance(state, dict):
        metrics = record.get("metrics", {})
        for key in ("best_upright_pass", "best_handoff", "best_capture", "best_any"):
            candidate = metrics.get(key) if isinstance(metrics, dict) else None
            if isinstance(candidate, dict):
                state = candidate
                break
    if not isinstance(state, dict):
        raise ValueError(f"{args.input} record has neither best_state nor metrics.best_upright_pass")
    if not isinstance(state.get("qpos"), list) or not isinstance(state.get("qvel"), list):
        raise ValueError("best_state must contain qpos and qvel lists")
    output = {
        "schema_version": 1,
        "source": str(Path(args.input)),
        "source_record": args.record,
        "states": [state],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
