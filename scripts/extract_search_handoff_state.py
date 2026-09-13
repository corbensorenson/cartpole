#!/usr/bin/env python
"""Materialize a measured best state from a search artifact as a state list.

Search artifacts often retain the selected MuJoCo state without retaining the
full rollout.  This converter only accepts state records already serialized by
the producer; it never synthesizes qpos/qvel values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from gcartpole.config import dump_json
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp


def find_state(payload: dict[str, Any], selector: str) -> dict[str, Any]:
    if selector.startswith("trajectory:"):
        try:
            index = int(selector.split(":", 1)[1])
        except ValueError as error:
            raise ValueError("trajectory selector must be trajectory:<index>") from error
        trajectory = payload.get("result", {}).get("trajectory")
        if not isinstance(trajectory, list) or not trajectory:
            raise ValueError("input artifact has no serialized result trajectory")
        if index < 0 or index >= len(trajectory):
            raise IndexError(f"trajectory index {index} outside 0..{len(trajectory) - 1}")
        return dict(trajectory[index])
    if selector in {"best", "best_state"}:
        best = payload.get("best")
        if isinstance(best, dict) and isinstance(best.get("best_state"), dict):
            return dict(best["best_state"])
        if isinstance(best, dict) and isinstance(best.get("metrics"), dict):
            candidate = best["metrics"].get("best_upright_pass")
            if isinstance(candidate, dict):
                return dict(candidate)
        if isinstance(best, dict) and isinstance(best.get("result"), dict):
            candidate = best["result"].get("best_row")
            if isinstance(candidate, dict):
                return dict(candidate)
        candidate = payload.get("best_upright_pass")
        if isinstance(candidate, dict):
            return dict(candidate)
    if selector in {"terminal", "terminal_state"}:
        candidate = payload.get("terminal_state")
        if isinstance(candidate, dict):
            return dict(candidate)
        candidate = payload.get("terminal_metrics")
        if isinstance(candidate, dict):
            return dict(candidate)
    states = payload.get("states")
    if isinstance(states, list) and states:
        index = int(selector)
        return dict(states[index])
    raise ValueError(f"could not find serialized state selector {selector!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--selector", default="best")
    args = parser.parse_args()

    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("input artifact must contain a JSON object")
    state = find_state(payload, args.selector)
    qpos = state.get("qpos")
    qvel = state.get("qvel")
    if not isinstance(qpos, list) or not isinstance(qvel, list) or len(qpos) != len(qvel):
        raise ValueError("selected record must contain equal-length qpos and qvel lists")

    state["source_artifact"] = str(source)
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Measured handoff state materialized from an existing search artifact.",
        "source_file": file_metadata(source),
        "selector": args.selector,
        "state_count": 1,
        "states_sha256": data_sha256([state]),
        "states": [state],
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, Path(args.out))
    print(f"Wrote {args.out} state_shape={(len(qpos), len(qvel))}")


if __name__ == "__main__":
    main()
