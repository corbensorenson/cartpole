#!/usr/bin/env python
"""Extract exact states from a recorded MuJoCo swing for capture training.

The output is intentionally a normal ``env.init_mode=state_list`` file.  It
contains only states that were actually visited by the source controller, so
the capture expert is trained on a real handoff distribution rather than a
synthetic angle/velocity box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json
from gcartpole.env import serial_absolute_angles
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp


def load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    rows = payload.get("trajectory")
    if not isinstance(rows, list) or not rows:
        rows = payload.get("trace")
    if not isinstance(rows, list) or not rows:
        result = payload.get("result")
        if isinstance(result, dict):
            rows = result.get("trajectory")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} does not contain non-empty trajectory or trace rows")
    records = [row for row in rows if isinstance(row, dict)]
    if len(records) != len(rows):
        raise ValueError(f"{path} contains non-object trajectory rows")
    return payload, records


def row_metrics(row: dict[str, Any]) -> dict[str, Any]:
    qpos = np.asarray(row.get("qpos", []), dtype=np.float64)
    qvel = np.asarray(row.get("qvel", []), dtype=np.float64)
    if qpos.ndim != 1 or qvel.ndim != 1 or qpos.shape != qvel.shape or qpos.size < 2:
        raise ValueError("each source row must contain equal one-dimensional qpos and qvel arrays")
    absolute_angles = np.asarray(row.get("absolute_angles", []), dtype=np.float64)
    if absolute_angles.shape != (qpos.size - 1,):
        absolute_angles = serial_absolute_angles(qpos[1:])
    absolute_rates = np.cumsum(qvel[1:])
    hinge_rms = float(row.get("hinge_velocity_rms", np.sqrt(np.mean(qvel[1:] ** 2))))
    absolute_rate_rms = float(
        row.get("absolute_angular_velocity_rms", row.get("absolute_rate_rms", np.sqrt(np.mean(absolute_rates**2))))
    )
    return {
        "max_abs_angle": float(row.get("max_abs_angle", np.max(np.abs(absolute_angles)))),
        "hinge_velocity_rms": hinge_rms,
        "absolute_angular_velocity_rms": absolute_rate_rms,
        "x": float(row.get("x", qpos[0])),
        "cart_velocity": float(row.get("cart_velocity", qvel[0])),
        "absolute_angles": absolute_angles.astype(float).tolist(),
    }


def selection_score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    metrics = row_metrics(row)
    return (
        metrics["max_abs_angle"],
        metrics["absolute_angular_velocity_rms"],
        metrics["hinge_velocity_rms"],
        abs(metrics["x"]),
        abs(metrics["cart_velocity"]),
    )


def make_state(row: dict[str, Any], source: str, source_index: int) -> dict[str, Any]:
    qpos = np.asarray(row["qpos"], dtype=np.float64)
    qvel = np.asarray(row["qvel"], dtype=np.float64)
    metrics = row_metrics(row)
    state = {
        "source": source,
        "source_row_index": int(source_index),
        "source_step": int(row.get("step", source_index + 1)),
        "source_time_seconds": float(row.get("time_seconds", 0.0)),
        "qpos": qpos.astype(float).tolist(),
        "qvel": qvel.astype(float).tolist(),
        **metrics,
        "action": None if row.get("action") is None else float(row["action"]),
        "is_upright": bool(row.get("is_upright", False)),
        "upright_streak_seconds": float(row.get("upright_streak_seconds", 0.0)),
        "max_upright_streak_seconds": float(row.get("max_upright_streak_seconds", 0.0)),
    }
    return state


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a real trajectory handoff state list")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-time", type=float, default=0.0)
    parser.add_argument("--max-time", type=float, default=None)
    parser.add_argument("--max-angle", type=float, default=None)
    parser.add_argument("--max-hinge-rms", type=float, default=None)
    parser.add_argument("--max-absolute-rate-rms", type=float, default=None)
    parser.add_argument("--max-cart-abs", type=float, default=None)
    parser.add_argument("--max-cart-velocity", type=float, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=None)
    args = parser.parse_args()

    if args.min_time < 0.0 or args.max_time is not None and args.max_time < args.min_time:
        raise ValueError("time bounds must satisfy 0 <= min-time <= max-time")
    if args.stride < 1:
        raise ValueError("--stride must be positive")
    if args.max_states is not None and args.max_states < 1:
        raise ValueError("--max-states must be positive")

    source_path = Path(args.input)
    payload, rows = load_rows(source_path)
    selected: list[tuple[int, dict[str, Any]]] = []
    rejected = 0
    for index, row in enumerate(rows):
        metrics = row_metrics(row)
        time_seconds = float(row.get("time_seconds", 0.0))
        if index % args.stride != 0:
            continue
        if time_seconds < args.min_time or args.max_time is not None and time_seconds > args.max_time:
            continue
        checks = (
            args.max_angle is None or metrics["max_abs_angle"] <= args.max_angle,
            args.max_hinge_rms is None or metrics["hinge_velocity_rms"] <= args.max_hinge_rms,
            args.max_absolute_rate_rms is None
            or metrics["absolute_angular_velocity_rms"] <= args.max_absolute_rate_rms,
            args.max_cart_abs is None or abs(metrics["x"]) <= args.max_cart_abs,
            args.max_cart_velocity is None or abs(metrics["cart_velocity"]) <= args.max_cart_velocity,
        )
        if all(checks):
            selected.append((index, row))
        else:
            rejected += 1

    if not selected:
        raise ValueError("filters selected no trajectory states")
    selected.sort(key=lambda item: selection_score(item[1]))
    if args.max_states is not None:
        selected = selected[: args.max_states]
    states = [make_state(row, str(source_path), index) for index, row in selected]

    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact visited states from a recorded swing, filtered for capture-expert initialization.",
        "source_file": file_metadata(source_path),
        "source_summary": payload.get("summary"),
        "source_config_path": payload.get("config_path"),
        "source_progress": payload.get("progress"),
        "selection": {
            "source_row_count": int(len(rows)),
            "selected_count": int(len(states)),
            "rejected_count": int(rejected),
            "min_time": float(args.min_time),
            "max_time": None if args.max_time is None else float(args.max_time),
            "max_angle": None if args.max_angle is None else float(args.max_angle),
            "max_hinge_rms": None if args.max_hinge_rms is None else float(args.max_hinge_rms),
            "max_absolute_rate_rms": None
            if args.max_absolute_rate_rms is None
            else float(args.max_absolute_rate_rms),
            "max_cart_abs": None if args.max_cart_abs is None else float(args.max_cart_abs),
            "max_cart_velocity": None if args.max_cart_velocity is None else float(args.max_cart_velocity),
            "stride": int(args.stride),
            "max_states": None if args.max_states is None else int(args.max_states),
        },
        "state_count": int(len(states)),
        "states_sha256": data_sha256(states),
        "states": states,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, Path(args.out))
    best = states[0]
    print(
        f"Wrote {args.out} states={len(states)} source_rows={len(rows)} "
        f"best_angle={best['max_abs_angle']:.6f} best_abs_rate={best['absolute_angular_velocity_rms']:.6f}"
    )


if __name__ == "__main__":
    main()
