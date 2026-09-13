#!/usr/bin/env python
"""Rank serialized swing states using morphology-derived upright modes.

The output is a standard repository state list and can be passed directly to
``search_fddp_capture.py``.  It replaces link-count-specific hand selection
with a deterministic, dimensionless ranking while preserving the original
measured MuJoCo states verbatim.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.generalized_modes import chain_normal_modes, modal_handoff_metrics
from gcartpole.generalized_solver import setup_from_config


def find_trace(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    candidates = (
        ("final_eval.trace", payload.get("final_eval", {}).get("trace")),
        ("result.trajectory", payload.get("result", {}).get("trajectory")),
        ("trace_rollout.trace", payload.get("trace_rollout", {}).get("trace")),
        ("trace", payload.get("trace")),
    )
    for name, value in candidates:
        if isinstance(value, list) and value:
            return name, [dict(row) for row in value]
    raise ValueError("input artifact has no supported nonempty serialized trace")


def rank_row(
    row: dict[str, Any],
    *,
    modes: Any,
    chain_length: float,
    natural_time: float,
    velocity_scale: float,
    energy_scale: float,
) -> dict[str, Any]:
    qpos = np.asarray(row.get("qpos"), dtype=np.float64)
    qvel = np.asarray(row.get("qvel"), dtype=np.float64)
    expected = modes.n_links + 1
    if qpos.shape != (expected,) or qvel.shape != (expected,):
        raise ValueError(
            "trace row qpos/qvel shape does not match requested morphology"
        )
    metrics = modal_handoff_metrics(
        modes,
        qpos,
        qvel,
        chain_length=chain_length,
        natural_time=natural_time,
        velocity_scale=velocity_scale,
        energy_scale=energy_scale,
    )
    return {
        **row,
        "modal_handoff": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--n-links", type=int, required=True)
    parser.add_argument("--max-states", type=int, default=12)
    parser.add_argument("--minimum-separation-seconds", type=float, default=0.08)
    parser.add_argument("--maximum-angle", type=float, default=1.20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.n_links, args.max_states) < 1:
        raise ValueError("link count and max states must be positive")
    if args.minimum_separation_seconds < 0.0 or args.maximum_angle <= 0.0:
        raise ValueError("separation must be nonnegative and angle must be positive")

    source = Path(args.input)
    payload = json.loads(source.read_text(encoding="utf-8"))
    trace_path, trace = find_trace(payload)
    cfg = copy.deepcopy(load_config(args.config))
    cfg["env"]["n_links"] = int(args.n_links)
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    modes = chain_normal_modes(env, equilibrium="upright")
    setup = setup_from_config(cfg)
    ranked = [
        rank_row(
            row,
            modes=modes,
            chain_length=setup.chain_length,
            natural_time=setup.natural_time,
            velocity_scale=np.sqrt(setup.gravity * setup.chain_length),
            energy_scale=float(env._energy_gap),
        )
        for row in trace
        if isinstance(row.get("qpos"), list) and isinstance(row.get("qvel"), list)
    ]
    env.close()
    ranked = [
        row
        for row in ranked
        if row["modal_handoff"]["max_abs_angle"] <= args.maximum_angle
    ]
    ranked.sort(key=lambda row: float(row["modal_handoff"]["score"]))
    selected: list[dict[str, Any]] = []
    for row in ranked:
        time_seconds = float(
            row.get("time_seconds", row.get("step", 0) * setup.policy_dt)
        )
        if all(
            abs(time_seconds - float(other.get("time_seconds", 0.0)))
            >= args.minimum_separation_seconds - 1.0e-12
            for other in selected
        ):
            row["source_artifact"] = str(source)
            row["source_trace_path"] = trace_path
            selected.append(row)
        if len(selected) >= args.max_states:
            break
    if not selected:
        raise ValueError("no trace states passed the requested angle filter")

    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "claim_status": "deterministic_modal_handoff_candidates_not_solution",
        "summary": "Measured states ranked by exact morphology-derived upright modal coordinates.",
        "source_file": file_metadata(source),
        "source_trace_path": trace_path,
        "config": file_metadata(Path(args.config)),
        "n_links": int(args.n_links),
        "normal_modes": modes.to_dict(),
        "selection": {
            "max_states": int(args.max_states),
            "minimum_separation_seconds": float(args.minimum_separation_seconds),
            "maximum_angle": float(args.maximum_angle),
        },
        "state_count": len(selected),
        "states_sha256": data_sha256(selected),
        "states": selected,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, args.out)
    print(
        f"wrote {args.out}: {len(selected)} states; "
        f"best_score={selected[0]['modal_handoff']['score']:.4f}"
    )


if __name__ == "__main__":
    main()
