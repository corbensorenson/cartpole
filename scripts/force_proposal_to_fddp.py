#!/usr/bin/env python
"""Convert an exact force-CEM proposal into an FDDP warm-start artifact.

The proposal contains only normalized force knots.  FDDP needs the exact
post-action state at every knot-expanded step, in the same dimensionless
coordinates used by the capture search.  This adapter replays the proposal
through serial MuJoCo and records those states without treating the proposal
as evidence of a solution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform
from gcartpole.env import NLinkCartPoleEnv


def normalized_force(knots: np.ndarray, step: int, step_count: int) -> float:
    phase = float(np.clip(step, 0, step_count - 1)) / float(max(1, step_count - 1))
    source = np.linspace(0.0, 1.0, len(knots), dtype=np.float64)
    return float(np.clip(np.interp(phase, source, knots), -1.0, 1.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--record-key", default="best")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    proposal_path = Path(args.proposal)
    proposal = json.loads(proposal_path.read_text(encoding="utf-8"))
    record = proposal.get(args.record_key)
    if not isinstance(record, dict):
        record = proposal.get("controller")
    if not isinstance(record, dict):
        raise ValueError(f"{proposal_path} does not contain a force proposal or controller")
    seconds = float(args.seconds if args.seconds is not None else proposal.get("search", {}).get("seconds", 0.0))
    if seconds <= 0.0:
        raise ValueError("seconds must be positive")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"], "init_mode": "hanging"}
    for noise_key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][noise_key] = 0.0
        cfg["env"][f"{noise_key}_start"] = 0.0
        cfg["env"][f"{noise_key}_end"] = 0.0

    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=0)
    spec = load_config(args.spec)
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        int(cfg["env"]["n_links"]),
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    transition = MujocoTransition(env, coordinate_transform=transform)
    env.reset(seed=0)
    policy_dt = float(env.dt)
    n_links = int(env.n)
    step_count = max(2, int(round(seconds / policy_dt)))
    knots = None if "knots" not in record else np.asarray(record["knots"], dtype=np.float64)
    if "stitched_controls" in record:
        source_controls = np.asarray(record["stitched_controls"], dtype=np.float64)
    elif "controls" in record:
        source_controls = np.asarray(record["controls"], dtype=np.float64)
    else:
        source_controls = None
    if knots is None and (source_controls is None or source_controls.ndim != 1):
        raise ValueError("proposal record must contain one-dimensional knots or controls")
    controls: list[float] = []
    physical_states = [data_state(env.data)]
    done_events: list[dict[str, Any]] = []
    try:
        for step in range(step_count):
            if source_controls is not None:
                source_seconds = float(record.get("horizon_seconds", seconds))
                source_t = np.linspace(0.0, max(source_seconds, 1e-9), len(source_controls), dtype=np.float64)
                target_t = np.linspace(0.0, seconds, step_count, dtype=np.float64)
                action = float(np.clip(np.interp(target_t[step], source_t, source_controls), -1.0, 1.0))
            else:
                action = normalized_force(knots, step, step_count)
            _, _, terminated, truncated, info = env.step([action])
            controls.append(action)
            physical_states.append(data_state(env.data))
            if terminated or truncated:
                done_events.append(
                    {
                        "step": int(step + 1),
                        "time_seconds": float((step + 1) * env.dt),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "termination_reason": info.get("termination_reason"),
                    }
                )
                break
    finally:
        env.close()

    if len(controls) != step_count:
        raise RuntimeError("proposal replay terminated before the requested FDDP horizon")
    nominal_states = np.asarray(
        [transition.to_coordinates(state) for state in physical_states],
        dtype=np.float64,
    )
    feedback_gains = np.zeros((step_count, nominal_states.shape[1]), dtype=np.float64)
    initial_physical_state = np.asarray(physical_states[0], dtype=np.float64)
    nq = n_links + 1
    out = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact-MuJoCo force-proposal replay converted to an FDDP warm start; not final evidence.",
        "selected_state": {
            "qpos": initial_physical_state[:nq].astype(float).tolist(),
            "qvel": initial_physical_state[nq:].astype(float).tolist(),
            "state_index": 0,
        },
        "terminal_state": {
            "qpos": np.asarray(physical_states[-1][:nq], dtype=np.float64).astype(float).tolist(),
            "qvel": np.asarray(physical_states[-1][nq:], dtype=np.float64).astype(float).tolist(),
        },
        "controller": {
            "type": "exact_mujoco_force_proposal_warm_start",
            "controls": np.asarray(controls, dtype=np.float64).astype(float).tolist(),
            "feedback_gains": feedback_gains.astype(float).tolist(),
            "horizon_steps": int(step_count),
            "horizon_seconds": float(step_count * policy_dt),
            "lqr_scale": 0.5,
            "source_proposal": file_metadata(proposal_path),
        },
        "search": {
            "converged": False,
            "is_feasible": False,
            "iterations": 0,
            "cost": None,
            "nominal_coordinate_states": nominal_states.astype(float).tolist(),
            "source_record_key": args.record_key if proposal.get(args.record_key) is not None else "controller",
            "source_record": {
                key: record.get(key)
                for key in ("cost", "best_time_seconds", "max_angle", "hinge_rms", "rail")
            },
        },
        "replay": {
            "progress": float(args.progress),
            "seed": 0,
            "steps": int(step_count),
            "done_events": done_events,
            "max_cart_excursion": float(max(abs(float(state[0])) for state in physical_states)),
            "nominal_sha256": data_sha256(nominal_states.tolist()),
        },
        "evidence": {
            "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(cfg)},
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(out, Path(args.out))
    print(f"Wrote {args.out} steps={step_count} max_cart={out['replay']['max_cart_excursion']:.3f}")


if __name__ == "__main__":
    main()
