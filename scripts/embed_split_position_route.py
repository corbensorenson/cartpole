#!/usr/bin/env python
"""Embed a solved route into a locked split at an arbitrary source link.

This is the config-aware counterpart to the older distal split diagnostic.  It
uses the same physical-state lift, coordinate transform, feedback projection,
and time resampling as ``generalized_swingup_solver split`` so interior split
positions are evaluated in a consistent state convention.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.generalized_solver import (
    force_action_scale,
    resample_controls,
    setup_from_config,
    split_state_lift_matrix,
    split_state_projection,
)
from gcartpole.modal import StateScales, dimensionless_absolute_transform


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_SCALES = StateScales(
    cart_position=3.0,
    absolute_angle=0.15,
    cart_velocity=4.0,
    hinge_velocity=15.0,
)


def coordinate_transform(n_links: int, spec: dict[str, Any] | None) -> np.ndarray:
    scales = DEFAULT_SCALES
    if spec is not None:
        distribution = spec["distribution"]
        scales = StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        )
    return dimensionless_absolute_transform(n_links, scales)


def time_interpolate_rows(rows: np.ndarray, source_count: int, target_count: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.shape[0] != source_count:
        raise ValueError("time-series row count does not match source controls")
    source_phase = (np.arange(source_count, dtype=np.float64) + 0.5) / source_count
    target_phase = (np.arange(target_count, dtype=np.float64) + 0.5) / target_count
    return np.column_stack(
        [np.interp(target_phase, source_phase, rows[:, column]) for column in range(rows.shape[1])]
    )


def time_interpolate_samples(rows: np.ndarray, target_count: int) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or target_count < 2:
        raise ValueError("state samples and target count must both have endpoints")
    source_phase = np.linspace(0.0, 1.0, rows.shape[0])
    target_phase = np.linspace(0.0, 1.0, target_count)
    return np.column_stack(
        [np.interp(target_phase, source_phase, rows[:, column]) for column in range(rows.shape[1])]
    )


def build_embedding(source_links: int, split_link: int) -> np.ndarray:
    """Return contiguous source-link assignments for a one-based split."""

    if source_links < 1 or not 1 <= split_link <= source_links:
        raise ValueError("split_link must be a valid one-based source link")
    return np.insert(np.arange(source_links, dtype=np.int64), split_link - 1, split_link - 1)


def embed_route(
    payload: dict[str, Any],
    source_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    *,
    split_link: int,
    spec: dict[str, Any] | None,
) -> dict[str, Any]:
    source = setup_from_config(source_cfg)
    target_start = setup_from_config(target_cfg, progress=0.0)
    if target_start.n_links != source.n_links + 1:
        raise ValueError("target config must have exactly one more link")

    controller = payload.get("controller")
    search = payload.get("search")
    if not isinstance(controller, dict) or not isinstance(search, dict):
        raise ValueError("source artifact must contain controller and search objects")
    controls_source = np.asarray(controller["controls"], dtype=np.float64)
    gains_source = np.asarray(controller["feedback_gains"], dtype=np.float64)
    states_source = np.asarray(search["nominal_coordinate_states"], dtype=np.float64)
    source_dim = 2 * (source.n_links + 1)
    if gains_source.shape != (controls_source.size, source_dim):
        raise ValueError("source feedback gains do not match source morphology")
    if states_source.shape != (controls_source.size + 1, source_dim):
        raise ValueError("source nominal states do not match source morphology")

    assignments = build_embedding(source.n_links, split_link)
    source_transform = coordinate_transform(source.n_links, spec)
    target_transform = coordinate_transform(target_start.n_links, spec)
    physical_lift = split_state_lift_matrix(
        source.n_links,
        assignments,
        length_scale=target_start.chain_length / source.chain_length,
    )
    coordinate_lift = target_transform @ physical_lift @ np.linalg.inv(source_transform)
    projection = split_state_projection(
        coordinate_lift,
        source.lengths[assignments] / target_start.chain_length,
    )
    action_scale = force_action_scale(source, target_start)
    controls = resample_controls(controls_source, source, target_start)
    spatial_gains = action_scale * gains_source @ projection
    feedback_gains = time_interpolate_rows(
        spatial_gains, controls_source.size, controls.size
    )
    lifted_states = states_source @ coordinate_lift.T
    states = time_interpolate_samples(lifted_states, controls.size + 1)
    initial_physical = physical_lift @ np.linalg.solve(source_transform, states_source[0])
    target_dim = 2 * (target_start.n_links + 1)
    if initial_physical.shape != (target_dim,):
        raise ValueError("lifted initial state has an unexpected dimension")
    invariant_error = float(np.max(np.abs(projection @ coordinate_lift - np.eye(source_dim))))
    return {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_locked_split_position_embedding_not_solution",
        "not_solution": True,
        "summary": "Config-aware locked split route embedding using the validated physical lift and coordinate projection.",
        "source": file_metadata(payload["_source_path"]),
        "selected_state": {
            "qpos": initial_physical[: target_start.n_links + 1].astype(float).tolist(),
            "qvel": initial_physical[target_start.n_links + 1 :].astype(float).tolist(),
            "state_index": 0,
        },
        "controller": {
            "type": "config_aware_locked_split_position_embedding",
            "controls": controls.astype(float).tolist(),
            "feedback_gains": feedback_gains.astype(float).tolist(),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(controls.size * target_start.policy_dt),
            "policy_dt": float(target_start.policy_dt),
            "coordinate_transform": target_transform.astype(float).tolist(),
        },
        "search": {
            "nominal_coordinate_states": states.astype(float).tolist(),
            "is_feasible": False,
            "iterations": 0,
            "cost": None,
        },
        "embedding": {
            "source_links": int(source.n_links),
            "target_links": int(target_start.n_links),
            "split_link_one_based": int(split_link),
            "segment_source_links": assignments.astype(int).tolist(),
            "inserted_joint_initially_locked": True,
            "physical_lift": physical_lift.astype(float).tolist(),
            "coordinate_lift": coordinate_lift.astype(float).tolist(),
            "coordinate_projection": projection.astype(float).tolist(),
            "feedback_invariance_max_abs_error": invariant_error,
            "force_action_scale": float(action_scale),
            "target_start_lengths": target_start.lengths.astype(float).tolist(),
            "target_start_masses": target_start.masses.astype(float).tolist(),
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(ROOT),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--split-link", type=int, required=True)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    source_path = Path(args.controller)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    payload["_source_path"] = source_path
    spec = load_config(args.spec) if args.spec else None
    result = embed_route(
        payload,
        load_config(args.source_config),
        load_config(args.target_config),
        split_link=args.split_link,
        spec=spec,
    )
    dump_json(result, Path(args.out))
    print(
        f"wrote {args.out}: n={result['embedding']['source_links']}->"
        f"{result['embedding']['target_links']} split={args.split_link}"
    )


if __name__ == "__main__":
    main()
