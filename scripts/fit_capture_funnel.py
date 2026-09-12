#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.capture_funnel import (
    binary_metrics,
    conservative_threshold,
    deterministic_stratified_split,
    effective_state,
    fit_capture_funnel,
    normalized_capture_coordinates,
    polynomial_features,
)
from gcartpole.config import dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp


def evaluation_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    evaluation = payload.get("evaluation", payload)
    rows = evaluation.get("episode_results") if isinstance(evaluation, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("rollout evidence must contain evaluation.episode_results or episode_results")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit a conservative empirical capture-funnel model from exact rollouts")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/train.json")
    parser.add_argument("--rollouts", required=True)
    parser.add_argument("--progress", type=float, required=True)
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    spec = load_config(args.spec)
    dataset_path = Path(args.dataset)
    rollout_path = Path(args.rollouts)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rollouts = json.loads(rollout_path.read_text(encoding="utf-8"))
    if dataset.get("split") != "train":
        raise ValueError("capture funnel fitting is restricted to the frozen training split")
    states = dataset.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("dataset must contain states")

    rows_by_index: dict[int, dict[str, Any]] = {}
    for row in evaluation_rows(rollouts):
        index = int(row["state_index"])
        if index in rows_by_index:
            raise ValueError(f"duplicate rollout label for state index {index}")
        if index < 0 or index >= len(states):
            raise ValueError(f"rollout state index {index} is outside the source dataset")
        rows_by_index[index] = row

    distribution = spec["distribution"]
    env_cfg = cfg["env"]
    qpos_power = float(env_cfg.get("init_qpos_scale_power", 1.0))
    qvel_power = float(env_cfg.get("init_qvel_scale_power", 1.0))
    qpos_start = float(env_cfg.get("init_qpos_scale_start", env_cfg.get("init_qpos_scale", 1.0)))
    qpos_end = float(env_cfg.get("init_qpos_scale_end", env_cfg.get("init_qpos_scale", 1.0)))
    qvel_start = float(env_cfg.get("init_qvel_scale_start", env_cfg.get("init_qvel_scale", 1.0)))
    qvel_end = float(env_cfg.get("init_qvel_scale_end", env_cfg.get("init_qvel_scale", 1.0)))
    qpos_scale = qpos_start + (qpos_end - qpos_start) * float(args.progress) ** qpos_power
    qvel_scale = qvel_start + (qvel_end - qvel_start) * float(args.progress) ** qvel_power
    coordinate_bounds = {
        "cart_position_bound": float(distribution["cart_position_abs_max"]) * qpos_scale,
        "angle_bound": float(distribution["absolute_link_angle_abs_max"]) * qpos_scale,
        "cart_velocity_bound": float(distribution["cart_velocity_abs_max"]) * qvel_scale,
        "hinge_velocity_bound": float(distribution["hinge_velocity_rms_max"]) * qvel_scale,
    }
    if min(coordinate_bounds.values()) <= 0.0:
        raise ValueError("funnel fitting requires nonzero effective coordinate bounds")

    records: list[dict[str, Any]] = []
    coordinates = []
    labels = []
    identifiers = []
    for index in sorted(rows_by_index):
        state = states[index]
        row = rows_by_index[index]
        qpos, qvel = effective_state(
            np.asarray(state["qpos"], dtype=np.float64),
            np.asarray(state["qvel"], dtype=np.float64),
            progress=float(args.progress),
            qpos_scale_power=qpos_power,
            qvel_scale_power=qvel_power,
            qpos_scale_start=qpos_start,
            qpos_scale_end=qpos_end,
            qvel_scale_start=qvel_start,
            qvel_scale_end=qvel_end,
            cart_target=float(env_cfg.get("init_qpos_cart_target", 0.0)),
        )
        coordinate = normalized_capture_coordinates(qpos, qvel, **coordinate_bounds)
        label = int(bool(row["success"]))
        identifiers.append(str(state["state_id"]))
        labels.append(label)
        coordinates.append(coordinate)
        records.append(
            {
                "source_index": index,
                "state_id": state["state_id"],
                "qpos": qpos.astype(float).tolist(),
                "qvel": qvel.astype(float).tolist(),
                "success": bool(label),
                "termination_reason": row.get("termination_reason"),
                "max_upright_streak_seconds": float(row.get("max_upright_streak_seconds", 0.0)),
                "max_cart_excursion": float(row.get("max_cart_excursion", 0.0)),
            }
        )

    labels_array = np.asarray(labels, dtype=np.int64)
    coordinates_array = np.asarray(coordinates, dtype=np.float64)
    development, holdout = deterministic_stratified_split(labels_array, identifiers, float(args.holdout_fraction))
    model, optimizer = fit_capture_funnel(
        coordinates_array,
        labels_array,
        development,
        l2=float(args.l2),
        max_iterations=int(args.max_iterations),
    )
    model.coordinate_bounds = coordinate_bounds
    model.coordinate_abs_limits = 1.05 * np.max(np.abs(coordinates_array[development]), axis=0)
    all_features = polynomial_features(coordinates_array)
    standardized = (all_features - model.feature_mean) / model.feature_scale
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(standardized @ model.weights + model.bias, -60.0, 60.0)))
    in_domain = np.all(np.abs(coordinates_array) <= model.coordinate_abs_limits, axis=1)
    probabilities = np.where(in_domain, probabilities, 0.0)
    threshold, threshold_metrics = conservative_threshold(
        labels_array[holdout], probabilities[holdout], float(args.minimum_precision)
    )
    model.acceptance_threshold = threshold
    for record, probability in zip(records, probabilities, strict=True):
        record["predicted_capture_probability"] = float(probability)
        record["predicted_funnel_member"] = bool(probability >= threshold)

    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Training-only empirical capture-funnel fit; membership is a policy-specific predictor, not P1 evidence.",
        "benchmark": dataset.get("benchmark"),
        "source_split": dataset.get("split"),
        "progress": float(args.progress),
        "effective_scales": {"qpos": qpos_scale, "qvel": qvel_scale},
        "coordinate_bounds": coordinate_bounds,
        "label_count": int(labels_array.size),
        "success_count": int(np.sum(labels_array)),
        "failure_count": int(np.sum(1 - labels_array)),
        "development_indices": development.astype(int).tolist(),
        "holdout_indices": holdout.astype(int).tolist(),
        "minimum_holdout_precision": float(args.minimum_precision),
        "out_of_domain_label_count": int(np.sum(~in_domain)),
        "model": model.to_dict(),
        "fit": {
            **optimizer,
            "feature_count": int(model.weights.size),
            "development": binary_metrics(labels_array[development], probabilities[development], threshold),
            "holdout_at_half": binary_metrics(labels_array[holdout], probabilities[holdout], 0.5),
            "holdout_at_acceptance_threshold": threshold_metrics,
        },
        "labels_sha256": data_sha256(records),
        "labels": records,
        "evidence": {
            "config": file_metadata(args.config),
            "spec": file_metadata(args.spec),
            "dataset": file_metadata(dataset_path),
            "rollouts": file_metadata(rollout_path),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    holdout_metrics = payload["fit"]["holdout_at_acceptance_threshold"]
    print(
        f"labels={labels_array.size} successes={int(np.sum(labels_array))} "
        f"holdout_auc={payload['fit']['holdout_at_half']['roc_auc']:.3f} "
        f"threshold={threshold:.6f} precision={holdout_metrics['precision']:.3f} "
        f"recall={holdout_metrics['recall']:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
