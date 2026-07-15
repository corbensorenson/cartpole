#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_lyapunov
from scipy.stats import rankdata

from gcartpole.capture_envelope import validate_capture_config, validate_capture_states
from gcartpole.capture_funnel import effective_state
from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.modal import (
    StateScales,
    dimensionless_absolute_transform,
    modal_decomposition,
    real_schur_decomposition,
    transform_dynamics,
    transform_feedback_gain,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_swingup_capture import lqr_gain


def binary_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positive = labels == 1
    positive_count = int(np.sum(positive))
    negative_count = int(labels.size - positive_count)
    if positive_count == 0 or negative_count == 0:
        return None
    ranks = rankdata(scores, method="average")
    return float(
        (np.sum(ranks[positive]) - positive_count * (positive_count + 1) / 2)
        / (positive_count * negative_count)
    )


def class_summary(values: np.ndarray, failure: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    failure = np.asarray(failure, dtype=bool)
    succeeded = values[~failure]
    failed = values[failure]
    success_mean = float(np.mean(succeeded)) if succeeded.size else None
    failure_mean = float(np.mean(failed)) if failed.size else None
    pooled_std = float(np.std(values))
    return {
        "success_count": int(succeeded.size),
        "failure_count": int(failed.size),
        "success_mean": success_mean,
        "failure_mean": failure_mean,
        "failure_to_success_mean_ratio": (
            None
            if success_mean is None or failure_mean is None
            else float(failure_mean / max(success_mean, 1e-15))
        ),
        "standardized_failure_mean_shift": (
            None
            if success_mean is None or failure_mean is None
            else float((failure_mean - success_mean) / max(pooled_std, 1e-15))
        ),
        "failure_roc_auc": binary_auc(failure.astype(np.int64), values),
    }


def complex_payload(value: complex) -> dict[str, float]:
    return {
        "real": float(np.real(value)),
        "imag": float(np.imag(value)),
        "magnitude": float(abs(value)),
    }


def state_scales(spec: dict[str, Any]) -> StateScales:
    distribution = spec["distribution"]
    return StateScales(
        cart_position=float(distribution["cart_position_abs_max"]),
        absolute_angle=float(distribution["absolute_link_angle_abs_max"]),
        cart_velocity=float(distribution["cart_velocity_abs_max"]),
        hinge_velocity=float(distribution["hinge_velocity_rms_max"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze capture failures in dimensionless LQR modal coordinates")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument(
        "--evaluation",
        default="runs/p1_capture_target_teachers/eval_adaptive_validation256_p0065.json",
    )
    parser.add_argument("--split", default="validation")
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--out", default="runs/p1_capture_modal/modal_analysis_validation256_p0065.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    spec = load_config(args.spec)
    dataset_path = Path(args.dataset)
    evaluation_path = Path(args.evaluation)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    errors = validate_capture_config(cfg, spec)
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    if args.split not in spec.get("splits", {}):
        errors.append(f"split {args.split!r} is not declared by the capture specification")
    else:
        errors.extend(validate_capture_states(dataset, spec, args.split))
    if evaluation.get("split") != args.split:
        errors.append("evaluation split does not match requested split")
    if evaluation.get("benchmark") != dataset.get("benchmark"):
        errors.append("evaluation benchmark does not match dataset")
    if evaluation.get("evidence", {}).get("config", {}).get("resolved_sha256") != data_sha256(cfg):
        errors.append("evaluation resolved config hash does not match current config")
    current_dataset_hash = file_metadata(dataset_path).get("sha256")
    if evaluation.get("evidence", {}).get("dataset", {}).get("sha256") != current_dataset_hash:
        errors.append("evaluation dataset hash does not match current dataset")
    rows = evaluation.get("episode_results")
    state_indices = evaluation.get("evidence", {}).get("state_indices")
    if not isinstance(rows, list) or not rows:
        errors.append("evaluation must contain episode_results")
    if not isinstance(state_indices, list) or len(state_indices) != len(rows or []):
        errors.append("evaluation state_indices must match episode_results")
    elif any(int(row.get("state_index", -1)) != int(index) for row, index in zip(rows, state_indices)):
        errors.append("evaluation row state indices do not match recorded state_indices")
    progress = float(evaluation.get("progress", -1.0))
    if not 0.0 <= progress <= 1.0:
        errors.append("evaluation progress must be in [0, 1]")
    if errors:
        raise ValueError("Capture modal analysis validation failed:\n- " + "\n- ".join(errors))

    n_links = int(cfg["env"]["n_links"])
    coordinate_labels = (
        ["cart_position"]
        + [f"absolute_angle_{index + 1}" for index in range(n_links)]
        + ["cart_velocity"]
        + [f"hinge_velocity_{index + 1}" for index in range(n_links)]
    )
    scales = state_scales(spec)
    transform = dimensionless_absolute_transform(n_links, scales)
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, args.fd_eps)
    dimensionless_a, dimensionless_b = transform_dynamics(state_matrix, input_matrix, transform)
    gain = lqr_gain(
        cfg,
        progress=1.0,
        fd_eps=args.fd_eps,
        control_cost=args.lqr_control_cost,
    )
    dimensionless_gain = transform_feedback_gain(gain, transform)
    closed_loop = dimensionless_a - dimensionless_b @ (
        float(args.lqr_scale) * dimensionless_gain.reshape(1, -1)
    )
    open_modes = modal_decomposition(dimensionless_a, dimensionless_b)
    closed_modes = modal_decomposition(closed_loop, dimensionless_b)
    schur_modes = real_schur_decomposition(closed_loop, dimensionless_b)
    spectral_radius = float(np.max(np.abs(closed_modes.eigenvalues)))
    if spectral_radius >= 1.0:
        raise ValueError(f"closed-loop dimensionless system is not linearly stable: radius={spectral_radius}")
    lyapunov = solve_discrete_lyapunov(closed_loop.T, np.eye(closed_loop.shape[0], dtype=np.float64))

    env_cfg = cfg["env"]
    states = dataset["states"]
    records: list[dict[str, Any]] = []
    for row in rows:
        state_index = int(row["state_index"])
        source_state = states[state_index]
        qpos, qvel = effective_state(
            np.asarray(source_state["qpos"], dtype=np.float64),
            np.asarray(source_state["qvel"], dtype=np.float64),
            progress=progress,
            qpos_scale_power=float(env_cfg.get("init_qpos_scale_power", 1.0)),
            qvel_scale_power=float(env_cfg.get("init_qvel_scale_power", 1.0)),
            qpos_scale_start=float(env_cfg.get("init_qpos_scale_start", env_cfg.get("init_qpos_scale", 1.0))),
            qpos_scale_end=float(env_cfg.get("init_qpos_scale_end", env_cfg.get("init_qpos_scale", 1.0))),
            qvel_scale_start=float(env_cfg.get("init_qvel_scale_start", env_cfg.get("init_qvel_scale", 1.0))),
            qvel_scale_end=float(env_cfg.get("init_qvel_scale_end", env_cfg.get("init_qvel_scale", 1.0))),
            cart_target=float(env_cfg.get("init_qpos_cart_target", 0.0)),
        )
        dimensionless_state = transform @ np.r_[qpos, qvel]
        amplitudes = schur_modes.grouped_amplitudes(dimensionless_state)
        shares = amplitudes / max(np.linalg.norm(amplitudes), 1e-15)
        records.append(
            {
                "episode": int(row["episode"]),
                "state_index": state_index,
                "state_id": source_state.get("state_id"),
                "baseline_success": bool(row["baseline"]["success"]),
                "planner_invoked": bool(row["planner_invoked"]),
                "planner_recovered": bool(row["planner_invoked"] and row["success"]),
                "success": bool(row["success"]),
                "rail_hit": bool(row["rail_hit"]),
                "max_upright_streak_seconds": float(row["max_upright_streak_seconds"]),
                "dimensionless_state": dimensionless_state.astype(float).tolist(),
                "dimensionless_state_norm": float(np.linalg.norm(dimensionless_state)),
                "dimensionless_lyapunov_value": float(dimensionless_state @ lyapunov @ dimensionless_state),
                "closed_loop_group_amplitudes": amplitudes.astype(float).tolist(),
                "closed_loop_group_shares": shares.astype(float).tolist(),
            }
        )

    amplitudes = np.asarray([record["closed_loop_group_amplitudes"] for record in records], dtype=np.float64)
    shares = np.asarray([record["closed_loop_group_shares"] for record in records], dtype=np.float64)
    final_failure = np.asarray([not record["success"] for record in records], dtype=bool)
    baseline_failure = np.asarray([not record["baseline_success"] for record in records], dtype=bool)
    planner_mask = np.asarray([record["planner_invoked"] for record in records], dtype=bool)
    planner_failure = final_failure[planner_mask]
    group_summaries = []
    for group_index, group in enumerate(schur_modes.groups):
        group_eigenvalues = schur_modes.group_eigenvalues[group_index]
        representative = next(
            (value for value in group_eigenvalues if np.imag(value) >= 0.0),
            group_eigenvalues[0],
        )
        basis_energy = np.sum(schur_modes.orthogonal_basis[:, list(group)] ** 2, axis=1)
        basis_fraction = basis_energy / max(float(np.sum(basis_energy)), 1e-15)
        top_coordinates = sorted(
            (
                {"coordinate": label, "basis_energy_fraction": float(fraction)}
                for label, fraction in zip(coordinate_labels, basis_fraction)
            ),
            key=lambda item: -item["basis_energy_fraction"],
        )[:5]
        summary = {
            "group_index": group_index,
            "component_indices": list(group),
            "eigenvalue": complex_payload(complex(representative)),
            "eigenvalues": [complex_payload(complex(value)) for value in group_eigenvalues],
            "input_coupling": float(schur_modes.group_input_coupling[group_index]),
            "top_coordinate_loadings": top_coordinates,
            "amplitude_all_final_outcomes": class_summary(amplitudes[:, group_index], final_failure),
            "amplitude_all_baseline_outcomes": class_summary(amplitudes[:, group_index], baseline_failure),
            "amplitude_planner_outcomes": (
                class_summary(amplitudes[planner_mask, group_index], planner_failure)
                if np.any(planner_mask)
                else None
            ),
            "share_all_final_outcomes": class_summary(shares[:, group_index], final_failure),
            "share_all_baseline_outcomes": class_summary(shares[:, group_index], baseline_failure),
            "share_planner_outcomes": (
                class_summary(shares[planner_mask, group_index], planner_failure)
                if np.any(planner_mask)
                else None
            ),
        }
        group_summaries.append(summary)
    group_summaries.sort(
        key=lambda summary: (
            -abs((summary["share_planner_outcomes"]["failure_roc_auc"] or 0.5) - 0.5),
            -abs(summary["share_planner_outcomes"]["standardized_failure_mean_shift"] or 0.0),
        )
    )

    state_norms = np.asarray([record["dimensionless_state_norm"] for record in records], dtype=np.float64)
    lyapunov_values = np.asarray(
        [record["dimensionless_lyapunov_value"] for record in records],
        dtype=np.float64,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Dimensionless modal diagnosis of a partial P1 validation frontier; not gate evidence.",
        "benchmark": dataset["benchmark"],
        "split": args.split,
        "progress": progress,
        "episodes": len(records),
        "success_count": int(np.sum(~final_failure)),
        "failure_count": int(np.sum(final_failure)),
        "baseline_failure_count": int(np.sum(baseline_failure)),
        "planner_invocation_count": int(np.sum(planner_mask)),
        "planner_recovery_count": int(np.sum(planner_mask & ~final_failure)),
        "coordinate_definition": {
            "order": [
                "cart_position",
                "absolute_link_angles",
                "cart_velocity",
                "relative_hinge_velocities",
            ],
            "labels": coordinate_labels,
            "scales": {
                "cart_position": scales.cart_position,
                "absolute_angle": scales.absolute_angle,
                "cart_velocity": scales.cart_velocity,
                "hinge_velocity": scales.hinge_velocity,
            },
            "transform": transform.astype(float).tolist(),
        },
        "linear_model": {
            "fd_eps": float(args.fd_eps),
            "lqr_control_cost": float(args.lqr_control_cost),
            "lqr_scale": float(args.lqr_scale),
            "state_matrix_sha256": data_sha256(state_matrix.astype(float).tolist()),
            "input_matrix_sha256": data_sha256(input_matrix.astype(float).tolist()),
            "gain": gain.astype(float).tolist(),
            "open_loop_spectral_radius": float(np.max(np.abs(open_modes.eigenvalues))),
            "closed_loop_spectral_radius": spectral_radius,
            "open_loop_eigenvector_condition": float(open_modes.eigenvector_condition),
            "closed_loop_eigenvector_condition": float(closed_modes.eigenvector_condition),
            "modal_coordinate_method": "orthonormal_real_schur_blocks",
            "open_loop_eigenvalues": [complex_payload(complex(value)) for value in open_modes.eigenvalues],
            "closed_loop_eigenvalues": [complex_payload(complex(value)) for value in closed_modes.eigenvalues],
        },
        "aggregate_diagnostics": {
            "dimensionless_state_norm": class_summary(state_norms, final_failure),
            "dimensionless_lyapunov_value": class_summary(lyapunov_values, final_failure),
            "closed_loop_mode_groups_ranked_by_planner_share_separation": group_summaries,
        },
        "records": records,
        "evidence": {
            "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(cfg)},
            "overrides": list(args.override),
            "spec": file_metadata(args.spec),
            "dataset": file_metadata(dataset_path),
            "evaluation": file_metadata(evaluation_path),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"states={len(records)} failures={payload['failure_count']} planner_recoveries={payload['planner_recovery_count']} "
        f"closed_loop_radius={spectral_radius:.9f}"
    )
    for summary in group_summaries[:5]:
        eigenvalue = summary["eigenvalue"]
        diagnostics = summary["share_planner_outcomes"]
        print(
            f"mode={summary['group_index']:02d} eig={eigenvalue['real']:+.6f}{eigenvalue['imag']:+.6f}j "
            f"planner_share_auc={diagnostics['failure_roc_auc']:.4f} "
            f"shift={diagnostics['standardized_failure_mean_shift']:+.3f} "
            f"coupling={summary['input_coupling']:.3e}"
        )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
