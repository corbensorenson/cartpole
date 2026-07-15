#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.modal import (
    StateScales,
    dimensionless_absolute_transform,
    real_schur_decomposition,
    scale_feedback_by_schur_group,
    transform_dynamics,
    transform_feedback_gain,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_capture_target_schedule import evaluate_target_schedule
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_capture_sequence import fixed_state_cfg
    from search_capture_target_schedule import evaluate_target_schedule
    from search_swingup_capture import lqr_gain


def parse_multipliers(value: str) -> list[float]:
    result = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not result or any(not np.isfinite(item) or item < 0.0 for item in result):
        raise ValueError("multipliers must be a comma-separated list of finite nonnegative values")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep one closed-loop Schur feedback group on capture failures")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument(
        "--evaluation",
        default="runs/p1_capture_target_teachers/eval_adaptive_validation256_p0065.json",
    )
    parser.add_argument(
        "--analysis",
        default="runs/p1_capture_modal/modal_analysis_validation256_p0065.json",
    )
    parser.add_argument("--mode-group", type=int, default=None)
    parser.add_argument("--multipliers", default="0.25,0.5,0.75,1.0,1.25,1.5,2.0,3.0")
    parser.add_argument("--selection", choices=("baseline_failures", "all"), default="baseline_failures")
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--out", default="runs/p1_capture_modal/modal_group10_gain_sweep_p0065.json")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    dataset_path = Path(args.dataset)
    evaluation_path = Path(args.evaluation)
    analysis_path = Path(args.analysis)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if evaluation.get("evidence", {}).get("config", {}).get("resolved_sha256") != data_sha256(cfg):
        errors.append("evaluation config hash does not match current config")
    if evaluation.get("evidence", {}).get("dataset", {}).get("sha256") != file_metadata(dataset_path).get("sha256"):
        errors.append("evaluation dataset hash does not match current dataset")
    if analysis.get("evidence", {}).get("evaluation", {}).get("sha256") != file_metadata(evaluation_path).get("sha256"):
        errors.append("modal analysis does not reference the current evaluation artifact")
    if analysis.get("evidence", {}).get("dataset", {}).get("sha256") != file_metadata(dataset_path).get("sha256"):
        errors.append("modal analysis does not reference the current dataset")
    rows = evaluation.get("episode_results")
    if not isinstance(rows, list) or not rows:
        errors.append("evaluation must contain episode_results")
    if errors:
        raise ValueError("Modal feedback sweep validation failed:\n- " + "\n- ".join(errors))

    coordinate_scales = analysis["coordinate_definition"]["scales"]
    scales = StateScales(
        cart_position=float(coordinate_scales["cart_position"]),
        absolute_angle=float(coordinate_scales["absolute_angle"]),
        cart_velocity=float(coordinate_scales["cart_velocity"]),
        hinge_velocity=float(coordinate_scales["hinge_velocity"]),
    )
    transform = dimensionless_absolute_transform(int(cfg["env"]["n_links"]), scales)
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, args.fd_eps)
    dimensionless_a, dimensionless_b = transform_dynamics(state_matrix, input_matrix, transform)
    gain = lqr_gain(
        cfg,
        progress=1.0,
        fd_eps=args.fd_eps,
        control_cost=args.lqr_control_cost,
    )
    dimensionless_gain = float(args.lqr_scale) * transform_feedback_gain(gain, transform)
    base_closed_loop = dimensionless_a - dimensionless_b @ dimensionless_gain.reshape(1, -1)
    decomposition = real_schur_decomposition(base_closed_loop, dimensionless_b)
    mode_group = args.mode_group
    if mode_group is None:
        ranked = analysis["aggregate_diagnostics"][
            "closed_loop_mode_groups_ranked_by_planner_share_separation"
        ]
        if not ranked:
            raise ValueError("modal analysis does not contain a ranked Schur group")
        target_payload = ranked[0]["eigenvalue"]
        target_eigenvalue = complex(float(target_payload["real"]), float(target_payload["imag"]))
        mode_group = min(
            range(len(decomposition.groups)),
            key=lambda index: min(
                abs(complex(value) - target_eigenvalue)
                for value in decomposition.group_eigenvalues[index]
            ),
        )
    if mode_group < 0 or mode_group >= len(decomposition.groups):
        raise IndexError(f"mode group {mode_group} is outside 0..{len(decomposition.groups) - 1}")

    selected_rows = [row for row in rows if args.selection == "all" or not bool(row["baseline"]["success"])]
    multipliers = parse_multipliers(args.multipliers)
    variants: list[dict[str, Any]] = []
    for multiplier in multipliers:
        modified_dimensionless_gain = scale_feedback_by_schur_group(
            dimensionless_gain,
            decomposition,
            {mode_group: multiplier},
        )
        modified_physical_gain = modified_dimensionless_gain @ transform
        closed_loop = state_matrix - input_matrix @ modified_physical_gain.reshape(1, -1)
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(closed_loop))))
        episode_results = []
        for ordinal, row in enumerate(selected_rows):
            state_index = int(row["state_index"])
            fixed_cfg = fixed_state_cfg(cfg, dataset["states"][state_index], float(cfg["env"]["episode_seconds"]))
            metrics = evaluate_target_schedule(
                fixed_cfg,
                progress=float(evaluation["progress"]),
                seed=int(row["seed"]),
                gain=modified_physical_gain,
                target_knots=np.zeros(2, dtype=np.float64),
                schedule_seconds=0.02,
                lqr_scale=1.0,
            )
            episode_results.append(
                {
                    "ordinal": ordinal,
                    "state_index": state_index,
                    "state_id": row.get("state_id"),
                    "seed": int(row["seed"]),
                    "success": bool(metrics["success"]),
                    "max_upright_streak_seconds": float(metrics["max_upright_streak_seconds"]),
                    "max_cart_excursion": float(metrics["max_cart_excursion"]),
                    "rail_hit": bool(metrics["rail_hit"]),
                    "termination_reason": metrics["termination_reason"],
                }
            )
        successes = np.asarray([row["success"] for row in episode_results], dtype=np.float64)
        holds = np.asarray([row["max_upright_streak_seconds"] for row in episode_results], dtype=np.float64)
        variant = {
            "mode_group_multiplier": multiplier,
            "linear_closed_loop_spectral_radius": spectral_radius,
            "success_count": int(np.sum(successes)),
            "success_rate": float(np.mean(successes)),
            "max_upright_streak_median": float(np.median(holds)),
            "max_upright_streak_mean": float(np.mean(holds)),
            "rail_hit_count": int(sum(bool(row["rail_hit"]) for row in episode_results)),
            "physical_gain": modified_physical_gain.astype(float).tolist(),
            "episode_results": episode_results,
        }
        variants.append(variant)
        print(
            f"multiplier={multiplier:.3f} success={variant['success_count']}/{len(episode_results)} "
            f"median_hold={variant['max_upright_streak_median']:.3f}s "
            f"rail_hits={variant['rail_hit_count']} radius={spectral_radius:.9f}"
        )

    best = max(
        variants,
        key=lambda variant: (
            variant["success_count"],
            variant["max_upright_streak_median"],
            -variant["rail_hit_count"],
        ),
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Targeted closed-loop Schur feedback ablation on a partial P1 validation frontier.",
        "benchmark": dataset.get("benchmark"),
        "split": evaluation.get("split"),
        "progress": float(evaluation["progress"]),
        "selection": args.selection,
        "selected_episode_count": len(selected_rows),
        "mode_group": int(mode_group),
        "mode_group_components": list(decomposition.groups[mode_group]),
        "mode_group_eigenvalues": [
            {"real": float(np.real(value)), "imag": float(np.imag(value)), "magnitude": float(abs(value))}
            for value in decomposition.group_eigenvalues[mode_group]
        ],
        "multipliers": multipliers,
        "best_multiplier": float(best["mode_group_multiplier"]),
        "variants": variants,
        "evidence": {
            "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(cfg)},
            "overrides": list(args.override),
            "dataset": file_metadata(dataset_path),
            "evaluation": file_metadata(evaluation_path),
            "analysis": file_metadata(analysis_path),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(f"best_multiplier={payload['best_multiplier']} Wrote {args.out}")


if __name__ == "__main__":
    main()
