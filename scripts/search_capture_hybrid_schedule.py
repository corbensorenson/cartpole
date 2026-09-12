#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_capture_recovery import recovery_residual
    from scripts.search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from scripts.search_capture_target_schedule import scheduled_cart_target
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from search_capture_recovery import recovery_residual
    from search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from search_capture_target_schedule import scheduled_cart_target
    from search_swingup_capture import lqr_action, lqr_gain


def dimensionless_state(env: NLinkCartPoleEnv, transform: np.ndarray) -> np.ndarray:
    qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    qpos[1:] = wrap_angle(qpos[1:])
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    return transform @ np.r_[qpos, qvel]


def lyapunov_value(env: NLinkCartPoleEnv, transform: np.ndarray, lyapunov: np.ndarray) -> float:
    state = dimensionless_state(env, transform)
    return float(max(0.0, state @ lyapunov @ state))


def evaluate_hybrid_schedule(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    gain: np.ndarray,
    transform: np.ndarray,
    lyapunov: np.ndarray,
    target_knots: np.ndarray,
    target_seconds: float,
    residual_knots: np.ndarray,
    residual_seconds: float,
    fade_fraction: float,
    lqr_scale: float,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    env.reset(seed=seed)
    shaping_seconds = max(float(target_seconds), float(residual_seconds))
    initial_lyapunov = lyapunov_value(env, transform, lyapunov)
    minimum_lyapunov = initial_lyapunov
    last_lyapunov = initial_lyapunov
    shaping_end_lyapunov: float | None = None
    episode_return = 0.0
    residual_energy = 0.0
    saturation_steps = 0
    trajectory: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    length = 0
    try:
        while not (terminated or truncated):
            t = float(env.step_count * env.dt)
            cart_target = scheduled_cart_target(t, target_knots, target_seconds)
            baseline = lqr_action(env, gain, scale=lqr_scale, cart_target=cart_target)
            residual = recovery_residual(
                t,
                residual_knots,
                recovery_seconds=residual_seconds,
                fade_fraction=fade_fraction,
            )
            unclipped_action = baseline + residual
            action = float(np.clip(unclipped_action, -1.0, 1.0))
            saturation_steps += int(abs(unclipped_action) > 1.0)
            _, reward, terminated, truncated, info = env.step([action])
            length = int(env.step_count)
            episode_return += float(reward)
            residual_energy += residual * residual
            value = lyapunov_value(env, transform, lyapunov)
            last_lyapunov = value
            minimum_lyapunov = min(minimum_lyapunov, value)
            if shaping_end_lyapunov is None and env.step_count * env.dt >= shaping_seconds:
                shaping_end_lyapunov = value
            if record_trajectory:
                row = row_from_env(env, step=env.step_count, action=action, reward=reward, info=info)
                row.update(
                    {
                        "cart_target": float(cart_target),
                        "lqr_action": float(baseline),
                        "recovery_residual": float(residual),
                        "dimensionless_lyapunov_value": value,
                    }
                )
                trajectory.append(row)
            final_info = dict(info)
    finally:
        env.close()

    if shaping_end_lyapunov is None:
        shaping_end_lyapunov = last_lyapunov
    success = bool(final_info.get("success", False))
    hold = float(final_info.get("max_upright_streak_seconds", 0.0))
    low_momentum_hold = float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0))
    max_cart = float(final_info.get("max_cart_excursion", 0.0))
    rail_hit = final_info.get("termination_reason") == "rail_violation"
    saturation_fraction = float(saturation_steps / max(1, length))
    score_terms = {
        "success": -10_000_000.0 * float(success),
        "upright_hold": -20_000.0 * hold,
        "low_momentum_hold": -2_000.0 * low_momentum_hold,
        "survival": -20.0 * length,
        "return": -0.2 * episode_return,
        "rail_margin": 1_000.0 * max(0.0, max_cart - 2.5) ** 2,
        "rail_hit": 2_500.0 * float(rail_hit),
        "minimum_lyapunov": 1_000.0 * float(np.log1p(minimum_lyapunov)),
        "shaping_end_lyapunov": 250.0 * float(np.log1p(shaping_end_lyapunov)),
        "residual_energy": 2.0 * residual_energy,
        "saturation": 1_000.0 * saturation_fraction,
    }
    score = float(sum(score_terms.values()))
    return {
        "score": score,
        "score_terms": score_terms,
        "success": success,
        "return": float(episode_return),
        "length": length,
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "time_to_capture": final_info.get("time_to_capture"),
        "capture_start_time": final_info.get("capture_start_time"),
        "max_upright_streak_seconds": hold,
        "final_upright_streak_seconds": float(final_info.get("upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": low_momentum_hold,
        "max_cart_excursion": max_cart,
        "rail_hit": bool(rail_hit),
        "termination_reason": final_info.get("termination_reason"),
        "initial_lyapunov": float(initial_lyapunov),
        "minimum_lyapunov": float(minimum_lyapunov),
        "shaping_end_lyapunov": float(shaping_end_lyapunov),
        "residual_energy": float(residual_energy),
        "saturation_fraction": saturation_fraction,
        "final_info": final_info,
        "trajectory": trajectory if record_trajectory else None,
    }


def vector_to_controller(
    vector: np.ndarray,
    *,
    target_count: int,
    residual_count: int,
    target_limit: float,
    residual_limit: float,
) -> dict[str, Any]:
    return {
        "target_knots": np.clip(vector[:target_count], -target_limit, target_limit).astype(float).tolist(),
        "residual_knots": np.clip(
            vector[target_count : target_count + residual_count],
            -residual_limit,
            residual_limit,
        )
        .astype(float)
        .tolist(),
        "lqr_scale": float(np.clip(vector[target_count + residual_count], 0.6, 2.2)),
    }


def controller_to_vector(controller: dict[str, Any]) -> np.ndarray:
    return np.r_[
        np.asarray(controller["target_knots"], dtype=np.float64),
        np.asarray(controller["residual_knots"], dtype=np.float64),
        float(controller["lqr_scale"]),
    ]


def search_hybrid_schedule(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    gain: np.ndarray,
    transform: np.ndarray,
    lyapunov: np.ndarray,
    initial_controller: dict[str, Any],
    iterations: int,
    population: int,
    elites: int,
    target_count: int,
    target_seconds: float,
    target_limit: float,
    target_sigma: float,
    residual_count: int,
    residual_seconds: float,
    residual_limit: float,
    residual_sigma: float,
    fade_fraction: float,
    scale_sigma: float,
    sigma_decay: float,
    target_sigma_floor: float,
    residual_sigma_floor: float,
    scale_sigma_floor: float,
    success_polish_iterations: int,
    verbose: bool = True,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    center = controller_to_vector(initial_controller)
    if center.shape != (target_count + residual_count + 1,):
        raise ValueError("initial controller dimensions do not match hybrid search")
    sigma = np.r_[
        np.full(target_count, target_sigma, dtype=np.float64),
        np.full(residual_count, residual_sigma, dtype=np.float64),
        float(scale_sigma),
    ]
    sigma_floor = np.r_[
        np.full(target_count, target_sigma_floor, dtype=np.float64),
        np.full(residual_count, residual_sigma_floor, dtype=np.float64),
        float(scale_sigma_floor),
    ]
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    first_success_iteration: int | None = None

    for iteration in range(max(0, iterations) + 1):
        candidates = [center.copy() if best is None else controller_to_vector(best["controller"])]
        if iteration > 0:
            candidates.extend(center + rng.normal(0.0, sigma) for _ in range(population - 1))
        records = []
        for candidate in candidates:
            controller = vector_to_controller(
                candidate,
                target_count=target_count,
                residual_count=residual_count,
                target_limit=target_limit,
                residual_limit=residual_limit,
            )
            metrics = evaluate_hybrid_schedule(
                cfg,
                progress=progress,
                seed=seed,
                gain=gain,
                transform=transform,
                lyapunov=lyapunov,
                target_knots=np.asarray(controller["target_knots"], dtype=np.float64),
                target_seconds=target_seconds,
                residual_knots=np.asarray(controller["residual_knots"], dtype=np.float64),
                residual_seconds=residual_seconds,
                fade_fraction=fade_fraction,
                lqr_scale=float(controller["lqr_scale"]),
            )
            records.append({"score": float(metrics["score"]), "controller": controller, "metrics": metrics})
        records.sort(key=lambda row: float(row["score"]))
        if best is None or float(records[0]["score"]) < float(best["score"]):
            best = records[0]
        top = records[0]
        history.append(
            {
                "iteration": int(iteration),
                "score": float(top["score"]),
                "success": bool(top["metrics"]["success"]),
                "max_upright_streak_seconds": float(top["metrics"]["max_upright_streak_seconds"]),
                "minimum_lyapunov": float(top["metrics"]["minimum_lyapunov"]),
                "shaping_end_lyapunov": float(top["metrics"]["shaping_end_lyapunov"]),
                "max_cart_excursion": float(top["metrics"]["max_cart_excursion"]),
                "saturation_fraction": float(top["metrics"]["saturation_fraction"]),
            }
        )
        if verbose:
            print(
                f"iter={iteration:03d} score={top['score']:.3f} success={top['metrics']['success']} "
                f"hold={top['metrics']['max_upright_streak_seconds']:.3f}s "
                f"min_v={top['metrics']['minimum_lyapunov']:.2f} "
                f"end_v={top['metrics']['shaping_end_lyapunov']:.2f} "
                f"cart={top['metrics']['max_cart_excursion']:.3f}"
            )
        if bool(best["metrics"]["success"]) and first_success_iteration is None:
            first_success_iteration = iteration
        if first_success_iteration is not None and iteration >= first_success_iteration + success_polish_iterations:
            break
        if iteration > 0:
            elite = np.asarray(
                [controller_to_vector(row["controller"]) for row in records[:elites]],
                dtype=np.float64,
            )
            center = elite.mean(axis=0)
            sigma = np.maximum(elite.std(axis=0) * sigma_decay, sigma_floor)

    assert best is not None
    return {
        "best": best,
        "history": history,
        "first_success_iteration": first_success_iteration,
    }


def resample_target_controller(
    controller: dict[str, Any],
    *,
    old_seconds: float,
    target_count: int,
    target_seconds: float,
    residual_count: int,
) -> dict[str, Any]:
    old_knots = np.asarray(controller["target_knots"], dtype=np.float64)
    new_times = np.linspace(0.0, target_seconds, target_count, dtype=np.float64)
    new_targets = [scheduled_cart_target(float(t), old_knots, old_seconds) for t in new_times]
    return {
        "target_knots": new_targets,
        "residual_knots": np.zeros(residual_count, dtype=np.float64).tolist(),
        "lqr_scale": float(controller["lqr_scale"]),
    }


def load_initial_controller(
    evaluation_path: str | Path | None,
    state_index: int,
    *,
    target_count: int,
    target_seconds: float,
    residual_count: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if evaluation_path is None:
        return {
            "target_knots": np.zeros(target_count, dtype=np.float64).tolist(),
            "residual_knots": np.zeros(residual_count, dtype=np.float64).tolist(),
            "lqr_scale": 1.30,
        }, None
    payload = json.loads(Path(evaluation_path).read_text(encoding="utf-8"))
    matches = [row for row in payload.get("episode_results", []) if int(row["state_index"]) == state_index]
    if len(matches) != 1:
        raise ValueError(f"initial evaluation must contain exactly one row for state {state_index}")
    planner = payload.get("controller", {}).get("planner") or payload.get("controller", {}).get("refinement_planner")
    if not isinstance(planner, dict):
        raise ValueError("initial evaluation does not declare planner schedule seconds")
    return (
        resample_target_controller(
            matches[0]["selected_controller"],
            old_seconds=float(planner["schedule_seconds"]),
            target_count=target_count,
            target_seconds=target_seconds,
            residual_count=residual_count,
        ),
        payload,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Search transient cart-target and residual schedules before LQR capture")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--state-json", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--initial-evaluation", default=None)
    parser.add_argument("--progress", type=float, default=0.065)
    parser.add_argument("--seed", type=int, default=63001)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--population", type=int, default=256)
    parser.add_argument("--elites", type=int, default=20)
    parser.add_argument("--target-count", type=int, default=8)
    parser.add_argument("--target-seconds", type=float, default=3.0)
    parser.add_argument("--target-limit", type=float, default=3.0)
    parser.add_argument("--target-sigma", type=float, default=0.50)
    parser.add_argument("--residual-count", type=int, default=12)
    parser.add_argument("--residual-seconds", type=float, default=3.0)
    parser.add_argument("--residual-limit", type=float, default=0.50)
    parser.add_argument("--residual-sigma", type=float, default=0.15)
    parser.add_argument("--fade-fraction", type=float, default=0.25)
    parser.add_argument("--scale-sigma", type=float, default=0.10)
    parser.add_argument("--success-polish-iterations", type=int, default=2)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.population < 2 or not 1 <= args.elites <= args.population:
        raise ValueError("--population must be >= 2 and --elites must be in 1..population")
    if min(args.target_count, args.residual_count) < 2:
        raise ValueError("target and residual knot counts must be at least two")
    if min(args.target_seconds, args.residual_seconds, args.target_limit, args.residual_limit) <= 0.0:
        raise ValueError("schedule durations and limits must be positive")
    if not 0.0 < args.fade_fraction <= 1.0:
        raise ValueError("--fade-fraction must be in (0, 1]")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    selected_state, selected_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"]))
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
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
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix,
        input_matrix,
        gain,
        transform,
        feedback_scale=1.30,
    )
    initial_controller, initial_evaluation = load_initial_controller(
        args.initial_evaluation,
        selected_index,
        target_count=args.target_count,
        target_seconds=args.target_seconds,
        residual_count=args.residual_count,
    )
    search = search_hybrid_schedule(
        cfg,
        progress=args.progress,
        seed=args.seed,
        gain=gain,
        transform=transform,
        lyapunov=lyapunov,
        initial_controller=initial_controller,
        iterations=args.iterations,
        population=args.population,
        elites=args.elites,
        target_count=args.target_count,
        target_seconds=args.target_seconds,
        target_limit=args.target_limit,
        target_sigma=args.target_sigma,
        residual_count=args.residual_count,
        residual_seconds=args.residual_seconds,
        residual_limit=args.residual_limit,
        residual_sigma=args.residual_sigma,
        fade_fraction=args.fade_fraction,
        scale_sigma=args.scale_sigma,
        sigma_decay=0.90,
        target_sigma_floor=0.03,
        residual_sigma_floor=0.01,
        scale_sigma_floor=0.01,
        success_polish_iterations=args.success_polish_iterations,
    )
    best = search["best"]
    verified = evaluate_hybrid_schedule(
        cfg,
        progress=args.progress,
        seed=args.seed,
        gain=gain,
        transform=transform,
        lyapunov=lyapunov,
        target_knots=np.asarray(best["controller"]["target_knots"], dtype=np.float64),
        target_seconds=args.target_seconds,
        residual_knots=np.asarray(best["controller"]["residual_knots"], dtype=np.float64),
        residual_seconds=args.residual_seconds,
        fade_fraction=args.fade_fraction,
        lqr_scale=float(best["controller"]["lqr_scale"]),
        record_trajectory=True,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "State-specific hybrid target/residual recovery teacher; not P1 gate evidence.",
        "config_path": str(Path(args.config)),
        "config_sha256": data_sha256(cfg),
        "state_source": file_metadata(args.state_json),
        "state_index": selected_index,
        "selected_state": selected_state,
        "progress": float(args.progress),
        "seed": int(args.seed),
        "search": {
            "iterations_requested": int(args.iterations),
            "iterations_completed": int(len(search["history"]) - 1),
            "population": int(args.population),
            "elites": int(args.elites),
            "target_count": int(args.target_count),
            "target_seconds": float(args.target_seconds),
            "target_limit": float(args.target_limit),
            "residual_count": int(args.residual_count),
            "residual_seconds": float(args.residual_seconds),
            "residual_limit": float(args.residual_limit),
            "fade_fraction": float(args.fade_fraction),
            "first_success_iteration": search["first_success_iteration"],
        },
        "lyapunov": {
            "coordinate_source": str(Path(args.spec)),
            "nominal_lqr_scale": 1.30,
            "closed_loop_spectral_radius": spectral_radius,
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "initial_controller": initial_controller,
        "initial_evaluation": (
            file_metadata(args.initial_evaluation) if initial_evaluation is not None else None
        ),
        "best": {"score": float(best["score"]), "controller": best["controller"]},
        "verified_rollout": verified,
        "history": search["history"],
        "evidence": {
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
