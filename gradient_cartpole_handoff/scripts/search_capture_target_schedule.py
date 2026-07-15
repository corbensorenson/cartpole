#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv

try:
    from scripts.search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg, load_state, row_from_env
    from search_swingup_capture import lqr_action, lqr_gain


def scheduled_cart_target(t: float, knots: np.ndarray, schedule_seconds: float) -> float:
    if t >= schedule_seconds:
        return 0.0
    knot_times = np.linspace(0.0, schedule_seconds, len(knots), dtype=np.float64)
    return float(np.interp(t, knot_times, knots))


def evaluate_target_schedule(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    gain: np.ndarray,
    target_knots: np.ndarray,
    schedule_seconds: float,
    lqr_scale: float,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    env.reset(seed=seed)
    episode_return = 0.0
    trajectory: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    length = 0
    try:
        while not (terminated or truncated):
            t = float(env.step_count * env.dt)
            cart_target = scheduled_cart_target(t, target_knots, schedule_seconds)
            action = lqr_action(env, gain, scale=lqr_scale, cart_target=cart_target)
            _, reward, terminated, truncated, info = env.step([action])
            length = int(env.step_count)
            episode_return += float(reward)
            if record_trajectory:
                row = row_from_env(env, step=env.step_count, action=action, reward=reward, info=info)
                row["cart_target"] = cart_target
                row["lqr_scale"] = float(lqr_scale)
                trajectory.append(row)
            final_info = dict(info)
    finally:
        env.close()

    success = bool(final_info.get("success", False))
    hold = float(final_info.get("max_upright_streak_seconds", 0.0))
    low_momentum_hold = float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0))
    max_cart = float(final_info.get("max_cart_excursion", 0.0))
    rail_hit = final_info.get("termination_reason") == "rail_violation"
    score = (
        -10_000_000.0 * float(success)
        -20_000.0 * hold
        -2_000.0 * low_momentum_hold
        -20.0 * length
        -0.2 * episode_return
        +1_000.0 * max(0.0, max_cart - 2.5) ** 2
        +2_500.0 * float(rail_hit)
    )
    return {
        "score": float(score),
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
        "final_info": final_info,
        "trajectory": trajectory if record_trajectory else None,
    }


def vector_to_controller(vector: np.ndarray, knot_count: int, target_limit: float) -> dict[str, Any]:
    return {
        "target_knots": np.clip(vector[:knot_count], -target_limit, target_limit).astype(float).tolist(),
        "lqr_scale": float(np.clip(vector[knot_count], 0.6, 2.2)),
    }


def controller_to_vector(controller: dict[str, Any]) -> np.ndarray:
    return np.r_[
        np.asarray(controller["target_knots"], dtype=np.float64),
        float(controller["lqr_scale"]),
    ]


def search_target_schedule(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    gain: np.ndarray,
    iterations: int,
    population: int,
    elites: int,
    knot_count: int,
    schedule_seconds: float,
    target_limit: float,
    target_sigma: float,
    scale_sigma: float,
    sigma_decay: float,
    target_sigma_floor: float,
    scale_sigma_floor: float,
    initial_lqr_scale: float,
    success_polish_iterations: int,
    verbose: bool = True,
    initial_controller: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    if initial_controller is None:
        center = np.r_[np.zeros(knot_count, dtype=np.float64), float(initial_lqr_scale)]
    else:
        center = controller_to_vector(initial_controller)
        if center.shape != (knot_count + 1,):
            raise ValueError("initial controller knot count does not match search knot count")
    sigma = np.r_[np.full(knot_count, float(target_sigma), dtype=np.float64), float(scale_sigma)]
    sigma_floor = np.r_[
        np.full(knot_count, float(target_sigma_floor), dtype=np.float64),
        float(scale_sigma_floor),
    ]
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    first_success_iteration: int | None = None

    for iteration in range(max(0, iterations) + 1):
        candidates = [center.copy()]
        if best is not None:
            candidates[0] = controller_to_vector(best["controller"])
        if iteration > 0:
            candidates.extend(center + rng.normal(0.0, sigma) for _ in range(population - 1))
        records = []
        for candidate in candidates:
            controller = vector_to_controller(candidate, knot_count, target_limit)
            metrics = evaluate_target_schedule(
                cfg,
                progress=progress,
                seed=seed,
                gain=gain,
                target_knots=np.asarray(controller["target_knots"], dtype=np.float64),
                schedule_seconds=schedule_seconds,
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
                "length": int(top["metrics"]["length"]),
                "max_cart_excursion": float(top["metrics"]["max_cart_excursion"]),
                "lqr_scale": float(top["controller"]["lqr_scale"]),
            }
        )
        if verbose:
            print(
                f"iter={iteration:03d} score={top['score']:.3f} success={top['metrics']['success']} "
                f"hold={top['metrics']['max_upright_streak_seconds']:.3f}s "
                f"length={top['metrics']['length']} cart={top['metrics']['max_cart_excursion']:.3f}"
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
            center = controller_to_vector(best["controller"])
            sigma = np.maximum(elite.std(axis=0) * float(sigma_decay), sigma_floor)

    assert best is not None
    return {
        "best": best,
        "history": history,
        "first_success_iteration": first_success_iteration,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search a short cart-target schedule before exact LQR capture")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--state-json", default="runs/p1_capture_envelope/train.json")
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--progress", type=float, default=0.0625)
    parser.add_argument("--seed", type=int, default=62001)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=12)
    parser.add_argument("--knot-count", type=int, default=8)
    parser.add_argument("--schedule-seconds", type=float, default=2.0)
    parser.add_argument("--target-limit", type=float, default=3.0)
    parser.add_argument("--target-sigma", type=float, default=1.0)
    parser.add_argument("--scale-sigma", type=float, default=0.25)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--target-sigma-floor", type=float, default=0.03)
    parser.add_argument("--scale-sigma-floor", type=float, default=0.01)
    parser.add_argument("--initial-lqr-scale", type=float, default=1.30)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--success-polish-iterations", type=int, default=2)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.population < 2 or not 1 <= args.elites <= args.population:
        raise ValueError("--population must be >= 2 and --elites must be in 1..population")
    if args.knot_count < 2 or args.schedule_seconds <= 0.0 or args.target_limit <= 0.0:
        raise ValueError("target schedule dimensions and limits must be positive")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    selected_state, selected_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"]))
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=args.lqr_control_cost)
    search = search_target_schedule(
        cfg,
        progress=args.progress,
        seed=args.seed,
        gain=gain,
        iterations=args.iterations,
        population=args.population,
        elites=args.elites,
        knot_count=args.knot_count,
        schedule_seconds=args.schedule_seconds,
        target_limit=args.target_limit,
        target_sigma=args.target_sigma,
        scale_sigma=args.scale_sigma,
        sigma_decay=args.sigma_decay,
        target_sigma_floor=args.target_sigma_floor,
        scale_sigma_floor=args.scale_sigma_floor,
        initial_lqr_scale=args.initial_lqr_scale,
        success_polish_iterations=args.success_polish_iterations,
    )
    best = search["best"]
    history = search["history"]
    first_success_iteration = search["first_success_iteration"]
    verified = evaluate_target_schedule(
        cfg,
        progress=args.progress,
        seed=args.seed,
        gain=gain,
        target_knots=np.asarray(best["controller"]["target_knots"], dtype=np.float64),
        schedule_seconds=args.schedule_seconds,
        lqr_scale=float(best["controller"]["lqr_scale"]),
        record_trajectory=True,
    )
    payload = {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "State-specific scheduled-target recovery teacher; not held-out P1 gate evidence.",
        "config_path": str(Path(args.config)),
        "config_sha256": data_sha256(cfg),
        "state_source": file_metadata(args.state_json),
        "state_index": selected_index,
        "selected_state": selected_state,
        "progress": float(args.progress),
        "seed": int(args.seed),
        "search": {
            "iterations_requested": int(args.iterations),
            "iterations_completed": int(len(history) - 1),
            "population": int(args.population),
            "elites": int(args.elites),
            "knot_count": int(args.knot_count),
            "schedule_seconds": float(args.schedule_seconds),
            "target_limit": float(args.target_limit),
            "first_success_iteration": first_success_iteration,
        },
        "lqr": {
            "control_cost": float(args.lqr_control_cost),
            "gain": gain.astype(float).tolist(),
        },
        "best": {"score": float(best["score"]), "controller": best["controller"]},
        "verified_rollout": verified,
        "history": history,
        "evidence": {
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
