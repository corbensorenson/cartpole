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


def recovery_residual(
    t: float,
    knots: np.ndarray,
    *,
    recovery_seconds: float,
    fade_fraction: float,
) -> float:
    if t >= recovery_seconds:
        return 0.0
    knot_times = np.linspace(0.0, recovery_seconds, len(knots), dtype=np.float64)
    residual = float(np.interp(t, knot_times, knots))
    fade_start = recovery_seconds * (1.0 - fade_fraction)
    if t > fade_start:
        residual *= max(0.0, (recovery_seconds - t) / max(1e-9, recovery_seconds - fade_start))
    return float(residual)


def evaluate_recovery(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    gain: np.ndarray,
    lqr_scale: float,
    residual_knots: np.ndarray,
    recovery_seconds: float,
    fade_fraction: float,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    env.reset(seed=seed)
    episode_return = 0.0
    running_cost = 0.0
    max_cart_abs = abs(float(env.data.qpos[0]))
    residual_energy = 0.0
    trajectory: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False

    while not (terminated or truncated):
        t = float(env.step_count * env.dt)
        baseline = lqr_action(env, gain, scale=lqr_scale, cart_target=0.0)
        residual = recovery_residual(
            t,
            residual_knots,
            recovery_seconds=recovery_seconds,
            fade_fraction=fade_fraction,
        )
        action = float(np.clip(baseline + residual, -1.0, 1.0))
        _, reward, terminated, truncated, info = env.step([action])
        episode_return += float(reward)
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        residual_energy += residual * residual
        angle = float(info["max_abs_angle"])
        hinge_rms = float(info["hinge_velocity_rms"])
        running_cost += (
            12.0 * angle * angle
            + 0.08 * hinge_rms * hinge_rms
            + 2.0 * (float(info["x"]) / env.rail_limit) ** 2
            + 0.05 * (float(env.data.qvel[0]) / 0.5) ** 2
        )
        if record_trajectory:
            trajectory.append(row_from_env(env, step=env.step_count, action=action, reward=reward, info=info))
            trajectory[-1]["lqr_action"] = float(baseline)
            trajectory[-1]["recovery_residual"] = float(residual)
        final_info = dict(info)

    env.close()
    success = bool(final_info.get("success", False))
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    low_momentum_streak = float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0))
    rail_hit = final_info.get("termination_reason") == "rail_violation"
    score = (
        -250000.0 * float(success)
        -8000.0 * max_streak
        -1000.0 * low_momentum_streak
        + 0.02 * running_cost
        + 0.05 * residual_energy
        + 1000.0 * max_cart_abs * max_cart_abs
        + 25000.0 * float(rail_hit)
    )
    return {
        "score": float(score),
        "success": success,
        "return": float(episode_return),
        "length": int(final_info.get("step", env.step_count)),
        "max_upright_streak_seconds": max_streak,
        "max_low_momentum_upright_streak_seconds": low_momentum_streak,
        "max_cart_abs": float(max_cart_abs),
        "rail_hit": bool(rail_hit),
        "termination_reason": final_info.get("termination_reason"),
        "final_info": final_info,
        "trajectory": trajectory if record_trajectory else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search a short nonlinear residual recovery prefix before exact LQR stabilization"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--state-json", default="runs/p1_capture_envelope/train_hard_00625.json")
    parser.add_argument("--state-index", default="0")
    parser.add_argument("--progress", type=float, default=0.0625)
    parser.add_argument("--seed", type=int, default=61711)
    parser.add_argument("--out", required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--residual-count", type=int, default=16)
    parser.add_argument("--recovery-seconds", type=float, default=1.5)
    parser.add_argument("--fade-fraction", type=float, default=0.25)
    parser.add_argument("--residual-sigma", type=float, default=0.30)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--success-polish-iterations", type=int, default=3)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.population < 2 or not 1 <= args.elites <= args.population:
        raise ValueError("--population must be >= 2 and --elites must be in 1..population")
    if args.residual_count < 2 or args.recovery_seconds <= 0.0:
        raise ValueError("--residual-count must be >= 2 and --recovery-seconds must be positive")
    if not 0.0 < args.fade_fraction <= 1.0:
        raise ValueError("--fade-fraction must be in (0, 1]")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    selected_state, selected_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"]))
    gain = lqr_gain(
        cfg,
        progress=args.progress,
        fd_eps=1e-7,
        control_cost=args.lqr_control_cost,
    )
    rng = np.random.default_rng(args.seed)
    center = np.zeros(args.residual_count, dtype=np.float64)
    sigma = np.full(args.residual_count, float(args.residual_sigma), dtype=np.float64)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    first_success_iteration: int | None = None

    for iteration in range(max(0, args.iterations) + 1):
        candidates = [center.copy()]
        if best is not None:
            candidates[0] = np.asarray(best["residual_knots"], dtype=np.float64)
        if iteration > 0:
            candidates.extend(
                np.clip(center + rng.normal(0.0, sigma), -1.0, 1.0)
                for _ in range(args.population - 1)
            )
        records = []
        for candidate in candidates:
            metrics = evaluate_recovery(
                cfg,
                progress=args.progress,
                seed=args.seed,
                gain=gain,
                lqr_scale=args.lqr_scale,
                residual_knots=np.asarray(candidate, dtype=np.float64),
                recovery_seconds=args.recovery_seconds,
                fade_fraction=args.fade_fraction,
            )
            records.append(
                {
                    "score": float(metrics["score"]),
                    "residual_knots": np.asarray(candidate, dtype=float).tolist(),
                    "metrics": metrics,
                }
            )
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
                "max_cart_abs": float(top["metrics"]["max_cart_abs"]),
                "rail_hit": bool(top["metrics"]["rail_hit"]),
            }
        )
        print(
            f"iter={iteration:03d} score={top['score']:.3f} "
            f"success={top['metrics']['success']} "
            f"hold={top['metrics']['max_upright_streak_seconds']:.3f}s "
            f"cart={top['metrics']['max_cart_abs']:.3f}"
        )
        if bool(best["metrics"]["success"]) and first_success_iteration is None:
            first_success_iteration = iteration
        if first_success_iteration is not None and iteration >= first_success_iteration + args.success_polish_iterations:
            break
        if iteration > 0:
            elite = np.asarray([row["residual_knots"] for row in records[: args.elites]], dtype=np.float64)
            center = np.asarray(best["residual_knots"], dtype=np.float64)
            sigma = np.maximum(elite.std(axis=0) * args.sigma_decay, args.sigma_floor)

    assert best is not None
    verified = evaluate_recovery(
        cfg,
        progress=args.progress,
        seed=args.seed,
        gain=gain,
        lqr_scale=args.lqr_scale,
        residual_knots=np.asarray(best["residual_knots"], dtype=np.float64),
        recovery_seconds=args.recovery_seconds,
        fade_fraction=args.fade_fraction,
        record_trajectory=True,
    )
    payload = {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Training-state nonlinear recovery teacher; not held-out P1 gate evidence.",
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
            "residual_count": int(args.residual_count),
            "recovery_seconds": float(args.recovery_seconds),
            "fade_fraction": float(args.fade_fraction),
            "residual_sigma": float(args.residual_sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "first_success_iteration": first_success_iteration,
        },
        "lqr": {
            "scale": float(args.lqr_scale),
            "control_cost": float(args.lqr_control_cost),
            "gain": gain.astype(float).tolist(),
        },
        "best": {"score": float(best["score"]), "residual_knots": best["residual_knots"]},
        "verified_rollout": verified,
        "history": history,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
