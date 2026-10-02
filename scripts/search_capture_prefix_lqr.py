#!/usr/bin/env python
"""Search an uninterrupted capture prefix followed by exact upright LQR.

The prefix is allowed to cool the chain and recenter the cart.  Candidates are
ranked by the full nonlinear rollout after the switch, so a brief upright
crossing cannot win over a state that actually survives the stabilizer tail.
This is a development controller search; its output is not final benchmark
evidence until an independent held-out evaluator accepts the resulting expert.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp

try:
    from scripts.search_capture_sequence import fixed_state_cfg, load_state
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg, load_state
    from search_swingup_capture import lqr_action, lqr_gain


def evaluate_prefix_lqr(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    prefix_seconds: float,
    total_seconds: float,
    action_knots: np.ndarray,
    lqr_gain_vector: np.ndarray,
    lqr_scale: float,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    env.reset(seed=seed)
    prefix_seconds = float(prefix_seconds)
    total_seconds = float(total_seconds)
    action_knots = np.asarray(action_knots, dtype=np.float64)
    action_times = np.linspace(0.0, prefix_seconds, action_knots.size, dtype=np.float64)
    steps = min(env.max_steps, int(total_seconds / env.dt))
    rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    best_angle = float("inf")
    best_row: dict[str, Any] | None = None
    prefix_steps = min(steps, int(prefix_seconds / env.dt))
    try:
        for step in range(steps):
            t = step * env.dt
            if step < prefix_steps:
                action = float(np.interp(t, action_times, action_knots))
            else:
                action = lqr_action(env, lqr_gain_vector, scale=lqr_scale, cart_target=0.0)
            action = float(np.clip(action, -1.0, 1.0))
            _, reward, terminated, truncated, info = env.step([action])
            final_info = dict(info)
            row = {
                "step": int(step + 1),
                "time_seconds": float((step + 1) * env.dt),
                "action": action,
                "reward": float(reward),
                "x": float(env.data.qpos[0]),
                "cart_velocity": float(env.data.qvel[0]),
                "max_abs_angle": float(info.get("max_abs_angle", np.inf)),
                "hinge_velocity_rms": float(info.get("hinge_velocity_rms", np.inf)),
                "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
                "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
            }
            if best_row is None or row["max_abs_angle"] < best_angle:
                best_angle = row["max_abs_angle"]
                best_row = row
            if record_trajectory:
                rows.append(row)
            if terminated or truncated:
                break
    finally:
        env.close()

    hold = float(final_info.get("max_upright_streak_seconds", 0.0))
    success = bool(final_info.get("success", False))
    rail_hit = final_info.get("termination_reason") == "rail_violation"
    max_cart = float(final_info.get("max_cart_excursion", 0.0))
    terminal_angle = float(final_info.get("max_abs_angle", np.inf))
    terminal_rate = float(final_info.get("hinge_velocity_rms", np.inf))
    score = (
        -20_000_000.0 * float(success)
        -50_000.0 * hold
        -5_000.0 * float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0))
        +250.0 * terminal_angle
        +20.0 * terminal_rate
        +10.0 * abs(float(final_info.get("x", 0.0)))
        +5.0 * abs(float(final_info.get("cart_velocity", 0.0) or 0.0))
        +2_000.0 * float(rail_hit)
        +1_000.0 * max(0.0, max_cart - 2.8) ** 2
    )
    return {
        "score": float(score),
        "success": success,
        "termination_reason": final_info.get("termination_reason"),
        "max_upright_streak_seconds": hold,
        "max_low_momentum_upright_streak_seconds": float(
            final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": max_cart,
        "terminal_max_abs_angle": terminal_angle,
        "terminal_hinge_velocity_rms": terminal_rate,
        "best_pass": best_row,
        "final_info": final_info,
        "trajectory": rows if record_trajectory else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--state-index", default="0")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--prefix-seconds", type=float, default=3.0)
    parser.add_argument("--total-seconds", type=float, default=15.0)
    parser.add_argument("--action-count", type=int, default=31)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--population", type=int, default=96)
    parser.add_argument("--elites", type=int, default=12)
    parser.add_argument("--action-sigma", type=float, default=0.45)
    parser.add_argument("--sigma-decay", type=float, default=0.88)
    parser.add_argument("--sigma-floor", type=float, default=0.015)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--lqr-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=62901)
    parser.add_argument("--out", required=True)
    parser.add_argument("--record-best", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.action_count < 2 or args.population < 2 or not 1 <= args.elites <= args.population:
        raise ValueError("invalid action search dimensions")
    if args.prefix_seconds <= 0.0 or args.total_seconds <= args.prefix_seconds:
        raise ValueError("total seconds must exceed prefix seconds")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    state, state_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(base_cfg, state, args.total_seconds)
    gain = lqr_gain(cfg, progress=args.progress, fd_eps=1e-7, control_cost=args.lqr_control_cost)
    rng = np.random.default_rng(args.seed)
    center = np.zeros(args.action_count, dtype=np.float64)
    sigma = np.full(args.action_count, args.action_sigma, dtype=np.float64)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for iteration in range(max(1, args.iterations) + 1):
        candidates = [center.copy()]
        if iteration > 0:
            candidates.extend(
                np.clip(center + rng.normal(0.0, sigma), -1.0, 1.0)
                for _ in range(args.population - 1)
            )
        records = []
        for candidate in candidates:
            metrics = evaluate_prefix_lqr(
                cfg,
                progress=args.progress,
                seed=args.seed,
                prefix_seconds=args.prefix_seconds,
                total_seconds=args.total_seconds,
                action_knots=candidate,
                lqr_gain_vector=gain,
                lqr_scale=args.lqr_scale,
            )
            records.append({"score": float(metrics["score"]), "action_knots": candidate.astype(float).tolist(), "metrics": metrics})
        records.sort(key=lambda row: row["score"])
        if best is None or records[0]["score"] < best["score"]:
            best = records[0]
        top = records[0]
        tm = top["metrics"]
        history.append({
            "iteration": iteration,
            "score": top["score"],
            "success": tm["success"],
            "max_upright_streak_seconds": tm["max_upright_streak_seconds"],
            "max_low_momentum_upright_streak_seconds": tm["max_low_momentum_upright_streak_seconds"],
            "max_cart_excursion": tm["max_cart_excursion"],
            "terminal_max_abs_angle": tm["terminal_max_abs_angle"],
            "terminal_hinge_velocity_rms": tm["terminal_hinge_velocity_rms"],
        })
        print(
            f"iter={iteration:03d} score={top['score']:.3f} success={tm['success']} "
            f"hold={tm['max_upright_streak_seconds']:.3f}s cart={tm['max_cart_excursion']:.3f} "
            f"terminal_angle={tm['terminal_max_abs_angle']:.3f} rate={tm['terminal_hinge_velocity_rms']:.3f}"
        )
        if tm["success"]:
            break
        elite = np.asarray([row["action_knots"] for row in records[: args.elites]], dtype=np.float64)
        center = np.asarray(best["action_knots"], dtype=np.float64)
        sigma = np.maximum(np.std(elite, axis=0) * args.sigma_decay, args.sigma_floor)

    assert best is not None
    best_metrics = evaluate_prefix_lqr(
        cfg,
        progress=args.progress,
        seed=args.seed,
        prefix_seconds=args.prefix_seconds,
        total_seconds=args.total_seconds,
        action_knots=np.asarray(best["action_knots"], dtype=np.float64),
        lqr_gain_vector=gain,
        lqr_scale=args.lqr_scale,
        record_trajectory=args.record_best,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "State-conditioned uninterrupted action prefix followed by exact LQR tail; development diagnostic.",
        "state_index": state_index,
        "selected_state": state,
        "progress": float(args.progress),
        "search": {
            "prefix_seconds": float(args.prefix_seconds),
            "total_seconds": float(args.total_seconds),
            "action_count": int(args.action_count),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "lqr_control_cost": float(args.lqr_control_cost),
            "lqr_scale": float(args.lqr_scale),
            "seed": int(args.seed),
        },
        "best": {"score": float(best["score"]), "action_knots": best["action_knots"], "metrics": best_metrics},
        "history": history,
        "evidence": {
            "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(cfg)},
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"success={best_metrics['success']} hold={best_metrics['max_upright_streak_seconds']:.3f}s "
        f"termination={best_metrics['termination_reason']} wrote={args.out}"
    )


if __name__ == "__main__":
    main()
