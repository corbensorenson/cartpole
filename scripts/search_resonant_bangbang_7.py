#!/usr/bin/env python
"""Try the deliberately simple resonant ``push, push, push`` controller.

This is a falsification experiment, not a final policy.  The cart receives a
single chirped square wave from a small parameter family.  Optionally, a
fixed-time handoff to the ordinary exact-MuJoCo LQR is tested.  All candidates
are replayed through the serial environment loop so a short upright crossing
cannot be mistaken for a solution.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles, wrap_angle
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics


def lqr_gain(cfg: dict[str, Any]) -> np.ndarray:
    a, b = finite_difference_dynamics(cfg, 1.0, 1e-7)
    q = absolute_angle_cost(
        int(cfg["env"]["n_links"]),
        {
            "cart_position": 0.1,
            "absolute_angle": 100.0,
            "cart_velocity": 0.1,
            "absolute_angular_velocity": 1.0,
            "relative_angle": 1.0,
            "relative_angular_velocity": 0.01,
        },
    )
    r = np.array([[1000.0]], dtype=np.float64)
    p = solve_discrete_are(a, b, q, r)
    return np.linalg.solve(b.T @ p @ b + r, b.T @ p @ a).reshape(-1)


def lqr_action(env: NLinkCartPoleEnv, gain: np.ndarray) -> float:
    qpos = np.asarray(env.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    state = np.r_[qpos[0], wrap_angle(qpos[1:]), qvel]
    return float(np.clip(-gain @ state, -1.0, 1.0))


def waveform_action(params: np.ndarray, t: float, seconds: float) -> float:
    """Chirped square wave with one optional harmonic and a taper."""
    amplitude, f_start, f_end, phase, harmonic, bias, taper_start, taper_end = params
    progress = float(np.clip(t / max(seconds, 1e-9), 0.0, 1.0))
    theta = 2.0 * np.pi * (f_start * t + 0.5 * (f_end - f_start) * t * t / max(seconds, 1e-9)) + phase
    carrier = np.sin(theta) + harmonic * np.sin(2.0 * theta + 0.5 * phase) + bias
    square = 1.0 if carrier >= 0.0 else -1.0
    if t <= taper_start:
        envelope = 1.0
    elif t >= taper_end:
        envelope = 0.0
    else:
        envelope = 1.0 - (t - taper_start) / max(1e-9, taper_end - taper_start)
    return float(amplitude * envelope * square)


def evaluate(
    cfg: dict[str, Any],
    params: np.ndarray,
    *,
    seconds: float,
    gain: np.ndarray | None,
    return_trace: bool = False,
) -> dict[str, Any]:
    env_cfg = {
        **cfg["env"],
        "init_mode": "hanging",
        "episode_seconds": float(seconds),
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "terminate_abs_angle": None,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        env_cfg[f"{key}_start"] = 0.0
        env_cfg[f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv({**cfg, "env": env_cfg}, progress=1.0, seed=0)
    env.reset(seed=0)
    switch_time = float(params[6])
    rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    max_cart = 0.0
    action_sq = 0.0
    previous = 0.0
    best_cost = float("inf")
    best_row: dict[str, Any] | None = None
    steps = min(env.max_steps, int(seconds / env.dt))
    for step in range(steps):
        t = step * env.dt
        if gain is not None and t >= switch_time:
            action = lqr_action(env, gain)
            mode = "lqr_after_fixed_switch"
        else:
            action = waveform_action(params, t, seconds)
            mode = "resonant_square_wave"
        _, _, terminated, truncated, info = env.step([action])
        absolute = serial_absolute_angles(np.asarray(env.data.qpos[1 : 1 + env.n], dtype=np.float64))
        absolute_rate = np.cumsum(np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64))
        row = {
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "controller_mode": mode,
            "action": float(action),
            "max_abs_angle": float(np.max(np.abs(absolute))),
            "hinge_velocity_rms": float(np.sqrt(np.mean(env.data.qvel[1:] ** 2))),
            "absolute_rate_rms": float(np.sqrt(np.mean(absolute_rate**2))),
            "x": float(env.data.qpos[0]),
            "cart_velocity": float(env.data.qvel[0]),
            "is_upright": bool(info.get("is_upright", False)),
            "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
            "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
            "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        }
        local_cost = (
            35.0 * (row["max_abs_angle"] / 0.15) ** 2
            + 12.0 * (row["hinge_velocity_rms"] / 0.75) ** 2
            + 16.0 * (row["absolute_rate_rms"] / 0.75) ** 2
            + 3.0 * (abs(row["x"]) / 1.25) ** 2
            + 3.0 * (abs(row["cart_velocity"]) / 0.50) ** 2
        )
        if local_cost < best_cost:
            best_cost = local_cost
            best_row = dict(row)
        if return_trace and (step % 2 == 0 or row["is_upright"]):
            rows.append(row)
        max_cart = max(max_cart, abs(row["x"]))
        action_sq += float(action) ** 2
        previous = float(action)
        final_info = dict(info)
        if terminated or truncated:
            break
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    centered_streak = float(final_info.get("max_centered_upright_streak_seconds", 0.0))
    low_streak = float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0))
    terminal_cost = float("inf")
    if final_info:
        terminal_cost = (
            35.0 * (float(final_info.get("max_abs_angle", np.inf)) / 0.15) ** 2
            + 12.0 * (float(final_info.get("hinge_velocity_rms", np.inf)) / 0.75) ** 2
            + 16.0 * (float(final_info.get("absolute_angular_velocity_rms", np.inf)) / 0.75) ** 2
            + 3.0 * (abs(float(final_info.get("x", np.inf))) / 1.25) ** 2
            + 3.0 * (abs(float(env.data.qvel[0])) / 0.50) ** 2
        )
    rail_penalty = 200000.0 * max(0.0, max_cart / float(env.rail_limit) - 0.97) ** 2
    score = (
        0.35 * best_cost
        + 0.35 * terminal_cost
        + 0.30 * (best_cost + terminal_cost) / 2.0
        - 18000.0 * max_streak
        - 3500.0 * centered_streak
        - 1500.0 * low_streak
        + rail_penalty
        + 0.01 * action_sq
    )
    result = {
        "score": float(score),
        "success": bool(final_info.get("success", False)),
        "best_cost": float(best_cost),
        "terminal_cost": float(terminal_cost),
        "max_upright_streak_seconds": max_streak,
        "max_centered_upright_streak_seconds": centered_streak,
        "max_low_momentum_upright_streak_seconds": low_streak,
        "max_cart_abs": float(max_cart),
        "termination_reason": final_info.get("termination_reason"),
        "best_row": best_row,
        "final_info": final_info,
        "trace": rows,
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact serial resonant square-wave search for 7-link swing-up")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--no-lqr-after-switch", action="store_true")
    parser.add_argument("--seed", type=int, default=20792)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.elites < 1 or args.elites > args.population or args.iterations < 1:
        raise ValueError("invalid search dimensions")
    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "rail_limit": float(cfg["env"].get("rail_limit", 3.0)),
        "rail_limit_start": float(cfg["env"].get("rail_limit", 3.0)),
        "rail_limit_end": float(cfg["env"].get("rail_limit", 3.0)),
        "plant_progress": 1.0,
    }
    gain = None if args.no_lqr_after_switch else lqr_gain(cfg)
    # amplitude, f_start, f_end, phase, harmonic, bias, taper_start, taper_end
    center = np.asarray([1.0, 0.55, 0.95, 0.0, 0.0, 0.0, 9.0, 13.0], dtype=np.float64)
    sigma = np.asarray([0.20, 0.25, 0.25, 0.80, 0.15, 0.10, 1.0, 1.0], dtype=np.float64)
    lower = np.asarray([0.10, 0.05, 0.05, -np.pi, -0.75, -0.75, 1.0, 1.1], dtype=np.float64)
    upper = np.asarray([1.0, 3.0, 3.0, np.pi, 0.75, 0.75, args.seconds - 1.0, args.seconds])
    rng = np.random.default_rng(args.seed)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    for iteration in range(args.iterations):
        candidates = np.clip(center[None, :] + rng.normal(0.0, sigma, size=(args.population, center.size)), lower, upper)
        candidates[0] = center
        records: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            metrics = evaluate(cfg, candidate, seconds=args.seconds, gain=gain)
            records.append({"index": index, "params": candidate, "metrics": metrics})
        records.sort(key=lambda row: float(row["metrics"]["score"]))
        top = records[0]
        elite = np.asarray([row["params"] for row in records[: args.elites]], dtype=np.float64)
        center = np.mean(elite, axis=0)
        sigma = np.maximum(np.std(elite, axis=0) * args.sigma_decay, args.sigma_floor)
        tm = top["metrics"]
        record = {
            "iteration": iteration + 1,
            "score": float(tm["score"]),
            "streak": float(tm["max_upright_streak_seconds"]),
            "centered_streak": float(tm["max_centered_upright_streak_seconds"]),
            "low_momentum_streak": float(tm["max_low_momentum_upright_streak_seconds"]),
            "best_cost": float(tm["best_cost"]),
            "terminal_cost": float(tm["terminal_cost"]),
            "max_cart_abs": float(tm["max_cart_abs"]),
        }
        history.append(record)
        if best is None or record["score"] < best["score"]:
            best = {**record, "params": top["params"].astype(float).tolist(), "metrics": tm}
        print(
            f"iter={iteration + 1:03d} score={record['score']:.2f} "
            f"streak={record['streak']:.3f}s centered={record['centered_streak']:.3f}s "
            f"low={record['low_momentum_streak']:.3f}s best_angle={tm.get('best_row', {}).get('max_abs_angle', np.nan):.3f} "
            f"rail={record['max_cart_abs']:.3f}",
            flush=True,
        )
    assert best is not None
    best_metrics = evaluate(cfg, np.asarray(best["params"], dtype=np.float64), seconds=args.seconds, gain=gain, return_trace=True)
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact serial canonical 7-link resonant square-wave/chirp search; discovery only.",
        "config_path": str(Path(args.config)),
        "config_sha256": data_sha256(cfg),
        "controller": {
            "type": "chirped_resonant_square_wave_then_optional_fixed_time_lqr",
            "params": np.asarray(best["params"], dtype=np.float64).astype(float).tolist(),
            "lqr_after_switch": not args.no_lqr_after_switch,
        },
        "search": {
            "seconds": float(args.seconds),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "sigma": float(args.sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "seed": int(args.seed),
            "wall_time_seconds": float(time.time() - started),
        },
        "best_search": best,
        "final_eval": best_metrics,
        "history": history,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(f"success={best_metrics['success']} hold={best_metrics['max_upright_streak_seconds']:.3f}s low={best_metrics['max_low_momentum_upright_streak_seconds']:.3f}s")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
