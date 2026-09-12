#!/usr/bin/env python
"""Search a mode-frequency multisine swing-up proposal in exact MuJoCo.

The drive frequencies are measured from the hanging linearization of the
actual plant instead of guessed as a blind chirp.  This is a discovery
controller: every candidate is evaluated through ``NLinkCartPoleEnv.step``
and the saved waveform still requires a downstream feedback replay.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.ilqr import MujocoTransition, data_state


def hanging_frequencies(env: NLinkCartPoleEnv) -> np.ndarray:
    """Return positive conjugate-pair frequencies from the hanging Jacobian."""
    transition = MujocoTransition(env)
    hanging_state = transition.to_coordinates(data_state(env.data))
    state_matrix, _ = transition.linearize(
        hanging_state,
        0.0,
        state_epsilon=1e-6,
        action_epsilon=1e-5,
    )
    eigenvalues = np.linalg.eigvals(state_matrix)
    frequencies = np.abs(np.angle(eigenvalues)) / (2.0 * np.pi * env.dt)
    frequencies = np.asarray(
        sorted(
            float(value)
            for value in frequencies
            if float(value) > 1e-3 and float(value) < 0.5 / env.dt
        ),
        dtype=np.float64,
    )
    # A conjugate pair contributes twice.  Retain one copy per physical mode.
    unique: list[float] = []
    for frequency in frequencies:
        if not unique or abs(frequency - unique[-1]) > 0.05:
            unique.append(frequency)
    if len(unique) < env.n:
        raise RuntimeError(f"hanging linearization exposed only {len(unique)} modes for n={env.n}: {unique}")
    return np.asarray(unique[: env.n], dtype=np.float64)


def action_from_params(
    params: np.ndarray,
    frequencies: np.ndarray,
    t: float,
    x: float,
    xdot: float,
    *,
    seconds: float,
    sequential: bool,
) -> float:
    n = len(frequencies)
    amplitudes = params[:n]
    phases = params[n : 2 * n]
    cart_kp, cart_kd, bias, harmonic = params[2 * n : 2 * n + 4]
    carriers = np.sin(2.0 * np.pi * frequencies * t + phases)
    if sequential:
        # Cross-fade neighboring stages so a mode is not switched off at a
        # discontinuity. The search can still choose each stage's amplitude
        # and phase independently.
        stage_position = np.clip(t / max(seconds, 1e-9) * n - 0.5, 0.0, n - 1.0)
        weights = np.maximum(0.0, 1.0 - np.abs(np.arange(n, dtype=np.float64) - stage_position))
        carrier = float(np.sum(amplitudes * weights * carriers))
    else:
        carrier = float(np.sum(amplitudes * carriers))
    carrier += float(harmonic) * np.sin(2.0 * np.pi * frequencies[0] * 2.0 * t + phases[0] / 2.0)
    return float(np.clip(np.tanh(carrier + bias - cart_kp * x - cart_kd * xdot), -1.0, 1.0))


def evaluate(
    cfg: dict[str, Any],
    params: np.ndarray,
    *,
    progress: float,
    seconds: float,
    rail_limit: float | None,
    frequencies: np.ndarray,
    sequential: bool,
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
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    if rail_limit is not None:
        env_cfg["rail_limit"] = float(rail_limit)
        env_cfg["rail_limit_start"] = float(rail_limit)
        env_cfg["rail_limit_end"] = float(rail_limit)
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        env_cfg[f"{key}_start"] = 0.0
        env_cfg[f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv({**cfg, "env": env_cfg}, progress=progress, seed=0)
    env.reset(seed=0)
    rows: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    best_cost = float("inf")
    best_row: dict[str, Any] | None = None
    max_cart = 0.0
    max_energy_fraction = 0.0
    action_sq = 0.0
    action_slew = 0.0
    previous_action = 0.0
    steps = min(env.max_steps, int(seconds / env.dt))
    for step in range(steps):
        t = step * env.dt
        action = action_from_params(
            params,
            frequencies,
            t,
            float(env.data.qpos[0]),
            float(env.data.qvel[0]),
            seconds=seconds,
            sequential=sequential,
        )
        _, _, terminated, truncated, info = env.step([action])
        absolute = serial_absolute_angles(np.asarray(env.data.qpos[1 : 1 + env.n], dtype=np.float64))
        absolute_rate = np.cumsum(np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64))
        row = {
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "action": float(action),
            "max_abs_angle": float(np.max(np.abs(absolute))),
            "hinge_velocity_rms": float(np.sqrt(np.mean(np.asarray(env.data.qvel[1 : 1 + env.n]) ** 2))),
            "absolute_angular_velocity_rms": float(np.sqrt(np.mean(absolute_rate**2))),
            "energy_fraction": float(env._energy_fraction()),
            "x": float(env.data.qpos[0]),
            "cart_velocity": float(env.data.qvel[0]),
            "absolute_angles": absolute.astype(float).tolist(),
            "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
            "is_upright": bool(info.get("is_upright", False)),
            "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
            "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
        }
        local_cost = (
            24.0 * (row["max_abs_angle"] / 0.15) ** 2
            + 8.0 * (row["hinge_velocity_rms"] / 0.75) ** 2
            + 10.0 * (row["absolute_angular_velocity_rms"] / 0.75) ** 2
            + 2.0 * (abs(row["x"]) / 1.25) ** 2
            + 2.0 * (abs(row["cart_velocity"]) / 0.50) ** 2
        )
        if step >= int(2.0 / env.dt) and local_cost < best_cost:
            best_cost = local_cost
            best_row = dict(row)
        if return_trace:
            rows.append(row)
        max_cart = max(max_cart, abs(row["x"]))
        max_energy_fraction = max(max_energy_fraction, row["energy_fraction"])
        action_sq += action * action
        action_slew += (action - previous_action) ** 2
        previous_action = action
        final_info = dict(info)
        if terminated or truncated:
            break

    if best_row is None:
        best_row = rows[-1]
        best_cost = float("inf")
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    max_centered = float(final_info.get("max_centered_upright_streak_seconds", 0.0))
    max_low = float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0))
    terminal_cost = (
        24.0 * (float(final_info.get("max_abs_angle", np.inf)) / 0.15) ** 2
        + 8.0 * (float(final_info.get("hinge_velocity_rms", np.inf)) / 0.75) ** 2
        + 10.0 * (float(final_info.get("absolute_angular_velocity_rms", np.inf)) / 0.75) ** 2
        + 2.0 * (abs(float(final_info.get("x", np.inf))) / 1.25) ** 2
        + 2.0 * (abs(float(env.data.qvel[0])) / 0.50) ** 2
    )
    rail = float(env.rail_limit)
    rail_penalty = 100000.0 * max(0.0, max_cart / max(1e-9, rail) - 0.95) ** 2
    score = (
        0.60 * best_cost
        + 0.25 * terminal_cost
        + 0.15 * terminal_cost
        + rail_penalty
        - 18000.0 * max_streak
        - 3500.0 * max_centered
        - 1500.0 * max_low
        + 0.01 * action_sq
        + 0.03 * action_slew
    )
    result = {
        "score": float(score),
        "success": bool(final_info.get("success", False)),
        "best_cost": float(best_cost),
        "terminal_cost": float(terminal_cost),
        "best_row": best_row,
        "max_upright_streak_seconds": max_streak,
        "max_centered_upright_streak_seconds": max_centered,
        "max_low_momentum_upright_streak_seconds": max_low,
        "max_cart_abs": float(max_cart),
        "max_energy_fraction": float(max_energy_fraction),
        "termination_reason": final_info.get("termination_reason"),
        "final_info": final_info,
        "trace": rows,
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Exact serial mode-frequency multisine search")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--rail-limit", type=float, default=None)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--reverse-order", action="store_true")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("--progress must be in [0, 1]")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("--elites must be in 1..population")

    cfg = apply_overrides(load_config(args.config), args.override)
    probe_cfg = {**cfg, "env": {**cfg["env"], "init_mode": "hanging", "action_lqr_residual": {"enabled": False}, "action_lqr_switch": {"enabled": False}}}
    probe = NLinkCartPoleEnv(probe_cfg, progress=args.progress, seed=0)
    probe.reset(seed=0)
    frequencies = hanging_frequencies(probe)
    if args.reverse_order:
        frequencies = frequencies[::-1].copy()
    n = len(frequencies)
    probe.close()

    # amplitudes, phases, cart_kp, cart_kd, bias, second-harmonic amplitude
    center = np.r_[np.full(n, 0.10), np.zeros(n), 0.04, 0.06, 0.0, 0.0].astype(np.float64)
    sigma = np.r_[np.full(n, args.sigma), np.full(n, 0.80), 0.05, 0.05, 0.08, 0.10].astype(np.float64)
    lower = np.r_[np.full(n, 0.0), np.full(n, -np.pi), 0.0, 0.0, -0.35, -0.50].astype(np.float64)
    upper = np.r_[np.full(n, 0.80), np.full(n, np.pi), 0.80, 0.80, 0.35, 0.50].astype(np.float64)
    rng = np.random.default_rng(args.seed)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    for iteration in range(args.iterations + 1):
        if iteration == 0:
            candidates = [center.copy()]
        else:
            candidates = [center]
            candidates.extend(
                np.clip(center + rng.normal(0.0, sigma), lower, upper)
                for _ in range(args.population - 1)
            )
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            metrics = evaluate(
                cfg,
                np.asarray(candidate, dtype=np.float64),
                progress=args.progress,
                seconds=args.seconds,
                rail_limit=args.rail_limit,
                frequencies=frequencies,
                sequential=args.sequential,
            )
            records.append({"params": np.asarray(candidate, dtype=np.float64), "metrics": metrics})
        records.sort(key=lambda row: float(row["metrics"]["score"]))
        top = records[0]
        if best is None or float(top["metrics"]["score"]) < float(best["score"]):
            best = {
                "score": float(top["metrics"]["score"]),
                "params": top["params"].astype(float).tolist(),
                "metrics": top["metrics"],
            }
        elite = np.asarray([row["params"] for row in records[: args.elites]], dtype=np.float64)
        center = np.mean(elite, axis=0)
        sigma = np.maximum(np.std(elite, axis=0) * args.sigma_decay, args.sigma_floor)
        tm = top["metrics"]
        history.append(
            {
                "iteration": int(iteration),
                "score": float(tm["score"]),
                "best_score": float(best["score"]),
                "best_angle": float(tm["best_row"]["max_abs_angle"]),
                "best_hinge": float(tm["best_row"]["hinge_velocity_rms"]),
                "best_absolute_rate": float(tm["best_row"]["absolute_angular_velocity_rms"]),
                "max_energy_fraction": float(tm["max_energy_fraction"]),
                "max_cart_abs": float(tm["max_cart_abs"]),
                "streak": float(tm["max_upright_streak_seconds"]),
            }
        )
        print(
            f"iter={iteration:03d} score={tm['score']:.2f} angle={tm['best_row']['max_abs_angle']:.4f} "
            f"hinge={tm['best_row']['hinge_velocity_rms']:.3f} abs={tm['best_row']['absolute_angular_velocity_rms']:.3f} "
            f"height={tm['max_energy_fraction']:.3f} rail={tm['max_cart_abs']:.2f} hold={tm['max_upright_streak_seconds']:.3f}s",
            flush=True,
        )

    assert best is not None
    final = evaluate(
        cfg,
        np.asarray(best["params"], dtype=np.float64),
        progress=args.progress,
        seconds=args.seconds,
        rail_limit=args.rail_limit,
        frequencies=frequencies,
        sequential=args.sequential,
        return_trace=True,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Mode-frequency multisine exact-MuJoCo swing discovery; downstream capture replay required.",
        "config_path": str(Path(args.config)),
        "progress": float(args.progress),
        "frequencies_hz": frequencies.astype(float).tolist(),
        "best": best,
        "final_eval": final,
        "search": {
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "sigma": float(args.sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "seconds": float(args.seconds),
            "rail_limit": args.rail_limit,
            "sequential": bool(args.sequential),
            "reverse_order": bool(args.reverse_order),
            "seed": int(args.seed),
            "wall_time_seconds": float(time.time() - started),
        },
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
        "history": history,
    }
    dump_json(payload, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
