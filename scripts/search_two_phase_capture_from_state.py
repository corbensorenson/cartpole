#!/usr/bin/env python
"""CEM-search two reset-free affine capture experts from a real state."""

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


def load_state(path: str, index: int) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    return dict(states[index])


def features(env: NLinkCartPoleEnv) -> np.ndarray:
    qpos = np.asarray(env.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    absolute = serial_absolute_angles(qpos[1 : 1 + env.n])
    absolute_omega = np.cumsum(qvel[1 : 1 + env.n])
    return np.r_[
        qpos[0] / 1.25,
        np.sin(absolute),
        np.cos(absolute) - 1.0,
        qvel[0] / 0.50,
        absolute_omega / 0.75,
    ].astype(np.float64)


def row(env: NLinkCartPoleEnv, info: dict[str, Any], step: int, action: float, mode: str) -> dict[str, Any]:
    _, absolute = env._angles()
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "mode": mode,
        "action": float(action),
        "max_abs_angle": float(np.max(np.abs(absolute))),
        "hinge_velocity_rms": float(np.sqrt(np.mean(np.asarray(env.data.qvel[1 : 1 + env.n]) ** 2))),
        "x": float(env.data.qpos[0]),
        "cart_velocity": float(env.data.qvel[0]),
        "absolute_angles": absolute.astype(float).tolist(),
        "is_upright": bool(info.get("is_upright", False)),
        "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
        "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
        "max_centered_upright_streak_seconds": float(info.get("max_centered_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(info.get("max_low_momentum_upright_streak_seconds", 0.0)),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
    }


def rollout(
    cfg: dict[str, Any],
    state: dict[str, Any],
    vector: np.ndarray,
    switch_seconds: float,
    seconds: float,
    trace: bool = False,
    progress: float = 0.0,
) -> dict[str, Any]:
    env_cfg = {
        **cfg["env"],
        "init_mode": "fixed_state",
        "init_qpos": state["qpos"],
        "init_qvel": state["qvel"],
        "episode_seconds": float(seconds),
        "terminate_abs_angle": None,
        "obs_include_capture_features": False,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        env_cfg[f"{key}_start"] = 0.0
        env_cfg[f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv({**cfg, "env": env_cfg}, progress=progress, seed=0)
    env.reset(seed=0)
    dim = int(features(env).size)
    block = dim + 1
    if vector.shape != (2 * block,):
        raise ValueError(f"expected vector {(2 * block,)}, got {vector.shape}")
    weights = [vector[:dim], vector[block : block + dim]]
    biases = [float(vector[dim]), float(vector[-1])]
    switch_step = max(1, int(round(switch_seconds / env.dt)))
    rows: list[dict[str, Any]] = []
    best_cost = float("inf")
    best: dict[str, Any] | None = None
    previous = 0.0
    action_sq = 0.0
    action_slew = 0.0
    max_cart = abs(float(env.data.qpos[0]))
    final_info: dict[str, Any] = {}
    for step in range(min(env.max_steps, int(seconds / env.dt))):
        phase = 0 if step < switch_step else 1
        action = float(np.tanh(features(env) @ weights[phase] + biases[phase]))
        _, _, terminated, truncated, info = env.step([action])
        current = row(env, info, step + 1, action, "arrest" if phase == 0 else "stabilize")
        local_cost = (
            30.0 * (current["max_abs_angle"] / 0.15) ** 2
            + 8.0 * (current["hinge_velocity_rms"] / 0.75) ** 2
            + 3.0 * (abs(current["x"]) / 1.25) ** 2
            + 3.0 * (abs(current["cart_velocity"]) / 0.50) ** 2
        )
        if local_cost < best_cost:
            best_cost = local_cost
            best = current
        if trace and (step % 2 == 0 or current["is_upright"]):
            rows.append(current)
        max_cart = max(max_cart, abs(current["x"]))
        action_sq += action * action
        action_slew += (action - previous) ** 2
        previous = action
        final_info = dict(info)
        if terminated or truncated:
            break
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    max_centered = float(final_info.get("max_centered_upright_streak_seconds", 0.0))
    max_low = float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0))
    terminal_cost = (
        30.0 * (float(final_info.get("max_abs_angle", np.inf)) / 0.15) ** 2
        + 8.0 * (float(final_info.get("hinge_velocity_rms", np.inf)) / 0.75) ** 2
        + 3.0 * (abs(float(final_info.get("x", np.inf))) / 1.25) ** 2
        + 3.0 * (abs(float(env.data.qvel[0])) / 0.50) ** 2
    )
    rail_over = max(0.0, max_cart - float(env.rail_limit))
    score = (
        0.25 * best_cost
        + 0.35 * terminal_cost
        + 0.40 * (best_cost + terminal_cost) / 2.0
        - 12000.0 * max_streak
        - 7000.0 * max_centered
        - 9000.0 * max_low
        + 150.0 * rail_over * rail_over
        + 0.01 * action_sq
        + 0.04 * action_slew
    )
    result = {
        "score": float(score),
        "success": bool(final_info.get("success", False)),
        "best_cost": float(best_cost),
        "terminal_cost": float(terminal_cost),
        "max_upright_streak_seconds": max_streak,
        "max_centered_upright_streak_seconds": max_centered,
        "max_low_momentum_upright_streak_seconds": max_low,
        "max_cart_abs": float(max_cart),
        "termination_reason": final_info.get("termination_reason"),
        "best_row": best,
        "final_info": final_info,
        "trace": rows,
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-phase affine capture CEM from a real state")
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--switch-seconds", type=float, default=0.20)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    cfg = apply_overrides(load_config(args.config), args.override)
    state = load_state(args.state_json, args.state_index)
    probe_cfg = {**cfg, "env": {**cfg["env"], "init_mode": "fixed_state", "init_qpos": state["qpos"], "init_qvel": state["qvel"]}}
    probe = NLinkCartPoleEnv(probe_cfg, progress=args.progress, seed=0)
    dim = int(features(probe).size)
    probe.close()
    rng = np.random.default_rng(args.seed)
    center = np.zeros(2 * (dim + 1), dtype=np.float64)
    sigma = np.full(center.shape, float(args.sigma), dtype=np.float64)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    for iteration in range(args.iterations + 1):
        candidates = [center.copy()] if iteration == 0 else [center + rng.normal(0.0, sigma, size=center.shape) for _ in range(args.population)]
        records = []
        for vector in candidates:
            metrics = rollout(
                cfg,
                state,
                np.asarray(vector, dtype=np.float64),
                args.switch_seconds,
                args.seconds,
                progress=args.progress,
            )
            records.append({"vector": np.asarray(vector, dtype=np.float64), "metrics": metrics})
        records.sort(key=lambda item: float(item["metrics"]["score"]))
        top = records[0]
        if best is None or float(top["metrics"]["score"]) < float(best["score"]):
            best = {"score": float(top["metrics"]["score"]), "vector": top["vector"].astype(float).tolist(), "metrics": top["metrics"]}
        elite = np.asarray([item["vector"] for item in records[: args.elites]], dtype=np.float64)
        center = np.mean(elite, axis=0)
        sigma = np.maximum(np.std(elite, axis=0) * args.sigma_decay, args.sigma_floor)
        tm = top["metrics"]
        history.append({"iteration": iteration, "score": float(tm["score"]), "best_score": float(best["score"]), "streak": float(tm["max_upright_streak_seconds"]), "centered": float(tm["max_centered_upright_streak_seconds"]), "low_momentum": float(tm["max_low_momentum_upright_streak_seconds"]), "best_cost": float(tm["best_cost"])})
        print(f"iter={iteration:03d} score={tm['score']:.2f} streak={tm['max_upright_streak_seconds']:.3f}s centered={tm['max_centered_upright_streak_seconds']:.3f}s low={tm['max_low_momentum_upright_streak_seconds']:.3f}s best={tm['best_cost']:.1f} terminal={tm['terminal_cost']:.1f}", flush=True)
    assert best is not None
    final = rollout(
        cfg,
        state,
        np.asarray(best["vector"], dtype=np.float64),
        args.switch_seconds,
        args.seconds,
        trace=True,
        progress=args.progress,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Two reset-free affine capture experts with a fixed arrest-to-stabilize switch; discovery only.",
        "config_path": str(Path(args.config)),
        "state_json": str(Path(args.state_json)),
        "state_index": int(args.state_index),
        "switch_seconds": float(args.switch_seconds),
        "seconds": float(args.seconds),
        "search": {"seed": int(args.seed), "iterations": int(args.iterations), "population": int(args.population), "elites": int(args.elites), "sigma": float(args.sigma), "sigma_decay": float(args.sigma_decay), "sigma_floor": float(args.sigma_floor), "wall_time_seconds": float(time.time() - started)},
        "feature_names": ["cart_position"] + [f"sin_absolute_angle_{i}" for i in range(int(cfg["env"]["n_links"]))] + [f"cos_minus_one_absolute_angle_{i}" for i in range(int(cfg["env"]["n_links"]))] + ["cart_velocity"] + [f"absolute_angular_velocity_{i}" for i in range(int(cfg["env"]["n_links"]))],
        "best_vector": best["vector"],
        "best_search": best,
        "final_eval": final,
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
