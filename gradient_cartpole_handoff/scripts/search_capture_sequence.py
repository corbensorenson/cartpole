#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from search_swingup_capture import lqr_action, lqr_gain


def state_quality(state: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(state.get("max_abs_angle", np.pi)),
        float(state.get("hinge_velocity_rms", np.inf)),
        abs(float(state.get("x", 0.0))),
        abs(float(state.get("cart_velocity", 0.0))),
    )


def load_state(path: str, state_index: str) -> tuple[dict[str, Any], int]:
    with open(Path(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    states = payload.get("states", payload) if isinstance(payload, dict) else payload
    if not isinstance(states, list) or not states:
        raise ValueError(f"{path} does not contain a non-empty states list")
    if state_index == "best":
        idx, state = min(enumerate(states), key=lambda item: state_quality(item[1]))
        return dict(state), int(idx)
    idx = int(state_index)
    if idx < 0 or idx >= len(states):
        raise IndexError(f"state index {idx} outside 0..{len(states) - 1}")
    return dict(states[idx]), idx


def fixed_state_cfg(cfg: dict[str, Any], state: dict[str, Any], seconds: float) -> dict[str, Any]:
    env_cfg = {
        **cfg["env"],
        "init_mode": "fixed_state",
        "init_qpos": state["qpos"],
        "init_qvel": state["qvel"],
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "episode_seconds": float(seconds),
        "terminate_abs_angle": None,
        "success_upright_threshold": cfg["env"].get("success_upright_threshold", 0.15),
        "success_sustain_seconds": cfg["env"].get("success_sustain_seconds", 5.0),
    }
    return {**cfg, "env": env_cfg}


def row_from_env(env: NLinkCartPoleEnv, *, step: int, action: float, reward: float, info: dict[str, Any]) -> dict[str, Any]:
    rel, abs_angles = env._angles()
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "reward": float(reward),
        "action": float(action),
        "x": float(info["x"]),
        "cart_velocity": float(env.data.qvel[0]),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "max_abs_angle": float(info["max_abs_angle"]),
        "mean_abs_angle": float(info["mean_abs_angle"]),
        "hinge_velocity_rms": float(info.get("hinge_velocity_rms", np.sqrt(np.mean(env.data.qvel[1 : 1 + env.n] ** 2)))),
        "capture_quality": float(info.get("capture_quality", 0.0)),
        "relative_angles": rel.astype(float).tolist(),
        "absolute_angles": abs_angles.astype(float).tolist(),
        "is_upright": bool(info["is_upright"]),
        "upright_streak_seconds": float(info["upright_streak_seconds"]),
        "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
        "time_to_first_upright": info["time_to_first_upright"],
    }


def evaluate_actions(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    seconds: float,
    action_knots: np.ndarray,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    _, reset_info = env.reset()
    del reset_info
    steps = min(env.max_steps, int(seconds / env.dt))
    action_times = np.linspace(0.0, seconds, len(action_knots), dtype=np.float64)
    best: dict[str, Any] | None = None
    upright_event_count = 0
    max_cart_abs = 0.0
    action_abs_max = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False

    for step in range(steps):
        t = step * env.dt
        action = float(np.interp(min(t, seconds), action_times, action_knots))
        action = float(np.clip(action, -1.0, 1.0))
        action_abs_max = max(action_abs_max, abs(action))
        _, reward, terminated, truncated, info = env.step([action])
        row = row_from_env(env, step=step + 1, action=action, reward=reward, info=info)
        max_cart_abs = max(max_cart_abs, abs(float(row["x"])))
        if best is None or state_quality(row) < state_quality(best):
            best = row
        if row["is_upright"]:
            upright_event_count += 1
        final_info = dict(info)
        if terminated or truncated:
            break

    env.close()
    assert best is not None
    success = bool(final_info.get("success", False))
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    terminal_penalty = 0.0 if truncated and not terminated else 650.0
    score = (
        -20000.0 * float(success)
        - 3500.0 * max_streak
        - 3.0 * float(upright_event_count)
        + 40.0 * float(best["max_abs_angle"])
        + 18.0 * float(best["hinge_velocity_rms"])
        + 8.0 * abs(float(best["cart_velocity"]))
        + 5.0 * abs(float(best["x"]))
        + 25.0 * max(0.0, max_cart_abs - float(cfg["env"]["rail_limit"])) ** 2
        + terminal_penalty
    )
    return {
        "score": float(score),
        "success": success,
        "simulated_steps": int(step + 1),
        "simulated_seconds": float((step + 1) * env.dt),
        "upright_event_count": int(upright_event_count),
        "max_cart_abs": float(max_cart_abs),
        "action_abs_max": float(action_abs_max),
        "best_upright_pass": best,
        "final_info": final_info,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def lqr_baseline_actions(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    seconds: float,
    action_count: int,
    fd_eps: float,
    control_cost: float,
    lqr_scale: float,
) -> np.ndarray:
    gain = lqr_gain(cfg, progress=progress, fd_eps=fd_eps, control_cost=control_cost)
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    env.reset()
    steps = min(env.max_steps, int(seconds / env.dt))
    sample_steps = np.linspace(0, max(1, steps - 1), action_count).astype(int)
    actions: list[float] = []
    sample_idx = 0
    last_action = 0.0
    for step in range(steps):
        action = lqr_action(env, gain, scale=lqr_scale, cart_target=0.0)
        if sample_idx < len(sample_steps) and step >= int(sample_steps[sample_idx]):
            actions.append(float(action))
            sample_idx += 1
        last_action = float(action)
        _, _, terminated, truncated, _ = env.step([action])
        if terminated or truncated:
            break
    env.close()
    while len(actions) < action_count:
        actions.append(last_action)
    return np.asarray(actions[:action_count], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Search an open-loop capture action sequence from a replayed handoff state")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--state-json", default="runs/swingup6_expert_chain/swing_states_low_momentum.json")
    parser.add_argument("--state-index", default="best")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--out", default="runs/swingup6_capture_sequence/search.json")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--action-count", type=int, default=41)
    parser.add_argument("--init-sequence", choices=["zero", "lqr"], default="lqr")
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--lqr-scale", type=float, default=1.0)
    parser.add_argument("--action-sigma", type=float, default=0.55)
    parser.add_argument("--sigma-decay", type=float, default=0.86)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.population < 1:
        raise ValueError("--population must be >= 1")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("--elites must be between 1 and --population")
    if args.action_count < 2:
        raise ValueError("--action-count must be >= 2")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    selected_state, selected_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(base_cfg, selected_state, args.seconds)
    rng = np.random.default_rng(args.seed)
    if args.init_sequence == "lqr":
        center = lqr_baseline_actions(
            cfg,
            progress=args.progress,
            seed=args.seed,
            seconds=args.seconds,
            action_count=args.action_count,
            fd_eps=args.fd_eps,
            control_cost=args.lqr_control_cost,
            lqr_scale=args.lqr_scale,
        )
    else:
        center = np.zeros(args.action_count, dtype=np.float64)
    sigma = np.full(args.action_count, float(args.action_sigma), dtype=np.float64)

    best: dict[str, Any] | None = None
    best_by_max_streak: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for iteration in range(max(1, args.iterations + 1)):
        candidates = [center]
        if iteration > 0:
            candidates.extend(center + rng.normal(0.0, sigma) for _ in range(max(0, args.population - 1)))
        records = []
        for candidate in candidates:
            actions = np.clip(candidate, -1.0, 1.0)
            metrics = evaluate_actions(
                cfg,
                progress=args.progress,
                seed=args.seed,
                seconds=args.seconds,
                action_knots=actions,
            )
            records.append({"score": metrics["score"], "action_knots": actions.astype(float).tolist(), "metrics": metrics})
        records.sort(key=lambda row: float(row["score"]))
        if best is None or float(records[0]["score"]) < float(best["score"]):
            best = records[0]
        for row in records:
            row_streak = float(row["metrics"]["final_info"].get("max_upright_streak_seconds", 0.0))
            if best_by_max_streak is None:
                best_by_max_streak = row
                continue
            incumbent_streak = float(best_by_max_streak["metrics"]["final_info"].get("max_upright_streak_seconds", 0.0))
            if (row_streak, -float(row["score"])) > (incumbent_streak, -float(best_by_max_streak["score"])):
                best_by_max_streak = row
        top = records[0]
        top_metrics = top["metrics"]
        best_pass = top_metrics["best_upright_pass"]
        history.append(
            {
                "iteration": int(iteration),
                "score": float(top["score"]),
                "success": bool(top_metrics["success"]),
                "max_upright_streak_seconds": float(top_metrics["final_info"].get("max_upright_streak_seconds", 0.0)),
                "best_max_abs_angle": float(best_pass["max_abs_angle"]),
                "best_hinge_velocity_rms": float(best_pass["hinge_velocity_rms"]),
                "max_cart_abs": float(top_metrics["max_cart_abs"]),
            }
        )
        print(
            f"iter={iteration:03d} score={top['score']:.6f} "
            f"streak={top_metrics['final_info'].get('max_upright_streak_seconds', 0.0):.3f}s "
            f"angle={best_pass['max_abs_angle']:.6f} "
            f"hinge={best_pass['hinge_velocity_rms']:.3f} "
            f"success={top_metrics['success']}"
        )
        if iteration > 0:
            elite_arr = np.asarray([row["action_knots"] for row in records[: args.elites]], dtype=np.float64)
            center = np.asarray(best["action_knots"], dtype=np.float64)
            sigma = np.maximum(elite_arr.std(axis=0), 1e-3) * float(args.sigma_decay)

    assert best is not None
    assert best_by_max_streak is not None
    result = {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Open-loop capture action-sequence diagnostic from a real swing handoff state; not final benchmark evidence.",
        "seed": int(args.seed),
        "progress": float(args.progress),
        "state_json": str(Path(args.state_json)),
        "state_index": selected_index,
        "selected_state": selected_state,
        "search": {
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "seconds": float(args.seconds),
            "action_count": int(args.action_count),
            "init_sequence": str(args.init_sequence),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "lqr_control_cost": float(args.lqr_control_cost),
            "lqr_scale": float(args.lqr_scale),
        },
        "best": best,
        "best_by": {"score": best, "max_upright_streak": best_by_max_streak},
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(result, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
