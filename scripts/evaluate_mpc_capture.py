#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from search_capture_sequence import fixed_state_cfg, load_state, row_from_env, state_quality
from search_swingup_capture import lqr_action, lqr_gain


def set_state(data: mujoco.MjData, qpos: np.ndarray, qvel: np.ndarray) -> None:
    data.time = 0.0
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = 0.0


def rollout_score(
    env: NLinkCartPoleEnv,
    data: mujoco.MjData,
    *,
    start_qpos: np.ndarray,
    start_qvel: np.ndarray,
    actions: np.ndarray,
    current_streak_steps: int,
) -> tuple[float, dict[str, Any]]:
    set_state(data, start_qpos, start_qvel)
    mujoco.mj_forward(env.model, data)
    threshold = float(env.env_cfg.get("success_upright_threshold", env.env_cfg.get("reward", {}).get("upright_threshold", 0.15)))
    max_streak = int(current_streak_steps)
    streak = int(current_streak_steps)
    best_row: dict[str, Any] | None = None
    max_cart_abs = abs(float(data.qpos[0]))
    terminal = False
    upright_count = 0
    angle_sum = 0.0
    hinge_sum = 0.0
    cart_cost_sum = 0.0

    for idx, action in enumerate(actions):
        data.ctrl[0] = float(np.clip(action, -1.0, 1.0)) * env.force_limit
        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, data)
        rel = wrap_angle(np.asarray(data.qpos[1 : 1 + env.n], dtype=np.float64))
        abs_angles = wrap_angle(np.cumsum(np.asarray(data.qpos[1 : 1 + env.n], dtype=np.float64)))
        max_abs_angle = float(np.max(np.abs(abs_angles)))
        hinge_rms = float(np.sqrt(np.mean(data.qvel[1 : 1 + env.n] ** 2)))
        row = {
            "step": int(idx + 1),
            "time_seconds": float((idx + 1) * env.dt),
            "action": float(action),
            "x": float(data.qpos[0]),
            "cart_velocity": float(data.qvel[0]),
            "qpos": np.asarray(data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.asarray(data.qvel, dtype=np.float64).astype(float).tolist(),
            "max_abs_angle": max_abs_angle,
            "mean_abs_angle": float(np.mean(np.abs(abs_angles))),
            "hinge_velocity_rms": hinge_rms,
            "relative_angles": rel.astype(float).tolist(),
            "absolute_angles": abs_angles.astype(float).tolist(),
            "is_upright": bool(max_abs_angle < threshold),
            "upright_streak_seconds": 0.0,
            "max_upright_streak_seconds": float(max_streak * env.dt),
            "time_to_first_upright": None,
        }
        if row["is_upright"]:
            streak += 1
            upright_count += 1
        else:
            streak = 0
        max_streak = max(max_streak, streak)
        row["upright_streak_seconds"] = float(streak * env.dt)
        row["max_upright_streak_seconds"] = float(max_streak * env.dt)
        if best_row is None or state_quality(row) < state_quality(best_row):
            best_row = row
        max_cart_abs = max(max_cart_abs, abs(float(data.qpos[0])))
        angle_sum += max_abs_angle * max_abs_angle
        hinge_sum += hinge_rms * hinge_rms
        cart_cost_sum += (float(data.qpos[0]) / env.rail_limit) ** 2 + 0.10 * float(data.qvel[0]) ** 2
        if abs(float(data.qpos[0])) > env.rail_limit or not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            terminal = True
            break

    assert best_row is not None
    score = (
        - 4000.0 * float(max_streak * env.dt)
        - 20.0 * float(upright_count)
        + 28.0 * float(best_row["max_abs_angle"])
        + 10.0 * float(best_row["hinge_velocity_rms"])
        + 2.0 * abs(float(best_row["x"]))
        + 1.0 * abs(float(best_row["cart_velocity"]))
        + 0.20 * angle_sum
        + 0.03 * hinge_sum
        + 0.25 * cart_cost_sum
        + 500.0 * float(terminal)
        + 100.0 * max(0.0, max_cart_abs - env.rail_limit) ** 2
    )
    summary = {
        "best_row": best_row,
        "max_streak_seconds": float(max_streak * env.dt),
        "upright_count": int(upright_count),
        "terminal": bool(terminal),
        "max_cart_abs": float(max_cart_abs),
    }
    return float(score), summary


def plan_action_sequence(
    env: NLinkCartPoleEnv,
    scratch: mujoco.MjData,
    *,
    rng: np.random.Generator,
    mean: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    current_streak_steps: int,
    samples: int,
    elites: int,
    iterations: int,
    sigma: float,
    lqr_gain_arr: np.ndarray | None,
    lqr_scale: float,
    lqr_target: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    best_actions = np.clip(mean, -1.0, 1.0)
    best_score = float("inf")
    best_summary: dict[str, Any] = {}
    cur_mean = np.asarray(mean, dtype=np.float64)
    cur_sigma = np.full_like(cur_mean, float(sigma), dtype=np.float64)
    for _ in range(max(1, iterations)):
        candidates = [np.clip(cur_mean, -1.0, 1.0)]
        if lqr_gain_arr is not None:
            live_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
            live_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
            live_ctrl = np.asarray(env.data.ctrl, dtype=np.float64).copy()
            lqr_actions = []
            set_state(scratch, qpos, qvel)
            mujoco.mj_forward(env.model, scratch)
            for _step in range(len(cur_mean)):
                env.data.qpos[:] = scratch.qpos
                env.data.qvel[:] = scratch.qvel
                env.data.ctrl[:] = scratch.ctrl
                action = lqr_action(env, lqr_gain_arr, scale=lqr_scale, cart_target=lqr_target)
                lqr_actions.append(action)
                scratch.ctrl[0] = action * env.force_limit
                for _ in range(env.frame_skip):
                    mujoco.mj_step(env.model, scratch)
            env.data.qpos[:] = live_qpos
            env.data.qvel[:] = live_qvel
            env.data.ctrl[:] = live_ctrl
            mujoco.mj_forward(env.model, env.data)
            candidates.append(np.asarray(lqr_actions, dtype=np.float64))
        for _ in range(max(0, samples - len(candidates))):
            candidates.append(np.clip(cur_mean + rng.normal(0.0, cur_sigma), -1.0, 1.0))
        records = []
        for actions in candidates:
            score, summary = rollout_score(
                env,
                scratch,
                start_qpos=qpos,
                start_qvel=qvel,
                actions=np.asarray(actions, dtype=np.float64),
                current_streak_steps=current_streak_steps,
            )
            records.append((score, np.asarray(actions, dtype=np.float64), summary))
        records.sort(key=lambda item: item[0])
        if records[0][0] < best_score:
            best_score = float(records[0][0])
            best_actions = records[0][1]
            best_summary = records[0][2]
        elite_arr = np.asarray([row[1] for row in records[: max(1, elites)]], dtype=np.float64)
        cur_mean = elite_arr.mean(axis=0)
        cur_sigma = np.maximum(elite_arr.std(axis=0), 1e-3) * 0.85
    return np.clip(best_actions, -1.0, 1.0), {"score": best_score, **best_summary}


def evaluate_mpc(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    seconds: float,
    horizon_steps: int,
    replan_steps: int,
    samples: int,
    elites: int,
    iterations: int,
    sigma: float,
    lqr_gain_arr: np.ndarray | None,
    lqr_scale: float,
    lqr_target: float,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    env.reset()
    scratch = mujoco.MjData(env.model)
    rng = np.random.default_rng(seed)
    total_steps = min(env.max_steps, int(seconds / env.dt))
    mean = np.zeros(horizon_steps, dtype=np.float64)
    plan: np.ndarray | None = None
    plan_index = 0
    rows: list[dict[str, Any]] = []
    plan_events: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False

    for step in range(total_steps):
        if plan is None or plan_index >= min(replan_steps, len(plan)):
            if plan is not None:
                remaining = plan[plan_index:]
                mean = np.r_[remaining, np.zeros(max(0, horizon_steps - len(remaining)))][:horizon_steps]
            plan, plan_summary = plan_action_sequence(
                env,
                scratch,
                rng=rng,
                mean=mean,
                qpos=np.asarray(env.data.qpos, dtype=np.float64).copy(),
                qvel=np.asarray(env.data.qvel, dtype=np.float64).copy(),
                current_streak_steps=env.upright_streak_steps,
                samples=samples,
                elites=elites,
                iterations=iterations,
                sigma=sigma,
                lqr_gain_arr=lqr_gain_arr,
                lqr_scale=lqr_scale,
                lqr_target=lqr_target,
            )
            plan_index = 0
            mean = plan.copy()
            plan_events.append({"step": int(step), "time_seconds": float(step * env.dt), **plan_summary})
        action = float(plan[plan_index])
        plan_index += 1
        _, reward, terminated, truncated, info = env.step([action])
        row = row_from_env(env, step=step + 1, action=action, reward=reward, info=info)
        rows.append(row)
        final_info = dict(info)
        if terminated or truncated:
            break

    env.close()
    best = min(rows, key=state_quality)
    return {
        "success": bool(final_info.get("success", False)),
        "simulated_steps": int(len(rows)),
        "simulated_seconds": float(len(rows) * float(cfg["env"]["timestep"]) * int(cfg["env"].get("frame_skip", 1))),
        "max_cart_abs": float(max(abs(float(row["x"])) for row in rows)),
        "action_abs_max": float(max(abs(float(row["action"])) for row in rows)),
        "best_upright_pass": best,
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "upright_event_count": int(sum(1 for row in rows if row["is_upright"])),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "final_info": final_info,
        "plan_events": plan_events,
        "trajectory_sample": rows[:10] + rows[-10:] if len(rows) > 20 else rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate receding-horizon random-shooting MPC from a replayed handoff state")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--state-json", default="runs/swingup6_expert_chain/swing_states_low_momentum.json")
    parser.add_argument("--state-index", default="best")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--out", default="runs/swingup6_mpc_capture/eval.json")
    parser.add_argument("--horizon-steps", type=int, default=30)
    parser.add_argument("--replan-steps", type=int, default=3)
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=0.65)
    parser.add_argument("--lqr-baseline", action="store_true")
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--lqr-scale", type=float, default=1.0)
    parser.add_argument("--lqr-target", type=float, default=0.0)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.horizon_steps < 1 or args.replan_steps < 1:
        raise ValueError("--horizon-steps and --replan-steps must be >= 1")
    if args.samples < 1 or args.elites < 1 or args.elites > args.samples:
        raise ValueError("--elites must be in 1..samples")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    selected_state, selected_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(base_cfg, selected_state, args.seconds)
    lqr_gain_arr = None
    if args.lqr_baseline:
        lqr_gain_arr = lqr_gain(cfg, progress=args.progress, fd_eps=1e-7, control_cost=args.lqr_control_cost)
    result = evaluate_mpc(
        cfg,
        progress=args.progress,
        seed=args.seed,
        seconds=args.seconds,
        horizon_steps=args.horizon_steps,
        replan_steps=args.replan_steps,
        samples=args.samples,
        elites=args.elites,
        iterations=args.iterations,
        sigma=args.sigma,
        lqr_gain_arr=lqr_gain_arr,
        lqr_scale=args.lqr_scale,
        lqr_target=args.lqr_target,
    )
    payload = {
        "generated_at": utc_timestamp(),
        "not_solution": not bool(result["success"]),
        "summary": "Receding-horizon random-shooting MPC capture diagnostic from a real swing handoff state.",
        "seed": int(args.seed),
        "progress": float(args.progress),
        "state_json": str(Path(args.state_json)),
        "state_index": selected_index,
        "selected_state": selected_state,
        "mpc": {
            "seconds": float(args.seconds),
            "horizon_steps": int(args.horizon_steps),
            "horizon_seconds": float(args.horizon_steps * float(cfg["env"]["timestep"]) * int(cfg["env"].get("frame_skip", 1))),
            "replan_steps": int(args.replan_steps),
            "samples": int(args.samples),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "sigma": float(args.sigma),
            "lqr_baseline": bool(args.lqr_baseline),
            "lqr_control_cost": float(args.lqr_control_cost),
            "lqr_scale": float(args.lqr_scale),
            "lqr_target": float(args.lqr_target),
        },
        **result,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    best = payload["best_upright_pass"]
    print(
        f"success={payload['success']} "
        f"streak={payload['max_upright_streak_seconds']:.3f}s "
        f"best_angle={best['max_abs_angle']:.6f} "
        f"hinge={best['hinge_velocity_rms']:.3f} "
        f"steps={payload['simulated_steps']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
