#!/usr/bin/env python
"""Search tail actions using reset-free downstream LQR capture value.

Unlike the earlier endpoint-only tail search, this evaluates the capture
controller from several actual states along every candidate tail.  The chosen
state is therefore a real point on the uninterrupted swing trajectory, not a
planned state written back into the live environment.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from mujoco import rollout as mujoco_rollout

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from search_swingup_capture import lqr_action, lqr_gain
from search_swingup_tail_action_cem import (
    interpolation_matrix,
    load_controller,
    load_tail_center,
    physical_metrics,
    replay_to_tail,
    score_batch,
)


def capture_cost(info: dict[str, Any], env: NLinkCartPoleEnv) -> float:
    return float(
        35.0 * (float(info["max_abs_angle"]) / 0.15) ** 2
        + 14.0 * (float(info["hinge_velocity_rms"]) / 0.75) ** 2
        + 4.0 * (abs(float(info["x"])) / 1.25) ** 2
        + 4.0 * (abs(float(env.data.qvel[0])) / 0.50) ** 2
    )


def evaluate_lqr_from_state(
    env: NLinkCartPoleEnv,
    qpos: np.ndarray,
    qvel: np.ndarray,
    gain: np.ndarray,
    *,
    scale: float,
    steps: int,
) -> dict[str, float]:
    env.reset(seed=0, options={"qpos": qpos, "qvel": qvel})
    best_cost = float("inf")
    terminal_cost = float("inf")
    max_streak = 0.0
    final_info: dict[str, Any] = {}
    for _ in range(steps):
        action = lqr_action(env, gain, scale=scale, cart_target=0.0)
        _, _, terminated, truncated, info = env.step([action])
        value = capture_cost(info, env)
        best_cost = min(best_cost, value)
        max_streak = max(max_streak, float(info.get("max_upright_streak_seconds", 0.0)))
        final_info = info
        if terminated or truncated:
            break
    terminal_cost = capture_cost(final_info, env) if final_info else float("inf")
    return {
        "capture_cost": float(0.65 * best_cost + 0.35 * terminal_cost - 1500.0 * max_streak),
        "capture_best_cost": float(best_cost),
        "capture_terminal_cost": float(terminal_cost),
        "capture_max_streak": float(max_streak),
        "capture_final_angle": float(final_info.get("max_abs_angle", np.inf)),
        "capture_final_hinge": float(final_info.get("hinge_velocity_rms", np.inf)),
        "capture_final_cart": abs(float(final_info.get("x", np.inf))),
        "capture_final_cart_velocity": abs(float(env.data.qvel[0])) if final_info else float("inf"),
    }


def evaluate_candidate_capture(
    env: NLinkCartPoleEnv,
    qpos: np.ndarray,
    qvel: np.ndarray,
    gain: np.ndarray,
    *,
    scale: float,
    capture_steps: int,
    candidate_stride: int,
    min_candidate_step: int,
) -> dict[str, float | int]:
    best: dict[str, float | int] | None = None
    for source_step in range(min_candidate_step, qpos.shape[0], max(1, candidate_stride)):
        result = evaluate_lqr_from_state(
            env,
            qpos[source_step],
            qvel[source_step],
            gain,
            scale=scale,
            steps=capture_steps,
        )
        result["source_step"] = int(source_step)
        if best is None or float(result["capture_cost"]) < float(best["capture_cost"]):
            best = result
    if best is None:
        raise ValueError("tail did not contain a candidate capture state")
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--swing-controller-json", required=True)
    parser.add_argument("--controller-key", default=None)
    parser.add_argument("--init-tail-json", default=None)
    parser.add_argument("--init-tail-key", default="best")
    parser.add_argument("--tail-start-seconds", type=float, default=11.0)
    parser.add_argument("--tail-seconds", type=float, default=5.0)
    parser.add_argument("--capture-seconds", type=float, default=1.5)
    parser.add_argument("--candidate-stride", type=int, default=10)
    parser.add_argument("--min-candidate-seconds", type=float, default=0.20)
    parser.add_argument("--lqr-scale", type=float, default=1.0)
    parser.add_argument("--control-cost", type=float, default=1000.0)
    parser.add_argument("--knot-count", type=int, default=32)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--action-sigma", type=float, default=0.12)
    parser.add_argument("--sigma-decay", type=float, default=0.92)
    parser.add_argument("--sigma-floor", type=float, default=0.005)
    parser.add_argument("--physical-weight", type=float, default=0.20)
    parser.add_argument("--capture-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.knot_count < 2 or args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "obs_include_capture_features": False,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0

    controller = load_controller(args.swing_controller_json, args.controller_key)
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    initial_state = replay_to_tail(env, controller, args.tail_start_seconds)
    horizon_steps = max(2, int(round(args.tail_seconds / env.dt)))
    capture_steps = max(1, int(round(args.capture_seconds / env.dt)))
    interpolation = interpolation_matrix(args.knot_count, horizon_steps)
    pool = [mujoco.MjData(env.model) for _ in range(min(32, max(1, args.population // 16)))]
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=args.control_cost)
    rng = np.random.default_rng(args.seed)
    center = load_tail_center(args.init_tail_json, args.knot_count, args.init_tail_key)
    sigma = np.full(args.knot_count, args.action_sigma, dtype=np.float64)
    best_record: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()

    for iteration in range(args.iterations):
        knots = np.clip(
            center[None, :] + rng.normal(0.0, sigma, size=(args.population, args.knot_count)),
            -1.0,
            1.0,
        )
        knots[0] = center
        physical_cost, metrics = score_batch(
            env,
            initial_state,
            knots,
            interpolation,
            pool,
            min_tail_steps=max(1, int(round(0.20 / env.dt))),
            angle_weight=15.0,
            hinge_weight=100.0,
            max_hinge_weight=12.0,
            absolute_velocity_weight=0.0,
            cart_weight=1.5,
            cart_velocity_weight=2.0,
            rail_penalty_limit=2.90,
            rail_penalty_weight=20_000.0,
            best_score_weight=0.55,
            terminal_score_weight=0.25,
            tail_average_weight=0.20,
            robust_window_steps=0,
        )
        capture_rows = [
            evaluate_candidate_capture(
                env,
                metrics["qpos"][index],
                metrics["qvel"][index],
                gain,
                scale=args.lqr_scale,
                capture_steps=capture_steps,
                candidate_stride=args.candidate_stride,
                min_candidate_step=max(1, int(round(args.min_candidate_seconds / env.dt))),
            )
            for index in range(len(knots))
        ]
        capture_cost = np.asarray([float(row["capture_cost"]) for row in capture_rows])
        cost = args.physical_weight * physical_cost + args.capture_weight * capture_cost
        order = np.argsort(cost)
        top = int(order[0])
        elite_knots = knots[order[: args.elites]]
        center = np.mean(elite_knots, axis=0)
        sigma = np.maximum(np.std(elite_knots, axis=0) * args.sigma_decay, args.sigma_floor)
        capture = capture_rows[top]
        source_step = int(capture["source_step"])
        record = {
            "iteration": iteration + 1,
            "cost": float(cost[top]),
            "physical_cost": float(physical_cost[top]),
            "capture_cost": float(capture["capture_cost"]),
            "capture_max_streak": float(capture["capture_max_streak"]),
            "capture_best_cost": float(capture["capture_best_cost"]),
            "source_step": source_step,
            "source_time_seconds": float((source_step + 1) * env.dt),
            "angle": float(metrics["max_angle"][top, source_step]),
            "hinge_rms": float(metrics["hinge_rms"][top, source_step]),
            "max_hinge": float(metrics["max_hinge"][top, source_step]),
            "cart_abs": float(metrics["qpos"][top, source_step, 0]),
            "cart_velocity": float(metrics["qvel"][top, source_step, 0]),
            "capture_final_angle": float(capture["capture_final_angle"]),
            "capture_final_hinge": float(capture["capture_final_hinge"]),
            "capture_final_cart": float(capture["capture_final_cart"]),
            "capture_final_cart_velocity": float(capture["capture_final_cart_velocity"]),
            "rail": float(metrics["rail"][top]),
        }
        candidate = {
            **record,
            "knots": knots[top].astype(float).tolist(),
            "state": {
                "qpos": metrics["qpos"][top, source_step].astype(float).tolist(),
                "qvel": metrics["qvel"][top, source_step].astype(float).tolist(),
                "absolute_angles": metrics["absolute_angles"][top, source_step].astype(float).tolist(),
            },
        }
        history.append(record)
        if best_record is None or candidate["cost"] < best_record["cost"]:
            best_record = candidate
        feasible = bool(
            record["angle"] <= 0.15
            and record["hinge_rms"] <= 0.75
            and abs(record["cart_abs"]) <= 1.25
            and abs(record["cart_velocity"]) <= 0.50
        )
        if feasible and (best_feasible is None or candidate["cost"] < best_feasible["cost"]):
            best_feasible = candidate
        print(
            f"iter={iteration + 1:03d} cost={record['cost']:.3f} "
            f"capture={record['capture_cost']:.3f} streak={record['capture_max_streak']:.3f}s "
            f"t={record['source_time_seconds']:.2f}s angle={record['angle']:.3f} "
            f"hinge={record['hinge_rms']:.3f} x={record['cart_abs']:.3f} "
            f"final_angle={record['capture_final_angle']:.3f}",
            flush=True,
        )

    assert best_record is not None
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact-MuJoCo tail CEM ranked by LQR capture from every sampled actual tail state.",
        "source_controller": controller,
        "source_controller_json": str(args.swing_controller_json),
        "controller_key": args.controller_key,
        "init_tail_json": args.init_tail_json,
        "init_tail_key": args.init_tail_key,
        "tail_start_seconds": float(args.tail_start_seconds),
        "tail_horizon_steps": int(horizon_steps),
        "tail_horizon_seconds": float(horizon_steps * env.dt),
        "capture_horizon_steps": int(capture_steps),
        "capture_horizon_seconds": float(capture_steps * env.dt),
        "capture_candidate_stride": int(args.candidate_stride),
        "lqr": {"scale": float(args.lqr_scale), "control_cost": float(args.control_cost), "gain": gain.astype(float).tolist()},
        "search": {
            "seed": int(args.seed),
            "knot_count": int(args.knot_count),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "physical_weight": float(args.physical_weight),
            "capture_weight": float(args.capture_weight),
            "wall_time_seconds": float(time.time() - started),
        },
        "best": best_record,
        "best_feasible": best_feasible,
        "history": history,
        "initial_state": {
            "qpos": initial_state[1 : 1 + env.model.nq].astype(float).tolist(),
            "qvel": initial_state[1 + env.model.nq : 1 + env.model.nq + env.model.nv].astype(float).tolist(),
        },
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(result, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
