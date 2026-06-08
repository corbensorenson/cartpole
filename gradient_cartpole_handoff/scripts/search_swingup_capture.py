#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics
from probe_swingup_trajectory import (
    DEFAULT_KD,
    DEFAULT_KNOTS,
    DEFAULT_KP,
    DEFAULT_TRAJECTORY_SECONDS,
    trajectory_action,
)


def lqr_gain(cfg: dict[str, Any], *, progress: float, fd_eps: float, control_cost: float) -> np.ndarray:
    a, b = finite_difference_dynamics(cfg, progress, fd_eps)
    n = int(cfg["env"]["n_links"])
    q_weights = {
        "cart_position": 0.1,
        "absolute_angle": 100.0,
        "cart_velocity": 0.1,
        "absolute_angular_velocity": 1.0,
        "relative_angle": 1.0,
        "relative_angular_velocity": 0.01,
    }
    q = absolute_angle_cost(n, q_weights)
    r = np.array([[float(control_cost)]], dtype=np.float64)
    p = solve_discrete_are(a, b, q, r)
    return np.linalg.solve(b.T @ p @ b + r, b.T @ p @ a).reshape(-1)


def lqr_action(env: NLinkCartPoleEnv, gain: np.ndarray, *, scale: float, cart_target: float) -> float:
    n = env.n
    d = n + 1
    qpos = np.array(env.data.qpos, dtype=np.float64)
    qvel = np.array(env.data.qvel, dtype=np.float64)
    state = np.zeros(2 * d, dtype=np.float64)
    state[0] = qpos[0] - cart_target
    state[1 : 1 + n] = wrap_angle(qpos[1 : 1 + n])
    state[d:] = qvel
    return float(np.clip(-scale * float(gain @ state), -1.0, 1.0))


def vector_to_controller(vec: np.ndarray, feedforward_count: int) -> dict[str, Any]:
    return {
        "switch_time": float(np.clip(vec[0], 5.0, 7.0)),
        "lqr_scale": float(np.clip(vec[1], 0.0, 8.0)),
        "cart_target": float(np.clip(vec[2], -1.5, 1.5)),
        "blend_seconds": float(np.clip(vec[3], 0.0, 1.5)),
        "feedforward_seconds": float(np.clip(vec[4], 2.0, 10.0)),
        "feedforward_actions": np.clip(vec[5 : 5 + feedforward_count], -1.0, 1.0).astype(float).tolist(),
    }


def evaluate_controller(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    seconds: float,
    zero_noise: bool,
    gain: np.ndarray,
    controller: dict[str, Any],
) -> dict[str, Any]:
    cfg = {**cfg, "env": {**cfg["env"]}}
    if zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0

    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    _, reset_info = env.reset()
    threshold = float(
        cfg["env"].get(
            "success_upright_threshold",
            cfg["env"].get("reward", {}).get("upright_threshold", 0.10),
        )
    )

    switch_time = float(controller["switch_time"])
    blend_seconds = float(controller["blend_seconds"])
    ff_seconds = float(controller["feedforward_seconds"])
    ff_actions = np.asarray(controller["feedforward_actions"], dtype=np.float64)
    ff_times = np.linspace(0.0, ff_seconds, len(ff_actions), dtype=np.float64)

    steps = min(env.max_steps, int(seconds / env.dt))
    best_pass: dict[str, Any] | None = None
    done_events: list[dict[str, Any]] = []
    action_abs_max = 0.0
    max_cart_abs = abs(float(reset_info["x"]))
    post_switch_cost = 0.0
    post_switch_count = 0
    final_info: dict[str, Any] = {}

    for step in range(steps):
        t = step * env.dt
        swing_action = trajectory_action(env, t, DEFAULT_KNOTS, DEFAULT_TRAJECTORY_SECONDS, DEFAULT_KP, DEFAULT_KD)
        if t < switch_time:
            action = swing_action
        else:
            tau = max(0.0, t - switch_time)
            ff = float(np.interp(min(tau, ff_seconds), ff_times, ff_actions))
            capture_action = float(
                np.clip(
                    lqr_action(
                        env,
                        gain,
                        scale=float(controller["lqr_scale"]),
                        cart_target=float(controller["cart_target"]),
                    )
                    + ff,
                    -1.0,
                    1.0,
                )
            )
            if blend_seconds > 0.0:
                blend = float(np.clip(tau / blend_seconds, 0.0, 1.0))
                action = float(np.clip((1.0 - blend) * swing_action + blend * capture_action, -1.0, 1.0))
            else:
                action = capture_action

        action_abs_max = max(action_abs_max, abs(action))
        _, reward, terminated, truncated, info = env.step([action])
        rel, abs_angles = env._angles()
        hinge_rms = float(np.sqrt(np.mean(env.data.qvel[1 : 1 + env.n] ** 2)))
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        row = {
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "reward": float(reward),
            "action": float(action),
            "x": float(info["x"]),
            "cart_velocity": float(env.data.qvel[0]),
            "qpos": np.array(env.data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.array(env.data.qvel, dtype=np.float64).astype(float).tolist(),
            "max_abs_angle": float(info["max_abs_angle"]),
            "mean_abs_angle": float(info["mean_abs_angle"]),
            "hinge_velocity_rms": hinge_rms,
            "relative_angles": rel.astype(float).tolist(),
            "absolute_angles": abs_angles.astype(float).tolist(),
            "is_upright": bool(info["is_upright"]),
            "upright_streak_seconds": float(info["upright_streak_seconds"]),
            "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
            "time_to_first_upright": info["time_to_first_upright"],
        }
        if best_pass is None or row["max_abs_angle"] < best_pass["max_abs_angle"]:
            best_pass = row
        if t >= switch_time:
            angle_violation = max(0.0, row["max_abs_angle"] - threshold)
            post_switch_cost += (
                angle_violation * angle_violation
                + 0.02 * row["mean_abs_angle"]
                + 0.001 * hinge_rms * hinge_rms
                + 0.002 * (row["x"] / float(cfg["env"]["rail_limit"])) ** 2
            )
            post_switch_count += 1
        final_info = dict(info)
        if terminated or truncated:
            done_events.append(
                {
                    "step": int(step + 1),
                    "time_seconds": float((step + 1) * env.dt),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success": bool(info.get("success", False)),
                    "x": float(info["x"]),
                    "max_abs_angle": float(info["max_abs_angle"]),
                    "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
                }
            )
            break

    env.close()
    assert best_pass is not None
    success = bool(final_info.get("success", False))
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    ever_upright = final_info.get("time_to_first_upright") is not None
    avg_post_switch_cost = post_switch_cost / max(1, post_switch_count)
    score = (
        -1000.0 * float(success)
        - 200.0 * max_streak
        - 50.0 * float(ever_upright)
        + 10.0 * avg_post_switch_cost
        + 0.05 * max_cart_abs
        + 0.02 * float(final_info.get("max_abs_angle", 0.0))
    )
    if not ever_upright:
        score += 50.0
    if max_cart_abs > float(cfg["env"]["rail_limit"]):
        score += 100.0 * (max_cart_abs - float(cfg["env"]["rail_limit"]))

    return {
        "score": float(score),
        "success": success,
        "ever_upright": bool(ever_upright),
        "simulated_steps": int(step + 1),
        "simulated_seconds": float((step + 1) * env.dt),
        "max_cart_abs": float(max_cart_abs),
        "action_abs_max": float(action_abs_max),
        "avg_post_switch_cost": float(avg_post_switch_cost),
        "best_upright_pass": best_pass,
        "done_events": done_events,
        "final_info": final_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search LQR-plus-feedforward capture handoffs after hanging-start swing-up")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default="runs/swingup6_capture_search/search.json")
    parser.add_argument("--zero-noise", action="store_true", help="Search the exact hanging state by overriding init noise to zero")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--feedforward-count", type=int, default=12)
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--control-cost", type=float, default=1000.0)
    parser.add_argument("--sigma-decay", type=float, default=0.85)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.population < 1:
        raise ValueError("--population must be >= 1")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("--elites must be between 1 and --population")
    if args.feedforward_count < 2:
        raise ValueError("--feedforward-count must be >= 2")

    cfg = apply_overrides(load_config(args.config), args.override)
    gain = lqr_gain(cfg, progress=args.progress, fd_eps=args.fd_eps, control_cost=args.control_cost)
    rng = np.random.default_rng(args.seed)

    center = np.concatenate(
        [
            np.asarray([5.84, 1.0, 0.0, 0.20, 6.0], dtype=np.float64),
            np.zeros(args.feedforward_count, dtype=np.float64),
        ]
    )
    sigma = np.concatenate(
        [
            np.asarray([0.25, 0.60, 0.55, 0.25, 1.0], dtype=np.float64),
            np.full(args.feedforward_count, 0.35, dtype=np.float64),
        ]
    )

    best: dict[str, Any] | None = None
    best_by_max_streak: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for iteration in range(max(1, args.iterations + 1)):
        candidates = [center] if iteration == 0 else [center + rng.normal(0.0, sigma) for _ in range(args.population)]
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            controller = vector_to_controller(candidate, args.feedforward_count)
            metrics = evaluate_controller(
                cfg,
                progress=args.progress,
                seed=args.seed,
                seconds=args.seconds,
                zero_noise=args.zero_noise,
                gain=gain,
                controller=controller,
            )
            records.append({"score": metrics["score"], "controller": controller, "metrics": metrics})

        records.sort(key=lambda row: float(row["score"]))
        if best is None or float(records[0]["score"]) < float(best["score"]):
            best = records[0]
        for row in records:
            row_streak = float(row["metrics"]["final_info"].get("max_upright_streak_seconds", 0.0))
            if best_by_max_streak is None:
                best_by_max_streak = row
            else:
                incumbent_streak = float(best_by_max_streak["metrics"]["final_info"].get("max_upright_streak_seconds", 0.0))
                if (row_streak, -float(row["score"])) > (incumbent_streak, -float(best_by_max_streak["score"])):
                    best_by_max_streak = row

        best_metrics = records[0]["metrics"]
        history.append(
            {
                "iteration": int(iteration),
                "score": float(records[0]["score"]),
                "success": bool(best_metrics["success"]),
                "max_upright_streak_seconds": float(best_metrics["final_info"].get("max_upright_streak_seconds", 0.0)),
                "time_to_first_upright": best_metrics["final_info"].get("time_to_first_upright"),
                "best_max_abs_angle": float(best_metrics["best_upright_pass"]["max_abs_angle"]),
                "max_cart_abs": float(best_metrics["max_cart_abs"]),
            }
        )
        print(
            f"iter={iteration:03d} score={records[0]['score']:.6f} "
            f"streak={best_metrics['final_info'].get('max_upright_streak_seconds', 0.0):.3f}s "
            f"angle={best_metrics['best_upright_pass']['max_abs_angle']:.6f} "
            f"cart={best_metrics['max_cart_abs']:.3f} "
            f"success={best_metrics['success']}"
        )

        if iteration > 0:
            elite_vectors = []
            for row in records[: args.elites]:
                ctrl = row["controller"]
                elite_vectors.append(
                    np.concatenate(
                        [
                            np.asarray(
                                [
                                    ctrl["switch_time"],
                                    ctrl["lqr_scale"],
                                    ctrl["cart_target"],
                                    ctrl["blend_seconds"],
                                    ctrl["feedforward_seconds"],
                                ],
                                dtype=np.float64,
                            ),
                            np.asarray(ctrl["feedforward_actions"], dtype=np.float64),
                        ]
                    )
                )
            elite_arr = np.asarray(elite_vectors, dtype=np.float64)
            center = elite_arr.mean(axis=0)
            sigma = np.maximum(elite_arr.std(axis=0), 1e-3) * float(args.sigma_decay)

    assert best is not None
    assert best_by_max_streak is not None
    result = {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Capture search after the hanging-start pass; success still requires held-out eval/video/checkpoint evidence.",
        "seed": int(args.seed),
        "progress": float(args.progress),
        "zero_noise": bool(args.zero_noise),
        "search": {
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "feedforward_count": int(args.feedforward_count),
            "fd_eps": float(args.fd_eps),
            "control_cost": float(args.control_cost),
            "sigma_decay": float(args.sigma_decay),
        },
        "lqr": {
            "gain": gain.astype(float).tolist(),
            "control_cost": float(args.control_cost),
        },
        "best": best,
        "best_by": {
            "score": best,
            "max_upright_streak": best_by_max_streak,
        },
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(result, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    # Import check: this script computes a MuJoCo LQR and uses mj_forward inside
    # finite_difference_dynamics, so keep the dependency explicit for failures.
    if mujoco is None:  # pragma: no cover
        raise RuntimeError("mujoco import failed")
    main()
