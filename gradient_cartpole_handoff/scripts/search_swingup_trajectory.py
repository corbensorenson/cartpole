#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from probe_swingup_trajectory import (
    DEFAULT_KD,
    DEFAULT_KNOTS,
    DEFAULT_KP,
    DEFAULT_TRAJECTORY_SECONDS,
    trajectory_action,
)


def evaluate_controller(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    seconds: float,
    zero_noise: bool,
    knots: np.ndarray,
    trajectory_seconds: float,
    kp: float,
    kd: float,
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
    steps = min(env.max_steps, int(seconds / env.dt))

    best: dict[str, Any] | None = None
    upright_event_count = 0
    action_abs_max = 0.0
    max_cart_abs = abs(float(reset_info["x"]))
    final_info: dict[str, Any] = {}
    score = float("inf")

    for step in range(steps):
        t = step * env.dt
        action = trajectory_action(env, t, knots, trajectory_seconds, kp, kd)
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

        angle_gap = max(0.0, row["max_abs_angle"] - threshold)
        row_score = (
            50.0 * angle_gap
            + row["mean_abs_angle"]
            + 0.55 * row["hinge_velocity_rms"]
            + 0.08 * abs(row["x"])
            + 0.04 * abs(row["cart_velocity"])
        )
        if row["max_abs_angle"] < threshold:
            row_score -= 0.50
        if row_score < score:
            score = row_score
            best = row
        if info["is_upright"]:
            upright_event_count += 1
        final_info = dict(info)
        if terminated or truncated:
            break

    env.close()
    assert best is not None
    success = bool(final_info.get("success", False))
    return {
        "score": float(score),
        "success": success,
        "upright_event_count": int(upright_event_count),
        "max_cart_abs": float(max_cart_abs),
        "action_abs_max": float(action_abs_max),
        "simulated_steps": int(step + 1),
        "simulated_seconds": float(
            (step + 1) * (float(cfg["env"]["timestep"]) * int(cfg["env"].get("frame_skip", 1)))
        ),
        "best_upright_pass": best,
        "final_info": final_info,
    }


def vector_to_controller(vec: np.ndarray, knot_count: int, rail_target_limit: float) -> tuple[np.ndarray, float, float, float]:
    knots = np.concatenate([[0.0], np.clip(vec[: knot_count - 1], -rail_target_limit, rail_target_limit)])
    kp = float(np.clip(vec[knot_count - 1], 0.05, 3.0))
    kd = float(np.clip(vec[knot_count], 0.0, 2.0))
    trajectory_seconds = float(np.clip(vec[knot_count + 1], 4.0, 16.0))
    return knots, kp, kd, trajectory_seconds


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search fixed cart-position PD trajectories for six-link hanging-start reachability"
    )
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default="runs/swingup6_trajectory_search/search.json")
    parser.add_argument("--zero-noise", action="store_true", help="Search the exact hanging state by overriding init noise to zero")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--rail-target-limit", type=float, default=2.6)
    parser.add_argument("--knot-sigma", type=float, default=0.45)
    parser.add_argument("--gain-sigma", type=float, default=0.20)
    parser.add_argument("--time-sigma", type=float, default=0.75)
    parser.add_argument("--sigma-decay", type=float, default=0.80)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.population < 1:
        raise ValueError("--population must be >= 1")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("--elites must be between 1 and --population")

    cfg = apply_overrides(load_config(args.config), args.override)
    rng = np.random.default_rng(args.seed)
    knot_count = len(DEFAULT_KNOTS)
    center = np.concatenate([DEFAULT_KNOTS[1:], [DEFAULT_KP, DEFAULT_KD, DEFAULT_TRAJECTORY_SECONDS]]).astype(np.float64)
    sigma = np.concatenate(
        [
            np.full(knot_count - 1, args.knot_sigma, dtype=np.float64),
            np.asarray([args.gain_sigma, args.gain_sigma, args.time_sigma], dtype=np.float64),
        ]
    )

    best_record: dict[str, Any] | None = None
    best_by_max_streak: dict[str, Any] | None = None
    best_by_min_angle: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for iteration in range(max(1, args.iterations + 1)):
        candidates = [center] if iteration == 0 else [center + rng.normal(0.0, sigma) for _ in range(args.population)]
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            knots, kp, kd, trajectory_seconds = vector_to_controller(candidate, knot_count, args.rail_target_limit)
            metrics = evaluate_controller(
                cfg,
                progress=args.progress,
                seed=args.seed,
                seconds=args.seconds,
                zero_noise=args.zero_noise,
                knots=knots,
                trajectory_seconds=trajectory_seconds,
                kp=kp,
                kd=kd,
            )
            records.append(
                {
                    "score": metrics["score"],
                    "controller": {
                        "type": "cart_position_pd_fixed_knots",
                        "trajectory_seconds": float(trajectory_seconds),
                        "kp": float(kp),
                        "kd": float(kd),
                        "knots": knots.astype(float).tolist(),
                    },
                    "metrics": metrics,
                }
            )

        records.sort(key=lambda row: float(row["score"]))
        if best_record is None or float(records[0]["score"]) < float(best_record["score"]):
            best_record = records[0]
        for row in records:
            row_pass = row["metrics"]["best_upright_pass"]
            row_streak = float(row["metrics"]["final_info"].get("max_upright_streak_seconds", 0.0))
            if best_by_max_streak is None:
                best_by_max_streak = row
            else:
                incumbent_pass = best_by_max_streak["metrics"]["best_upright_pass"]
                incumbent_streak = float(best_by_max_streak["metrics"]["final_info"].get("max_upright_streak_seconds", 0.0))
                if (
                    row_streak,
                    -float(row_pass["max_abs_angle"]),
                    -float(row["score"]),
                ) > (
                    incumbent_streak,
                    -float(incumbent_pass["max_abs_angle"]),
                    -float(best_by_max_streak["score"]),
                ):
                    best_by_max_streak = row
            if best_by_min_angle is None:
                best_by_min_angle = row
            else:
                incumbent_pass = best_by_min_angle["metrics"]["best_upright_pass"]
                if (
                    float(row_pass["max_abs_angle"]),
                    float(row["score"]),
                ) < (
                    float(incumbent_pass["max_abs_angle"]),
                    float(best_by_min_angle["score"]),
                ):
                    best_by_min_angle = row

        best_pass = records[0]["metrics"]["best_upright_pass"]
        history.append(
            {
                "iteration": int(iteration),
                "score": float(records[0]["score"]),
                "max_abs_angle": float(best_pass["max_abs_angle"]),
                "hinge_velocity_rms": float(best_pass["hinge_velocity_rms"]),
                "max_upright_streak_seconds": float(
                    records[0]["metrics"]["final_info"].get("max_upright_streak_seconds", 0.0)
                ),
                "success": bool(records[0]["metrics"]["success"]),
            }
        )
        print(
            f"iter={iteration:03d} score={records[0]['score']:.6f} "
            f"angle={best_pass['max_abs_angle']:.6f} "
            f"hinge_rms={best_pass['hinge_velocity_rms']:.3f} "
            f"streak={records[0]['metrics']['final_info'].get('max_upright_streak_seconds', 0.0):.3f}s "
            f"success={records[0]['metrics']['success']}"
        )

        if iteration > 0:
            elite_vectors = []
            for row in records[: args.elites]:
                controller = row["controller"]
                elite_vectors.append(
                    np.concatenate(
                        [
                            np.asarray(controller["knots"][1:], dtype=np.float64),
                            [controller["kp"], controller["kd"], controller["trajectory_seconds"]],
                        ]
                    )
                )
            elite_arr = np.asarray(elite_vectors, dtype=np.float64)
            center = elite_arr.mean(axis=0)
            sigma = np.maximum(elite_arr.std(axis=0), 1e-3) * args.sigma_decay

    assert best_record is not None
    assert best_by_max_streak is not None
    assert best_by_min_angle is not None
    result = {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "CEM search for reachability/capture candidates; success still requires the separate held-out eval and video gates.",
        "seed": int(args.seed),
        "progress": float(args.progress),
        "zero_noise": bool(args.zero_noise),
        "search": {
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "rail_target_limit": float(args.rail_target_limit),
            "knot_sigma": float(args.knot_sigma),
            "gain_sigma": float(args.gain_sigma),
            "time_sigma": float(args.time_sigma),
            "sigma_decay": float(args.sigma_decay),
        },
        "best": best_record,
        "best_by": {
            "score": best_record,
            "max_upright_streak": best_by_max_streak,
            "min_best_pass_angle": best_by_min_angle,
        },
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(result, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
