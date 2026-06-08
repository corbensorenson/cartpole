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
from probe_swingup_trajectory import (
    DEFAULT_KD,
    DEFAULT_KNOTS,
    DEFAULT_KP,
    DEFAULT_TRAJECTORY_SECONDS,
    trajectory_action,
)


def state_quality(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(row.get("max_abs_angle", np.pi)),
        float(row.get("hinge_velocity_rms", np.inf)),
        abs(float(row.get("cart_velocity", 0.0))),
        abs(float(row.get("x", 0.0))),
        -float(row.get("capture_quality", 0.0)),
    )


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


def controller_from_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(Path(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "best_by" in payload and "score" in payload["best_by"]:
        return dict(payload["best_by"]["score"]["controller"])
    if isinstance(payload, dict) and "best" in payload:
        return dict(payload["best"]["controller"])
    if isinstance(payload, dict) and "controller" in payload:
        return dict(payload["controller"])
    if isinstance(payload, dict) and ("action_knots" in payload or "knots" in payload):
        return dict(payload)
    raise ValueError(f"Could not load controller from {path}")


def center_actions_from_cart_pd(cfg: dict[str, Any], *, progress: float, seed: int, seconds: float, action_count: int) -> np.ndarray:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    env.reset()
    action_times = np.linspace(0.0, seconds, action_count, dtype=np.float64)
    actions: list[float] = []
    sample_idx = 0
    last_action = 0.0
    steps = min(env.max_steps, int(seconds / env.dt))
    for step in range(steps):
        t = step * env.dt
        action = trajectory_action(env, t, DEFAULT_KNOTS, DEFAULT_TRAJECTORY_SECONDS, DEFAULT_KP, DEFAULT_KD)
        while sample_idx < len(action_times) and t >= action_times[sample_idx]:
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


def initial_actions(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    seconds: float,
    action_count: int,
    init_controller_json: str | None,
) -> np.ndarray:
    controller = controller_from_json(init_controller_json)
    if controller is None:
        return center_actions_from_cart_pd(cfg, progress=progress, seed=seed, seconds=seconds, action_count=action_count)
    if "action_knots" in controller:
        src = np.asarray(controller["action_knots"], dtype=np.float64)
        src_times = np.linspace(0.0, float(controller.get("seconds", seconds)), len(src), dtype=np.float64)
        dst_times = np.linspace(0.0, seconds, action_count, dtype=np.float64)
        return np.interp(dst_times, src_times, src)
    if "knots" in controller:
        cart_cfg = {
            **controller,
            "trajectory_seconds": float(controller.get("trajectory_seconds", DEFAULT_TRAJECTORY_SECONDS)),
            "kp": float(controller.get("kp", DEFAULT_KP)),
            "kd": float(controller.get("kd", DEFAULT_KD)),
        }
        env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
        env.reset()
        action_times = np.linspace(0.0, seconds, action_count, dtype=np.float64)
        actions: list[float] = []
        sample_idx = 0
        last_action = 0.0
        steps = min(env.max_steps, int(seconds / env.dt))
        knots = np.asarray(cart_cfg["knots"], dtype=np.float64)
        for step in range(steps):
            t = step * env.dt
            action = trajectory_action(env, t, knots, cart_cfg["trajectory_seconds"], cart_cfg["kp"], cart_cfg["kd"])
            while sample_idx < len(action_times) and t >= action_times[sample_idx]:
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
    raise ValueError(f"Unsupported controller type in {init_controller_json}")


def evaluate_actions(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    seconds: float,
    zero_noise: bool,
    action_knots: np.ndarray,
    score_mode: str,
    handoff_min_time: float,
    handoff_max_cart_abs: float,
    collect_states: bool,
    state_max_angle: float,
    state_max_hinge_rms: float,
    state_max_cart_abs: float,
    state_max_cart_velocity: float,
) -> dict[str, Any]:
    cfg = {**cfg, "env": {**cfg["env"]}}
    if zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    _, reset_info = env.reset()
    threshold = float(cfg["env"].get("success_upright_threshold", cfg["env"].get("reward", {}).get("upright_threshold", 0.15)))
    steps = min(env.max_steps, int(seconds / env.dt))
    action_times = np.linspace(0.0, seconds, len(action_knots), dtype=np.float64)

    best: dict[str, Any] | None = None
    best_row_score = float("inf")
    final_info: dict[str, Any] = {}
    upright_event_count = 0
    max_cart_abs = abs(float(reset_info["x"]))
    action_abs_max = 0.0
    action_smooth_cost = 0.0
    candidate_states: list[dict[str, Any]] = []

    previous_action = 0.0
    terminated = False
    truncated = False
    for step in range(steps):
        t = step * env.dt
        action = float(np.interp(min(t, seconds), action_times, action_knots))
        action = float(np.clip(action, -1.0, 1.0))
        action_abs_max = max(action_abs_max, abs(action))
        action_smooth_cost += (action - previous_action) ** 2
        previous_action = action
        _, reward, terminated, truncated, info = env.step([action])
        row = row_from_env(env, step=step + 1, action=action, reward=reward, info=info)
        max_cart_abs = max(max_cart_abs, abs(float(row["x"])))
        angle_gap = max(0.0, float(row["max_abs_angle"]) - threshold)
        cart_over = max(0.0, abs(float(row["x"])) - handoff_max_cart_abs)
        time_shortfall = max(0.0, handoff_min_time - float(row["time_seconds"]))
        if score_mode == "low_momentum":
            row_score = (
                260.0 * angle_gap
                + 8.0 * float(row["max_abs_angle"])
                + 3.0 * float(row["hinge_velocity_rms"])
                + 1.2 * abs(float(row["cart_velocity"]))
                + 0.9 * abs(float(row["x"]))
                + 16.0 * cart_over * cart_over
                + 25.0 * time_shortfall
                - 5.0 * float(row["is_upright"])
                - 2.5 * float(row["capture_quality"])
                - 1.2 * float(row["max_upright_streak_seconds"])
            )
        elif score_mode == "sustain":
            row_score = (
                180.0 * angle_gap
                + 4.0 * float(row["max_abs_angle"])
                + 1.5 * float(row["hinge_velocity_rms"])
                + 0.8 * abs(float(row["cart_velocity"]))
                + 0.5 * abs(float(row["x"]))
                + 10.0 * cart_over * cart_over
                - 4.0 * float(row["upright_streak_seconds"])
                - 2.5 * float(row["capture_quality"])
            )
        else:
            row_score = (
                60.0 * angle_gap
                + float(row["mean_abs_angle"])
                + 0.5 * float(row["hinge_velocity_rms"])
                + 0.1 * abs(float(row["x"]))
                + 0.05 * abs(float(row["cart_velocity"]))
                - 0.75 * float(row["is_upright"])
            )
        if row_score < best_row_score:
            best_row_score = float(row_score)
            best = row
        if bool(row["is_upright"]):
            upright_event_count += 1
        if (
            collect_states
            and float(row["time_seconds"]) >= handoff_min_time
            and float(row["max_abs_angle"]) <= state_max_angle
            and float(row["hinge_velocity_rms"]) <= state_max_hinge_rms
            and abs(float(row["x"])) <= state_max_cart_abs
            and abs(float(row["cart_velocity"])) <= state_max_cart_velocity
        ):
            candidate_states.append(row)
        final_info = dict(info)
        if terminated or truncated:
            break

    env.close()
    assert best is not None
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    success = bool(final_info.get("success", False))
    ever_upright = final_info.get("time_to_first_upright") is not None
    terminal_penalty = 0.0
    if terminated and not success:
        terminal_penalty = 800.0
    if score_mode == "sustain":
        score = (
            -25000.0 * float(success)
            - 3500.0 * max_streak
            - 250.0 * float(ever_upright)
            - 2.0 * float(upright_event_count)
            + 35.0 * float(best["max_abs_angle"])
            + 18.0 * float(best["hinge_velocity_rms"])
            + 7.0 * abs(float(best["cart_velocity"]))
            + 4.0 * abs(float(best["x"]))
            + 40.0 * max(0.0, max_cart_abs - float(cfg["env"]["rail_limit"])) ** 2
            + 0.02 * action_smooth_cost
            + terminal_penalty
        )
    elif score_mode == "low_momentum":
        score = best_row_score + 0.01 * action_smooth_cost + 35.0 * max(0.0, max_cart_abs - float(cfg["env"]["rail_limit"])) ** 2
    else:
        score = best_row_score + 0.01 * action_smooth_cost

    candidate_states.sort(key=state_quality)
    return {
        "score": float(score),
        "success": success,
        "ever_upright": bool(ever_upright),
        "simulated_steps": int(step + 1),
        "simulated_seconds": float((step + 1) * env.dt),
        "upright_event_count": int(upright_event_count),
        "max_cart_abs": float(max_cart_abs),
        "action_abs_max": float(action_abs_max),
        "action_smooth_cost": float(action_smooth_cost),
        "best_upright_pass": best,
        "candidate_states": candidate_states,
        "final_info": final_info,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search direct open-loop force sequences for hanging-start swing-up handoffs")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--out", default="runs/swingup6_action_search/search.json")
    parser.add_argument("--state-out", default=None)
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--init-controller-json", default=None)
    parser.add_argument("--score-mode", choices=["reachability", "low_momentum", "sustain"], default="low_momentum")
    parser.add_argument("--handoff-min-time", type=float, default=4.0)
    parser.add_argument("--handoff-max-cart-abs", type=float, default=1.25)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--action-count", type=int, default=81)
    parser.add_argument("--action-sigma", type=float, default=0.45)
    parser.add_argument("--sigma-decay", type=float, default=0.84)
    parser.add_argument("--state-max-angle", type=float, default=0.35)
    parser.add_argument("--state-max-hinge-rms", type=float, default=2.5)
    parser.add_argument("--state-max-cart-abs", type=float, default=2.75)
    parser.add_argument("--state-max-cart-velocity", type=float, default=2.0)
    parser.add_argument("--keep-best-states", type=int, default=96)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.population < 1:
        raise ValueError("--population must be >= 1")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("--elites must be in 1..population")
    if args.action_count < 2:
        raise ValueError("--action-count must be >= 2")

    cfg = apply_overrides(load_config(args.config), args.override)
    rng = np.random.default_rng(args.seed)
    center = np.clip(
        initial_actions(
            cfg,
            progress=args.progress,
            seed=args.seed,
            seconds=args.seconds,
            action_count=args.action_count,
            init_controller_json=args.init_controller_json,
        ),
        -1.0,
        1.0,
    )
    sigma = np.full(args.action_count, float(args.action_sigma), dtype=np.float64)

    best_record: dict[str, Any] | None = None
    best_by_max_streak: dict[str, Any] | None = None
    best_by_min_angle: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for iteration in range(max(1, args.iterations + 1)):
        if iteration == 0:
            candidates = [center]
        else:
            candidates = [center]
            for _ in range(max(0, args.population - 1)):
                candidates.append(np.clip(center + rng.normal(0.0, sigma), -1.0, 1.0))
        records = []
        for candidate in candidates:
            metrics = evaluate_actions(
                cfg,
                progress=args.progress,
                seed=args.seed,
                seconds=args.seconds,
                zero_noise=args.zero_noise,
                action_knots=np.asarray(candidate, dtype=np.float64),
                score_mode=args.score_mode,
                handoff_min_time=args.handoff_min_time,
                handoff_max_cart_abs=args.handoff_max_cart_abs,
                collect_states=False,
                state_max_angle=args.state_max_angle,
                state_max_hinge_rms=args.state_max_hinge_rms,
                state_max_cart_abs=args.state_max_cart_abs,
                state_max_cart_velocity=args.state_max_cart_velocity,
            )
            records.append(
                {
                    "score": float(metrics["score"]),
                    "controller": {
                        "type": "direct_action_knots",
                        "seconds": float(args.seconds),
                        "action_knots": np.asarray(candidate, dtype=np.float64).astype(float).tolist(),
                    },
                    "metrics": {k: v for k, v in metrics.items() if k != "candidate_states"},
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
                "cart_velocity": float(best_pass["cart_velocity"]),
                "x": float(best_pass["x"]),
                "max_upright_streak_seconds": float(records[0]["metrics"]["final_info"].get("max_upright_streak_seconds", 0.0)),
                "success": bool(records[0]["metrics"]["success"]),
            }
        )
        print(
            f"iter={iteration:03d} score={records[0]['score']:.6f} "
            f"angle={best_pass['max_abs_angle']:.6f} "
            f"hinge={best_pass['hinge_velocity_rms']:.3f} "
            f"x={best_pass['x']:.3f} "
            f"cartv={best_pass['cart_velocity']:.3f} "
            f"streak={records[0]['metrics']['final_info'].get('max_upright_streak_seconds', 0.0):.3f}s "
            f"success={records[0]['metrics']['success']}"
        )

        if iteration > 0:
            elite_arr = np.asarray([row["controller"]["action_knots"] for row in records[: args.elites]], dtype=np.float64)
            center = np.clip(elite_arr.mean(axis=0), -1.0, 1.0)
            sigma = np.maximum(elite_arr.std(axis=0), 1e-3) * float(args.sigma_decay)

    assert best_record is not None
    assert best_by_max_streak is not None
    assert best_by_min_angle is not None

    state_payload = None
    if args.state_out:
        state_records = [
            ("score", best_record),
            ("max_upright_streak", best_by_max_streak),
            ("min_best_pass_angle", best_by_min_angle),
        ]
        states: list[dict[str, Any]] = []
        for label, record in state_records:
            state_metrics = evaluate_actions(
                cfg,
                progress=args.progress,
                seed=args.seed,
                seconds=args.seconds,
                zero_noise=args.zero_noise,
                action_knots=np.asarray(record["controller"]["action_knots"], dtype=np.float64),
                score_mode=args.score_mode,
                handoff_min_time=args.handoff_min_time,
                handoff_max_cart_abs=args.handoff_max_cart_abs,
                collect_states=True,
                state_max_angle=args.state_max_angle,
                state_max_hinge_rms=args.state_max_hinge_rms,
                state_max_cart_abs=args.state_max_cart_abs,
                state_max_cart_velocity=args.state_max_cart_velocity,
            )
            for state in state_metrics["candidate_states"]:
                states.append({**state, "source_record": label})
        states.sort(key=state_quality)
        states = states[: max(0, int(args.keep_best_states))]
        state_payload = {
            "generated_at": utc_timestamp(),
            "source": "direct_action_sequence_search",
            "controller": best_record["controller"],
            "states": states,
            "state_filters": {
                "handoff_min_time": float(args.handoff_min_time),
                "state_max_angle": float(args.state_max_angle),
                "state_max_hinge_rms": float(args.state_max_hinge_rms),
                "state_max_cart_abs": float(args.state_max_cart_abs),
                "state_max_cart_velocity": float(args.state_max_cart_velocity),
            },
            "state_count": int(len(states)),
            "config_sha256": data_sha256(cfg),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        }
        dump_json(state_payload, Path(args.state_out))
        print(f"Wrote {args.state_out} states={len(states)}")

    result = {
        "generated_at": utc_timestamp(),
        "not_solution": not bool(best_by_max_streak["metrics"]["success"]),
        "summary": "Direct action-sequence CEM search for hanging-start swing-up handoff states.",
        "seed": int(args.seed),
        "progress": float(args.progress),
        "zero_noise": bool(args.zero_noise),
        "search": {
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "action_count": int(args.action_count),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "score_mode": str(args.score_mode),
            "handoff_min_time": float(args.handoff_min_time),
            "handoff_max_cart_abs": float(args.handoff_max_cart_abs),
            "init_controller_json": args.init_controller_json,
            "state_out": args.state_out,
            "state_count": None if state_payload is None else int(state_payload["state_count"]),
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
