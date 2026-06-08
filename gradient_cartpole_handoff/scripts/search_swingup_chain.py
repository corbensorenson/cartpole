#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, sample_action
from evaluate_expert_chain import hinge_velocity_rms, load_checkpoint_policy
from probe_swingup_trajectory import (
    DEFAULT_KD,
    DEFAULT_KNOTS,
    DEFAULT_KP,
    DEFAULT_TRAJECTORY_SECONDS,
    trajectory_action,
)
from search_swingup_capture import lqr_action, lqr_gain


def vector_to_controller(vec: np.ndarray, knot_count: int, rail_target_limit: float) -> dict[str, Any]:
    return {
        "type": "cart_position_pd_fixed_knots",
        "knots": np.concatenate([[0.0], np.clip(vec[: knot_count - 1], -rail_target_limit, rail_target_limit)])
        .astype(float)
        .tolist(),
        "kp": float(np.clip(vec[knot_count - 1], 0.05, 3.0)),
        "kd": float(np.clip(vec[knot_count], 0.0, 2.0)),
        "trajectory_seconds": float(np.clip(vec[knot_count + 1], 4.0, 16.0)),
        "capture_min_time": float(np.clip(vec[knot_count + 2], 3.0, 10.0)),
        "capture_enter_angle": float(np.clip(vec[knot_count + 3], 0.08, 0.75)),
    }


def controller_to_vector(controller: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(controller["knots"][1:], dtype=np.float64),
            np.asarray(
                [
                    controller["kp"],
                    controller["kd"],
                    controller["trajectory_seconds"],
                    controller.get("capture_min_time", 5.70),
                    controller.get("capture_enter_angle", 0.16),
                ],
                dtype=np.float64,
            ),
        ]
    )


def load_initial_controller(path: str | None) -> dict[str, Any]:
    def normalize(controller: dict[str, Any]) -> dict[str, Any]:
        controller = dict(controller)
        controller.setdefault("capture_min_time", 5.70)
        controller.setdefault("capture_enter_angle", 0.16)
        return controller

    if not path:
        return normalize({
            "knots": DEFAULT_KNOTS.astype(float).tolist(),
            "kp": float(DEFAULT_KP),
            "kd": float(DEFAULT_KD),
            "trajectory_seconds": float(DEFAULT_TRAJECTORY_SECONDS),
            "capture_min_time": 5.70,
            "capture_enter_angle": 0.16,
        })
    with open(Path(path), "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "best" in payload:
        return normalize(payload["best"]["controller"])
    if isinstance(payload, dict) and "controller" in payload:
        return normalize(payload["controller"])
    if isinstance(payload, dict) and "knots" in payload:
        return normalize(payload)
    raise ValueError(f"Could not load swing controller from {path}")


def row_from_env(env: NLinkCartPoleEnv, *, step: int, stage: str, action: float, reward: float, info: dict[str, Any]) -> dict[str, Any]:
    rel, abs_angles = env._angles()
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "stage": stage,
        "reward": float(reward),
        "action": float(action),
        "x": float(info["x"]),
        "cart_velocity": float(env.data.qvel[0]),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "max_abs_angle": float(info["max_abs_angle"]),
        "mean_abs_angle": float(info["mean_abs_angle"]),
        "hinge_velocity_rms": float(info.get("hinge_velocity_rms", hinge_velocity_rms(env))),
        "capture_quality": float(info.get("capture_quality", 0.0)),
        "relative_angles": rel.astype(float).tolist(),
        "absolute_angles": abs_angles.astype(float).tolist(),
        "is_upright": bool(info["is_upright"]),
        "upright_streak_seconds": float(info["upright_streak_seconds"]),
        "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
        "time_to_first_upright": info["time_to_first_upright"],
    }


def handoff_quality(row: dict[str, Any], rail_limit: float) -> float:
    return float(
        8.0 * row["max_abs_angle"]
        + 0.75 * row["hinge_velocity_rms"]
        + 0.50 * abs(row["x"]) / rail_limit
        + 0.35 * abs(row["cart_velocity"])
        - 2.0 * row["capture_quality"]
        - 1.0 * float(row["is_upright"])
    )


def evaluate_controller(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    seconds: float,
    zero_noise: bool,
    swing_controller: dict[str, Any],
    capture_model: ActorCritic | None,
    capture_gain: np.ndarray,
    stabilize_gain: np.ndarray,
    min_capture_seconds: float,
    stabilize_enter_angle: float,
    stabilize_enter_streak: float,
    stabilize_hinge_rms: float,
    lqr_capture_scale: float,
    lqr_stabilize_scale: float,
    collect_states: bool = False,
    state_min_time: float = 0.0,
    state_max_angle: float = 0.60,
    state_max_hinge_rms: float | None = None,
    state_stride: int = 1,
) -> dict[str, Any]:
    cfg = {**cfg, "env": {**cfg["env"]}}
    if zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0

    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    obs, reset_info = env.reset()
    del reset_info

    knots = np.asarray(swing_controller["knots"], dtype=np.float64)
    trajectory_seconds = float(swing_controller["trajectory_seconds"])
    kp = float(swing_controller["kp"])
    kd = float(swing_controller["kd"])
    capture_min_time = float(swing_controller["capture_min_time"])
    capture_enter_angle = float(swing_controller["capture_enter_angle"])

    stage = "swing"
    stage_enter_time = 0.0
    stage_counts = {"swing": 0, "capture": 0, "stabilize": 0}
    stage_events: list[dict[str, Any]] = [{"time_seconds": 0.0, "stage": stage, "reason": "reset"}]
    best_any: dict[str, Any] | None = None
    best_capture: dict[str, Any] | None = None
    done_events: list[dict[str, Any]] = []
    max_cart_abs = 0.0
    action_abs_max = 0.0
    selected_states: list[dict[str, Any]] = []
    steps = min(env.max_steps, int(seconds / env.dt))
    final_info: dict[str, Any] = {}
    threshold = float(cfg["env"].get("success_upright_threshold", cfg["env"].get("reward", {}).get("upright_threshold", 0.10)))

    for step in range(steps):
        t = step * env.dt
        _, abs_angles = env._angles()
        max_abs_angle = float(np.max(np.abs(abs_angles)))
        hinge_rms = hinge_velocity_rms(env)
        if stage == "swing" and t >= capture_min_time and max_abs_angle <= capture_enter_angle:
            stage = "capture"
            stage_enter_time = t
            stage_events.append(
                {
                    "time_seconds": float(t),
                    "stage": stage,
                    "reason": "capture_enter_angle",
                    "max_abs_angle": max_abs_angle,
                    "hinge_velocity_rms": hinge_rms,
                    "x": float(env.data.qpos[0]),
                }
            )
        if (
            stage == "capture"
            and t - stage_enter_time >= min_capture_seconds
            and max_abs_angle <= stabilize_enter_angle
            and float(env._info()["upright_streak_seconds"]) >= stabilize_enter_streak
            and hinge_rms <= stabilize_hinge_rms
        ):
            stage = "stabilize"
            stage_enter_time = t
            stage_events.append(
                {
                    "time_seconds": float(t),
                    "stage": stage,
                    "reason": "stabilize_gate",
                    "max_abs_angle": max_abs_angle,
                    "hinge_velocity_rms": hinge_rms,
                    "x": float(env.data.qpos[0]),
                }
            )

        if stage == "swing":
            action = trajectory_action(env, t, knots, trajectory_seconds, kp, kd)
        elif stage == "capture":
            if capture_model is None:
                action = lqr_action(env, capture_gain, scale=lqr_capture_scale, cart_target=0.0)
            else:
                action_arr, _, _ = sample_action(capture_model, obs[None, :], deterministic=True)
                action = float(action_arr[0, 0])
        else:
            action = lqr_action(env, stabilize_gain, scale=lqr_stabilize_scale, cart_target=0.0)

        stage_counts[stage] += 1
        action_abs_max = max(action_abs_max, abs(float(action)))
        obs, reward, terminated, truncated, info = env.step([action])
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        row = row_from_env(env, step=step + 1, stage=stage, action=action, reward=reward, info=info)
        if (
            collect_states
            and stage in {"capture", "stabilize"}
            and row["time_seconds"] >= state_min_time
            and row["max_abs_angle"] <= state_max_angle
            and (state_max_hinge_rms is None or row["hinge_velocity_rms"] <= state_max_hinge_rms)
            and int(step + 1) % max(1, state_stride) == 0
        ):
            selected_states.append(row)
        if best_any is None or handoff_quality(row, float(cfg["env"]["rail_limit"])) < handoff_quality(
            best_any, float(cfg["env"]["rail_limit"])
        ):
            best_any = row
        if stage in {"capture", "stabilize"} and (
            best_capture is None
            or handoff_quality(row, float(cfg["env"]["rail_limit"])) < handoff_quality(
                best_capture, float(cfg["env"]["rail_limit"])
            )
        ):
            best_capture = row
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
    assert best_any is not None
    best = best_capture or best_any
    best_quality = handoff_quality(best, float(cfg["env"]["rail_limit"]))
    success = bool(final_info.get("success", False))
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    ever_upright = final_info.get("time_to_first_upright") is not None
    capture_reached = stage_counts["capture"] + stage_counts["stabilize"] > 0
    terminated = bool(done_events and done_events[-1]["terminated"])
    final_angle = float(final_info.get("max_abs_angle", np.pi))
    final_capture_quality = float(final_info.get("capture_quality", 0.0))
    angle_gap = max(0.0, float(best["max_abs_angle"]) - threshold)
    score = (
        -5000.0 * float(success)
        - 1800.0 * max_streak
        - 120.0 * float(ever_upright)
        - 80.0 * float(capture_reached)
        - 40.0 * float(final_capture_quality)
        + 60.0 * angle_gap
        + 20.0 * best_quality
        + 4.0 * final_angle
        + 2.0 * max_cart_abs
        + 120.0 * float(terminated)
    )
    if not capture_reached:
        score += 200.0
    return {
        "score": float(score),
        "success": success,
        "ever_upright": bool(ever_upright),
        "capture_reached": bool(capture_reached),
        "simulated_steps": int(step + 1),
        "simulated_seconds": float((step + 1) * env.dt),
        "stage_counts": stage_counts,
        "stage_events": stage_events,
        "max_cart_abs": float(max_cart_abs),
        "action_abs_max": float(action_abs_max),
        "best_handoff": best,
        "best_any": best_any,
        "best_capture": best_capture,
        "best_handoff_quality": float(best_quality),
        "selected_states": selected_states,
        "done_events": done_events,
        "final_info": final_info,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search swing trajectories against the actual capture/stabilize chain")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--out", default="runs/swingup6_chain_search/search.json")
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--capture-config", default=None)
    parser.add_argument("--capture-checkpoint", default=None)
    parser.add_argument("--init-controller-json", default=None)
    parser.add_argument("--state-out", default=None)
    parser.add_argument("--state-min-time", type=float, default=0.0)
    parser.add_argument("--state-max-angle", type=float, default=0.60)
    parser.add_argument("--state-max-hinge-rms", type=float, default=None)
    parser.add_argument("--state-stride", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elites", type=int, default=4)
    parser.add_argument("--rail-target-limit", type=float, default=2.6)
    parser.add_argument("--knot-sigma", type=float, default=0.35)
    parser.add_argument("--gain-sigma", type=float, default=0.18)
    parser.add_argument("--time-sigma", type=float, default=0.65)
    parser.add_argument("--capture-time-sigma", type=float, default=0.35)
    parser.add_argument("--capture-angle-sigma", type=float, default=0.10)
    parser.add_argument("--sigma-decay", type=float, default=0.82)
    parser.add_argument("--min-capture-seconds", type=float, default=0.50)
    parser.add_argument("--stabilize-enter-angle", type=float, default=0.15)
    parser.add_argument("--stabilize-enter-streak", type=float, default=0.02)
    parser.add_argument("--stabilize-hinge-rms", type=float, default=1.0)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--lqr-capture-scale", type=float, default=1.0)
    parser.add_argument("--lqr-stabilize-scale", type=float, default=1.0)
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.population < 1:
        raise ValueError("--population must be >= 1")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("--elites must be between 1 and --population")

    cfg = apply_overrides(load_config(args.config), args.override)
    capture_model = None
    capture_evidence: dict[str, Any] | None = None
    if args.capture_checkpoint:
        if not args.capture_config:
            raise ValueError("--capture-checkpoint requires --capture-config")
        _, capture_model = load_checkpoint_policy(args.capture_config, args.capture_checkpoint, args.progress)
        capture_evidence = {
            "config": str(Path(args.capture_config)),
            "checkpoint": file_metadata(args.capture_checkpoint),
        }

    capture_gain = lqr_gain(cfg, progress=args.progress, fd_eps=args.fd_eps, control_cost=args.lqr_control_cost)
    stabilize_gain = capture_gain
    rng = np.random.default_rng(args.seed)
    knot_count = len(DEFAULT_KNOTS)
    center_controller = load_initial_controller(args.init_controller_json)
    center = controller_to_vector(center_controller)
    sigma = np.concatenate(
        [
            np.full(knot_count - 1, args.knot_sigma, dtype=np.float64),
            np.asarray(
                [
                    args.gain_sigma,
                    args.gain_sigma,
                    args.time_sigma,
                    args.capture_time_sigma,
                    args.capture_angle_sigma,
                ],
                dtype=np.float64,
            ),
        ]
    )

    best: dict[str, Any] | None = None
    best_by_max_streak: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for iteration in range(max(1, args.iterations + 1)):
        candidates = [center]
        if iteration > 0:
            candidates.extend(center + rng.normal(0.0, sigma) for _ in range(max(0, args.population - 1)))
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            controller = vector_to_controller(candidate, knot_count, args.rail_target_limit)
            metrics = evaluate_controller(
                cfg,
                progress=args.progress,
                seed=args.seed,
                seconds=args.seconds,
                zero_noise=args.zero_noise,
                swing_controller=controller,
                capture_model=capture_model,
                capture_gain=capture_gain,
                stabilize_gain=stabilize_gain,
                min_capture_seconds=args.min_capture_seconds,
                stabilize_enter_angle=args.stabilize_enter_angle,
                stabilize_enter_streak=args.stabilize_enter_streak,
                stabilize_hinge_rms=args.stabilize_hinge_rms,
                lqr_capture_scale=args.lqr_capture_scale,
                lqr_stabilize_scale=args.lqr_stabilize_scale,
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

        top = records[0]
        top_metrics = top["metrics"]
        top_handoff = top_metrics["best_handoff"]
        history.append(
            {
                "iteration": int(iteration),
                "score": float(top["score"]),
                "success": bool(top_metrics["success"]),
                "max_upright_streak_seconds": float(top_metrics["final_info"].get("max_upright_streak_seconds", 0.0)),
                "best_handoff_quality": float(top_metrics["best_handoff_quality"]),
                "best_handoff_angle": float(top_handoff["max_abs_angle"]),
                "best_handoff_hinge_rms": float(top_handoff["hinge_velocity_rms"]),
                "best_handoff_x": float(top_handoff["x"]),
                "capture_reached": bool(top_metrics["capture_reached"]),
            }
        )
        print(
            f"iter={iteration:03d} score={top['score']:.6f} "
            f"streak={top_metrics['final_info'].get('max_upright_streak_seconds', 0.0):.3f}s "
            f"angle={top_handoff['max_abs_angle']:.6f} "
            f"hinge={top_handoff['hinge_velocity_rms']:.3f} "
            f"x={top_handoff['x']:.3f} "
            f"capture={top_metrics['capture_reached']} "
            f"success={top_metrics['success']}"
        )

        if iteration > 0:
            elite_vectors = [controller_to_vector(row["controller"]) for row in records[: args.elites]]
            elite_vectors.append(controller_to_vector(best["controller"]))
            elite_arr = np.asarray(elite_vectors, dtype=np.float64)
            center = controller_to_vector(best["controller"])
            sigma = np.maximum(elite_arr.std(axis=0), 1e-3) * float(args.sigma_decay)

    assert best is not None
    assert best_by_max_streak is not None
    result = {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Chain-level swing trajectory search; success still requires final held-out eval, checkpoint, hashes, and reset-free video.",
        "seed": int(args.seed),
        "progress": float(args.progress),
        "zero_noise": bool(args.zero_noise),
        "search": {
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "seconds": float(args.seconds),
            "rail_target_limit": float(args.rail_target_limit),
            "sigma_decay": float(args.sigma_decay),
        },
        "experts": {
            "swing": {"type": "searched_cart_position_pd"},
            "capture": {"type": "checkpoint" if capture_model is not None else "lqr", "checkpoint": capture_evidence},
            "stabilize": {"type": "lqr", "control_cost": float(args.lqr_control_cost)},
        },
        "best": best,
        "best_by": {"score": best, "max_upright_streak": best_by_max_streak},
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(result, Path(args.out))
    if args.state_out:
        state_metrics = evaluate_controller(
            cfg,
            progress=args.progress,
            seed=args.seed,
            seconds=args.seconds,
            zero_noise=args.zero_noise,
            swing_controller=best["controller"],
            capture_model=capture_model,
            capture_gain=capture_gain,
            stabilize_gain=stabilize_gain,
            min_capture_seconds=args.min_capture_seconds,
            stabilize_enter_angle=args.stabilize_enter_angle,
            stabilize_enter_streak=args.stabilize_enter_streak,
            stabilize_hinge_rms=args.stabilize_hinge_rms,
            lqr_capture_scale=args.lqr_capture_scale,
            lqr_stabilize_scale=args.lqr_stabilize_scale,
            collect_states=True,
            state_min_time=args.state_min_time,
            state_max_angle=args.state_max_angle,
            state_max_hinge_rms=args.state_max_hinge_rms,
            state_stride=args.state_stride,
        )
        states = state_metrics.pop("selected_states")
        state_payload = {
            "generated_at": utc_timestamp(),
            "not_solution": True,
            "summary": "Replayable chain-generated states for transition/capture training; not final swing-up evidence.",
            "source_search": str(Path(args.out)),
            "selection": {
                "state_min_time": float(args.state_min_time),
                "state_max_angle": float(args.state_max_angle),
                "state_max_hinge_rms": args.state_max_hinge_rms,
                "state_stride": int(args.state_stride),
            },
            "controller": best["controller"],
            "metrics": state_metrics,
            "state_count": int(len(states)),
            "states": states,
            "config_sha256": data_sha256(cfg),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        }
        dump_json(state_payload, Path(args.state_out))
        print(f"Wrote {args.state_out} states={len(states)}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
