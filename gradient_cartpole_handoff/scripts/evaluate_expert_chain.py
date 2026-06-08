#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, load_model, sample_action
from probe_swingup_trajectory import DEFAULT_KD, DEFAULT_KNOTS, DEFAULT_KP, DEFAULT_TRAJECTORY_SECONDS, trajectory_action
from search_swingup_capture import lqr_action, lqr_gain


def load_checkpoint_policy(config_path: str, checkpoint_path: str, progress: float) -> tuple[dict[str, Any], ActorCritic]:
    import mlx.core as mx

    cfg = load_config(config_path)
    probe = NLinkCartPoleEnv(cfg, progress=progress, seed=int(cfg["experiment"].get("seed", 0)))
    obs_dim = probe.observation_space.shape[0]
    act_dim = probe.action_space.shape[0]
    probe.close()
    ppo = cfg["ppo"]
    model = ActorCritic(obs_dim, act_dim, list(ppo.get("hidden_sizes", [256, 256])), float(ppo.get("action_std_init", 0.7)))
    mx.eval(model.parameters())
    load_model(model, checkpoint_path)
    return cfg, model


def hinge_velocity_rms(env: NLinkCartPoleEnv) -> float:
    return float(np.sqrt(np.mean(env.data.qvel[1 : 1 + env.n] ** 2)))


def evaluate_chain(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    seconds: float,
    zero_noise: bool,
    capture_model: ActorCritic | None,
    capture_gain: np.ndarray,
    stabilize_gain: np.ndarray,
    capture_enter_angle: float,
    capture_min_time: float,
    stabilize_enter_angle: float,
    stabilize_enter_streak: float,
    stabilize_hinge_rms: float,
    min_capture_seconds: float,
    lqr_capture_scale: float,
    lqr_stabilize_scale: float,
) -> dict[str, Any]:
    cfg = {**cfg, "env": {**cfg["env"]}}
    if zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0

    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    obs, reset_info = env.reset()
    del reset_info

    stage = "swing"
    stage_enter_time = 0.0
    stage_counts = {"swing": 0, "capture": 0, "stabilize": 0}
    stage_events: list[dict[str, Any]] = [{"time_seconds": 0.0, "stage": stage, "reason": "reset"}]
    best_pass: dict[str, Any] | None = None
    done_events: list[dict[str, Any]] = []
    max_cart_abs = 0.0
    action_abs_max = 0.0
    steps = min(env.max_steps, int(seconds / env.dt))

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
            action = trajectory_action(env, t, DEFAULT_KNOTS, DEFAULT_TRAJECTORY_SECONDS, DEFAULT_KP, DEFAULT_KD)
        elif stage == "capture":
            if capture_model is not None:
                action_arr, _, _ = sample_action(capture_model, obs[None, :], deterministic=True)
                action = float(action_arr[0, 0])
            else:
                action = lqr_action(env, capture_gain, scale=lqr_capture_scale, cart_target=0.0)
        else:
            action = lqr_action(env, stabilize_gain, scale=lqr_stabilize_scale, cart_target=0.0)

        stage_counts[stage] += 1
        action_abs_max = max(action_abs_max, abs(float(action)))
        obs, reward, terminated, truncated, info = env.step([action])
        rel, abs_after = env._angles()
        hinge_after = hinge_velocity_rms(env)
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        row = {
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "stage": stage,
            "reward": float(reward),
            "action": float(action),
            "x": float(info["x"]),
            "cart_velocity": float(env.data.qvel[0]),
            "qpos": np.array(env.data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.array(env.data.qvel, dtype=np.float64).astype(float).tolist(),
            "max_abs_angle": float(info["max_abs_angle"]),
            "mean_abs_angle": float(info["mean_abs_angle"]),
            "hinge_velocity_rms": hinge_after,
            "relative_angles": rel.astype(float).tolist(),
            "absolute_angles": abs_after.astype(float).tolist(),
            "is_upright": bool(info["is_upright"]),
            "upright_streak_seconds": float(info["upright_streak_seconds"]),
            "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
            "time_to_first_upright": info["time_to_first_upright"],
        }
        if best_pass is None or row["max_abs_angle"] < best_pass["max_abs_angle"]:
            best_pass = row
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

    final_info = dict(info)
    env.close()
    return {
        "generated_at": utc_timestamp(),
        "not_solution": not bool(final_info.get("success", False)),
        "summary": "Reset-free expert chain evaluation: swing expert, capture expert, then stabilize expert.",
        "success": bool(final_info.get("success", False)),
        "seed": int(seed),
        "progress": float(progress),
        "zero_noise": bool(zero_noise),
        "simulated_steps": int(step + 1),
        "simulated_seconds": float((step + 1) * (float(cfg["env"]["timestep"]) * int(cfg["env"].get("frame_skip", 1)))),
        "stage_counts": stage_counts,
        "stage_events": stage_events,
        "max_cart_abs": float(max_cart_abs),
        "action_abs_max": float(action_abs_max),
        "best_upright_pass": best_pass,
        "done_events": done_events,
        "final_info": final_info,
        "gates": {
            "capture_enter_angle": float(capture_enter_angle),
            "capture_min_time": float(capture_min_time),
            "stabilize_enter_angle": float(stabilize_enter_angle),
            "stabilize_enter_streak": float(stabilize_enter_streak),
            "stabilize_hinge_rms": float(stabilize_hinge_rms),
            "min_capture_seconds": float(min_capture_seconds),
        },
        "environment": {
            "n_links": int(cfg["env"]["n_links"]),
            "init_mode": str(cfg["env"].get("init_mode", "upright")),
            "episode_seconds": float(cfg["env"]["episode_seconds"]),
            "force_limit": float(cfg["env"]["force_limit"]),
            "rail_limit": float(cfg["env"]["rail_limit"]),
            "success_upright_threshold": float(
                cfg["env"].get("success_upright_threshold", cfg["env"].get("reward", {}).get("upright_threshold", 0.10))
            ),
            "success_sustain_seconds": float(cfg["env"].get("success_sustain_seconds", 0.0)),
        },
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a chained swing/capture/stabilize expert controller")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default="runs/swingup6_expert_chain/eval_chain.json")
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--capture-config", default=None)
    parser.add_argument("--capture-checkpoint", default=None)
    parser.add_argument("--capture-enter-angle", type=float, default=0.16)
    parser.add_argument("--capture-min-time", type=float, default=5.70)
    parser.add_argument("--stabilize-enter-angle", type=float, default=0.15)
    parser.add_argument("--stabilize-enter-streak", type=float, default=0.02)
    parser.add_argument("--stabilize-hinge-rms", type=float, default=1.00)
    parser.add_argument("--min-capture-seconds", type=float, default=0.50)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--lqr-capture-scale", type=float, default=1.0)
    parser.add_argument("--lqr-stabilize-scale", type=float, default=1.0)
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

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
    result = evaluate_chain(
        cfg,
        progress=args.progress,
        seed=args.seed,
        seconds=args.seconds,
        zero_noise=args.zero_noise,
        capture_model=capture_model,
        capture_gain=capture_gain,
        stabilize_gain=stabilize_gain,
        capture_enter_angle=args.capture_enter_angle,
        capture_min_time=args.capture_min_time,
        stabilize_enter_angle=args.stabilize_enter_angle,
        stabilize_enter_streak=args.stabilize_enter_streak,
        stabilize_hinge_rms=args.stabilize_hinge_rms,
        min_capture_seconds=args.min_capture_seconds,
        lqr_capture_scale=args.lqr_capture_scale,
        lqr_stabilize_scale=args.lqr_stabilize_scale,
    )
    result["experts"] = {
        "swing": {
            "type": "cart_position_pd_fixed_knots",
            "trajectory_seconds": float(DEFAULT_TRAJECTORY_SECONDS),
            "kp": float(DEFAULT_KP),
            "kd": float(DEFAULT_KD),
            "knots": DEFAULT_KNOTS.astype(float).tolist(),
        },
        "capture": {"type": "checkpoint" if capture_model is not None else "lqr", "checkpoint": capture_evidence},
        "stabilize": {"type": "lqr", "control_cost": float(args.lqr_control_cost)},
    }
    dump_json(result, Path(args.out))
    best = result.get("best_upright_pass") or {}
    print(
        f"success={result['success']} "
        f"stages={result['stage_counts']} "
        f"best_angle={best.get('max_abs_angle', float('nan')):.6f} "
        f"max_streak={result['final_info'].get('max_upright_streak_seconds', 0.0):.3f}s"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
