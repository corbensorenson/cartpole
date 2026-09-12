#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np

from torch_runtime import prepare_runtime

prepare_runtime()

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_torch import ActorCritic, load_model, sample_action


def handoff_score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        -float(row["max_abs_angle"]),
        -float(row["hinge_velocity_rms"]),
        -abs(float(row["cart_velocity"])),
        -abs(float(row["x"])) / max(1e-6, float(row.get("rail_limit", 1.0))),
        float(row.get("capture_quality", 0.0)),
    )


def make_state_row(
    env: NLinkCartPoleEnv,
    *,
    episode: int,
    seed: int,
    step: int,
    action: float,
    reward: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    rel, abs_angles = env._angles()
    hinge_rms = float(np.sqrt(np.mean(env.data.qvel[1 : 1 + env.n] ** 2)))
    return {
        "source": "learned_policy_maintenance_rollout",
        "episode": int(episode),
        "seed": int(seed),
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "reward": float(reward),
        "action": float(action),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "x": float(info["x"]),
        "cart_velocity": float(env.data.qvel[0]),
        "max_abs_angle": float(info["max_abs_angle"]),
        "mean_abs_angle": float(info["mean_abs_angle"]),
        "hinge_velocity_rms": hinge_rms,
        "capture_quality": float(info.get("capture_quality", 0.0)),
        "relative_angles": rel.astype(float).tolist(),
        "absolute_angles": abs_angles.astype(float).tolist(),
        "is_upright": bool(info.get("is_upright", False)),
        "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
        "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
        "time_to_first_upright": info.get("time_to_first_upright"),
        "progress": float(info.get("progress", env.progress)),
        "plant_progress": float(info.get("plant_progress", env.plant_progress)),
        "rail_limit": float(info.get("rail_limit", env.rail_limit)),
        "lengths": list(info.get("lengths", [])),
        "masses": list(info.get("masses", [])),
        "damping": list(info.get("damping", [])),
        "frictionloss": list(info.get("frictionloss", [])),
    }


def export_policy_states(
    cfg: dict[str, Any],
    *,
    checkpoint: str,
    progress: float,
    episodes: int,
    seed: int,
    seconds: float,
    deterministic: bool,
    zero_noise: bool,
    min_angle: float,
    max_angle: float,
    max_hinge_rms: float,
    max_cart_velocity: float,
    max_cart_abs: float | None,
    min_time: float,
    stride: int,
    max_states: int | None,
    one_per_episode: bool,
    require_success: bool,
) -> dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    if zero_noise:
        for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
            cfg["env"][key] = 0.0

    probe = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    ppo = cfg["ppo"]
    model = ActorCritic(
        probe.observation_space.shape[0],
        probe.action_space.shape[0],
        list(ppo.get("hidden_sizes", [256, 256])),
        float(ppo.get("action_std_init", 0.7)),
    )
    load_model(model, checkpoint)
    model.eval()
    probe.close()

    states: list[dict[str, Any]] = []
    best_states: list[dict[str, Any]] = []
    selected_episodes = 0
    successful_episodes = 0
    for episode in range(int(episodes)):
        episode_seed = int(seed + 10_000 + episode)
        env = NLinkCartPoleEnv(cfg, progress=progress, seed=episode_seed)
        obs, _ = env.reset(seed=episode_seed)
        max_steps = min(env.max_steps, int(seconds / env.dt))
        candidates: list[dict[str, Any]] = []
        best_row: dict[str, Any] | None = None
        final_info: dict[str, Any] = {}
        for step in range(1, max_steps + 1):
            action_arr, _, _ = sample_action(
                model,
                obs[None, :],
                deterministic=deterministic,
            )
            action = float(action_arr[0, 0])
            obs, reward, terminated, truncated, final_info = env.step(action_arr[0])
            row = make_state_row(
                env,
                episode=episode,
                seed=episode_seed,
                step=step,
                action=action,
                reward=float(reward),
                info=final_info,
            )
            if best_row is None or handoff_score(row) > handoff_score(best_row):
                best_row = row
            hinge_rms = float(row["hinge_velocity_rms"])
            eligible = (
                not terminated
                and step * env.dt >= min_time
                and min_angle <= float(row["max_abs_angle"]) <= max_angle
                and hinge_rms <= max_hinge_rms
                and abs(float(row["cart_velocity"])) <= max_cart_velocity
                and (max_cart_abs is None or abs(float(row["x"])) <= max_cart_abs)
            )
            if eligible and step % max(1, stride) == 0:
                candidates.append(row)
            if terminated or truncated:
                break
        episode_success = bool(final_info.get("success", False))
        successful_episodes += int(episode_success)
        if require_success and not episode_success:
            candidates = []
        if candidates:
            selected_episodes += 1
            if one_per_episode:
                states.append(sorted(candidates, key=handoff_score, reverse=True)[0])
            else:
                states.extend(candidates)
        if best_row is not None:
            best_states.append(best_row)
        env.close()
        if max_states is not None and len(states) >= max_states:
            break

    states = sorted(states, key=handoff_score, reverse=True)
    if max_states is not None:
        states = states[:max_states]
    best_states = sorted(best_states, key=handoff_score, reverse=True)
    return {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Real MuJoCo maintenance-policy states for capture/recovery training; not final swing-up evidence.",
        "state_count": int(len(states)),
        "episodes": int(episodes),
        "episodes_with_selected_state": int(selected_episodes),
        "successful_source_episodes": int(successful_episodes),
        "progress": float(progress),
        "seed": int(seed),
        "deterministic_policy": bool(deterministic),
        "zero_noise": bool(zero_noise),
        "selection": {
            "min_angle": float(min_angle),
            "max_angle": float(max_angle),
            "max_hinge_rms": float(max_hinge_rms),
            "max_cart_velocity": float(max_cart_velocity),
            "max_cart_abs": None if max_cart_abs is None else float(max_cart_abs),
            "min_time": float(min_time),
            "stride": int(stride),
            "max_states": max_states,
            "one_per_episode": bool(one_per_episode),
            "require_success": bool(require_success),
        },
        "states": states,
        "best_states": best_states[: min(len(best_states), 64)],
        "checkpoint": file_metadata(checkpoint),
        "environment": {
            "n_links": int(cfg["env"]["n_links"]),
            "init_mode": str(cfg["env"].get("init_mode", "upright")),
            "episode_seconds": float(cfg["env"]["episode_seconds"]),
            "force_limit": float(cfg["env"]["force_limit"]),
            "target_rail_limit": float(cfg["env"]["rail_limit"]),
            "scheduled_rail_limit": None if not states else float(states[0]["rail_limit"]),
            "plant_progress": float(probe.plant_progress),
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
    parser = argparse.ArgumentParser(
        description="Export real CPU PyTorch policy states for the next expert"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--progress", type=float, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--min-angle", type=float, default=0.0)
    parser.add_argument("--max-angle", type=float, default=None)
    parser.add_argument("--max-hinge-rms", type=float, default=None)
    parser.add_argument("--max-cart-velocity", type=float, default=None)
    parser.add_argument("--max-cart-abs", type=float, default=None)
    parser.add_argument("--min-time", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=512)
    parser.add_argument("--one-per-episode", action="store_true")
    parser.add_argument("--require-success", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    reward_cfg = cfg.get("env", {}).get("reward", {})
    threshold = float(cfg["env"].get("success_upright_threshold", reward_cfg.get("upright_threshold", 0.15)))
    result = export_policy_states(
        cfg,
        checkpoint=args.checkpoint,
        progress=float(args.progress),
        episodes=int(args.episodes),
        seed=int(args.seed),
        seconds=float(args.seconds),
        deterministic=not args.stochastic,
        zero_noise=bool(args.zero_noise),
        min_angle=float(args.min_angle),
        max_angle=float(args.max_angle if args.max_angle is not None else threshold),
        max_hinge_rms=float(args.max_hinge_rms if args.max_hinge_rms is not None else reward_cfg.get("upright_hinge_vel_threshold", 0.75)),
        max_cart_velocity=float(args.max_cart_velocity if args.max_cart_velocity is not None else reward_cfg.get("upright_cart_vel_threshold", 0.50)),
        max_cart_abs=args.max_cart_abs,
        min_time=float(args.min_time),
        stride=int(args.stride),
        max_states=args.max_states,
        one_per_episode=bool(args.one_per_episode),
        require_success=bool(args.require_success),
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(result, out)
    best = (result.get("states") or result.get("best_states") or [{}])[0]
    print(
        f"states={result['state_count']} selected_episodes={result['episodes_with_selected_state']}/{result['episodes']} "
        f"successful_sources={result['successful_source_episodes']} progress={result['progress']:.3f} "
        f"best_angle={best.get('max_abs_angle', float('nan')):.6f} "
        f"best_hinge={best.get('hinge_velocity_rms', float('nan')):.3f}"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
