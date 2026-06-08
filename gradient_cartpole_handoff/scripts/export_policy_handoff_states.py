#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, load_model, sample_action


def checkpoint_progress(checkpoint: str | Path) -> float | None:
    path = Path(checkpoint)
    meta_path = path.with_suffix(".meta.json")
    if not meta_path.exists() and path.name in {"best.safetensors", "latest.safetensors"}:
        meta_path = path.with_name(f"{path.stem}.meta.json")
    if not meta_path.exists():
        return None
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    progress = meta.get("eval_progress", meta.get("progress"))
    if progress is None:
        return None
    return float(progress)


def state_row(
    env: NLinkCartPoleEnv,
    *,
    episode: int,
    seed: int,
    step: int,
    action: float,
    reward: float,
    info: dict[str, Any],
    low_momentum: bool,
) -> dict[str, Any]:
    rel, abs_angles = env._angles()
    hinge_rms = float(np.sqrt(np.mean(env.data.qvel[1 : 1 + env.n] ** 2)))
    return {
        "source": "learned_policy_low_momentum_handoff",
        "episode": int(episode),
        "seed": int(seed),
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "reward": float(reward),
        "action": float(action),
        "qpos": np.array(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.array(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "x": float(info["x"]),
        "cart_velocity": float(env.data.qvel[0]),
        "max_abs_angle": float(info["max_abs_angle"]),
        "mean_abs_angle": float(info["mean_abs_angle"]),
        "hinge_velocity_rms": hinge_rms,
        "capture_quality": float(info.get("capture_quality", 0.0)),
        "relative_angles": rel.astype(float).tolist(),
        "absolute_angles": abs_angles.astype(float).tolist(),
        "is_upright": bool(info["is_upright"]),
        "low_momentum_upright": bool(low_momentum),
        "upright_streak_seconds": float(info["upright_streak_seconds"]),
        "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
        "time_to_first_upright": info["time_to_first_upright"],
        "progress": float(info.get("progress", env.progress)),
        "rail_limit": float(info.get("rail_limit", env.rail_limit)),
    }


def handoff_score(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        -float(row["max_abs_angle"]),
        -float(row["hinge_velocity_rms"]),
        -abs(float(row["cart_velocity"])),
        -abs(float(row["x"])) / max(1e-6, float(row.get("rail_limit", 1.0))),
        float(row["capture_quality"]),
    )


def export_policy_handoffs(
    cfg: dict[str, Any],
    *,
    checkpoint: str,
    progress: float,
    episodes: int,
    seed: int,
    seconds: float,
    deterministic: bool,
    zero_noise: bool,
    max_angle: float,
    max_hinge_rms: float,
    max_cart_velocity: float,
    max_cart_abs: float | None,
    min_time: float,
    stride: int,
    max_states: int | None,
    one_per_episode: bool,
) -> dict[str, Any]:
    cfg = {**cfg, "env": {**cfg["env"]}}
    if zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0

    probe = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    ppo = cfg["ppo"]
    model = ActorCritic(
        probe.observation_space.shape[0],
        probe.action_space.shape[0],
        list(ppo.get("hidden_sizes", [256, 256])),
        float(ppo.get("action_std_init", 0.7)),
    )
    mx.eval(model.parameters())
    load_model(model, checkpoint)
    probe.close()

    states: list[dict[str, Any]] = []
    best_states: list[dict[str, Any]] = []
    selected_episodes = 0
    for ep in range(episodes):
        ep_seed = seed + 10_000 + ep
        env = NLinkCartPoleEnv(cfg, progress=progress, seed=ep_seed)
        obs, _ = env.reset()
        max_steps = min(env.max_steps, int(seconds / env.dt))
        best_row: dict[str, Any] | None = None
        ep_selected = False
        for step in range(1, max_steps + 1):
            action_arr, _, _ = sample_action(model, obs[None, :], deterministic=deterministic)
            action = float(action_arr[0][0])
            obs, reward, terminated, truncated, info = env.step(action_arr[0])
            hinge_rms = float(info.get("hinge_velocity_rms", np.inf))
            low_momentum = bool(
                info.get("is_upright", False)
                and float(info["max_abs_angle"]) <= max_angle
                and hinge_rms <= max_hinge_rms
                and abs(float(env.data.qvel[0])) <= max_cart_velocity
                and (max_cart_abs is None or abs(float(info["x"])) <= max_cart_abs)
                and step * env.dt >= min_time
            )
            row = state_row(
                env,
                episode=ep,
                seed=ep_seed,
                step=step,
                action=action,
                reward=float(reward),
                info=info,
                low_momentum=low_momentum,
            )
            if best_row is None or handoff_score(row) > handoff_score(best_row):
                best_row = row
            if low_momentum and step % max(1, stride) == 0:
                states.append(row)
                ep_selected = True
                if one_per_episode:
                    break
                if max_states is not None and len(states) >= max_states:
                    break
            if terminated or truncated or (max_states is not None and len(states) >= max_states):
                break
        if ep_selected:
            selected_episodes += 1
        if best_row is not None:
            best_states.append(best_row)
        env.close()
        if max_states is not None and len(states) >= max_states:
            break

    states = sorted(states, key=lambda row: handoff_score(row), reverse=True)
    if max_states is not None:
        states = states[:max_states]
    best_states = sorted(best_states, key=lambda row: handoff_score(row), reverse=True)
    return {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Learned swing-policy low-momentum handoff states for capture/stabilize expert training; not final benchmark evidence.",
        "state_count": int(len(states)),
        "episodes": int(episodes),
        "episodes_with_selected_state": int(selected_episodes),
        "progress": float(progress),
        "seed": int(seed),
        "deterministic_policy": bool(deterministic),
        "zero_noise": bool(zero_noise),
        "selection": {
            "max_angle": float(max_angle),
            "max_hinge_rms": float(max_hinge_rms),
            "max_cart_velocity": float(max_cart_velocity),
            "max_cart_abs": None if max_cart_abs is None else float(max_cart_abs),
            "min_time": float(min_time),
            "stride": int(stride),
            "max_states": max_states,
            "one_per_episode": bool(one_per_episode),
        },
        "states": states,
        "best_states": best_states[: min(len(best_states), 64)],
        "checkpoint": file_metadata(checkpoint),
        "environment": {
            "n_links": int(cfg["env"]["n_links"]),
            "init_mode": str(cfg["env"].get("init_mode", "upright")),
            "episode_seconds": float(cfg["env"]["episode_seconds"]),
            "force_limit": float(cfg["env"]["force_limit"]),
            "rail_limit": float(cfg["env"]["rail_limit"]),
            "scheduled_rail_limit": None if not states else float(states[0]["rail_limit"]),
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
    parser = argparse.ArgumentParser(description="Export low-momentum handoff states from a learned swing policy")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--progress", type=float, default=None)
    parser.add_argument("--use-checkpoint-progress", action="store_true")
    parser.add_argument("--episodes", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default="runs/swingup6_policy_handoff/swing_handoff_states.json")
    parser.add_argument("--stochastic", action="store_true", help="Sample the policy instead of using deterministic mean actions")
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--max-angle", type=float, default=None)
    parser.add_argument("--max-hinge-rms", type=float, default=None)
    parser.add_argument("--max-cart-velocity", type=float, default=None)
    parser.add_argument("--max-cart-abs", type=float, default=None)
    parser.add_argument("--min-time", type=float, default=0.5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-states", type=int, default=512)
    parser.add_argument("--one-per-episode", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    progress = args.progress
    if args.use_checkpoint_progress:
        inferred = checkpoint_progress(args.checkpoint)
        if inferred is None:
            raise ValueError("--use-checkpoint-progress requested, but checkpoint metadata did not contain progress")
        progress = inferred
    if progress is None:
        progress = 1.0

    reward_cfg = cfg.get("env", {}).get("reward", {})
    result = export_policy_handoffs(
        cfg,
        checkpoint=args.checkpoint,
        progress=float(progress),
        episodes=args.episodes,
        seed=args.seed,
        seconds=args.seconds,
        deterministic=not args.stochastic,
        zero_noise=bool(args.zero_noise),
        max_angle=float(args.max_angle if args.max_angle is not None else cfg["env"].get("success_upright_threshold", reward_cfg.get("upright_threshold", 0.15))),
        max_hinge_rms=float(args.max_hinge_rms if args.max_hinge_rms is not None else reward_cfg.get("upright_hinge_vel_threshold", 0.7)),
        max_cart_velocity=float(args.max_cart_velocity if args.max_cart_velocity is not None else reward_cfg.get("upright_cart_vel_threshold", 0.45)),
        max_cart_abs=args.max_cart_abs,
        min_time=float(args.min_time),
        stride=int(args.stride),
        max_states=args.max_states,
        one_per_episode=bool(args.one_per_episode),
    )
    out = Path(args.out)
    dump_json(result, out)
    best = (result.get("states") or result.get("best_states") or [{}])[0]
    print(
        f"states={result['state_count']} episodes_with_state={result['episodes_with_selected_state']}/{result['episodes']} "
        f"progress={result['progress']:.3f} best_angle={best.get('max_abs_angle', float('nan')):.6f} "
        f"best_hinge={best.get('hinge_velocity_rms', float('nan')):.3f}"
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
