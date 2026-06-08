#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config, save_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, evaluate_policy, load_model, save_model


def load_linear_actor(path: str | None, obs_dim: int) -> tuple[np.ndarray, float]:
    if path is None:
        return np.zeros(obs_dim, dtype=np.float64), 0.0
    raw = mx.load(str(path))
    weight = np.asarray(raw["actor_out.weight"], dtype=np.float64).reshape(-1)
    bias = float(np.asarray(raw["actor_out.bias"], dtype=np.float64).reshape(-1)[0])
    if weight.shape != (obs_dim,):
        raise ValueError(f"{path} actor_out.weight has shape {weight.shape}; expected {(obs_dim,)}")
    return weight, bias


def build_sigma(
    *,
    n_links: int,
    obs_dim: int,
    time_feature_count: int,
    cart_sigma: float,
    cart_vel_sigma: float,
    angle_sigma: float,
    hinge_vel_sigma: float,
    morphology_sigma: float,
    time_sigma: float,
    bias_sigma: float,
) -> np.ndarray:
    sigma = np.zeros(obs_dim + 1, dtype=np.float64)
    sigma[0] = cart_sigma
    sigma[1] = cart_vel_sigma
    sigma[2 : 2 + 2 * n_links] = angle_sigma
    rel_start = 2 + 2 * n_links
    vel_start = 2 + 3 * n_links
    sigma[rel_start : rel_start + n_links] = angle_sigma
    sigma[vel_start : vel_start + n_links] = hinge_vel_sigma
    sigma[2 + 4 * n_links : obs_dim] = morphology_sigma
    if time_feature_count > 0:
        sigma[obs_dim - time_feature_count : obs_dim] = time_sigma
    sigma[-1] = bias_sigma
    return sigma


def time_feature_count(cfg: dict[str, Any]) -> int:
    env_cfg = cfg.get("env", {})
    if not bool(env_cfg.get("obs_include_time", False)):
        return 0
    return 1 + 2 * len(list(env_cfg.get("obs_time_frequencies", [])))


def vector_to_actor(vec: np.ndarray, obs_dim: int) -> tuple[np.ndarray, float]:
    return np.asarray(vec[:obs_dim], dtype=np.float64), float(vec[obs_dim])


def set_linear_actor(model: ActorCritic, weight: np.ndarray, bias: float, action_std: float) -> None:
    obs_dim = int(weight.shape[0])
    params = {
        "actor_out": {
            "weight": mx.array(np.asarray(weight, dtype=np.float32).reshape(1, obs_dim)),
            "bias": mx.array([float(bias)], dtype=mx.float32),
        },
        "critic_out": {
            "weight": mx.zeros((1, obs_dim), dtype=mx.float32),
            "bias": mx.zeros((1,), dtype=mx.float32),
        },
        "log_std": mx.array([float(np.log(max(1e-6, action_std)))], dtype=mx.float32),
    }
    model.update(params)
    mx.eval(model.parameters())


def rollout_linear_actor(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    weight: np.ndarray,
    bias: float,
    zero_noise: bool,
    seconds: float | None,
) -> dict[str, Any]:
    cfg = {**cfg, "env": {**cfg["env"]}}
    if zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
    if seconds is not None:
        cfg["env"]["episode_seconds"] = float(seconds)

    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    obs, _ = env.reset()
    done = False
    ep_return = 0.0
    ep_len = 0
    action_sq = 0.0
    action_smooth = 0.0
    prev_action = 0.0
    best_angle = float("inf")
    best_hinge_rms = float("inf")
    best_cart_abs = float("inf")
    best_cart_vel_abs = float("inf")
    max_cart_abs = 0.0
    max_capture_quality = 0.0
    final_info: dict[str, Any] = {}
    terminated = False
    truncated = False

    while not done:
        action = float(np.tanh(float(np.dot(obs, weight) + bias)))
        obs, reward, terminated, truncated, info = env.step([action])
        ep_return += float(reward)
        ep_len += 1
        action_sq += action * action
        action_smooth += (action - prev_action) ** 2
        prev_action = action
        max_cart_abs = max(max_cart_abs, abs(float(info.get("x", 0.0))))
        max_capture_quality = max(max_capture_quality, float(info.get("capture_quality", 0.0)))
        angle = float(info.get("max_abs_angle", np.inf))
        hinge = float(info.get("hinge_velocity_rms", np.inf))
        cart_abs = abs(float(info.get("x", 0.0)))
        cart_vel_abs = abs(float(env.data.qvel[0]))
        if (angle, hinge, cart_abs, cart_vel_abs) < (best_angle, best_hinge_rms, best_cart_abs, best_cart_vel_abs):
            best_angle = angle
            best_hinge_rms = hinge
            best_cart_abs = cart_abs
            best_cart_vel_abs = cart_vel_abs
        final_info = dict(info)
        done = bool(terminated or truncated)

    success = bool(final_info.get("success", False))
    ever_upright = final_info.get("time_to_first_upright") is not None
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    max_centered_streak = float(final_info.get("max_centered_upright_streak_seconds", 0.0))
    max_low_momentum_streak = float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0))
    rail_over = max(0.0, max_cart_abs - float(cfg["env"]["rail_limit"]))
    score = (
        -float(ep_return)
        - 50000.0 * float(success)
        - 7000.0 * max_streak
        - 4500.0 * max_centered_streak
        - 6500.0 * max_low_momentum_streak
        - 1500.0 * float(ever_upright)
        - 1200.0 * max_capture_quality
        + 22.0 * best_angle
        + 6.0 * best_hinge_rms
        + 2.0 * best_cart_abs
        + 1.0 * best_cart_vel_abs
        + 90.0 * float(terminated and not success)
        + 140.0 * rail_over * rail_over
        + 0.01 * action_sq
        + 0.04 * action_smooth
    )
    env.close()
    return {
        "score": float(score),
        "return": float(ep_return),
        "length": int(ep_len),
        "success": success,
        "ever_upright": bool(ever_upright),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "best_max_abs_angle": float(best_angle),
        "best_hinge_velocity_rms": float(best_hinge_rms),
        "best_cart_abs": float(best_cart_abs),
        "best_cart_velocity_abs": float(best_cart_vel_abs),
        "max_cart_abs": float(max_cart_abs),
        "max_capture_quality": float(max_capture_quality),
        "max_upright_streak_seconds": max_streak,
        "max_centered_upright_streak_seconds": max_centered_streak,
        "max_low_momentum_upright_streak_seconds": max_low_momentum_streak,
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "final_info": final_info,
    }


def evaluate_candidate(
    cfg: dict[str, Any],
    *,
    progress: float,
    seed: int,
    episodes: int,
    weight: np.ndarray,
    bias: float,
    zero_noise: bool,
    seconds: float | None,
) -> dict[str, Any]:
    rows = [
        rollout_linear_actor(
            cfg,
            progress=progress,
            seed=seed + ep,
            weight=weight,
            bias=bias,
            zero_noise=zero_noise,
            seconds=seconds,
        )
        for ep in range(episodes)
    ]
    return {
        "score": float(np.mean([row["score"] for row in rows])),
        "return_mean": float(np.mean([row["return"] for row in rows])),
        "success_rate": float(np.mean([float(row["success"]) for row in rows])),
        "ever_upright_rate": float(np.mean([float(row["ever_upright"]) for row in rows])),
        "terminated_rate": float(np.mean([float(row["terminated"] and not row["success"]) for row in rows])),
        "max_upright_streak_mean": float(np.mean([row["max_upright_streak_seconds"] for row in rows])),
        "max_upright_streak_max": float(np.max([row["max_upright_streak_seconds"] for row in rows])),
        "max_low_momentum_upright_streak_mean": float(
            np.mean([row["max_low_momentum_upright_streak_seconds"] for row in rows])
        ),
        "max_capture_quality_mean": float(np.mean([row["max_capture_quality"] for row in rows])),
        "max_capture_quality_max": float(np.max([row["max_capture_quality"] for row in rows])),
        "best_max_abs_angle_min": float(np.min([row["best_max_abs_angle"] for row in rows])),
        "best_hinge_velocity_rms_min": float(np.min([row["best_hinge_velocity_rms"] for row in rows])),
        "episodes": rows,
    }


def evaluate_candidate_job(job: dict[str, Any]) -> dict[str, Any]:
    vector = np.asarray(job["vector"], dtype=np.float64)
    weight, bias = vector_to_actor(vector, int(job["obs_dim"]))
    metrics = evaluate_candidate(
        job["cfg"],
        progress=float(job["progress"]),
        seed=int(job["seed"]),
        episodes=int(job["episodes"]),
        weight=weight,
        bias=bias,
        zero_noise=bool(job["zero_noise"]),
        seconds=job["seconds"],
    )
    return {"score": float(metrics["score"]), "vector": vector, "metrics": metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="CEM search for a closed-loop linear tanh swing-up policy")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--out-dir", default="runs/swingup6_linear_policy_search")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--zero-noise", action="store_true")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--sigma-decay", type=float, default=0.88)
    parser.add_argument("--sigma-floor", type=float, default=0.002)
    parser.add_argument("--cart-sigma", type=float, default=0.25)
    parser.add_argument("--cart-vel-sigma", type=float, default=0.12)
    parser.add_argument("--angle-sigma", type=float, default=0.45)
    parser.add_argument("--hinge-vel-sigma", type=float, default=0.12)
    parser.add_argument("--morphology-sigma", type=float, default=0.0)
    parser.add_argument("--time-sigma", type=float, default=0.75)
    parser.add_argument("--bias-sigma", type=float, default=0.35)
    parser.add_argument("--action-std", type=float, default=0.03)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.population < 1:
        raise ValueError("--population must be >= 1")
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("--elites must be in 1..population")
    if args.episodes < 1 or args.eval_episodes < 1:
        raise ValueError("--episodes and --eval-episodes must be >= 1")

    cfg = apply_overrides(load_config(args.config), args.override)
    hidden_sizes = list(cfg.get("ppo", {}).get("hidden_sizes", []))
    if hidden_sizes:
        raise ValueError("search_linear_policy requires a linear policy; pass --override 'ppo.hidden_sizes=[]'")
    cfg.setdefault("ppo", {})["hidden_sizes"] = []
    cfg["ppo"]["action_std_init"] = float(args.action_std)
    if args.seconds is not None:
        cfg["env"]["episode_seconds"] = float(args.seconds)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.resolved.yaml")

    probe = NLinkCartPoleEnv(cfg, progress=args.progress, seed=int(cfg["experiment"].get("seed", 0)))
    obs_dim = int(probe.observation_space.shape[0])
    act_dim = int(probe.action_space.shape[0])
    n_links = int(probe.n)
    probe.close()
    if act_dim != 1:
        raise ValueError(f"expected one action, got {act_dim}")

    init_weight, init_bias = load_linear_actor(args.init_checkpoint, obs_dim)
    center = np.r_[init_weight, init_bias].astype(np.float64)
    sigma = build_sigma(
        n_links=n_links,
        obs_dim=obs_dim,
        time_feature_count=time_feature_count(cfg),
        cart_sigma=args.cart_sigma,
        cart_vel_sigma=args.cart_vel_sigma,
        angle_sigma=args.angle_sigma,
        hinge_vel_sigma=args.hinge_vel_sigma,
        morphology_sigma=args.morphology_sigma,
        time_sigma=args.time_sigma,
        bias_sigma=args.bias_sigma,
    )
    rng = np.random.default_rng(args.seed)
    best: dict[str, Any] | None = None
    best_vec = center.copy()
    history: list[dict[str, Any]] = []

    executor = ProcessPoolExecutor(max_workers=int(args.workers)) if int(args.workers) > 1 else None
    try:
        for iteration in range(max(1, args.iterations + 1)):
            if iteration == 0:
                candidates = [center.copy()]
            else:
                candidates = [center.copy()]
                for _ in range(args.population - 1):
                    candidates.append(center + rng.normal(0.0, sigma))
            jobs = [
                {
                    "cfg": cfg,
                    "progress": float(args.progress),
                    "seed": int(args.seed + 1000 * iteration),
                    "episodes": int(args.episodes),
                    "vector": np.asarray(candidate, dtype=np.float64),
                    "obs_dim": int(obs_dim),
                    "zero_noise": bool(args.zero_noise),
                    "seconds": args.seconds,
                }
                for candidate in candidates
            ]
            if executor is None:
                records = [evaluate_candidate_job(job) for job in jobs]
            else:
                records = list(executor.map(evaluate_candidate_job, jobs))
            records.sort(key=lambda row: float(row["score"]))
            if best is None or float(records[0]["score"]) < float(best["score"]):
                best = {k: v for k, v in records[0].items() if k != "vector"}
                best_vec = np.asarray(records[0]["vector"], dtype=np.float64).copy()

            top = records[0]["metrics"]
            history.append(
                {
                    "iteration": int(iteration),
                    "score": float(records[0]["score"]),
                    "best_score": None if best is None else float(best["score"]),
                    "return_mean": float(top["return_mean"]),
                    "success_rate": float(top["success_rate"]),
                    "ever_upright_rate": float(top["ever_upright_rate"]),
                    "terminated_rate": float(top["terminated_rate"]),
                    "max_upright_streak_mean": float(top["max_upright_streak_mean"]),
                    "max_upright_streak_max": float(top["max_upright_streak_max"]),
                    "max_capture_quality_max": float(top["max_capture_quality_max"]),
                    "best_max_abs_angle_min": float(top["best_max_abs_angle_min"]),
                }
            )
            print(
                f"iter={iteration:03d} score={records[0]['score']:.3f} "
                f"ret={top['return_mean']:.1f} succ={top['success_rate']:.2f} "
                f"ever={top['ever_upright_rate']:.2f} streak={top['max_upright_streak_max']:.3f}s "
                f"angle={top['best_max_abs_angle_min']:.3f}"
            )

            if iteration > 0:
                elite_arr = np.asarray([row["vector"] for row in records[: args.elites]], dtype=np.float64)
                center = elite_arr.mean(axis=0)
                sigma = np.maximum(elite_arr.std(axis=0), float(args.sigma_floor)) * float(args.sigma_decay)
    finally:
        if executor is not None:
            executor.shutdown()

    assert best is not None
    best_weight, best_bias = vector_to_actor(best_vec, obs_dim)
    model = ActorCritic(obs_dim, act_dim, [], float(args.action_std))
    set_linear_actor(model, best_weight, best_bias, float(args.action_std))
    checkpoint = ckpt_dir / "best.safetensors"
    save_model(model, checkpoint)
    eval_metrics = evaluate_policy(
        cfg,
        model,
        episodes=int(args.eval_episodes),
        seed=int(cfg["experiment"].get("seed", 0)) + 4444,
        progress=float(args.progress),
        return_episodes=True,
    )

    payload = {
        "generated_at": utc_timestamp(),
        "not_solution": bool(eval_metrics.get("success_rate", 0.0) < 0.80),
        "summary": "Closed-loop linear tanh CEM search for the hanging-start six-link swing-up task.",
        "config_path": str(Path(args.config)),
        "out_dir": str(out_dir),
        "checkpoint": file_metadata(checkpoint),
        "init_checkpoint": None if args.init_checkpoint is None else file_metadata(args.init_checkpoint),
        "progress": float(args.progress),
        "zero_noise": bool(args.zero_noise),
        "search": {
            "seed": int(args.seed),
            "episodes": int(args.episodes),
            "eval_episodes": int(args.eval_episodes),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "workers": int(args.workers),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
        },
        "best_search_record": best,
        "eval": eval_metrics,
        "actor": {
            "weight": best_weight.astype(float).tolist(),
            "bias": float(best_bias),
        },
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, out_dir / "search_linear_policy.json")
    dump_json(
        {
            "generated_at": payload["generated_at"],
            "method": "cem_linear_tanh_policy",
            "checkpoint": str(checkpoint),
            "config_resolved_sha256": payload["config_sha256"],
            "search_result": str(out_dir / "search_linear_policy.json"),
            "eval_success_rate": float(eval_metrics.get("success_rate", 0.0)),
            "eval_max_upright_streak_max": float(eval_metrics.get("max_upright_streak_max", 0.0)),
            "runtime": payload["runtime"],
            "git": payload["git"],
        },
        ckpt_dir / "best.meta.json",
    )
    print(
        f"wrote {checkpoint} eval_success={eval_metrics.get('success_rate', 0.0):.2f} "
        f"max_streak={eval_metrics.get('max_upright_streak_max', 0.0):.3f}s"
    )


if __name__ == "__main__":
    main()
