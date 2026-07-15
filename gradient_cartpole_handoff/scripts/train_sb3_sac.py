#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.capture_envelope import validate_capture_config
from gcartpole.config import apply_overrides, dump_json, load_config, save_config
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import select_evaluation_state_indices
from gcartpole.sac_trusted import TrustedResidualSAC


def require_sb3():
    try:
        from stable_baselines3 import SAC
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the optional backend with: pip install -r requirements-sb3.txt") from exc
    return SAC, BaseCallback, DummyVecEnv, VecMonitor


def evaluate_sb3_indexed(
    policy: Any,
    cfg: dict[str, Any],
    *,
    state_indices: list[int],
    seed: int,
    progress: float,
    batch_size: int,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    batch_size = max(1, min(int(batch_size), len(state_indices)))
    for batch_start in range(0, len(state_indices), batch_size):
        count = min(batch_size, len(state_indices) - batch_start)
        envs = [NLinkCartPoleEnv(cfg, progress=progress, seed=seed + batch_start + slot) for slot in range(count)]
        observations: dict[int, np.ndarray] = {}
        trackers = [{"return": 0.0, "length": 0} for _ in range(count)]
        active = set(range(count))
        try:
            for slot, env in enumerate(envs):
                state_index = int(state_indices[batch_start + slot])
                observations[slot], _ = env.reset(
                    seed=seed + batch_start + slot,
                    options={"state_index": state_index},
                )
            while active:
                slots = sorted(active)
                obs_batch = np.stack([observations[slot] for slot in slots])
                actions, _ = policy.predict(obs_batch, deterministic=True)
                for action_offset, slot in enumerate(slots):
                    env = envs[slot]
                    obs, reward, terminated, truncated, info = env.step(actions[action_offset])
                    observations[slot] = obs
                    trackers[slot]["return"] += float(reward)
                    trackers[slot]["length"] += 1
                    if not (terminated or truncated):
                        continue
                    active.remove(slot)
                    episode = batch_start + slot
                    records.append(
                        {
                            "episode": episode,
                            "state_index": int(state_indices[episode]),
                            "seed": seed + episode,
                            "return": float(trackers[slot]["return"]),
                            "length": int(trackers[slot]["length"]),
                            "success": bool(info.get("success", False)),
                            "termination_reason": info.get("termination_reason"),
                            "rail_hit": info.get("termination_reason") == "rail_violation",
                            "time_to_first_upright": info.get("time_to_first_upright"),
                            "time_to_capture": info.get("time_to_capture"),
                            "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
                            "final_upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
                            "max_cart_excursion": float(info.get("max_cart_excursion", 0.0)),
                        }
                    )
        finally:
            for env in envs:
                env.close()
    records.sort(key=lambda row: int(row["episode"]))
    successes = np.asarray([row["success"] for row in records], dtype=np.float64)
    holds = np.asarray([row["max_upright_streak_seconds"] for row in records], dtype=np.float64)
    returns = np.asarray([row["return"] for row in records], dtype=np.float64)
    return {
        "episodes": len(records),
        "success_rate": float(np.mean(successes)),
        "capture_success_count": int(np.sum(successes)),
        "max_upright_streak_mean": float(np.mean(holds)),
        "max_upright_streak_median": float(np.median(holds)),
        "return_mean": float(np.mean(returns)),
        "rail_hit_count": int(sum(bool(row["rail_hit"]) for row in records)),
        "episode_results": records,
    }


def evaluation_score(metrics: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(metrics["success_rate"]),
        float(metrics["max_upright_streak_median"]),
        float(metrics["max_upright_streak_mean"]),
        -float(metrics["rail_hit_count"]),
        float(metrics["return_mean"]),
    )


def main() -> None:
    SAC, BaseCallback, DummyVecEnv, VecMonitor = require_sb3()
    parser = argparse.ArgumentParser(description="Train off-policy SAC on a frozen capture-curriculum stage")
    parser.add_argument("--config", default="configs/swingup6_capture_sac_boundary.yaml")
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    sac_cfg = cfg["sac"]
    progress = float(sac_cfg["progress"])
    total_timesteps = int(args.total_timesteps or sac_cfg["total_timesteps"])
    out_dir = Path(args.out_dir or cfg["experiment"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.resolved.yaml")
    spec = load_config("benchmarks/p1_capture_envelope.yaml")
    errors = validate_capture_config(cfg, spec)
    if errors:
        raise ValueError("Capture benchmark validation failed:\n- " + "\n- ".join(errors))

    eval_cfg = copy.deepcopy(cfg)
    eval_cfg["env"]["init_mode"] = "state_list"
    eval_cfg["env"]["init_states_path"] = str(sac_cfg["eval_states_path"])
    eval_cfg["env"]["init_state_curriculum"] = "all"
    eval_indices = select_evaluation_state_indices(
        sac_cfg["eval_states_path"],
        int(sac_cfg["eval_episodes"]),
        int(sac_cfg["eval_seed"]),
    )

    seed = int(sac_cfg["seed"])

    def make_env(rank: int):
        def factory():
            return NLinkCartPoleEnv(copy.deepcopy(cfg), progress=progress, seed=seed + rank)

        return factory

    vec_env = VecMonitor(DummyVecEnv([make_env(rank) for rank in range(int(sac_cfg["num_envs"]))]))
    model = TrustedResidualSAC(
        "MlpPolicy",
        vec_env,
        learning_rate=float(sac_cfg["learning_rate"]),
        buffer_size=int(sac_cfg["buffer_size"]),
        learning_starts=int(sac_cfg["learning_starts"]),
        batch_size=int(sac_cfg["batch_size"]),
        tau=float(sac_cfg["tau"]),
        gamma=float(sac_cfg["gamma"]),
        train_freq=(int(sac_cfg["train_freq"]), "step"),
        gradient_steps=int(sac_cfg["gradient_steps"]),
        ent_coef=sac_cfg["ent_coef"],
        trusted_action_coef=float(sac_cfg.get("trusted_action_coef", 0.0)),
        policy_kwargs={"net_arch": list(sac_cfg["net_arch"])},
        device=str(sac_cfg["device"]),
        seed=seed,
        verbose=1,
    )
    # Preserve the known LQR scaffold at timestep zero: deterministic SAC
    # output is exactly zero residual until replay updates prove an improvement.
    actor_mu = model.actor.mu
    actor_mu.weight.data.zero_()
    actor_mu.bias.data.zero_()
    initial_metrics = evaluate_sb3_indexed(
        model,
        eval_cfg,
        state_indices=eval_indices,
        seed=int(sac_cfg["eval_seed"]),
        progress=progress,
        batch_size=int(sac_cfg["eval_batch_size"]),
    )
    initial_record = {"timesteps": 0, "score": list(evaluation_score(initial_metrics)), "metrics": initial_metrics}
    model.save(out_dir / "best_model.zip")
    dump_json(initial_record, out_dir / "eval_000000000.json")
    dump_json(initial_record, out_dir / "best_model.meta.json")
    print(
        f"eval steps=0 success={initial_metrics['success_rate']:.4f} "
        f"median_hold={initial_metrics['max_upright_streak_median']:.3f}s "
        f"rail={initial_metrics['rail_hit_count']}"
    )

    class CaptureEvalCallback(BaseCallback):
        def __init__(self) -> None:
            super().__init__(verbose=0)
            self.next_eval = int(sac_cfg["eval_every_timesteps"])
            self.best_score: tuple[float, ...] | None = evaluation_score(initial_metrics)
            self.evaluations: list[dict[str, Any]] = [initial_record]

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_eval:
                return True
            metrics = evaluate_sb3_indexed(
                self.model,
                eval_cfg,
                state_indices=eval_indices,
                seed=int(sac_cfg["eval_seed"]),
                progress=progress,
                batch_size=int(sac_cfg["eval_batch_size"]),
            )
            score = evaluation_score(metrics)
            record = {"timesteps": int(self.num_timesteps), "score": list(score), "metrics": metrics}
            self.evaluations.append(record)
            dump_json(record, out_dir / f"eval_{self.num_timesteps:09d}.json")
            if self.best_score is None or score > self.best_score:
                self.best_score = score
                self.model.save(out_dir / "best_model.zip")
                dump_json(record, out_dir / "best_model.meta.json")
            print(
                f"eval steps={self.num_timesteps} success={metrics['success_rate']:.4f} "
                f"median_hold={metrics['max_upright_streak_median']:.3f}s rail={metrics['rail_hit_count']}"
            )
            while self.next_eval <= self.num_timesteps:
                self.next_eval += int(sac_cfg["eval_every_timesteps"])
            return True

    callback = CaptureEvalCallback()
    started = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=False)
        final_metrics = evaluate_sb3_indexed(
            model,
            eval_cfg,
            state_indices=eval_indices,
            seed=int(sac_cfg["eval_seed"]),
            progress=progress,
            batch_size=int(sac_cfg["eval_batch_size"]),
        )
        model.save(out_dir / "final_model.zip")
    finally:
        vec_env.close()
    final_path = out_dir / "final_model.zip"
    payload = {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "SAC boundary-stage diagnostic; progress below 1.0 cannot pass P1.",
        "config_path": str(Path(args.config)),
        "progress": progress,
        "total_timesteps": total_timesteps,
        "wall_time_seconds": float(time.time() - started),
        "checkpoint": file_metadata(final_path),
        "eval_states": file_metadata(sac_cfg["eval_states_path"]),
        "eval_indices": eval_indices,
        "final_eval": final_metrics,
        "evaluations": callback.evaluations,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, out_dir / "training_receipt.json")
    print(f"Wrote {out_dir / 'training_receipt.json'}")


if __name__ == "__main__":
    main()
