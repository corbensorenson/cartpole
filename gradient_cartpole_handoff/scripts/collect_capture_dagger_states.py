#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, load_model, sample_action

try:
    from scripts.build_feedback_mpc_teachers import failed_state_indices
except ModuleNotFoundError:
    from build_feedback_mpc_teachers import failed_state_indices


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect learner-controlled capture states from the frozen training split"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_hybrid_policy.yaml")
    parser.add_argument(
        "--checkpoint",
        default="runs/swingup6_capture_hybrid_lyapunov_ppo_probe_100/checkpoints/best.safetensors",
    )
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/train.json")
    parser.add_argument("--rollouts", default="runs/p1_capture_envelope/train_hard_007.json")
    parser.add_argument("--progress", type=float, default=0.07)
    parser.add_argument("--failure-limit", type=int, default=32)
    parser.add_argument("--query-stride", type=int, default=5)
    parser.add_argument("--queries-per-source", type=int, default=8)
    parser.add_argument("--seed", type=int, default=79401)
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--out", default="runs/p1_capture_dagger/round1_train_queries.json")
    args = parser.parse_args()
    if min(args.failure_limit, args.query_stride, args.queries_per_source, args.round) < 1:
        raise ValueError("failure limit, query cadence, and round must be positive")

    config_path = Path(args.config)
    checkpoint_path = Path(args.checkpoint)
    dataset_path = Path(args.dataset)
    rollout_path = Path(args.rollouts)
    cfg = load_config(config_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rollouts = json.loads(rollout_path.read_text(encoding="utf-8"))
    if dataset.get("split") != "train":
        raise ValueError("DAgger state collection is restricted to the frozen training split")
    states = dataset.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("training dataset must contain states")
    source_indices = failed_state_indices(rollouts)[: args.failure_limit]
    cfg["env"]["init_states_path"] = str(dataset_path)
    switch_cfg = cfg["env"].get("action_lqr_switch", {})
    if not bool(switch_cfg.get("enabled", False)):
        raise ValueError("DAgger collection requires the hybrid policy/LQR switch")

    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    model = ActorCritic(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        list(cfg["ppo"].get("hidden_sizes", [256, 256])),
        float(cfg["ppo"].get("action_std_init", 0.20)),
    )
    mx.eval(model.parameters())
    load_model(model, checkpoint_path)
    queries: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    try:
        for ordinal, source_index in enumerate(source_indices):
            observation, _ = env.reset(
                seed=args.seed + ordinal,
                options={"state_index": source_index},
            )
            terminated = truncated = False
            episode_return = 0.0
            collected = 0
            final_info: dict[str, Any] = {}
            while not (terminated or truncated):
                action_batch, _, _ = sample_action(
                    model,
                    observation[None, :],
                    deterministic=True,
                )
                learner_action = float(action_batch[0, 0])
                policy_controlled = bool(
                    not env.lqr_switch_active
                    and not env._within_lqr_switch_bounds(switch_cfg, "enter")
                )
                if (
                    policy_controlled
                    and env.step_count % args.query_stride == 0
                    and collected < args.queries_per_source
                ):
                    lqr_action, _ = env._lqr_action_bias(switch_cfg)
                    qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
                    qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
                    state_payload = {
                        "round": int(args.round),
                        "source_ordinal": ordinal,
                        "source_index": source_index,
                        "source_state_id": states[source_index].get("state_id"),
                        "step": int(env.step_count),
                        "time_seconds": float(env.step_count * env.dt),
                        "qpos": qpos.astype(float).tolist(),
                        "qvel": qvel.astype(float).tolist(),
                        "observation": observation.astype(float).tolist(),
                        "learner_action": learner_action,
                        "lqr_action": float(lqr_action),
                        "action_disagreement_abs": abs(learner_action - float(lqr_action)),
                        "dimensionless_lyapunov_value": float(
                            env._lqr_switch_lyapunov_value(switch_cfg)
                        ),
                    }
                    state_payload["query_id"] = (
                        f"r{args.round}-s{source_index}-t{env.step_count}-"
                        f"{data_sha256(state_payload)[:12]}"
                    )
                    queries.append(state_payload)
                    collected += 1
                observation, reward, terminated, truncated, final_info = env.step(
                    [learner_action]
                )
                episode_return += float(reward)
            episodes.append(
                {
                    "source_ordinal": ordinal,
                    "source_index": source_index,
                    "source_state_id": states[source_index].get("state_id"),
                    "query_count": collected,
                    "success": bool(final_info.get("success", False)),
                    "return": float(episode_return),
                    "length": int(env.step_count),
                    "termination_reason": final_info.get("termination_reason"),
                    "max_upright_streak_seconds": float(
                        final_info.get("max_upright_streak_seconds", 0.0)
                    ),
                    "max_cart_excursion": float(final_info.get("max_cart_excursion", 0.0)),
                    "lqr_switch_entry_count": int(
                        final_info.get("lqr_switch_entry_count", 0)
                    ),
                    "lqr_switch_exit_count": int(final_info.get("lqr_switch_exit_count", 0)),
                    "lqr_switch_policy_steps": int(
                        final_info.get("lqr_switch_policy_steps", 0)
                    ),
                    "lqr_switch_lqr_steps": int(final_info.get("lqr_switch_lqr_steps", 0)),
                }
            )
            print(
                f"source={ordinal + 1}/{len(source_indices)} index={source_index} "
                f"queries={collected} success={episodes[-1]['success']} "
                f"hold={episodes[-1]['max_upright_streak_seconds']:.3f}s",
                flush=True,
            )
    finally:
        env.close()

    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Policy-controlled DAgger queries from frozen training failures only.",
        "split": "train",
        "round": int(args.round),
        "progress": float(args.progress),
        "source_count": len(source_indices),
        "query_count": len(queries),
        "successful_learner_episode_count": int(sum(row["success"] for row in episodes)),
        "query_stride": int(args.query_stride),
        "queries_per_source": int(args.queries_per_source),
        "queries_sha256": data_sha256(queries),
        "queries": queries,
        "episodes": episodes,
        "evidence": {
            "config": file_metadata(config_path),
            "checkpoint": file_metadata(checkpoint_path),
            "dataset": file_metadata(dataset_path),
            "rollouts": file_metadata(rollout_path),
            "source_indices": source_indices,
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(f"queries={len(queries)} sources={len(source_indices)} Wrote {args.out}")


if __name__ == "__main__":
    main()
