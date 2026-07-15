#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from gcartpole.capture_envelope import validate_capture_config, validate_capture_states
from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, load_model, sample_action
from gcartpole.trajectory_policy import trajectory_conditioned_features


def selected_indices(
    state_count: int,
    metadata: dict,
    source_partition: str | None,
    limit: int | None,
) -> list[int]:
    if source_partition is None:
        result = list(range(state_count))
    elif source_partition == "all":
        result = sorted(
            set(metadata["train_sources"]) | set(metadata["validation_sources"])
        )
    else:
        result = [int(value) for value in metadata[f"{source_partition}_sources"]]
    if any(index < 0 or index >= state_count for index in result):
        raise IndexError("selected source index is outside the dataset")
    return result if limit is None else result[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an initial-state and time-conditioned capture policy"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_hybrid_policy.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--source-partition", choices=("train", "validation", "all"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=61201)
    parser.add_argument("--progress", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    spec = load_config(args.spec)
    dataset_path = Path(args.dataset)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    errors = validate_capture_states(dataset, spec, args.split)
    errors.extend(validate_capture_config(cfg, spec))
    if errors:
        raise ValueError(f"capture benchmark validation failed: {errors[:20]}")
    metadata_path = Path(args.metadata)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    dataset_metadata = file_metadata(dataset_path)
    if (
        args.source_partition is not None
        and dataset_metadata["sha256"] != metadata["dataset"]["sha256"]
    ):
        raise ValueError("source partitions require the supervisor training dataset")
    progress = float(metadata["progress"] if args.progress is None else args.progress)
    states = dataset["states"]
    indices = selected_indices(
        len(states), metadata, args.source_partition, args.limit
    )
    if not indices:
        raise ValueError("evaluation selected no states")
    cfg["env"]["init_states_path"] = str(dataset_path)
    batch_size = max(1, min(args.batch_size, len(indices)))
    envs = [
        NLinkCartPoleEnv(cfg, progress=progress, seed=args.seed + slot)
        for slot in range(batch_size)
    ]
    model = ActorCritic(
        int(metadata["policy_input_dim"]),
        1,
        [int(value) for value in metadata["hidden_sizes"]],
        0.20,
    )
    mx.eval(model.parameters())
    load_model(model, args.checkpoint)
    maximum_steps = int(metadata["maximum_steps"])
    episode_results = []
    try:
        for batch_start in range(0, len(indices), batch_size):
            batch_indices = indices[batch_start : batch_start + batch_size]
            observations = []
            initial_observations = []
            returns = np.zeros(len(batch_indices), dtype=np.float64)
            steps = np.zeros(len(batch_indices), dtype=np.int64)
            active = set(range(len(batch_indices)))
            for slot, state_index in enumerate(batch_indices):
                observation, _ = envs[slot].reset(
                    seed=args.seed + state_index,
                    options={"state_index": state_index},
                )
                observations.append(np.asarray(observation, dtype=np.float32))
                initial_observations.append(np.asarray(observation, dtype=np.float32))
            while active:
                slots = sorted(active)
                features = trajectory_conditioned_features(
                    np.stack([observations[slot] for slot in slots]),
                    np.stack([initial_observations[slot] for slot in slots]),
                    steps[slots],
                    maximum_steps=maximum_steps,
                    include_initial_observation=(
                        metadata.get("feature_mode", "current_initial_time")
                        == "current_initial_time"
                    ),
                )
                actions, _, _ = sample_action(model, features, deterministic=True)
                for action_index, slot in enumerate(slots):
                    state_index = batch_indices[slot]
                    observation, reward, terminated, truncated, info = envs[slot].step(
                        actions[action_index]
                    )
                    observations[slot] = np.asarray(observation, dtype=np.float32)
                    returns[slot] += float(reward)
                    steps[slot] += 1
                    if not (terminated or truncated):
                        continue
                    active.remove(slot)
                    rail_hit = info.get("termination_reason") == "rail_violation"
                    episode_results.append(
                        {
                            "state_index": state_index,
                            "state_id": states[state_index]["state_id"],
                            "seed": args.seed + state_index,
                            "return": float(returns[slot]),
                            "length": int(steps[slot]),
                            "success": bool(info.get("success", False)),
                            "rail_hit": rail_hit,
                            "termination_reason": info.get("termination_reason"),
                            "time_to_first_upright": info.get("time_to_first_upright"),
                            "time_to_capture": info.get("time_to_capture"),
                            "max_upright_streak_seconds": float(
                                info.get("max_upright_streak_seconds", 0.0)
                            ),
                            "final_upright_streak_seconds": float(
                                info.get("upright_streak_seconds", 0.0)
                            ),
                            "max_cart_excursion": float(
                                info.get("max_cart_excursion", 0.0)
                            ),
                            "lqr_switch_entry_count": int(
                                info.get("lqr_switch_entry_count", 0)
                            ),
                            "lqr_switch_exit_count": int(
                                info.get("lqr_switch_exit_count", 0)
                            ),
                            "lqr_switch_policy_steps": int(
                                info.get("lqr_switch_policy_steps", 0)
                            ),
                            "lqr_switch_lqr_steps": int(
                                info.get("lqr_switch_lqr_steps", 0)
                            ),
                        }
                    )
    finally:
        for env in envs:
            env.close()
    episode_results.sort(key=lambda row: row["state_index"])
    successes = np.asarray([row["success"] for row in episode_results])
    holds = np.asarray(
        [row["max_upright_streak_seconds"] for row in episode_results]
    )
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Closed-loop trajectory-conditioned guided policy diagnostic.",
        "split": args.split,
        "source_partition": args.source_partition,
        "progress": progress,
        "episodes": len(episode_results),
        "capture_success_count": int(np.sum(successes)),
        "success_rate": float(np.mean(successes)),
        "max_upright_streak_median": float(np.median(holds)),
        "rail_hit_count": int(sum(row["rail_hit"] for row in episode_results)),
        "episode_results": episode_results,
        "evidence": {
            "config": {"path": args.config, "resolved_sha256": data_sha256(cfg)},
            "checkpoint": file_metadata(args.checkpoint),
            "metadata": file_metadata(metadata_path),
            "dataset": dataset_metadata,
            "spec": file_metadata(args.spec),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(result, args.out)
    print(
        f"success={result['capture_success_count']}/{result['episodes']} "
        f"rate={result['success_rate']:.4f} "
        f"median_hold={result['max_upright_streak_median']:.3f}s "
        f"rail_hits={result['rail_hit_count']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
