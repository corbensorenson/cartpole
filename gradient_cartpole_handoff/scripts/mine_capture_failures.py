#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, evaluate_indexed_policy_batched, load_model


def build_mining_mixture(
    states: list[dict[str, Any]],
    sampled_indices: list[int],
    failure_indices: list[int],
    hard_repeat: int,
) -> list[dict[str, Any]]:
    mixture = []
    failure_set = set(failure_indices)
    for index in sampled_indices:
        row = copy.deepcopy(states[index])
        row["mining"] = {"source_index": index, "hard_failure": index in failure_set, "copy": 0}
        mixture.append(row)
    for copy_index in range(1, hard_repeat + 1):
        for index in failure_indices:
            row = copy.deepcopy(states[index])
            row["state_id"] = f"{row['state_id']}-hard{copy_index}"
            row["mining"] = {"source_index": index, "hard_failure": True, "copy": copy_index}
            mixture.append(row)
    return mixture


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine failed capture states from the training split")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/train.json")
    parser.add_argument("--progress", type=float, required=True)
    parser.add_argument("--sample-count", type=int, default=2048)
    parser.add_argument("--hard-repeat", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=61301)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    dataset_path = Path(args.dataset)
    source = json.loads(dataset_path.read_text(encoding="utf-8"))
    states = source.get("states", source) if isinstance(source, dict) else source
    if not isinstance(states, list) or not states:
        raise ValueError("capture training dataset must contain states")
    if args.sample_count < 1 or args.sample_count > len(states):
        raise ValueError(f"--sample-count must be in 1..{len(states)}")
    if args.hard_repeat < 0:
        raise ValueError("--hard-repeat must be nonnegative")

    rng = np.random.default_rng(args.seed)
    sampled_indices = rng.permutation(len(states))[: args.sample_count].astype(int).tolist()
    cfg["env"]["init_mode"] = "state_list"
    cfg["env"]["init_states_path"] = str(dataset_path)
    probe = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    model = ActorCritic(
        probe.observation_space.shape[0],
        probe.action_space.shape[0],
        list(cfg["ppo"].get("hidden_sizes", [256, 256])),
        float(cfg["ppo"].get("action_std_init", 0.7)),
    )
    probe.close()
    mx.eval(model.parameters())
    load_model(model, args.checkpoint)
    metrics = evaluate_indexed_policy_batched(
        cfg,
        model,
        sampled_indices,
        seed=args.seed,
        progress=args.progress,
        batch_size=args.batch_size,
    )
    failure_indices = [
        int(row["state_index"]) for row in metrics["episode_results"] if not bool(row["success"])
    ]
    mixture = build_mining_mixture(states, sampled_indices, failure_indices, args.hard_repeat)
    payload = {
        "schema_version": 1,
        "benchmark": source.get("benchmark") if isinstance(source, dict) else None,
        "source": "capture_hard_negative_mining",
        "generated_at": utc_timestamp(),
        "progress": float(args.progress),
        "seed": int(args.seed),
        "sample_count": len(sampled_indices),
        "failure_count": len(failure_indices),
        "failure_rate": float(len(failure_indices) / len(sampled_indices)),
        "hard_repeat": int(args.hard_repeat),
        "mixture_count": len(mixture),
        "sampled_indices": sampled_indices,
        "failure_indices": failure_indices,
        "evaluation": metrics,
        "states": mixture,
        "evidence": {
            "config_path": str(Path(args.config)),
            "overrides": list(args.override),
            "checkpoint": file_metadata(args.checkpoint),
            "dataset": file_metadata(dataset_path),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"sampled={len(sampled_indices)} failures={len(failure_indices)} "
        f"failure_rate={payload['failure_rate']:.3f} mixture={len(mixture)}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
