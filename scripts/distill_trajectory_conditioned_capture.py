#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from gcartpole.config import dump_json, load_config, save_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, load_model, save_model
from gcartpole.trajectory_policy import trajectory_conditioned_features

try:
    from scripts.distill_capture_supervisor import (
        grouped_source_split,
        prediction_metrics,
        source_balancing_weights,
        weighted_distillation_loss,
    )
except ModuleNotFoundError:
    from distill_capture_supervisor import (
        grouped_source_split,
        prediction_metrics,
        source_balancing_weights,
        weighted_distillation_loss,
    )


def initial_observations_by_source(
    cfg: dict[str, Any],
    source_indices: np.ndarray,
    *,
    progress: float,
    seed: int,
) -> dict[int, np.ndarray]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    observations = {}
    try:
        for source in sorted(set(int(value) for value in source_indices)):
            observation, _ = env.reset(
                seed=seed + source,
                options={"state_index": source},
            )
            observations[source] = np.asarray(observation, dtype=np.float32)
    finally:
        env.close()
    return observations


def parse_hidden_sizes(text: str) -> list[int]:
    sizes = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("hidden sizes must be positive comma-separated integers")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill capture feedback with initial-state and trajectory-time context"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_hybrid_policy.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/train.json")
    parser.add_argument("--labels", default="runs/p1_capture_supervisor/train_labels_p007.npz")
    parser.add_argument("--out-dir", default="runs/swingup6_capture_trajectory_distilled")
    parser.add_argument("--progress", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--hidden-sizes", default="512,512,256")
    parser.add_argument(
        "--feature-mode",
        choices=("current_time", "current_initial_time"),
        default="current_initial_time",
    )
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--early-weight", type=float, default=40.0)
    parser.add_argument("--early-decay-steps", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=79301)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.learning_rate, args.early_decay_steps) <= 0:
        raise ValueError("training counts and scales must be positive")
    if args.early_weight < 0.0 or not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("invalid weighting or validation fraction")
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    cfg = load_config(args.config)
    cfg["env"]["init_states_path"] = args.dataset
    labels = np.load(args.labels)
    observations = np.asarray(labels["observations"], dtype=np.float32)
    targets = np.asarray(labels["actions"], dtype=np.float32)
    source_indices = np.asarray(labels["source_indices"], dtype=np.int64)
    source_steps = np.asarray(labels["source_steps"], dtype=np.int64)
    tiers = np.asarray(labels["tiers"]).astype(str)
    if observations.ndim != 2 or targets.shape != (len(observations), 1):
        raise ValueError("supervisor observations or actions have invalid shapes")
    initial_by_source = initial_observations_by_source(
        cfg, source_indices, progress=args.progress, seed=args.seed
    )
    initial_observations = np.stack(
        [initial_by_source[int(source)] for source in source_indices]
    )
    policy_dt = float(cfg["env"]["timestep"] * cfg["env"]["frame_skip"])
    maximum_steps = int(round(float(cfg["env"]["episode_seconds"]) / policy_dt))
    features = trajectory_conditioned_features(
        observations,
        initial_observations,
        source_steps,
        maximum_steps=maximum_steps,
        include_initial_observation=args.feature_mode == "current_initial_time",
    )
    train_indices, validation_indices = grouped_source_split(
        source_indices,
        tiers,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    weights = source_balancing_weights(
        source_indices,
        source_steps,
        early_weight=args.early_weight,
        early_decay_steps=args.early_decay_steps,
    )

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.resolved.yaml")
    model = ActorCritic(features.shape[1], 1, hidden_sizes, 0.20)
    mx.eval(model.parameters())
    rng = np.random.default_rng(args.seed)
    mx.random.seed(args.seed)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(model, weighted_distillation_loss)
    best_validation_mse = float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_indices = rng.permutation(train_indices)
        train_losses = []
        for start in range(0, len(epoch_indices), args.batch_size):
            batch = epoch_indices[start : start + args.batch_size]
            loss, grads = loss_and_grad(
                model,
                mx.array(features[batch]),
                mx.array(targets[batch]),
                mx.array(weights[batch]),
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            train_losses.append(float(loss))
        validation = prediction_metrics(
            model,
            features[validation_indices],
            targets[validation_indices],
            weights[validation_indices],
        )
        if validation["weighted_mse"] < best_validation_mse:
            best_validation_mse = validation["weighted_mse"]
            save_model(model, checkpoint_dir / "best.safetensors")
        row = {
            "epoch": epoch,
            "weighted_train_mse": float(np.mean(train_losses)),
            "validation_mse": validation["weighted_mse"],
        }
        history.append(row)
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d}/{args.epochs} train={row['weighted_train_mse']:.8f} "
                f"validation={row['validation_mse']:.8f}",
                flush=True,
            )

    load_model(model, checkpoint_dir / "best.safetensors")
    train_sources = sorted(set(int(source_indices[index]) for index in train_indices))
    validation_sources = sorted(
        set(int(source_indices[index]) for index in validation_indices)
    )
    evidence = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "method": "initial_state_and_time_conditioned_guided_policy_distillation",
        "progress": float(args.progress),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "hidden_sizes": hidden_sizes,
        "early_weight": float(args.early_weight),
        "early_decay_steps": float(args.early_decay_steps),
        "base_observation_dim": int(observations.shape[1]),
        "policy_input_dim": int(features.shape[1]),
        "maximum_steps": maximum_steps,
        "feature_mode": args.feature_mode,
        "feature_order": (
            ["current_observation", "initial_observation", "normalized_time"]
            if args.feature_mode == "current_initial_time"
            else ["current_observation", "normalized_time"]
        ),
        "label_count": int(len(features)),
        "train_sources": train_sources,
        "validation_sources": validation_sources,
        "train_sources_sha256": data_sha256(train_sources),
        "validation_sources_sha256": data_sha256(validation_sources),
        "train_metrics": prediction_metrics(
            model, features[train_indices], targets[train_indices], weights[train_indices]
        ),
        "validation_metrics": prediction_metrics(
            model,
            features[validation_indices],
            targets[validation_indices],
            weights[validation_indices],
        ),
        "best_validation_mse": float(best_validation_mse),
        "history": history,
        "checkpoint": file_metadata(checkpoint_dir / "best.safetensors"),
        "labels": file_metadata(args.labels),
        "dataset": file_metadata(args.dataset),
        "config": file_metadata(out_dir / "config.resolved.yaml"),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(evidence, out_dir / "trajectory_distillation.json")
    print(f"Wrote {checkpoint_dir / 'best.safetensors'}")


if __name__ == "__main__":
    main()
