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


def grouped_source_split(
    source_indices: np.ndarray,
    tiers: np.ndarray,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split by source state, stratified by supervisor tier to prevent trajectory leakage."""
    source_indices = np.asarray(source_indices, dtype=np.int64)
    tiers = np.asarray(tiers).astype(str)
    if source_indices.ndim != 1 or tiers.shape != source_indices.shape:
        raise ValueError("source indices and tiers must be equal-length vectors")
    source_tier: dict[int, str] = {}
    for source, tier in zip(source_indices, tiers):
        prior = source_tier.setdefault(int(source), str(tier))
        if prior != str(tier):
            raise ValueError(f"source state {source} has labels from multiple supervisor tiers")
    rng = np.random.default_rng(seed)
    validation_sources: set[int] = set()
    for tier in sorted(set(source_tier.values())):
        sources = np.asarray(sorted(source for source, value in source_tier.items() if value == tier))
        if len(sources) < 2:
            continue
        count = int(np.clip(round(len(sources) * validation_fraction), 1, len(sources) - 1))
        validation_sources.update(int(source) for source in rng.permutation(sources)[:count])
    if not validation_sources:
        sources = np.asarray(sorted(source_tier))
        if len(sources) < 2:
            raise ValueError("at least two source states are required for grouped validation")
        validation_sources.add(int(rng.choice(sources)))
    validation_mask = np.isin(source_indices, sorted(validation_sources))
    if np.all(validation_mask) or not np.any(validation_mask):
        raise RuntimeError("grouped source split produced an empty partition")
    return np.flatnonzero(~validation_mask), np.flatnonzero(validation_mask)


def source_balancing_weights(
    source_indices: np.ndarray,
    source_steps: np.ndarray | None = None,
    *,
    early_weight: float = 0.0,
    early_decay_steps: float = 1.0,
) -> np.ndarray:
    sources = np.asarray(source_indices, dtype=np.int64)
    if source_steps is None:
        temporal = np.ones(len(sources), dtype=np.float64)
    else:
        steps = np.asarray(source_steps, dtype=np.float64)
        if steps.shape != sources.shape or np.any(steps < 0.0):
            raise ValueError("source steps must be a nonnegative vector matching source indices")
        temporal = 1.0 + float(early_weight) * np.exp(
            -steps / max(1e-9, float(early_decay_steps))
        )
    weights = np.zeros(len(sources), dtype=np.float64)
    for source in np.unique(sources):
        mask = sources == source
        weights[mask] = temporal[mask] / float(np.sum(temporal[mask]))
    weights = weights.astype(np.float32)
    return weights / float(np.mean(weights))


def weighted_distillation_loss(model: ActorCritic, observations, targets, weights):
    means, _ = model(observations)
    squared_error = mx.mean((means - targets) ** 2, axis=-1)
    return mx.sum(weights * squared_error) / mx.sum(weights)


def prediction_metrics(
    model: ActorCritic,
    observations: np.ndarray,
    targets: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    predicted, _ = model(mx.array(observations))
    mx.eval(predicted)
    errors = np.asarray(predicted) - targets
    metrics = {
        "mse": float(np.mean(errors**2)),
        "mae": float(np.mean(np.abs(errors))),
        "max_abs_error": float(np.max(np.abs(errors))),
        "sign_agreement_rate": float(np.mean(np.sign(np.asarray(predicted)) == np.sign(targets))),
    }
    if weights is not None:
        sample_error = np.mean(errors**2, axis=-1)
        metrics["weighted_mse"] = float(np.sum(weights * sample_error) / np.sum(weights))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Distill successful train-only capture supervisors into the recovery actor"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_hybrid_policy.yaml")
    parser.add_argument("--labels", default="runs/p1_capture_supervisor/train_labels_p007.npz")
    parser.add_argument("--out-dir", default="runs/swingup6_capture_supervisor_distilled")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--validation-fraction", type=float, default=0.20)
    parser.add_argument("--early-weight", type=float, default=20.0)
    parser.add_argument("--early-decay-steps", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=79201)
    args = parser.parse_args()
    if (
        args.epochs < 1
        or args.batch_size < 1
        or args.learning_rate <= 0.0
        or args.early_weight < 0.0
        or args.early_decay_steps <= 0.0
    ):
        raise ValueError("epochs, batch size, and learning rate must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("validation fraction must be in (0, 1)")

    cfg = load_config(args.config)
    labels = np.load(args.labels)
    observations = np.asarray(labels["observations"], dtype=np.float32)
    targets = np.asarray(labels["actions"], dtype=np.float32)
    source_indices = np.asarray(labels["source_indices"], dtype=np.int64)
    source_steps = np.asarray(labels["source_steps"], dtype=np.int64)
    tiers = np.asarray(labels["tiers"]).astype(str)
    if observations.ndim != 2 or targets.shape != (len(observations), 1):
        raise ValueError("supervisor observations or actions have invalid shapes")
    if (
        source_indices.shape != (len(observations),)
        or source_steps.shape != source_indices.shape
        or tiers.shape != source_indices.shape
    ):
        raise ValueError("supervisor provenance arrays do not match observations")
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
    probe = NLinkCartPoleEnv(cfg, progress=0.07, seed=args.seed)
    try:
        if observations.shape[1] != probe.observation_space.shape[0]:
            raise ValueError("supervisor observation dimension does not match policy config")
        model = ActorCritic(
            observations.shape[1],
            probe.action_space.shape[0],
            list(cfg["ppo"].get("hidden_sizes", [256, 256])),
            float(cfg["ppo"].get("action_std_init", 0.20)),
        )
    finally:
        probe.close()
    mx.eval(model.parameters())
    if args.init_checkpoint:
        load_model(model, args.init_checkpoint)
    elif bool(cfg["ppo"].get("zero_init_actor_output", False)):
        model.update(
            {
                "actor_out": {
                    "weight": mx.zeros_like(model.actor_out.weight),
                    "bias": mx.zeros_like(model.actor_out.bias),
                }
            }
        )
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
                mx.array(observations[batch]),
                mx.array(targets[batch]),
                mx.array(weights[batch]),
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            train_losses.append(float(loss))
        validation = prediction_metrics(
            model,
            observations[validation_indices],
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
    train_metrics = prediction_metrics(
        model,
        observations[train_indices],
        targets[train_indices],
        weights[train_indices],
    )
    validation_metrics = prediction_metrics(
        model,
        observations[validation_indices],
        targets[validation_indices],
        weights[validation_indices],
    )
    train_sources = sorted(set(int(source_indices[index]) for index in train_indices))
    validation_sources = sorted(set(int(source_indices[index]) for index in validation_indices))
    evidence = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "method": "source_grouped_weighted_supervisor_distillation_warm_start",
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "early_weight": float(args.early_weight),
        "early_decay_steps": float(args.early_decay_steps),
        "label_count": int(len(observations)),
        "train_label_count": int(len(train_indices)),
        "validation_label_count": int(len(validation_indices)),
        "train_source_count": len(train_sources),
        "validation_source_count": len(validation_sources),
        "train_sources_sha256": data_sha256(train_sources),
        "validation_sources_sha256": data_sha256(validation_sources),
        "train_metrics": train_metrics,
        "validation_metrics": validation_metrics,
        "best_validation_mse": float(best_validation_mse),
        "target_saturation_rate": float(np.mean(np.abs(targets) >= 0.999)),
        "history": history,
        "checkpoint": file_metadata(checkpoint_dir / "best.safetensors"),
        "init_checkpoint": None if args.init_checkpoint is None else file_metadata(args.init_checkpoint),
        "labels": file_metadata(args.labels),
        "config": file_metadata(out_dir / "config.resolved.yaml"),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(evidence, out_dir / "distillation.json")
    print(f"Wrote {checkpoint_dir / 'best.safetensors'}")


if __name__ == "__main__":
    main()
