#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config, save_config
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, load_model, save_model


def distillation_loss(model: ActorCritic, observations, targets):
    means, _ = model(observations)
    return mx.mean((means - targets) ** 2)


def collect_teacher_dataset(
    cfg: dict,
    states: list[dict],
    *,
    sample_count: int,
    max_progress: float,
    teacher_scale: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(states), size=sample_count)
    progresses = rng.uniform(0.0, max_progress, size=sample_count)
    teacher_cfg = copy.deepcopy(cfg)
    residual_cfg = teacher_cfg["env"]["action_lqr_residual"]
    residual_cfg["scale_start"] = float(teacher_scale)
    residual_cfg["scale_end"] = float(teacher_scale)
    env = NLinkCartPoleEnv(teacher_cfg, progress=0.0, seed=seed)
    observations = []
    targets = []
    try:
        for state_index, progress in zip(indices, progresses):
            env.progress = float(progress)
            obs, _ = env.reset(
                options={"qpos": states[int(state_index)]["qpos"], "qvel": states[int(state_index)]["qvel"]}
            )
            action, _ = env._lqr_action_bias(residual_cfg)
            observations.append(obs)
            targets.append([action])
    finally:
        env.close()
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(targets, dtype=np.float32),
        np.asarray(progresses, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill the analytic capture LQR teacher into the nonlinear MLX actor")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/train.json")
    parser.add_argument("--out-dir", default="runs/swingup6_capture_lqr_distilled")
    parser.add_argument("--init-checkpoint", default=None)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--max-progress", type=float, default=0.15)
    parser.add_argument("--teacher-scale", type=float, default=1.30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=61401)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.samples < 2 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("samples, epochs, and batch size must be positive")
    if not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")
    if not 0.0 < args.max_progress <= 1.0:
        raise ValueError("--max-progress must be in (0, 1]")

    cfg = apply_overrides(load_config(args.config), args.override)
    source = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    states = source.get("states", source) if isinstance(source, dict) else source
    if not isinstance(states, list) or not states:
        raise ValueError("distillation dataset must contain capture states")
    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.resolved.yaml")

    observations, targets, progresses = collect_teacher_dataset(
        cfg,
        states,
        sample_count=args.samples,
        max_progress=args.max_progress,
        teacher_scale=args.teacher_scale,
        seed=args.seed,
    )
    rng = np.random.default_rng(args.seed + 1)
    order = rng.permutation(args.samples)
    validation_count = max(1, int(round(args.samples * args.validation_fraction)))
    validation_indices = order[:validation_count]
    train_indices = order[validation_count:]

    probe = NLinkCartPoleEnv(cfg, progress=0.0, seed=args.seed)
    model = ActorCritic(
        probe.observation_space.shape[0],
        probe.action_space.shape[0],
        list(cfg["ppo"].get("hidden_sizes", [256, 256])),
        float(cfg["ppo"].get("action_std_init", 0.08)),
    )
    probe.close()
    mx.eval(model.parameters())
    if args.init_checkpoint:
        load_model(model, args.init_checkpoint)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(model, distillation_loss)
    best_validation_mse = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        epoch_order = rng.permutation(train_indices)
        train_losses = []
        for start in range(0, len(epoch_order), args.batch_size):
            batch = epoch_order[start : start + args.batch_size]
            loss, grads = loss_and_grad(model, mx.array(observations[batch]), mx.array(targets[batch]))
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            train_losses.append(float(loss))
        validation_means, _ = model(mx.array(observations[validation_indices]))
        validation_mse = mx.mean((validation_means - mx.array(targets[validation_indices])) ** 2)
        mx.eval(validation_mse)
        validation_mse_value = float(validation_mse)
        if validation_mse_value < best_validation_mse:
            best_validation_mse = validation_mse_value
            save_model(model, checkpoint_dir / "best.safetensors")
        row = {
            "epoch": epoch,
            "train_mse": float(np.mean(train_losses)),
            "validation_mse": validation_mse_value,
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d}/{args.epochs} train_mse={row['train_mse']:.8f} "
                f"validation_mse={validation_mse_value:.8f}",
                flush=True,
            )

    load_model(model, checkpoint_dir / "best.safetensors")
    predicted, _ = model(mx.array(observations[validation_indices]))
    mx.eval(predicted)
    predicted_np = np.asarray(predicted)
    target_np = targets[validation_indices]
    errors = predicted_np - target_np
    evidence = {
        "generated_at": utc_timestamp(),
        "method": "supervised_lqr_policy_distillation",
        "teacher_scale": float(args.teacher_scale),
        "sample_count": int(args.samples),
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "max_progress": float(args.max_progress),
        "progress_mean": float(np.mean(progresses)),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "best_validation_mse": best_validation_mse,
        "validation_mae": float(np.mean(np.abs(errors))),
        "validation_max_abs_error": float(np.max(np.abs(errors))),
        "target_saturation_rate": float(np.mean(np.abs(target_np) >= 0.999)),
        "history": history,
        "checkpoint": file_metadata(checkpoint_dir / "best.safetensors"),
        "init_checkpoint": None if args.init_checkpoint is None else file_metadata(args.init_checkpoint),
        "dataset": file_metadata(args.dataset),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(evidence, out_dir / "distillation.json")
    print(f"Wrote {checkpoint_dir / 'best.safetensors'}")
    print(f"Wrote {out_dir / 'distillation.json'}")


if __name__ == "__main__":
    main()
