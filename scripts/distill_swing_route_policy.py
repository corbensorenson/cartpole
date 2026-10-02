#!/usr/bin/env python
"""Distill an exact assisted swing route into a time-aware state policy.

The route is only a development teacher. Its states are replayed from the
recorded actions, then re-encoded with the canonical target plant before the
actor is trained. The resulting checkpoint is intentionally marked as
non-solution evidence and must be evaluated from a fresh hanging reset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mujoco
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config, save_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.ppo_mlx import ActorCritic, save_model


def parse_hidden_sizes(text: str) -> list[int]:
    sizes = [int(value.strip()) for value in text.split(",") if value.strip()]
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("hidden sizes must be positive comma-separated integers")
    return sizes


def split_route_rows(row_count: int) -> tuple[np.ndarray, np.ndarray]:
    """Use deterministic interleaved validation rows without leaking a phase block."""
    if row_count < 5:
        raise ValueError("at least five route rows are required")
    indices = np.arange(row_count, dtype=np.int64)
    validation = (indices % 5) == 0
    if np.all(validation) or not np.any(validation):
        raise RuntimeError("route split produced an empty partition")
    return indices[~validation], indices[validation]


def set_physical_state(
    env: NLinkCartPoleEnv,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    step_count: int,
) -> None:
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    if qpos.shape != env.data.qpos.shape or qvel.shape != env.data.qvel.shape:
        raise ValueError("teacher state dimension does not match target environment")
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.step_count = int(step_count)
    mujoco.mj_forward(env.model, env.data)


def load_route(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("route artifact must contain a JSON object")
    trajectory = payload.get("result", {}).get("trajectory")
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError("route artifact must contain result.trajectory")
    if not isinstance(payload.get("selected_state"), dict):
        raise ValueError("route artifact must contain selected_state")
    return payload


def build_route_dataset(
    cfg: dict[str, Any],
    route: dict[str, Any],
    *,
    progress: float,
    seed: int,
    max_seconds: float | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    trajectory = route["result"]["trajectory"]
    source_config_path = route.get("evidence", {}).get("config", {}).get("path")
    if not source_config_path:
        raise ValueError("route evidence does not identify the source config")
    target_env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    observations: list[np.ndarray] = []
    actions: list[float] = []
    used_rows = 0
    try:
        selected = route["selected_state"]
        # The recorded trajectory rows are post-action states. Pair the
        # selected initial state with the first action, then pair each later
        # pre-action state (the preceding recorded row) with the next action.
        source_states = [selected] + list(trajectory[:-1])
        for state, row in zip(source_states, trajectory):
            time_seconds = float(row.get("time_seconds", used_rows * target_env.dt))
            if max_seconds is not None and time_seconds > max_seconds + 1e-12:
                break
            qpos = np.asarray(state["qpos"], dtype=np.float64)
            qvel = np.asarray(state["qvel"], dtype=np.float64)
            step_count = int(state.get("step", max(0, round(time_seconds / target_env.dt) - 1)))
            set_physical_state(
                target_env,
                qpos,
                qvel,
                step_count=step_count,
            )
            observations.append(target_env._get_obs().copy())
            actions.append(float(np.clip(row["action"], -1.0, 1.0)))
            used_rows += 1
    finally:
        target_env.close()

    if used_rows < 5:
        raise ValueError("route produced too few teacher rows")
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(actions, dtype=np.float32).reshape(-1, 1),
        {
            "source_config": str(source_config_path),
            "source_route_rows": int(len(trajectory)),
            "used_rows": int(used_rows),
            "state_alignment": "selected_initial_then_recorded_previous_row_to_next_action",
            "recorded_state_provenance": True,
            "source_route_sha256": data_sha256(route),
        },
    )


def imitation_loss(model: ActorCritic, observations, actions):
    means, _ = model(observations)
    return mx.mean((means - actions) ** 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Distill a saved assisted swing route into a time-aware actor")
    parser.add_argument("--config", default="configs/swingup11_uniform.yaml")
    parser.add_argument(
        "--route",
        default="runs/generalized_solver/n11_p39_feedback_a1rm1.json",
    )
    parser.add_argument(
        "--out-dir",
        default="runs/generalized_solver/n11_route_imitation_teacher",
    )
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=11011)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--hidden-sizes", default="256,256")
    parser.add_argument("--time-scale-seconds", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.learning_rate, args.time_scale_seconds) <= 0:
        raise ValueError("training counts and scales must be positive")
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    cfg = apply_overrides(load_config(args.config), args.override)
    # Keep the teacher checkpoint shape compatible with the follow-up PPO run.
    cfg["env"]["obs_include_time"] = True
    cfg["env"]["obs_time_scale_seconds"] = float(args.time_scale_seconds)
    cfg["env"]["obs_time_frequencies"] = []
    route = load_route(args.route)
    observations, actions, route_metadata = build_route_dataset(
        cfg,
        route,
        progress=args.progress,
        seed=args.seed,
        max_seconds=args.max_seconds,
    )
    train_indices, validation_indices = split_route_rows(len(observations))

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.resolved.yaml")
    np.savez_compressed(
        out_dir / "route_labels.npz",
        observations=observations,
        actions=actions,
        train_indices=train_indices,
        validation_indices=validation_indices,
    )

    model = ActorCritic(observations.shape[1], 1, hidden_sizes, 0.20)
    mx.eval(model.parameters())
    mx.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    optimizer = optim.Adam(learning_rate=args.learning_rate)
    loss_and_grad = nn.value_and_grad(model, imitation_loss)
    best_validation_mse = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        epoch_indices = rng.permutation(train_indices)
        losses: list[float] = []
        for start in range(0, len(epoch_indices), args.batch_size):
            batch = epoch_indices[start : start + args.batch_size]
            loss, grads = loss_and_grad(
                model,
                mx.array(observations[batch]),
                mx.array(actions[batch]),
            )
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            losses.append(float(loss))
        predicted, _ = model(mx.array(observations[validation_indices]))
        mx.eval(predicted)
        validation_mse = float(np.mean((np.asarray(predicted) - actions[validation_indices]) ** 2))
        row = {
            "epoch": float(epoch),
            "train_mse": float(np.mean(losses)),
            "validation_mse": validation_mse,
        }
        history.append(row)
        if validation_mse < best_validation_mse:
            best_validation_mse = validation_mse
            save_model(model, checkpoint_dir / "best.safetensors")
        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:04d}/{args.epochs} train={row['train_mse']:.8f} "
                f"validation={validation_mse:.8f}",
                flush=True,
            )

    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "claim_status": "development_route_imitation_teacher_not_solution",
        "summary": (
            "Time-aware supervised teacher distilled from a near-locked 11-link "
            "route; canonical hanging-start replay is required."
        ),
        "method": "canonical_observation_route_imitation_then_ppo_warm_start",
        "progress": float(args.progress),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "hidden_sizes": hidden_sizes,
        "time_scale_seconds": float(args.time_scale_seconds),
        "observation_dim": int(observations.shape[1]),
        "label_count": int(len(observations)),
        "train_label_count": int(len(train_indices)),
        "validation_label_count": int(len(validation_indices)),
        "best_validation_mse": float(best_validation_mse),
        "route": route_metadata,
        "config": file_metadata(out_dir / "config.resolved.yaml"),
        "labels": file_metadata(out_dir / "route_labels.npz"),
        "checkpoint": file_metadata(checkpoint_dir / "best.safetensors"),
        "history": history,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, out_dir / "route_imitation_teacher.json")
    print(f"Wrote {checkpoint_dir / 'best.safetensors'}")


if __name__ == "__main__":
    main()
