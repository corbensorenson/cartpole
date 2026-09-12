#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Iterable

import mujoco
import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, load_model, sample_action

try:
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_capture_target_schedule import evaluate_target_schedule
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg
    from search_capture_target_schedule import evaluate_target_schedule


def aligned_trajectory_steps(
    trajectory: list[dict[str, Any]],
    initial_qpos: Iterable[float],
    initial_qvel: Iterable[float],
) -> list[dict[str, Any]]:
    """Pair each action with the state before it, including legacy post-state traces."""
    previous_qpos = np.asarray(initial_qpos, dtype=np.float64)
    previous_qvel = np.asarray(initial_qvel, dtype=np.float64)
    if previous_qpos.ndim != 1 or previous_qvel.shape != previous_qpos.shape:
        raise ValueError("initial qpos and qvel must be equal-length vectors")
    aligned = []
    for row in trajectory:
        if "pre_action_qpos" in row or "pre_action_qvel" in row:
            if "pre_action_qpos" not in row or "pre_action_qvel" not in row:
                raise ValueError("trajectory row has only one pre-action state field")
            qpos = np.asarray(row["pre_action_qpos"], dtype=np.float64)
            qvel = np.asarray(row["pre_action_qvel"], dtype=np.float64)
        else:
            qpos = previous_qpos
            qvel = previous_qvel
        if qpos.shape != previous_qpos.shape or qvel.shape != previous_qvel.shape:
            raise ValueError("trajectory state shape changed")
        aligned.append(
            {
                "qpos": qpos.copy(),
                "qvel": qvel.copy(),
                "action": float(row["action"]),
                "controller_mode": row.get("controller_mode"),
                "source_step": int(row.get("step", len(aligned) + 1)) - 1,
            }
        )
        previous_qpos = np.asarray(row["qpos"], dtype=np.float64)
        previous_qvel = np.asarray(row["qvel"], dtype=np.float64)
    return aligned


def set_physical_state(env: NLinkCartPoleEnv, qpos: np.ndarray, qvel: np.ndarray) -> None:
    if qpos.shape != env.data.qpos.shape or qvel.shape != env.data.qvel.shape:
        raise ValueError("teacher state does not match supervisor environment")
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    mujoco.mj_forward(env.model, env.data)


def append_outside_funnel_labels(
    env: NLinkCartPoleEnv,
    switch_cfg: dict[str, Any],
    steps: list[dict[str, Any]],
    *,
    source_index: int,
    tier: str,
    observations: list[np.ndarray],
    actions: list[float],
    source_indices: list[int],
    source_steps: list[int],
    tiers: list[str],
) -> int:
    added = 0
    for step in steps:
        set_physical_state(env, step["qpos"], step["qvel"])
        if env._within_lqr_switch_bounds(switch_cfg, "enter"):
            continue
        observations.append(env._get_obs().copy())
        actions.append(float(np.clip(step["action"], -1.0, 1.0)))
        source_indices.append(int(source_index))
        source_steps.append(int(step.get("source_step", 0)))
        tiers.append(tier)
        added += 1
    return added


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build aligned train-only labels for the Lyapunov-gated capture recovery policy"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_hybrid_policy.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/train.json")
    parser.add_argument("--rollouts", default="runs/p1_capture_envelope/train_hard_007.json")
    parser.add_argument(
        "--target-teachers",
        default="runs/p1_capture_target_teachers/train_teachers_32_p007.json",
    )
    parser.add_argument(
        "--feedback-teachers",
        default="runs/p1_capture_feedback_mpc/train_teachers_8_p007.json",
    )
    parser.add_argument("--progress", type=float, default=0.07)
    parser.add_argument("--out", default="runs/p1_capture_supervisor/train_labels_p007.npz")
    parser.add_argument("--metadata-out", default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    dataset_path = Path(args.dataset)
    rollout_path = Path(args.rollouts)
    target_path = Path(args.target_teachers)
    feedback_path = Path(args.feedback_teachers)
    cfg = load_config(config_path)
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    rollouts = json.loads(rollout_path.read_text(encoding="utf-8"))
    target_payload = json.loads(target_path.read_text(encoding="utf-8"))
    feedback_payload = json.loads(feedback_path.read_text(encoding="utf-8"))
    if dataset.get("split") != "train":
        raise ValueError("supervisor labels are restricted to the frozen training split")
    for name, payload in (("target", target_payload), ("feedback", feedback_payload)):
        if not np.isclose(float(payload.get("progress", -1.0)), args.progress):
            raise ValueError(f"{name} teacher progress does not match requested progress")
    states = dataset.get("states")
    rollout_rows = rollouts.get("evaluation", rollouts).get("episode_results")
    if not isinstance(states, list) or not states or not isinstance(rollout_rows, list):
        raise ValueError("training dataset or rollout evidence is malformed")

    switch_cfg = cfg["env"].get("action_lqr_switch", {})
    if not bool(switch_cfg.get("enabled", False)) or switch_cfg.get("enter_lyapunov_max") is None:
        raise ValueError("supervisor config must enable the Lyapunov-gated LQR switch")
    observations: list[np.ndarray] = []
    actions: list[float] = []
    source_indices: list[int] = []
    source_steps: list[int] = []
    tiers: list[str] = []
    source_records: list[dict[str, Any]] = []
    baseline_replay_mismatches: list[dict[str, Any]] = []
    selected_sources: set[int] = set()
    hybrid_env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=79101)
    baseline_checkpoint = Path(rollouts["evidence"]["checkpoint"]["path"])
    baseline_config_path = baseline_checkpoint.parent / "config.resolved.yaml"
    baseline_cfg = load_config(baseline_config_path)
    baseline_cfg["env"]["init_states_path"] = str(dataset_path)
    baseline_env = NLinkCartPoleEnv(baseline_cfg, progress=args.progress, seed=79101)
    baseline_model = ActorCritic(
        baseline_env.observation_space.shape[0],
        baseline_env.action_space.shape[0],
        [],
        float(baseline_cfg["ppo"].get("action_std_init", 0.20)),
    )
    load_model(baseline_model, baseline_checkpoint)

    try:
        # Cheap successful LQR rollouts teach the recovery policy how to enter the strict funnel.
        for row in sorted(rollout_rows, key=lambda item: int(item["state_index"])):
            source_index = int(row["state_index"])
            if not bool(row.get("success", False)):
                continue
            observation, _ = baseline_env.reset(options={"state_index": source_index})
            set_physical_state(
                hybrid_env,
                np.asarray(baseline_env.data.qpos, dtype=np.float64),
                np.asarray(baseline_env.data.qvel, dtype=np.float64),
            )
            if hybrid_env._within_lqr_switch_bounds(switch_cfg, "enter"):
                continue
            tier_steps = []
            terminated = truncated = False
            final_info: dict[str, Any] = {}
            while not (terminated or truncated):
                qpos = np.asarray(baseline_env.data.qpos, dtype=np.float64).copy()
                qvel = np.asarray(baseline_env.data.qvel, dtype=np.float64).copy()
                action_batch, _, _ = sample_action(
                    baseline_model,
                    observation[None, :],
                    deterministic=True,
                )
                action = float(action_batch[0, 0])
                tier_steps.append(
                    {
                        "qpos": qpos,
                        "qvel": qvel,
                        "action": action,
                        "source_step": int(baseline_env.step_count),
                    }
                )
                observation, _, terminated, truncated, final_info = baseline_env.step([action])
            if not bool(final_info.get("success", False)):
                baseline_replay_mismatches.append(
                    {
                        "source_index": source_index,
                        "termination_reason": final_info.get("termination_reason"),
                        "max_upright_streak_seconds": float(
                            final_info.get("max_upright_streak_seconds", 0.0)
                        ),
                        "max_cart_excursion": float(final_info.get("max_cart_excursion", 0.0)),
                    }
                )
                continue
            added = append_outside_funnel_labels(
                hybrid_env,
                switch_cfg,
                tier_steps,
                source_index=source_index,
                tier="lqr",
                observations=observations,
                actions=actions,
                source_indices=source_indices,
                source_steps=source_steps,
                tiers=tiers,
            )
            selected_sources.add(source_index)
            source_records.append({"source_index": source_index, "tier": "lqr", "labels": added})

        # Scheduled-target teachers are the first nonlinear supervisor tier.
        target_cfg = copy.deepcopy(cfg)
        target_cfg["env"]["action_lqr_switch"]["enabled"] = False
        target_seed = int(target_payload["search"]["seed"])
        target_seconds = float(target_payload["search"]["schedule_seconds"])
        target_gain = np.asarray(target_payload["lqr_gain"], dtype=np.float64)
        for teacher in target_payload.get("teachers", []):
            source_index = int(teacher["source_index"])
            if not bool(teacher.get("success", False)) or source_index in selected_sources:
                continue
            controller = teacher["controller"]
            fixed_cfg = fixed_state_cfg(
                target_cfg,
                states[source_index],
                float(target_cfg["env"]["episode_seconds"]),
            )
            replay = evaluate_target_schedule(
                fixed_cfg,
                progress=args.progress,
                seed=target_seed + int(teacher["ordinal"]),
                gain=target_gain,
                target_knots=np.asarray(controller["target_knots"], dtype=np.float64),
                schedule_seconds=target_seconds,
                lqr_scale=float(controller["lqr_scale"]),
                record_trajectory=True,
            )
            if not replay["success"]:
                raise RuntimeError(f"target teacher no longer replays for training state {source_index}")
            steps = aligned_trajectory_steps(replay["trajectory"], teacher["qpos"], teacher["qvel"])
            added = append_outside_funnel_labels(
                hybrid_env,
                switch_cfg,
                steps,
                source_index=source_index,
                tier="target",
                observations=observations,
                actions=actions,
                source_indices=source_indices,
                source_steps=source_steps,
                tiers=tiers,
            )
            selected_sources.add(source_index)
            source_records.append({"source_index": source_index, "tier": "target", "labels": added})

        # Feedback MPC is accepted only after its complete uninterrupted rollout succeeds.
        for teacher in feedback_payload.get("teachers", []):
            source_index = int(teacher["source_index"])
            if not bool(teacher.get("success", False)) or source_index in selected_sources:
                continue
            hybrid_env.reset(options={"state_index": source_index})
            result = teacher["result"]
            initial_qpos = result.get("initial_qpos", np.asarray(hybrid_env.data.qpos).tolist())
            initial_qvel = result.get("initial_qvel", np.asarray(hybrid_env.data.qvel).tolist())
            steps = aligned_trajectory_steps(result["trajectory"], initial_qpos, initial_qvel)
            steps = [step for step in steps if step["controller_mode"] == "feedback_mpc"]
            added = append_outside_funnel_labels(
                hybrid_env,
                switch_cfg,
                steps,
                source_index=source_index,
                tier="feedback_mpc",
                observations=observations,
                actions=actions,
                source_indices=source_indices,
                source_steps=source_steps,
                tiers=tiers,
            )
            selected_sources.add(source_index)
            source_records.append(
                {"source_index": source_index, "tier": "feedback_mpc", "labels": added}
            )
    finally:
        hybrid_env.close()
        baseline_env.close()

    if not observations:
        raise RuntimeError("no successful outside-funnel supervisor labels were produced")
    observation_array = np.asarray(observations, dtype=np.float32)
    action_array = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
    source_array = np.asarray(source_indices, dtype=np.int32)
    source_step_array = np.asarray(source_steps, dtype=np.int32)
    tier_array = np.asarray(tiers, dtype="U16")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        observations=observation_array,
        actions=action_array,
        source_indices=source_array,
        source_steps=source_step_array,
        tiers=tier_array,
    )
    metadata_path = Path(args.metadata_out) if args.metadata_out else out_path.with_suffix(".json")
    tier_counts = {tier: int(np.sum(tier_array == tier)) for tier in sorted(set(tiers))}
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Aligned train-only successful supervisor labels outside the strict LQR funnel.",
        "progress": float(args.progress),
        "label_count": int(len(action_array)),
        "observation_dim": int(observation_array.shape[1]),
        "source_count": int(len(set(source_indices))),
        "tier_label_counts": tier_counts,
        "source_records": source_records,
        "source_records_sha256": data_sha256(source_records),
        "baseline_replay_mismatch_count": len(baseline_replay_mismatches),
        "baseline_replay_mismatches": baseline_replay_mismatches,
        "labels": file_metadata(out_path),
        "evidence": {
            "config": file_metadata(config_path),
            "dataset": file_metadata(dataset_path),
            "rollouts": file_metadata(rollout_path),
            "target_teachers": file_metadata(target_path),
            "feedback_teachers": file_metadata(feedback_path),
            "baseline_config": file_metadata(baseline_config_path),
            "baseline_checkpoint": file_metadata(baseline_checkpoint),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, metadata_path)
    print(
        f"sources={payload['source_count']} labels={payload['label_count']} "
        f"tiers={tier_counts} Wrote {out_path}"
    )


if __name__ == "__main__":
    main()
