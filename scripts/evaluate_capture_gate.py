#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import numpy as np

from gcartpole.capture_envelope import validate_capture_config, validate_capture_states
from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    text_sha256,
    utc_timestamp,
)
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import ActorCritic, load_model, sample_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a capture policy on every state in a frozen envelope split")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/test.json")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Diagnostic subset; cannot pass the P1 gate")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=61201)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    spec = load_config(args.spec)
    dataset_path = Path(args.dataset)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    dataset_errors = validate_capture_states(payload, spec, args.split)
    if dataset_errors:
        raise ValueError(f"invalid capture envelope: {dataset_errors[:20]}")
    config_errors = validate_capture_config(cfg, spec)
    if config_errors:
        raise ValueError(f"config violates frozen capture benchmark: {config_errors[:20]}")
    states = payload["states"]
    requested_count = len(states) if args.limit is None else min(len(states), int(args.limit))
    cfg["env"]["init_states_path"] = str(dataset_path)

    batch_size = max(1, min(int(args.batch_size), requested_count))
    envs = [NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed + slot) for slot in range(batch_size)]
    probe = envs[0]
    ppo = cfg["ppo"]
    model = ActorCritic(
        probe.observation_space.shape[0],
        probe.action_space.shape[0],
        list(ppo.get("hidden_sizes", [256, 256])),
        float(ppo.get("action_std_init", 0.7)),
    )
    mx.eval(model.parameters())
    load_model(model, args.checkpoint)

    episode_results = []
    for batch_start in range(0, requested_count, batch_size):
        count = min(batch_size, requested_count - batch_start)
        observations: dict[int, np.ndarray] = {}
        episode_returns = np.zeros(count, dtype=np.float64)
        episode_steps = np.zeros(count, dtype=np.int64)
        active = set(range(count))
        for slot in range(count):
            index = batch_start + slot
            observations[slot], _ = envs[slot].reset(
                seed=args.seed + index,
                options={"qpos": states[index]["qpos"], "qvel": states[index]["qvel"]},
            )
        while active:
            slots = sorted(active)
            observation_batch = np.stack([observations[slot] for slot in slots])
            actions, _, _ = sample_action(model, observation_batch, deterministic=True)
            for action_index, slot in enumerate(slots):
                index = batch_start + slot
                obs, reward, terminated, truncated, info = envs[slot].step(actions[action_index])
                observations[slot] = obs
                episode_returns[slot] += float(reward)
                episode_steps[slot] += 1
                if not (terminated or truncated):
                    continue
                active.remove(slot)
                final_info = dict(info)
                rail_hit = final_info.get("termination_reason") == "rail_violation"
                max_hold = float(final_info.get("max_upright_streak_seconds", 0.0))
                episode_results.append(
                    {
                        "episode": index,
                        "state_index": index,
                        "state_id": states[index]["state_id"],
                        "seed": args.seed + index,
                        "return": float(episode_returns[slot]),
                        "length": int(episode_steps[slot]),
                        "success": bool(final_info.get("success", False)),
                        "rail_hit": rail_hit,
                        "termination_reason": final_info.get("termination_reason"),
                        "time_to_first_upright": final_info.get("time_to_first_upright"),
                        "time_to_capture": final_info.get("time_to_capture"),
                        "capture_start_time": final_info.get("capture_start_time"),
                        "max_upright_streak_seconds": max_hold,
                        "final_upright_streak_seconds": float(final_info.get("upright_streak_seconds", 0.0)),
                        "max_cart_excursion": float(final_info.get("max_cart_excursion", 0.0)),
                        "final_x": float(final_info.get("x", np.nan)),
                        "lqr_switch_entry_count": int(final_info.get("lqr_switch_entry_count", 0)),
                        "lqr_switch_exit_count": int(final_info.get("lqr_switch_exit_count", 0)),
                        "lqr_switch_first_entry_time": final_info.get("lqr_switch_first_entry_time"),
                        "lqr_switch_lqr_steps": int(final_info.get("lqr_switch_lqr_steps", 0)),
                        "lqr_switch_policy_steps": int(final_info.get("lqr_switch_policy_steps", 0)),
                    }
                )
    episode_results.sort(key=lambda row: row["state_index"])
    for env in envs:
        env.close()

    successes = np.asarray([row["success"] for row in episode_results], dtype=np.float64)
    holds = np.asarray([row["max_upright_streak_seconds"] for row in episode_results], dtype=np.float64)
    rail_hits = np.asarray([row["rail_hit"] for row in episode_results], dtype=np.float64)
    policy_steps = np.asarray(
        [row["lqr_switch_policy_steps"] for row in episode_results],
        dtype=np.int64,
    )
    lqr_steps = np.asarray(
        [row["lqr_switch_lqr_steps"] for row in episode_results],
        dtype=np.int64,
    )
    gate_spec = spec["capture_gate"]
    full_gate_count = int(gate_spec["required_episodes"])
    success_rate = float(np.mean(successes))
    median_hold = float(np.median(holds))
    gate = {
        "required_episodes": full_gate_count,
        "required_success_rate": float(gate_spec["required_success_rate"]),
        "required_median_upright_hold_seconds": float(gate_spec["required_median_upright_hold_seconds"]),
        "successful_episode_rail_hits_required": int(gate_spec["successful_episode_rail_hits_required"]),
        "passed_episode_count": requested_count == full_gate_count,
        "passed_final_progress": args.progress == 1.0,
        "passed_success_rate": success_rate >= float(gate_spec["required_success_rate"]),
        "passed_median_hold": median_hold >= float(gate_spec["required_median_upright_hold_seconds"]),
        "passed_successful_rail_safety": not any(
            row["success"] and row["rail_hit"] for row in episode_results
        ),
    }
    gate["passed"] = all(value for key, value in gate.items() if key.startswith("passed_"))
    result = {
        "generated_at": utc_timestamp(),
        "benchmark": payload["benchmark"],
        "split": args.split,
        "progress": float(args.progress),
        "episodes": requested_count,
        "success_rate": success_rate,
        "capture_success_count": int(np.sum(successes)),
        "max_upright_streak_median": median_hold,
        "max_upright_streak_mean": float(np.mean(holds)),
        "max_upright_streak_min": float(np.min(holds)),
        "max_upright_streak_max": float(np.max(holds)),
        "rail_hit_rate": float(np.mean(rail_hits)),
        "rail_hit_count": int(np.sum(rail_hits)),
        "policy_control_episode_count": int(np.sum(policy_steps > 0)),
        "policy_control_success_count": int(
            sum(row["success"] and row["lqr_switch_policy_steps"] > 0 for row in episode_results)
        ),
        "lqr_only_episode_count": int(np.sum(policy_steps == 0)),
        "lqr_only_success_count": int(
            sum(row["success"] and row["lqr_switch_policy_steps"] == 0 for row in episode_results)
        ),
        "lqr_switch_entry_episode_count": int(
            sum(row["lqr_switch_entry_count"] > 0 for row in episode_results)
        ),
        "lqr_switch_exit_episode_count": int(
            sum(row["lqr_switch_exit_count"] > 0 for row in episode_results)
        ),
        "policy_control_step_count": int(np.sum(policy_steps)),
        "lqr_control_step_count": int(np.sum(lqr_steps)),
        "gate": gate,
        "episode_results": episode_results,
        "evidence": {
            "deterministic_policy": True,
            "config": {
                "path": str(Path(args.config)),
                "resolved_sha256": data_sha256(cfg),
                "overrides": list(args.override),
            },
            "checkpoint": file_metadata(args.checkpoint),
            "dataset": file_metadata(dataset_path),
            "dataset_spec": {
                "path": str(Path(args.spec)),
                "resolved_sha256": data_sha256(spec),
                "embedded_spec_sha256": payload["spec_sha256"],
            },
            "generated_xml_sha256": text_sha256(probe.xml),
            "environment": {
                "n_links": probe.n,
                "rail_limit": probe.rail_limit,
                "force_limit": probe.force_limit,
                "episode_seconds": cfg["env"]["episode_seconds"],
                "success_upright_threshold": cfg["env"]["success_upright_threshold"],
                "success_sustain_seconds": cfg["env"]["success_sustain_seconds"],
                "observation_dim": probe.observation_space.shape[0],
                "action_dim": probe.action_space.shape[0],
                "action_frequency_hz": 1.0 / probe.dt,
                "lengths": probe.morphology.lengths.astype(float).tolist(),
                "masses": probe.morphology.masses.astype(float).tolist(),
                "damping": probe.morphology.damping.astype(float).tolist(),
                "frictionloss": probe.morphology.frictionloss.astype(float).tolist(),
            },
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(result, args.out)
    print(
        f"episodes={requested_count} success_rate={success_rate:.3f} "
        f"median_hold={median_hold:.3f}s rail_hits={int(np.sum(rail_hits))} gate={gate['passed']}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
