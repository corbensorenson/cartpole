#!/usr/bin/env python
"""Characterize the released controller outside the canonical claim boundary."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from scripts.evaluate_fddp_two_expert import (
    hanging_lqr_gain,
    load_controller,
    run_episode,
)
from scripts.search_swingup_capture import lqr_gain


ROOT = Path(__file__).resolve().parents[1]


def scenario_config(base: dict, changes: dict) -> dict:
    cfg = copy.deepcopy(base)
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"].setdefault("action_lqr_switch", {"enabled": False})["enabled"] = False
    for key, value in changes.get("env", {}).items():
        cfg["env"][key] = value
    for endpoint in ("start", "end"):
        for key, value in changes.get("morphology", {}).items():
            cfg["morphology"][endpoint][key] = value
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--controller",
        default="runs/swingup7_uniform/seven_link_release_controller.json",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=60732)
    parser.add_argument("--out", default="runs/swingup7_uniform/robustness_sweep.json")
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("episodes must be positive")

    source_git = {
        key: value
        for key, value in git_metadata(ROOT, include_untracked=False).items()
        if key != "root"
    }
    base = load_config(args.config)
    spec = load_config(args.spec)
    controller = load_controller(Path(args.controller), int(base["env"]["n_links"]), spec)
    canonical_runtime = scenario_config(base, {})
    capture_gain = lqr_gain(canonical_runtime, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    settle_gain = hanging_lqr_gain(
        canonical_runtime, progress=1.0, fd_eps=1e-7, control_cost=1000.0
    )

    scenarios = [
        {"name": "initial_noise_sigma_0p10", "changes": {"env": {"init_angle_noise": 0.10, "init_vel_noise": 0.10}}},
        {"name": "initial_noise_sigma_0p20", "changes": {"env": {"init_angle_noise": 0.20, "init_vel_noise": 0.20}}},
        {"name": "force_limit_40N", "changes": {"env": {"force_limit": 40.0}}},
        {"name": "force_limit_20N", "changes": {"env": {"force_limit": 20.0}}},
        {"name": "total_mass_minus_5pct", "changes": {"env": {"total_mass": 0.95}}},
        {"name": "total_mass_plus_5pct", "changes": {"env": {"total_mass": 1.05}}},
        {"name": "total_length_minus_5pct", "changes": {"env": {"total_length": 2.85}}},
        {"name": "total_length_plus_5pct", "changes": {"env": {"total_length": 3.15}}},
        {"name": "joint_damping_half", "changes": {"morphology": {"total_damping": 0.0075}}},
        {"name": "joint_damping_double", "changes": {"morphology": {"total_damping": 0.0300}}},
        {"name": "sensor_noise_std_0p01", "measurement_noise_std": 0.01, "changes": {}},
        {"name": "control_delay_20ms", "control_delay_steps": 1, "changes": {}},
        {"name": "control_delay_40ms", "control_delay_steps": 2, "changes": {}},
        {"name": "settling_8s", "prelude_seconds": 8.0, "changes": {}},
        {"name": "settling_6s", "prelude_seconds": 6.0, "changes": {}},
    ]
    results = []
    for scenario in scenarios:
        cfg = scenario_config(base, scenario["changes"])
        prelude_seconds = float(scenario.get("prelude_seconds", 10.0))
        prelude_steps = int(round(prelude_seconds / (cfg["env"]["timestep"] * cfg["env"]["frame_skip"])))
        episodes = [
            run_episode(
                cfg,
                controller,
                capture_gain,
                seed=args.seed + index,
                episode=index,
                tracking_gain_scale=2.0,
                prelude_steps=prelude_steps,
                settle_mode="hanging_lqr",
                settle_gain=settle_gain,
                settle_scale=1.0,
                phase_adaptive=False,
                phase_window=12,
                shift_cart_nominal=True,
                measurement_noise_std=float(scenario.get("measurement_noise_std", 0.0)),
                control_delay_steps=int(scenario.get("control_delay_steps", 0)),
            )
            for index in range(args.episodes)
        ]
        successes = sum(bool(episode["success"]) for episode in episodes)
        result = {
            "name": scenario["name"],
            "changes": scenario["changes"],
            "measurement_noise_std": float(scenario.get("measurement_noise_std", 0.0)),
            "control_delay_steps": int(scenario.get("control_delay_steps", 0)),
            "prelude_seconds": prelude_seconds,
            "episodes": args.episodes,
            "seed_start": args.seed,
            "successes": successes,
            "success_rate": successes / args.episodes,
            "termination_counts": dict(Counter(str(episode["termination_reason"]) for episode in episodes)),
            "max_cart_excursion": float(max(episode["max_cart_excursion"] for episode in episodes)),
            "episode_results": episodes,
        }
        results.append(result)
        print(f"{scenario['name']}: {successes}/{args.episodes}")

    output = {
        "schema_version": 1,
        "claim_status": "noncanonical_robustness_characterization",
        "summary": "Paired-seed stress tests of the frozen canonical controller; these do not extend the canonical success claim.",
        "generated_at": utc_timestamp(),
        "config": file_metadata(args.config),
        "controller": controller["source"],
        "episodes_per_scenario": args.episodes,
        "seed_start": args.seed,
        "capture_and_settle_gains": "frozen from the canonical plant for every scenario",
        "runtime": runtime_metadata(),
        "git": source_git,
        "scenarios": results,
    }
    dump_json(output, args.out)


if __name__ == "__main__":
    main()
