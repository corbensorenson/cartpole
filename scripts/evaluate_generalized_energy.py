#!/usr/bin/env python
"""Evaluate the deterministic generalized energy/LQR controller."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.generalized_energy import (
    EnergySwingParameters,
    GeneralizedEnergyController,
    hanging_lqr_action,
    hanging_lqr_gain,
)
from gcartpole.generalized_solver import (
    dimensionless_setup,
    rail_requirement,
    setup_from_config,
)


def run_episode(
    cfg: dict[str, Any],
    parameters: EnergySwingParameters,
    seed: int,
    include_trace: bool,
    conditioning_seconds: float = 10.0,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
    _, reset_info = env.reset(seed=seed)
    controller = GeneralizedEnergyController(env, parameters)
    settle_gain = hanging_lqr_gain(env)
    conditioning_steps = round(conditioning_seconds / env.dt)
    cart_positions = [float(env.data.qpos[0])]
    trace: list[dict[str, Any]] = []
    final_info = reset_info
    for step in range(env.max_steps):
        if step < conditioning_steps:
            action = hanging_lqr_action(env, settle_gain)
            diagnostic = {"mode": "hanging_lqr"}
        else:
            action, diagnostic = controller.action(
                env, (step - conditioning_steps) * env.dt
            )
        _, _, terminated, truncated, final_info = env.step([action])
        cart_positions.append(float(env.data.qpos[0]))
        if include_trace:
            trace.append(
                {
                    "step": step + 1,
                    "time_seconds": (step + 1) * env.dt,
                    "action": action,
                    "qpos": np.asarray(env.data.qpos).astype(float).tolist(),
                    "qvel": np.asarray(env.data.qvel).astype(float).tolist(),
                    **diagnostic,
                }
            )
        if terminated or truncated:
            break
    setup = setup_from_config(cfg)
    result = {
        "seed": int(seed),
        "success": bool(final_info.get("success", False)),
        "termination_reason": final_info.get("termination_reason"),
        "conditioning_seconds": float(conditioning_steps * env.dt),
        "switch_time_after_conditioning": controller.switch_time,
        "max_upright_streak_seconds": float(
            final_info.get("max_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(max(abs(value) for value in cart_positions)),
        "rail_requirement": rail_requirement(np.asarray(cart_positions), setup),
        "final_info": final_info,
    }
    if include_trace:
        result["trace"] = trace
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--n-links", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=71001)
    parser.add_argument("--out", required=True)
    parser.add_argument("--include-traces", action="store_true")
    parser.add_argument("--conditioning-seconds", type=float, default=10.0)
    parser.add_argument("--collective-modal-gain", type=float, default=0.0)
    parser.add_argument("--internal-modal-damping-gain", type=float, default=0.0)
    parser.add_argument("--modal-acceleration-limit-ratio", type=float, default=2.0)
    args = parser.parse_args()
    if min(args.n_links, args.episodes) < 1:
        raise ValueError("link count and episode count must be positive")
    cfg = copy.deepcopy(load_config(args.config))
    cfg["env"]["n_links"] = args.n_links
    cfg["experiment"]["name"] = f"generalized_energy_n{args.n_links}"
    for name in ("lengths", "masses", "joint_stiffness", "joint_lock"):
        cfg["morphology"].pop(f"{name}_start", None)
        cfg["morphology"].pop(f"{name}_end", None)
    for endpoint in ("start", "end"):
        cfg["morphology"][endpoint]["alpha_length"] = 0.0
        cfg["morphology"][endpoint]["alpha_mass"] = 0.0
        cfg["morphology"][endpoint]["alpha_damping"] = 0.0
    cfg["env"]["init_mode"] = "hanging"
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"]["action_lqr_switch"] = {"enabled": False}
    if (
        min(
            args.collective_modal_gain,
            args.internal_modal_damping_gain,
            args.modal_acceleration_limit_ratio,
        )
        < 0.0
    ):
        raise ValueError("modal gains and acceleration limit must be nonnegative")
    parameters = EnergySwingParameters(
        collective_modal_gain=args.collective_modal_gain,
        internal_modal_damping_gain=args.internal_modal_damping_gain,
        modal_acceleration_limit_ratio=args.modal_acceleration_limit_ratio,
    )
    episodes = [
        run_episode(
            cfg,
            parameters,
            args.seed + index,
            args.include_traces,
            conditioning_seconds=args.conditioning_seconds,
        )
        for index in range(args.episodes)
    ]
    success_rate = float(np.mean([episode["success"] for episode in episodes]))
    setup = setup_from_config(cfg)
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_deterministic_baseline",
        "summary": "Exact mass-matrix energy shaping followed by exact-linearization LQR capture.",
        "source_config": file_metadata(Path(args.config)),
        "resolved_config_sha256": data_sha256(cfg),
        "n_links": args.n_links,
        "dimensionless_setup": dimensionless_setup(setup).to_dict(),
        "controller": {
            "type": "dimensionless_energy_pfl_then_exact_lqr",
            "parameters": parameters.to_dict(),
        },
        "episodes": args.episodes,
        "seed_start": args.seed,
        "conditioning_seconds": args.conditioning_seconds,
        "success_rate": success_rate,
        "termination_counts": dict(
            Counter(str(row["termination_reason"]) for row in episodes)
        ),
        "max_rail_ratio": float(
            max(row["rail_requirement"]["required_rail_ratio"] for row in episodes)
        ),
        "episode_results": episodes,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, args.out)
    print(
        f"n={args.n_links} episodes={args.episodes} success={success_rate:.3f} "
        f"max_required_rho={output['max_rail_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
