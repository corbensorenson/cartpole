#!/usr/bin/env python
"""Materialize an analytic normal-mode phase schedule through exact MuJoCo."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.generalized_energy import force_for_desired_cart_acceleration
from gcartpole.generalized_modes import (
    chain_normal_modes,
    minimum_energy_modal_phase_seed,
)
from gcartpole.generalized_solver import rail_requirement, setup_from_config
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform


def uniform_config(base: dict, n_links: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["env"]["n_links"] = int(n_links)
    cfg["env"]["init_mode"] = "hanging"
    cfg["env"]["terminate_abs_angle"] = None
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"]["action_lqr_switch"] = {"enabled": False}
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        cfg["env"][key] = 0.0
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    for name in (
        "lengths",
        "masses",
        "damping",
        "frictionloss",
        "joint_stiffness",
        "joint_lock",
    ):
        cfg["morphology"].pop(f"{name}_start", None)
        cfg["morphology"].pop(f"{name}_end", None)
    for endpoint in ("start", "end"):
        cfg["morphology"][endpoint]["alpha_length"] = 0.0
        cfg["morphology"][endpoint]["alpha_mass"] = 0.0
        cfg["morphology"][endpoint]["alpha_damping"] = 0.0
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--n-links", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--acceleration-limit-ratio", type=float, default=4.0)
    parser.add_argument("--regularization", type=float, default=1.0e-8)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.n_links < 1:
        raise ValueError("--n-links must be positive")
    cfg = uniform_config(
        apply_overrides(load_config(args.config), args.override), args.n_links
    )
    cfg["env"]["episode_seconds"] = max(
        float(cfg["env"]["episode_seconds"]), args.seconds + 1.0
    )
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    modes = chain_normal_modes(env, equilibrium="hanging")
    setup = setup_from_config(cfg)
    seed = minimum_energy_modal_phase_seed(
        modes,
        policy_dt=float(env.dt),
        horizon_seconds=args.seconds,
        gravity=setup.gravity,
        acceleration_limit_ratio=args.acceleration_limit_ratio,
        regularization=args.regularization,
    )
    spec = load_config(args.spec)
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        env.n,
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    transition = MujocoTransition(env, coordinate_transform=transform)
    physical_states = [data_state(env.data)]
    controls: list[float] = []
    cart_positions = [float(env.data.qpos[0])]
    done_events: list[dict[str, object]] = []
    trace: list[dict[str, object]] = []
    for step, acceleration in enumerate(seed.accelerations):
        force = force_for_desired_cart_acceleration(env, float(acceleration))
        action = float(np.clip(force / env.force_limit, -1.0, 1.0))
        _, _, terminated, truncated, info = env.step([action])
        controls.append(float(info["applied_action_norm"]))
        physical_states.append(data_state(env.data))
        cart_positions.append(float(env.data.qpos[0]))
        trace.append(
            {
                "step": step + 1,
                "time_seconds": (step + 1) * env.dt,
                "action": float(info["applied_action_norm"]),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64)
                .astype(float)
                .tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64)
                .astype(float)
                .tolist(),
                "max_abs_angle": float(info["max_abs_angle"]),
                "absolute_angular_velocity_rms": float(
                    info["absolute_angular_velocity_rms"]
                ),
            }
        )
        if terminated or truncated:
            done_events.append(
                {
                    "step": step + 1,
                    "termination_reason": info.get("termination_reason"),
                }
            )
    states = np.asarray(
        [transition.to_coordinates(state) for state in physical_states],
        dtype=np.float64,
    )
    initial = physical_states[0]
    nq = env.n + 1
    final_info = env._info()
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "analytic_modal_phase_warm_start_not_solution",
        "not_solution": True,
        "summary": "Analytic minimum-energy modal phase schedule replayed through exact target MuJoCo; warm start only.",
        "selected_state": {
            "qpos": initial[:nq].astype(float).tolist(),
            "qvel": initial[nq:].astype(float).tolist(),
            "state_index": 0,
        },
        "controller": {
            "type": "analytic_modal_phase_seed_exact_mujoco_replay",
            "controls": controls,
            "feedback_gains": np.zeros((len(controls), 2 * nq)).tolist(),
            "horizon_steps": len(controls),
            "horizon_seconds": len(controls) * env.dt,
            "policy_dt": env.dt,
            "coordinate_transform": transform.astype(float).tolist(),
        },
        "search": {
            "iterations": 0,
            "is_feasible": not done_events,
            "cost": None,
            "nominal_coordinate_states": states.astype(float).tolist(),
        },
        "analytic_seed": seed.to_dict(),
        "normal_modes": modes.to_dict(),
        "exact_replay": {
            "done_events": done_events,
            "final_info": final_info,
            "trace": trace,
            "rail_requirement": rail_requirement(
                np.asarray(cart_positions, dtype=np.float64), setup
            ),
            "controls_sha256": data_sha256(controls),
        },
        "source_config": file_metadata(Path(args.config)),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(output, Path(args.out))
    print(
        f"wrote {args.out}: linear_residual={np.linalg.norm(seed.terminal_residual):.3e} "
        f"exact_angle={final_info['max_abs_angle']:.3f} "
        f"max_cart={output['exact_replay']['rail_requirement']['max_cart_center_excursion']:.3f}"
    )


if __name__ == "__main__":
    main()
