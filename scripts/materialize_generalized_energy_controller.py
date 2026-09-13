#!/usr/bin/env python
"""Materialize the deterministic energy/LQR law as an exact FDDP warm start."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from gcartpole.config import dump_json, load_config, save_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, utc_timestamp
from gcartpole.generalized_energy import EnergySwingParameters, GeneralizedEnergyController
from gcartpole.ilqr import data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform


def uniform_config(base: dict, n_links: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["env"]["n_links"] = int(n_links)
    cfg["env"]["init_mode"] = "hanging"
    cfg["env"]["terminate_abs_angle"] = None
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"]["action_lqr_switch"] = {"enabled": False}
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][key] = 0.0
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    for name in ("lengths", "masses", "joint_stiffness", "joint_lock"):
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
    parser.add_argument("--resolved-config-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = uniform_config(load_config(args.config), args.n_links)
    cfg["env"]["episode_seconds"] = max(float(cfg["env"]["episode_seconds"]), args.seconds + 1.0)
    save_config(cfg, args.resolved_config_out)
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    parameters = EnergySwingParameters()
    controller = GeneralizedEnergyController(env, parameters)
    spec = load_config("benchmarks/p1_capture_envelope.yaml")
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
    states = [transform @ data_state(env.data)]
    physical_states = [data_state(env.data)]
    controls: list[float] = []
    modes: list[str] = []
    steps = int(round(args.seconds / env.dt))
    for step in range(steps):
        action, diagnostic = controller.action(env, step * env.dt)
        env.step([action])
        controls.append(float(action))
        modes.append(str(diagnostic["mode"]))
        physical = data_state(env.data)
        physical_states.append(physical)
        states.append(transform @ physical)
    final_info = env._info()
    nq = env.n + 1
    initial = physical_states[0]
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "warm_start_not_solution_evidence",
        "not_solution": True,
        "summary": "Exact deterministic energy/PFL/LQR nominal route for morphology continuation.",
        "source_config": file_metadata(Path(args.config)),
        "resolved_config": file_metadata(Path(args.resolved_config_out)),
        "selected_state": {
            "qpos": initial[:nq].astype(float).tolist(),
            "qvel": initial[nq:].astype(float).tolist(),
            "state_index": 0,
        },
        "controller": {
            "type": "materialized_dimensionless_energy_pfl_then_exact_lqr",
            "parameters": parameters.to_dict(),
            "controls": controls,
            "feedback_gains": np.zeros((steps, 2 * nq), dtype=np.float64).tolist(),
            "horizon_steps": steps,
            "horizon_seconds": steps * env.dt,
            "policy_dt": env.dt,
            "coordinate_transform": transform.tolist(),
            "mode_by_step": modes,
        },
        "search": {
            "nominal_coordinate_states": np.asarray(states).astype(float).tolist(),
            "iterations": 0,
            "is_feasible": bool(max(abs(float(state[0])) for state in physical_states) < env.rail_limit),
            "cost": None,
        },
        "terminal": final_info,
    }
    env.close()
    dump_json(output, args.out)
    print(
        f"wrote {args.out}: n={args.n_links}, switch={controller.switch_time}, "
        f"terminal angle={final_info['max_abs_angle']:.6f}, "
        f"hold={final_info['max_upright_streak_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
