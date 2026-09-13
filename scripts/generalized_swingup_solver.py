#!/usr/bin/env python
"""Analyze morphologies and generate exact-dynamics warm starts across link counts.

This is the public entry point for the count/length-agnostic solver work.  It
does not certify a swing-up by itself: ``transfer`` writes a clearly labelled
warm start that must be refined with exact MuJoCo trajectory optimization and
then evaluated under the repository's sustained-upright gate.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, utc_timestamp
from gcartpole.fddp import rollout_controls
from gcartpole.generalized_modes import chain_normal_modes
from gcartpole.generalized_solver import (
    dimensionless_setup,
    force_action_scale,
    rail_requirement,
    resample_controls,
    setup_from_config,
    state_transfer_matrix,
    transfer_feedback_gains,
    transfer_state,
)
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.linear import analyze_morphology
from gcartpole.modal import StateScales, dimensionless_absolute_transform

DEFAULT_SCALES = StateScales(
    cart_position=3.0,
    absolute_angle=0.15,
    cart_velocity=4.0,
    hinge_velocity=15.0,
)


def uniform_config(base: dict[str, Any], n_links: int) -> dict[str, Any]:
    cfg = copy.deepcopy(base)
    cfg["env"]["n_links"] = int(n_links)
    cfg["experiment"]["name"] = f"generalized_uniform_n{n_links}"
    cfg["morphology"].pop("lengths_start", None)
    cfg["morphology"].pop("lengths_end", None)
    cfg["morphology"].pop("masses_start", None)
    cfg["morphology"].pop("masses_end", None)
    for name in ("joint_stiffness", "joint_lock"):
        cfg["morphology"].pop(f"{name}_start", None)
        cfg["morphology"].pop(f"{name}_end", None)
    for endpoint in ("start", "end"):
        cfg["morphology"][endpoint]["alpha_length"] = 0.0
        cfg["morphology"][endpoint]["alpha_mass"] = 0.0
        cfg["morphology"][endpoint]["alpha_damping"] = 0.0
    return cfg


def coordinate_transform(n_links: int, spec: dict[str, Any] | None) -> np.ndarray:
    scales = DEFAULT_SCALES
    if spec is not None:
        distribution = spec["distribution"]
        scales = StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        )
    return dimensionless_absolute_transform(n_links, scales)


def controller_record(payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    controller = payload.get("controller")
    search = payload.get("search")
    if not isinstance(controller, dict) or not isinstance(search, dict):
        raise TypeError("source controller must contain controller and search objects")
    return controller, search


def time_interpolate_rows(
    rows: np.ndarray,
    source_count: int,
    target_count: int,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.shape[0] != source_count:
        raise ValueError("time-series row count does not match source controls")
    source_phase = (np.arange(source_count, dtype=np.float64) + 0.5) / source_count
    target_phase = (np.arange(target_count, dtype=np.float64) + 0.5) / target_count
    return np.column_stack(
        [
            np.interp(target_phase, source_phase, rows[:, column])
            for column in range(rows.shape[1])
        ]
    )


def analyze_command(args: argparse.Namespace) -> None:
    base = load_config(args.config)
    records: list[dict[str, Any]] = []
    for n_links in range(args.min_links, args.max_links + 1):
        cfg = uniform_config(base, n_links)
        physical = setup_from_config(cfg)
        pi = dimensionless_setup(physical)
        linear = analyze_morphology(
            n_links=n_links,
            total_length=physical.chain_length,
            total_mass=physical.link_mass,
            cart_mass=physical.cart_mass,
            alpha_length=0.0,
            alpha_mass=0.0,
            total_damping=float(np.sum(physical.joint_damping)),
        )
        record = pi.to_dict()
        record["upright_linearization"] = linear.to_dict()
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
        env.reset(seed=0)
        record["hanging_normal_modes"] = chain_normal_modes(
            env, equilibrium="hanging"
        ).to_dict()
        record["upright_normal_modes"] = chain_normal_modes(
            env, equilibrium="upright"
        ).to_dict()
        env.close()
        record["minimum_geometric_rail_ratio"] = (
            physical.cart_half_length / physical.chain_length
        )
        records.append(record)
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "analysis_not_solution_evidence",
        "summary": "Dimensionless morphology and exact normal-mode table for the generalized solver.",
        "source_config": file_metadata(Path(args.config)),
        "link_range": [args.min_links, args.max_links],
        "records": records,
    }
    dump_json(payload, args.out)
    print(f"wrote {args.out} with n={args.min_links}..{args.max_links}")


def transfer_command(args: argparse.Namespace) -> None:
    source_base = load_config(args.source_config)
    target_base = load_config(args.target_config or args.source_config)
    source_cfg = (
        source_base
        if int(source_base["env"]["n_links"]) == args.source_links
        else uniform_config(source_base, args.source_links)
    )
    target_cfg = (
        target_base
        if int(target_base["env"]["n_links"]) == args.target_links
        else uniform_config(target_base, args.target_links)
    )
    source_setup = setup_from_config(source_cfg)
    target_setup = setup_from_config(target_cfg)
    spec = None if args.spec is None else load_config(args.spec)
    source_transform = coordinate_transform(source_setup.n_links, spec)
    target_transform = coordinate_transform(target_setup.n_links, spec)

    source_path = Path(args.controller)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    controller, search = controller_record(payload)
    source_controls = np.asarray(controller["controls"], dtype=np.float64)
    source_gains = np.asarray(controller.get("feedback_gains", []), dtype=np.float64)
    source_coordinate_states = np.asarray(
        search["nominal_coordinate_states"], dtype=np.float64
    )
    source_dim = 2 * (source_setup.n_links + 1)
    if source_coordinate_states.shape != (source_controls.size + 1, source_dim):
        raise ValueError("source nominal states do not match source morphology")
    if source_gains.shape != (source_controls.size, source_dim):
        raise ValueError("source feedback gains do not match source morphology")

    controls = resample_controls(source_controls, source_setup, target_setup)
    spatial_gains = transfer_feedback_gains(source_gains, source_setup, target_setup)
    feedback_gains = time_interpolate_rows(
        spatial_gains, source_controls.size, controls.size
    )

    source_initial_physical = np.linalg.solve(
        source_transform, source_coordinate_states[0]
    )
    target_initial_physical = transfer_state(
        source_initial_physical, source_setup, target_setup
    )
    cfg = copy.deepcopy(target_cfg)
    cfg["env"]["init_mode"] = "fixed_state"
    nq = target_setup.n_links + 1
    cfg["env"]["init_qpos"] = target_initial_physical[:nq].tolist()
    cfg["env"]["init_qvel"] = target_initial_physical[nq:].tolist()
    cfg["env"]["episode_seconds"] = max(
        float(cfg["env"]["episode_seconds"]),
        controls.size * target_setup.policy_dt + 1.0,
    )
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        cfg["env"][key] = 0.0
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    transition = MujocoTransition(env, coordinate_transform=target_transform)
    initial_coordinate_state = transition.to_coordinates(data_state(env.data))
    target_states = rollout_controls(transition, initial_coordinate_state, controls)
    physical_cart = np.asarray(
        [transition.to_physical(state)[0] for state in target_states], dtype=np.float64
    )
    rail = rail_requirement(physical_cart, target_setup)
    env.close()

    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "warm_start_not_solution_evidence",
        "not_solution": True,
        "summary": "Morphology-normalized controller transfer replayed through exact target MuJoCo dynamics.",
        "source": {
            "config": file_metadata(Path(args.source_config)),
            "controller": file_metadata(source_path),
            "dimensionless": dimensionless_setup(source_setup).to_dict(),
        },
        "target": {
            "config": args.target_config,
            "n_links": target_setup.n_links,
            "dimensionless": dimensionless_setup(target_setup).to_dict(),
        },
        "transfer": {
            "state_matrix": state_transfer_matrix(source_setup, target_setup).tolist(),
            "force_action_scale": force_action_scale(source_setup, target_setup),
            "rail_requirement": rail,
        },
        "selected_state": {
            "qpos": target_initial_physical[:nq].tolist(),
            "qvel": target_initial_physical[nq:].tolist(),
            "state_index": 0,
        },
        "controller": {
            "type": "generalized_morphology_transfer_warm_start",
            "controls": controls.tolist(),
            "feedback_gains": feedback_gains.tolist(),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(controls.size * target_setup.policy_dt),
            "policy_dt": target_setup.policy_dt,
            "coordinate_transform": target_transform.tolist(),
        },
        "search": {
            "nominal_coordinate_states": target_states.tolist(),
            "is_feasible": bool(
                np.max(np.abs(physical_cart)) < target_setup.rail_half_length
            ),
            "iterations": 0,
            "cost": None,
        },
    }
    dump_json(output, args.out)
    print(
        f"wrote {args.out}: n={source_setup.n_links}->{target_setup.n_links}, "
        f"steps={controls.size}, max|x|={rail['max_cart_center_excursion']:.3f}, "
        f"required rho={rail['required_rail_ratio']:.3f}"
    )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser(
        "analyze", help="write a dimensionless morphology table"
    )
    analyze.add_argument("--config", default="configs/swingup7_uniform.yaml")
    analyze.add_argument("--min-links", type=int, default=1)
    analyze.add_argument("--max-links", type=int, default=20)
    analyze.add_argument("--out", required=True)
    analyze.set_defaults(run=analyze_command)

    transfer = commands.add_parser(
        "transfer", help="generate an exact target warm start"
    )
    transfer.add_argument("--source-config", default="configs/swingup7_uniform.yaml")
    transfer.add_argument("--target-config", default=None)
    transfer.add_argument("--source-links", type=int, required=True)
    transfer.add_argument("--target-links", type=int, required=True)
    transfer.add_argument("--controller", required=True)
    transfer.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    transfer.add_argument("--out", required=True)
    transfer.set_defaults(run=transfer_command)
    return root


def main() -> None:
    args = parser().parse_args()
    if getattr(args, "min_links", 1) < 1 or getattr(args, "max_links", 1) < getattr(
        args, "min_links", 1
    ):
        raise ValueError("link range must be positive and ordered")
    if getattr(args, "source_links", 1) < 1 or getattr(args, "target_links", 1) < 1:
        raise ValueError("link counts must be positive")
    args.run(args)


if __name__ == "__main__":
    main()
