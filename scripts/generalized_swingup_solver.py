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
import math
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config, save_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, utc_timestamp
from gcartpole.fddp import rollout_controls
from gcartpole.generalized_modes import chain_normal_modes
from gcartpole.generalized_solver import (
    PhysicalSetup,
    dimensionless_setup,
    force_action_scale,
    rail_requirement,
    resample_controls,
    setup_from_config,
    similarity_scaled_config,
    similarity_scaled_lqr_weights,
    split_embedding,
    split_joint_profile,
    split_state_lift_matrix,
    split_state_projection,
    state_transfer_matrix,
    transfer_coordinate_feedback_gains,
    transfer_coordinate_states,
    transfer_state,
)
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.linear import analyze_morphology
from gcartpole.modal import StateScales, dimensionless_absolute_transform
from gcartpole.morphology import build_morphology

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


def time_interpolate_samples(rows: np.ndarray, target_count: int) -> np.ndarray:
    """Interpolate state samples while preserving both time endpoints."""

    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 2 or target_count < 2:
        raise ValueError("state samples and target count must both have endpoints")
    source_phase = np.linspace(0.0, 1.0, rows.shape[0])
    target_phase = np.linspace(0.0, 1.0, target_count)
    return np.column_stack(
        [
            np.interp(target_phase, source_phase, rows[:, column])
            for column in range(rows.shape[1])
        ]
    )


def split_continuation_config(
    source_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a target-count homotopy whose start is a locked source chain."""

    source = setup_from_config(source_cfg)
    target = setup_from_config(target_cfg)
    if target.n_links <= source.n_links:
        raise ValueError("split continuation requires target links > source links")
    embedding = split_embedding(
        source.lengths,
        source.masses,
        target.lengths,
        target.masses,
    )
    source_morph = build_morphology(
        source_cfg["env"], source_cfg["morphology"], progress=1.0
    )
    target_morph = build_morphology(
        target_cfg["env"], target_cfg["morphology"], progress=1.0
    )
    source_rotational_scale = (
        source.system_mass * source.chain_length**2 / source.natural_time
    )
    target_rotational_scale = (
        target.system_mass * target.chain_length**2 / target.natural_time
    )
    start_damping = split_joint_profile(
        source.joint_damping
        * (target_rotational_scale / source_rotational_scale),
        embedding.segment_source_links,
    )
    torque_scale = (
        target.system_mass * target.gravity * target.chain_length
    ) / (source.system_mass * source.gravity * source.chain_length)
    start_friction = split_joint_profile(
        source_morph.frictionloss * torque_scale,
        embedding.segment_source_links,
    )
    start_stiffness = split_joint_profile(
        source_morph.joint_stiffness * torque_scale,
        embedding.segment_source_links,
    )

    cfg = copy.deepcopy(target_cfg)
    cfg["experiment"]["name"] = (
        f"split_continuation_n{source.n_links}_to_n{target.n_links}"
    )
    cfg["env"].pop("plant_progress", None)
    cfg["env"]["rigid_split_inertia"] = True
    cfg["env"]["joint_lock_impedance_schedule"] = "log_compliance"
    morph = cfg["morphology"]
    morph["schedule_mode"] = "all_linear"
    morph["lengths_start"] = embedding.source_lengths.tolist()
    morph["lengths_end"] = target.lengths.tolist()
    morph["masses_start"] = embedding.source_masses.tolist()
    morph["masses_end"] = target.masses.tolist()
    morph["damping_start"] = start_damping.tolist()
    morph["damping_end"] = target.joint_damping.tolist()
    morph["frictionloss_start"] = start_friction.tolist()
    morph["frictionloss_end"] = target_morph.frictionloss.tolist()
    morph["joint_stiffness_start"] = start_stiffness.tolist()
    morph["joint_stiffness_end"] = target_morph.joint_stiffness.tolist()
    morph["joint_lock_start"] = embedding.source_joint_locks.tolist()
    morph["joint_lock_end"] = embedding.target_joint_locks.tolist()
    for endpoint, damping, friction in (
        ("start", start_damping, start_friction),
        ("end", target.joint_damping, target_morph.frictionloss),
    ):
        morph.setdefault(endpoint, {})
        morph[endpoint]["alpha_length"] = 0.0
        morph[endpoint]["alpha_mass"] = 0.0
        morph[endpoint]["alpha_damping"] = 0.0
        morph[endpoint]["alpha_frictionloss"] = 0.0
        morph[endpoint]["total_damping"] = float(np.sum(damping))
        morph[endpoint]["total_frictionloss"] = float(np.sum(friction))

    source_pi = dimensionless_setup(source)
    start = setup_from_config(cfg, progress=0.0)
    start_pi = dimensionless_setup(start)
    embedded_source_damping = split_joint_profile(
        source_pi.joint_damping_ratios,
        embedding.segment_source_links,
    )
    compatibility = {
        "cart_to_link_mass_error": float(
            start_pi.cart_to_link_mass - source_pi.cart_to_link_mass
        ),
        "cart_damping_ratio_error": float(
            start_pi.cart_damping_ratio - source_pi.cart_damping_ratio
        ),
        "link_radius_ratio_error": float(
            start_pi.link_radius_ratio - source_pi.link_radius_ratio
        ),
        "joint_armature_ratio_error": float(
            start_pi.joint_armature_ratio - source_pi.joint_armature_ratio
        ),
        "joint_damping_ratio_max_error": float(
            np.max(
                np.abs(
                    start_pi.joint_damping_ratios - embedded_source_damping
                )
            )
        ),
    }
    compatibility["global_dynamic_similarity"] = bool(
        max(abs(float(value)) for value in compatibility.values()) <= 1.0e-10
    )
    metadata = embedding.to_dict()
    metadata["compatibility"] = compatibility
    return cfg, metadata


def split_warm_start(
    payload: dict[str, Any],
    source: PhysicalSetup,
    target_start: PhysicalSetup,
    embedding_metadata: dict[str, Any],
    source_transform: np.ndarray,
    target_transform: np.ndarray,
) -> dict[str, Any]:
    """Lift a route and its feedback through the locked-split injection."""

    controller, search = controller_record(payload)
    controls_source = np.asarray(controller["controls"], dtype=np.float64)
    gains_source = np.asarray(controller["feedback_gains"], dtype=np.float64)
    states_source = np.asarray(
        search["nominal_coordinate_states"], dtype=np.float64
    )
    source_dim = 2 * (source.n_links + 1)
    if gains_source.shape != (controls_source.size, source_dim):
        raise ValueError("source feedback gains do not match source morphology")
    if states_source.shape != (controls_source.size + 1, source_dim):
        raise ValueError("source nominal states do not match source morphology")

    assignments = np.asarray(
        embedding_metadata["segment_source_links"], dtype=np.int64
    )
    physical_lift = split_state_lift_matrix(
        source.n_links,
        assignments,
        length_scale=target_start.chain_length / source.chain_length,
    )
    coordinate_lift = (
        target_transform @ physical_lift @ np.linalg.inv(source_transform)
    )
    projection = split_state_projection(
        coordinate_lift,
        np.asarray(embedding_metadata["source_lengths"], dtype=np.float64)
        / target_start.chain_length,
    )
    action_scale = force_action_scale(source, target_start)
    controls = resample_controls(controls_source, source, target_start)
    spatial_gains = (
        action_scale * gains_source @ projection
    )
    feedback_gains = time_interpolate_rows(
        spatial_gains, controls_source.size, controls.size
    )
    lifted_states = states_source @ coordinate_lift.T
    states = time_interpolate_samples(lifted_states, controls.size + 1)
    initial_physical = physical_lift @ np.linalg.solve(
        source_transform, states_source[0]
    )
    nq = target_start.n_links + 1
    invariant_error = float(
        np.max(np.abs(projection @ coordinate_lift - np.eye(source_dim)))
    )
    return {
        "selected_state": {
            "qpos": initial_physical[:nq].tolist(),
            "qvel": initial_physical[nq:].tolist(),
            "state_index": 0,
        },
        "controller": {
            "type": "generalized_locked_split_warm_start",
            "controls": controls.tolist(),
            "feedback_gains": feedback_gains.tolist(),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(controls.size * target_start.policy_dt),
            "policy_dt": target_start.policy_dt,
            "coordinate_transform": target_transform.tolist(),
        },
        "search": {
            "nominal_coordinate_states": states.tolist(),
            "is_feasible": False,
            "iterations": 0,
            "cost": None,
        },
        "embedding": {
            **embedding_metadata,
            "state_lift_matrix": physical_lift.tolist(),
            "coordinate_lift_matrix": coordinate_lift.tolist(),
            "coordinate_projection_matrix": projection.tolist(),
            "feedback_invariance_max_abs_error": invariant_error,
            "force_action_scale": action_scale,
            "force_scaling_requires_clipping": bool(
                np.max(np.abs(controls_source)) * action_scale > 1.0
            ),
        },
    }


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


def similarity_command(args: argparse.Namespace) -> None:
    """Materialize a physical plant with identical dimensionless groups."""

    source_path = Path(args.config)
    source_cfg = apply_overrides(load_config(source_path), args.override)
    target_cfg = similarity_scaled_config(
        source_cfg,
        length_scale=args.length_scale,
        mass_scale=args.mass_scale,
    )
    source_pi = dimensionless_setup(setup_from_config(source_cfg))
    target_pi = dimensionless_setup(setup_from_config(target_cfg))
    scalar_names = (
        "cart_to_link_mass",
        "force_authority",
        "rail_ratio",
        "usable_rail_ratio",
        "cart_half_length_ratio",
        "link_radius_ratio",
        "policy_dt_ratio",
        "cart_damping_ratio",
        "joint_armature_ratio",
    )
    scalar_errors = {
        name: float(getattr(target_pi, name) - getattr(source_pi, name))
        for name in scalar_names
    }
    vector_errors = {
        "length_fractions": float(
            np.max(np.abs(target_pi.length_fractions - source_pi.length_fractions))
        ),
        "mass_fractions": float(
            np.max(np.abs(target_pi.mass_fractions - source_pi.mass_fractions))
        ),
        "joint_damping_ratios": float(
            np.max(
                np.abs(
                    target_pi.joint_damping_ratios
                    - source_pi.joint_damping_ratios
                )
            )
        ),
    }
    maximum_error = max(
        [abs(value) for value in scalar_errors.values()]
        + list(vector_errors.values())
    )
    save_config(target_cfg, args.out_config)
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "dynamic_similarity_plant_not_solution_evidence",
        "not_solution": True,
        "summary": (
            "Physical plant generated by deterministic Buckingham-pi scaling; "
            "controller transfer and exact nonlinear gates remain required."
        ),
        "source_config": file_metadata(source_path),
        "target_config": str(Path(args.out_config)),
        "scales": {
            "length": float(args.length_scale),
            "mass": float(args.mass_scale),
            "time": float(np.sqrt(args.length_scale)),
        },
        "source_dimensionless": source_pi.to_dict(),
        "target_dimensionless": target_pi.to_dict(),
        "dimensionless_errors": {
            "scalars": scalar_errors,
            "vectors_max_abs": vector_errors,
            "maximum_abs": float(maximum_error),
        },
    }
    dump_json(output, args.out)
    print(
        f"wrote {args.out_config} and {args.out}: "
        f"lambda={args.length_scale:g} mu={args.mass_scale:g} "
        f"max_pi_error={maximum_error:.3e}"
    )


def transfer_command(args: argparse.Namespace) -> None:
    source_base = apply_overrides(load_config(args.source_config), args.source_override)
    target_base = apply_overrides(
        load_config(args.target_config or args.source_config), args.target_override
    )
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
    spatial_gains = transfer_coordinate_feedback_gains(
        source_gains,
        source_setup,
        target_setup,
        source_transform,
        target_transform,
    )
    feedback_gains = time_interpolate_rows(
        spatial_gains, source_controls.size, controls.size
    )
    lqr_weights = similarity_scaled_lqr_weights(
        controller.get("lqr_weights"),
        length_scale=target_setup.chain_length / source_setup.chain_length,
    )
    length_scale = target_setup.chain_length / source_setup.chain_length

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
    transferred_nominal_states = transfer_coordinate_states(
        source_coordinate_states,
        source_setup,
        target_setup,
        source_transform,
        target_transform,
    )
    source_feedback_actions = np.einsum(
        "ij,ij->i", source_gains, source_coordinate_states[:-1]
    )
    target_feedback_actions = np.einsum(
        "ij,ij->i", feedback_gains, transferred_nominal_states[:-1]
    )
    feedback_invariance_error = float(
        np.max(
            np.abs(
                target_feedback_actions
                - force_action_scale(source_setup, target_setup)
                * source_feedback_actions
            )
        )
    )
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
            "coordinate_feedback_invariance_max_abs_error": (
                feedback_invariance_error
            ),
            "rail_requirement": rail,
        },
        "selected_state": {
            "qpos": target_initial_physical[:nq].tolist(),
            "qvel": target_initial_physical[nq:].tolist(),
            "state_index": 0,
        },
        "controller": {
            "type": "generalized_morphology_transfer_then_lqr_warm_start",
            "controls": controls.tolist(),
            "feedback_gains": feedback_gains.tolist(),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(controls.size * target_setup.policy_dt),
            "policy_dt": target_setup.policy_dt,
            "coordinate_transform": target_transform.tolist(),
            "lqr_scale": float(controller.get("lqr_scale", 1.0)),
            "lqr_control_cost": float(controller.get("lqr_control_cost", 1000.0)),
            "lqr_weights": lqr_weights,
            "defer_handoff_until_horizon": bool(
                controller.get("defer_handoff_until_horizon", True)
            ),
            "handoff_angle_abs": float(
                controller.get("handoff_angle_abs", 0.15)
            ),
            "handoff_cart_abs": float(
                controller.get("handoff_cart_abs", 0.5 * source_setup.rail_half_length)
            )
            * length_scale,
            "handoff_cart_velocity_abs": float(
                controller.get("handoff_cart_velocity_abs", 0.5)
            )
            * math.sqrt(length_scale),
            "handoff_hinge_velocity_rms": float(
                controller.get("handoff_hinge_velocity_rms", 0.75)
            )
            / math.sqrt(length_scale),
            "handoff_lyapunov": float(
                controller.get(
                    "handoff_lyapunov", controller.get("switch_lyapunov", math.inf)
                )
            ),
            "periodic_coordinate_errors": bool(
                controller.get("periodic_coordinate_errors", False)
            ),
        },
        "search": {
            "nominal_coordinate_states": transferred_nominal_states.tolist(),
            "exact_open_loop_coordinate_states": target_states.tolist(),
            "is_feasible": bool(
                np.max(np.abs(physical_cart)) < target_setup.rail_half_length
            ),
            "iterations": 0,
            "cost": None,
        },
    }
    if "materialized_swing_prefix_steps" in controller:
        step_ratio = controls.size / source_controls.size
        prefix_steps = round(
            float(controller["materialized_swing_prefix_steps"]) * step_ratio
        )
        prefix_steps = min(max(prefix_steps, 0), controls.size)
        output["controller"]["materialized_swing_prefix_steps"] = prefix_steps
        output["controller"]["materialized_lqr_tail_steps"] = int(
            controls.size - prefix_steps
        )
    dump_json(output, args.out)
    print(
        f"wrote {args.out}: n={source_setup.n_links}->{target_setup.n_links}, "
        f"steps={controls.size}, max|x|={rail['max_cart_center_excursion']:.3f}, "
        f"required rho={rail['required_rail_ratio']:.3f}"
    )


def split_command(args: argparse.Namespace) -> None:
    """Materialize a count-increase curriculum and its algebraic warm start."""

    source_base = apply_overrides(load_config(args.source_config), args.source_override)
    target_base = apply_overrides(load_config(args.target_config), args.target_override)
    source_cfg = (
        source_base
        if int(source_base["env"]["n_links"]) == args.source_links
        else uniform_config(source_base, args.source_links)
    )
    continuation_cfg, embedding = split_continuation_config(source_cfg, target_base)
    source = setup_from_config(source_cfg)
    target_start = setup_from_config(continuation_cfg, progress=0.0)
    spec = None if args.spec is None else load_config(args.spec)
    source_transform = coordinate_transform(source.n_links, spec)
    target_transform = coordinate_transform(target_start.n_links, spec)
    source_path = Path(args.controller)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    warm_start = split_warm_start(
        payload,
        source,
        target_start,
        embedding,
        source_transform,
        target_transform,
    )
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "locked_split_warm_start_not_solution_evidence",
        "not_solution": True,
        "summary": (
            "Deterministic locked-split count continuation; exact optimization "
            "and the sustained-upright gates are still required."
        ),
        "source": {
            "config": file_metadata(Path(args.source_config)),
            "controller": file_metadata(source_path),
            "dimensionless": dimensionless_setup(source).to_dict(),
        },
        "target": {
            "config": file_metadata(Path(args.target_config)),
            "continuation_config": str(Path(args.out_config)),
            "dimensionless_start": dimensionless_setup(target_start).to_dict(),
            "dimensionless_end": dimensionless_setup(
                setup_from_config(continuation_cfg, progress=1.0)
            ).to_dict(),
        },
        **warm_start,
    }
    save_config(continuation_cfg, args.out_config)
    dump_json(output, args.out)
    compatibility = embedding["compatibility"]["global_dynamic_similarity"]
    print(
        f"wrote {args.out_config} and {args.out}: "
        f"n={source.n_links}->{target_start.n_links}, "
        f"globally_similar={compatibility}, "
        f"feedback_error={warm_start['embedding']['feedback_invariance_max_abs_error']:.3e}"
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

    similarity = commands.add_parser(
        "similarity", help="generate a Buckingham-pi-equivalent physical plant"
    )
    similarity.add_argument("--config", default="configs/swingup7_uniform.yaml")
    similarity.add_argument("--length-scale", type=float, required=True)
    similarity.add_argument("--mass-scale", type=float, required=True)
    similarity.add_argument("--override", action="append", default=[])
    similarity.add_argument("--out-config", required=True)
    similarity.add_argument("--out", required=True)
    similarity.set_defaults(run=similarity_command)

    transfer = commands.add_parser(
        "transfer", help="generate an exact target warm start"
    )
    transfer.add_argument("--source-config", default="configs/swingup7_uniform.yaml")
    transfer.add_argument("--target-config", default=None)
    transfer.add_argument("--source-links", type=int, required=True)
    transfer.add_argument("--target-links", type=int, required=True)
    transfer.add_argument("--controller", required=True)
    transfer.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    transfer.add_argument("--source-override", action="append", default=[])
    transfer.add_argument("--target-override", action="append", default=[])
    transfer.add_argument("--out", required=True)
    transfer.set_defaults(run=transfer_command)

    split = commands.add_parser(
        "split", help="build a locked-split count continuation and warm start"
    )
    split.add_argument("--source-config", default="configs/swingup7_uniform.yaml")
    split.add_argument("--target-config", required=True)
    split.add_argument("--source-links", type=int, required=True)
    split.add_argument("--controller", required=True)
    split.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    split.add_argument("--source-override", action="append", default=[])
    split.add_argument("--target-override", action="append", default=[])
    split.add_argument("--out-config", required=True)
    split.add_argument("--out", required=True)
    split.set_defaults(run=split_command)
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
