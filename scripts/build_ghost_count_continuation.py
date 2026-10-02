#!/usr/bin/env python
"""Build a fixed-count, low-inertia ghost-link continuation config.

The source route remains kinematically exact at progress zero: the final
source link is split into a real proximal piece and a short, nearly massless
distal ghost, and the inserted joint is locked.  The continuation then grows
the ghost into the requested target segment while releasing that lock.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from gcartpole.config import load_config, save_config
from gcartpole.generalized_solver import setup_from_config, split_joint_profile
from gcartpole.morphology import build_morphology


def build_config(
    source_cfg: dict,
    target_cfg: dict,
    *,
    ghost_length: float,
    ghost_mass: float,
    unlocked_start: bool = False,
    ghost_damping: float = 0.0,
    ghost_stiffness: float = 0.0,
    hold_ghost_support: bool = False,
    rail_start: float | None = None,
    rail_end: float | None = None,
) -> dict:
    source = setup_from_config(source_cfg)
    target = setup_from_config(target_cfg)
    if target.n_links != source.n_links + 1:
        raise ValueError("ghost continuation currently supports one added link")
    if not np.isclose(source.chain_length, target.chain_length):
        raise ValueError("source and target chain lengths must match")
    if not np.isclose(source.link_mass, target.link_mass):
        raise ValueError("source and target link masses must match")
    if ghost_length <= 0.0 or ghost_length >= source.lengths[-1]:
        raise ValueError("ghost length must be positive and smaller than the final source link")
    if ghost_mass <= 0.0 or ghost_mass >= source.masses[-1]:
        raise ValueError("ghost mass must be positive and smaller than the final source link")
    if ghost_damping < 0.0 or ghost_stiffness < 0.0:
        raise ValueError("ghost damping and stiffness must be nonnegative")

    assignments = np.arange(target.n_links, dtype=np.int64)
    assignments[-1] = source.n_links - 1
    lengths_start = np.r_[
        source.lengths[:-1],
        source.lengths[-1] - ghost_length,
        ghost_length,
    ]
    masses_start = np.r_[
        source.masses[:-1],
        source.masses[-1] - ghost_mass,
        ghost_mass,
    ]
    target_morph = build_morphology(
        target_cfg["env"], target_cfg["morphology"], progress=1.0
    )
    source_rotational_scale = (
        source.system_mass * source.chain_length**2 / source.natural_time
    )
    target_rotational_scale = (
        target.system_mass * target.chain_length**2 / target.natural_time
    )
    damping_start = split_joint_profile(
        source.joint_damping
        * (target_rotational_scale / source_rotational_scale),
        assignments,
    )
    torque_scale = (
        target.system_mass * target.gravity * target.chain_length
    ) / (source.system_mass * source.gravity * source.chain_length)
    source_morph = build_morphology(
        source_cfg["env"], source_cfg["morphology"], progress=1.0
    )
    friction_start = split_joint_profile(
        source_morph.frictionloss * torque_scale, assignments
    )
    stiffness_start = split_joint_profile(
        source_morph.joint_stiffness * torque_scale, assignments
    )
    damping_start[-1] = float(ghost_damping)
    stiffness_start[-1] = float(ghost_stiffness)

    cfg = copy.deepcopy(target_cfg)
    cfg["experiment"]["name"] = (
        f"ghost_count_continuation_n{source.n_links}_to_n{target.n_links}"
    )
    cfg["env"]["rigid_split_inertia"] = True
    cfg["env"]["joint_lock_impedance_schedule"] = "log_compliance"
    if rail_start is not None or rail_end is not None:
        end_rail = float(target_cfg["env"]["rail_limit"] if rail_end is None else rail_end)
        start_rail = float(end_rail if rail_start is None else rail_start)
        if start_rail <= 0.0 or end_rail <= 0.0:
            raise ValueError("rail schedule values must be positive")
        cfg["env"]["rail_limit"] = end_rail
        cfg["env"]["rail_limit_start"] = start_rail
        cfg["env"]["rail_limit_end"] = end_rail
    morph = cfg["morphology"]
    morph["schedule_mode"] = "all_linear"
    morph["lengths_start"] = lengths_start.astype(float).tolist()
    morph["lengths_end"] = target.lengths.astype(float).tolist()
    morph["masses_start"] = masses_start.astype(float).tolist()
    morph["masses_end"] = target.masses.astype(float).tolist()
    morph["damping_start"] = damping_start.astype(float).tolist()
    morph["damping_end"] = target.joint_damping.astype(float).tolist()
    morph["frictionloss_start"] = friction_start.astype(float).tolist()
    morph["frictionloss_end"] = target_morph.frictionloss.astype(float).tolist()
    morph["joint_stiffness_start"] = stiffness_start.astype(float).tolist()
    morph["joint_stiffness_end"] = target_morph.joint_stiffness.astype(float).tolist()
    morph["joint_lock_start"] = (
        [0.0] * target.n_links
        if unlocked_start
        else [0.0] * source.n_links + [1.0]
    )
    morph["joint_lock_end"] = [0.0] * target.n_links
    if hold_ghost_support:
        damping_end = np.asarray(damping_start, dtype=np.float64).copy()
        stiffness_end = np.asarray(stiffness_start, dtype=np.float64).copy()
        damping_end[-1] = float(ghost_damping)
        stiffness_end[-1] = float(ghost_stiffness)
        morph["damping_end"] = damping_end.astype(float).tolist()
        morph["joint_stiffness_end"] = stiffness_end.astype(float).tolist()
        morph["end"]["total_damping"] = float(np.sum(damping_end))
    for endpoint, damping, friction in (
        ("start", damping_start, friction_start),
        (
            "end",
            damping_end if hold_ghost_support else target.joint_damping,
            target_morph.frictionloss,
        ),
    ):
        morph.setdefault(endpoint, {})
        morph[endpoint]["alpha_length"] = 0.0
        morph[endpoint]["alpha_mass"] = 0.0
        morph[endpoint]["alpha_damping"] = 0.0
        morph[endpoint]["alpha_frictionloss"] = 0.0
        morph[endpoint]["total_damping"] = float(np.sum(damping))
        morph[endpoint]["total_frictionloss"] = float(np.sum(friction))
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--ghost-length", type=float, default=0.02)
    # MuJoCo rejects moving bodies below its inertia floor; 0.01 is still a
    # tenfold lighter distal segment than the uniform target link.
    parser.add_argument("--ghost-mass", type=float, default=0.01)
    parser.add_argument(
        "--unlocked-start",
        action="store_true",
        help="leave the inserted ghost joint free at continuation progress zero",
    )
    parser.add_argument(
        "--ghost-damping",
        type=float,
        default=0.0,
        help="temporary inserted-joint damping at continuation progress zero",
    )
    parser.add_argument(
        "--ghost-stiffness",
        type=float,
        default=0.0,
        help="temporary inserted-joint straightening stiffness at progress zero",
    )
    parser.add_argument(
        "--hold-ghost-support",
        action="store_true",
        help="keep inserted-joint damping and stiffness at their start values through the endpoint",
    )
    parser.add_argument("--rail-start", type=float, default=None)
    parser.add_argument("--rail-end", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = build_config(
        load_config(args.source_config),
        load_config(args.target_config),
        ghost_length=args.ghost_length,
        ghost_mass=args.ghost_mass,
        unlocked_start=args.unlocked_start,
        ghost_damping=args.ghost_damping,
        ghost_stiffness=args.ghost_stiffness,
        hold_ghost_support=args.hold_ghost_support,
        rail_start=args.rail_start,
        rail_end=args.rail_end,
    )
    save_config(cfg, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
