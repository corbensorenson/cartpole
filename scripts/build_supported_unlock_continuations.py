#!/usr/bin/env python
"""Build topology-safe release and support-relaxation continuations.

An equality-constrained split plant is not continuous at the exact point where
the equality disappears.  This builder makes that boundary explicit in three
count-agnostic stages:

1. ramp a dimensionless physical spring/damper while the equality stays fixed;
2. hold that support while the equality constraint is removed;
3. keep the plant fully unlocked while the support and remaining morphology
   differences are annealed to the measured target.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import load_config, save_config
from gcartpole.generalized_solver import setup_from_config
from gcartpole.morphology import Morphology, build_morphology


def _set_profiles(
    cfg: dict[str, Any],
    start: Morphology,
    end: Morphology,
) -> None:
    morph = cfg["morphology"]
    morph["schedule_mode"] = "all_linear"
    for name in (
        "lengths",
        "masses",
        "damping",
        "frictionloss",
        "joint_stiffness",
        "joint_lock",
    ):
        morph[f"{name}_start"] = np.asarray(getattr(start, name), dtype=float).tolist()
        morph[f"{name}_end"] = np.asarray(getattr(end, name), dtype=float).tolist()
    for endpoint, profile in (("start", start), ("end", end)):
        values = morph.setdefault(endpoint, {})
        for alpha in (
            "alpha_length",
            "alpha_mass",
            "alpha_damping",
            "alpha_frictionloss",
        ):
            values[alpha] = 0.0
        values["total_damping"] = float(np.sum(profile.damping))
        values["total_frictionloss"] = float(np.sum(profile.frictionloss))


def _replace_profiles(
    base: Morphology,
    *,
    damping: np.ndarray,
    stiffness: np.ndarray,
    locks: np.ndarray,
) -> Morphology:
    return replace(
        base,
        damping=damping,
        joint_stiffness=stiffness,
        joint_lock=locks,
        total_damping=float(np.sum(damping)),
    )


def build_supported_unlock_continuations(
    locked_cfg: dict[str, Any],
    target_cfg: dict[str, Any],
    *,
    stiffness_ratio: float,
    damping_ratio: float,
    initial_stiffness_ratio: float = 0.0,
    initial_damping_ratio: float = 0.0,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return support-ramp, equality-release, and support-relaxation configs."""

    if stiffness_ratio <= 0.0 or damping_ratio <= 0.0:
        raise ValueError("support ratios must be positive")
    if initial_stiffness_ratio < 0.0 or initial_damping_ratio < 0.0:
        raise ValueError("initial support ratios must be nonnegative")
    if initial_stiffness_ratio > stiffness_ratio:
        raise ValueError("initial stiffness ratio must not exceed the final ratio")
    if initial_damping_ratio > damping_ratio:
        raise ValueError("initial damping ratio must not exceed the final ratio")
    if np.isclose(initial_stiffness_ratio, stiffness_ratio) and np.isclose(
        initial_damping_ratio, damping_ratio
    ):
        raise ValueError("at least one support ratio must increase")
    locked = build_morphology(
        locked_cfg["env"], locked_cfg["morphology"], progress=1.0
    )
    target = build_morphology(
        target_cfg["env"], target_cfg["morphology"], progress=1.0
    )
    if locked.n_links != target.n_links:
        raise ValueError("locked and target configurations must have equal link count")
    releasing = (locked.joint_lock > 0.0) & np.isclose(target.joint_lock, 0.0)
    if not np.any(releasing):
        raise ValueError("no positive joint lock releases to the target")

    setup = setup_from_config(target_cfg)
    stiffness_scale = setup.system_mass * setup.gravity * setup.chain_length
    damping_scale = (
        setup.system_mass * setup.chain_length**2 / setup.natural_time
    )
    initial_stiffness = locked.joint_stiffness.copy()
    initial_damping = locked.damping.copy()
    initial_stiffness[releasing] += initial_stiffness_ratio * stiffness_scale
    initial_damping[releasing] += initial_damping_ratio * damping_scale
    support_stiffness = locked.joint_stiffness.copy()
    support_damping = locked.damping.copy()
    support_stiffness[releasing] += stiffness_ratio * stiffness_scale
    support_damping[releasing] += damping_ratio * damping_scale

    initially_supported_locked = _replace_profiles(
        locked,
        damping=initial_damping,
        stiffness=initial_stiffness,
        locks=locked.joint_lock,
    )
    supported_locked = _replace_profiles(
        locked,
        damping=support_damping,
        stiffness=support_stiffness,
        locks=locked.joint_lock,
    )
    supported_unlocked = _replace_profiles(
        locked,
        damping=support_damping,
        stiffness=support_stiffness,
        locks=np.zeros(locked.n_links, dtype=np.float64),
    )

    ramp = copy.deepcopy(target_cfg)
    ramp["experiment"]["name"] = "supported_unlock_ramp"
    _set_profiles(ramp, initially_supported_locked, supported_locked)
    release = copy.deepcopy(target_cfg)
    release["experiment"]["name"] = "supported_unlock_release"
    _set_profiles(release, supported_locked, supported_unlocked)
    relaxation = copy.deepcopy(target_cfg)
    relaxation["experiment"]["name"] = "supported_unlock_relaxation"
    _set_profiles(relaxation, supported_unlocked, target)
    metadata = {
        "releasing_joint_indices": np.flatnonzero(releasing).astype(int).tolist(),
        "initial_stiffness_ratio": float(initial_stiffness_ratio),
        "initial_damping_ratio": float(initial_damping_ratio),
        "stiffness_ratio": float(stiffness_ratio),
        "damping_ratio": float(damping_ratio),
        "stiffness_scale": float(stiffness_scale),
        "damping_scale": float(damping_scale),
    }
    ramp["supported_unlock"] = {**metadata, "stage": "ramp_support"}
    release["supported_unlock"] = {**metadata, "stage": "release_equality"}
    relaxation["supported_unlock"] = {**metadata, "stage": "relax_support"}
    return ramp, release, relaxation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locked-config", required=True)
    parser.add_argument("--target-config", required=True)
    parser.add_argument("--stiffness-ratio", type=float, default=1.0)
    parser.add_argument("--damping-ratio", type=float, default=0.01)
    parser.add_argument("--initial-stiffness-ratio", type=float, default=0.0)
    parser.add_argument("--initial-damping-ratio", type=float, default=0.0)
    parser.add_argument("--ramp-out", required=True)
    parser.add_argument("--release-out", required=True)
    parser.add_argument("--relaxation-out", required=True)
    args = parser.parse_args()
    ramp, release, relaxation = build_supported_unlock_continuations(
        load_config(args.locked_config),
        load_config(args.target_config),
        stiffness_ratio=args.stiffness_ratio,
        damping_ratio=args.damping_ratio,
        initial_stiffness_ratio=args.initial_stiffness_ratio,
        initial_damping_ratio=args.initial_damping_ratio,
    )
    save_config(ramp, Path(args.ramp_out))
    save_config(release, Path(args.release_out))
    save_config(relaxation, Path(args.relaxation_out))
    print(f"wrote {args.ramp_out}")
    print(f"wrote {args.release_out}")
    print(f"wrote {args.relaxation_out}")


if __name__ == "__main__":
    main()
