#!/usr/bin/env python
"""Build a locked n-to-n+1 continuation with a forced split position.

The default split embedding assigns extra target segments by a morphology
distance objective and therefore normally puts the new joint at the distal end
for equal links.  This helper keeps the physical continuation explicit while
allowing a diagnostic comparison of where the new joint is introduced.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import load_config, save_config
from gcartpole.generalized_solver import setup_from_config
from gcartpole.morphology import build_morphology


def split_profile(values: np.ndarray, split_link: int) -> np.ndarray:
    """Split one one-based source-link value into two equal pieces."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not 1 <= split_link <= values.size:
        raise ValueError("split_link must be a valid one-based source index")
    value = float(values[split_link - 1])
    index = split_link - 1
    return np.concatenate(
        (values[:index], np.asarray([value * 0.5, value * 0.5]), values[index + 1 :])
    )


def insert_internal_profile(values: np.ndarray, split_link: int) -> np.ndarray:
    """Split a source profile and set the newly internal joint to zero."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not 1 <= split_link <= values.size:
        raise ValueError("split_link must be a valid one-based source index")
    return np.insert(values, split_link, 0.0)


def build_config(
    source_cfg: dict[str, Any],
    continuation_cfg: dict[str, Any],
    split_link: int,
    *,
    hold_split_lock: bool = False,
) -> dict[str, Any]:
    source = setup_from_config(source_cfg)
    target = setup_from_config(continuation_cfg)
    target_morph = build_morphology(
        continuation_cfg["env"], continuation_cfg["morphology"], progress=1.0
    )
    if target.n_links != source.n_links + 1:
        raise ValueError("continuation config must have exactly one more link")
    if not 1 <= split_link <= source.n_links:
        raise ValueError("split_link must be a one-based source index")

    source_damping = np.asarray(source.joint_damping, dtype=np.float64)
    source_morph = source_cfg["morphology"]
    source_friction = np.asarray(
        source_morph.get("frictionloss_start", source_morph.get("frictionloss", [0.0] * source.n_links)),
        dtype=np.float64,
    )
    source_stiffness = np.asarray(
        source_morph.get("joint_stiffness_start", source_morph.get("joint_stiffness", [0.0] * source.n_links)),
        dtype=np.float64,
    )
    if source_friction.shape != (source.n_links,) or source_stiffness.shape != (source.n_links,):
        raise ValueError("source friction and stiffness profiles must match source links")

    cfg = copy.deepcopy(continuation_cfg)
    cfg["experiment"]["name"] = (
        f"forced_split_n{source.n_links}_to_n{target.n_links}_link{split_link}"
    )
    cfg["env"]["rigid_split_inertia"] = True
    cfg["env"]["joint_lock_impedance_schedule"] = "log_compliance"
    morphology = cfg["morphology"]
    morphology["schedule_mode"] = "all_linear"
    morphology["lengths_start"] = split_profile(source.lengths, split_link).tolist()
    morphology["lengths_end"] = target.lengths.astype(float).tolist()
    morphology["masses_start"] = split_profile(source.masses, split_link).tolist()
    morphology["masses_end"] = target.masses.astype(float).tolist()
    morphology["damping_start"] = insert_internal_profile(source_damping, split_link).tolist()
    morphology["damping_end"] = target.joint_damping.astype(float).tolist()
    morphology["frictionloss_start"] = insert_internal_profile(source_friction, split_link).tolist()
    morphology["frictionloss_end"] = target_morph.frictionloss.astype(float).tolist()
    morphology["joint_stiffness_start"] = insert_internal_profile(source_stiffness, split_link).tolist()
    morphology["joint_stiffness_end"] = target_morph.joint_stiffness.astype(float).tolist()
    lock_start = np.zeros(target.n_links, dtype=np.float64)
    lock_start[split_link] = 1.0
    morphology["joint_lock_start"] = lock_start.tolist()
    morphology["joint_lock_end"] = (
        lock_start if hold_split_lock else np.zeros(target.n_links, dtype=np.float64)
    ).tolist()
    for endpoint, damping, friction in (
        ("start", np.asarray(morphology["damping_start"], dtype=np.float64), np.asarray(morphology["frictionloss_start"], dtype=np.float64)),
        ("end", target.joint_damping, target_morph.frictionloss),
    ):
        morphology.setdefault(endpoint, {})
        morphology[endpoint]["alpha_length"] = 0.0
        morphology[endpoint]["alpha_mass"] = 0.0
        morphology[endpoint]["alpha_damping"] = 0.0
        morphology[endpoint]["alpha_frictionloss"] = 0.0
        morphology[endpoint]["total_damping"] = float(np.sum(damping))
        morphology[endpoint]["total_frictionloss"] = float(np.sum(friction))
    cfg["split_probe"] = {
        "source_links": int(source.n_links),
        "target_links": int(target.n_links),
        "split_link_one_based": int(split_link),
        "locked_joint_index_zero_based": int(split_link),
        "preserves_total_length": bool(np.isclose(np.sum(morphology["lengths_start"]), target.chain_length)),
        "preserves_total_mass": bool(np.isclose(np.sum(morphology["masses_start"]), target.link_mass)),
        "hold_split_lock": bool(hold_split_lock),
    }
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--continuation-config", required=True)
    parser.add_argument("--split-link", type=int, required=True)
    parser.add_argument(
        "--hold-split-lock",
        action="store_true",
        help="keep the inserted split joint locked while morphology moves to the target",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source_cfg = load_config(args.source_config)
    continuation_cfg = load_config(args.continuation_config)
    cfg = build_config(
        source_cfg,
        continuation_cfg,
        args.split_link,
        hold_split_lock=args.hold_split_lock,
    )
    save_config(cfg, Path(args.out))
    print(f"wrote {args.out}: split_link={args.split_link}")


if __name__ == "__main__":
    main()
