#!/usr/bin/env python
"""Build a physically free split-link stiffness curriculum config."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from build_split_position_continuation import build_config
from gcartpole.config import load_config, save_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--continuation-config", required=True)
    parser.add_argument("--split-link", type=int, required=True)
    parser.add_argument("--stiffness", type=float, required=True)
    parser.add_argument("--damping", type=float, required=True)
    parser.add_argument(
        "--end-stiffness",
        type=float,
        default=0.0,
        help="inserted-joint stiffness at morphology progress 1",
    )
    parser.add_argument(
        "--end-damping",
        type=float,
        default=0.0,
        help="inserted-joint damping at morphology progress 1",
    )
    parser.add_argument("--rail-limit", type=float, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if min(args.stiffness, args.damping, args.end_stiffness, args.end_damping) < 0.0:
        raise ValueError("stiffness and damping values must be nonnegative")

    cfg = build_config(
        load_config(args.source_config),
        load_config(args.continuation_config),
        args.split_link,
        hold_split_lock=False,
    )
    n_links = int(cfg["env"]["n_links"])
    inserted = int(args.split_link)
    stiffness = np.zeros(n_links, dtype=np.float64)
    damping = np.asarray(cfg["morphology"]["damping_start"], dtype=np.float64)
    stiffness[inserted] = float(args.stiffness)
    damping[inserted] = float(args.damping)
    end_stiffness = np.zeros(n_links, dtype=np.float64)
    end_damping = np.asarray(cfg["morphology"]["damping_end"], dtype=np.float64)
    end_stiffness[inserted] = float(args.end_stiffness)
    end_damping[inserted] = float(args.end_damping)
    cfg["experiment"]["name"] = (
        f"free_split_stiffness_n{n_links}_link{args.split_link}_"
        f"k{args.stiffness:g}_d{args.damping:g}"
    )
    cfg["env"]["rigid_split_inertia"] = False
    cfg["morphology"]["joint_lock_start"] = np.zeros(n_links).tolist()
    cfg["morphology"]["joint_lock_end"] = np.zeros(n_links).tolist()
    cfg["morphology"]["joint_stiffness_start"] = stiffness.tolist()
    cfg["morphology"]["joint_stiffness_end"] = end_stiffness.tolist()
    cfg["morphology"]["damping_start"] = damping.tolist()
    cfg["morphology"]["damping_end"] = end_damping.tolist()
    cfg["morphology"]["start"]["total_damping"] = float(np.sum(damping))
    cfg["morphology"]["end"]["total_damping"] = float(np.sum(end_damping))
    if args.rail_limit is not None:
        cfg["env"]["rail_limit"] = float(args.rail_limit)
        cfg["env"]["rail_limit_start"] = float(args.rail_limit)
        cfg["env"]["rail_limit_end"] = float(args.rail_limit)
    cfg["free_split_stiffness_probe"] = {
        "source_config": str(Path(args.source_config)),
        "continuation_config": str(Path(args.continuation_config)),
        "split_link_one_based": int(args.split_link),
        "inserted_joint_index_zero_based": inserted,
        "stiffness_start": float(args.stiffness),
        "damping_start": float(args.damping),
        "stiffness_end": float(args.end_stiffness),
        "damping_end": float(args.end_damping),
        "joint_lock_start": False,
        "joint_lock_end": False,
        "rigid_split_inertia": False,
    }
    save_config(cfg, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
