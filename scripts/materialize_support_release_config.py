#!/usr/bin/env python
"""Freeze an accepted supported plant and expose its support as one axis.

This is a development continuation helper.  It preserves the accepted
intermediate link lengths and masses exactly, then interpolates only the
distal damping and straightening stiffness toward the target values.  The
resulting config is suitable for exact replay or Box-FDDP; it is never final
uniform-plant evidence by itself.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from gcartpole.config import load_config, save_config


def _as_vector(morphology: dict, name: str, n: int) -> np.ndarray:
    values = morphology.get(name)
    if not isinstance(values, list) or len(values) != n:
        raise ValueError(f"{name} must contain {n} values")
    vector = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite")
    return vector


def materialize(path: str | Path, progress: float, out: str | Path) -> None:
    if not 0.0 <= float(progress) <= 1.0:
        raise ValueError("progress must lie in [0, 1]")
    base = load_config(path)
    cfg = copy.deepcopy(base)
    cfg["experiment"]["name"] = f"support_release_p{float(progress):.9f}"
    cfg["experiment"]["out_dir"] = str(Path(out).resolve().parent)
    morph = cfg["morphology"]
    n = int(cfg["env"]["n_links"])

    lengths = _as_vector(morph, "lengths_start", n)
    masses = _as_vector(morph, "masses_start", n)
    damping_start = _as_vector(morph, "damping_start", n)
    stiffness_start = _as_vector(morph, "joint_stiffness_start", n)
    friction = _as_vector(morph, "frictionloss_start", n)
    locks = _as_vector(morph, "joint_lock_start", n)
    if not np.isclose(float(np.sum(lengths)), float(cfg["env"]["total_length"])):
        raise ValueError("frozen lengths do not preserve total_length")
    if not np.isclose(float(np.sum(masses)), float(cfg["env"]["total_mass"])):
        raise ValueError("frozen masses do not preserve total_mass")

    # The canonical plant has uniform 0.015 total joint damping and no
    # straightening stiffness.  Keep these values explicit so every stage is
    # independently reproducible and the morphology validator can check sums.
    damping_end = np.full(n, 0.015 / n, dtype=np.float64)
    stiffness_end = np.zeros(n, dtype=np.float64)
    morph["lengths_start"] = lengths.tolist()
    morph["lengths_end"] = lengths.tolist()
    morph["masses_start"] = masses.tolist()
    morph["masses_end"] = masses.tolist()
    morph["damping_start"] = damping_start.tolist()
    morph["damping_end"] = damping_end.tolist()
    morph["frictionloss_start"] = friction.tolist()
    morph["frictionloss_end"] = friction.tolist()
    morph["joint_stiffness_start"] = stiffness_start.tolist()
    morph["joint_stiffness_end"] = stiffness_end.tolist()
    morph["joint_lock_start"] = locks.tolist()
    morph["joint_lock_end"] = locks.tolist()
    morph.setdefault("start", {})["total_damping"] = float(np.sum(damping_start))
    morph.setdefault("end", {})["total_damping"] = float(np.sum(damping_end))
    morph.setdefault("start", {})["total_frictionloss"] = float(np.sum(friction))
    morph.setdefault("end", {})["total_frictionloss"] = float(np.sum(friction))
    morph["schedule_mode"] = "all_linear"
    cfg["env"]["rail_limit"] = float(cfg["env"]["rail_limit"])
    cfg["env"].pop("rail_limit_start", None)
    cfg["env"].pop("rail_limit_end", None)
    cfg["release_progress"] = float(progress)
    save_config(cfg, out)
    print(f"wrote {out} at support progress {float(progress):.9f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--progress", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    materialize(args.base_config, args.progress, args.out)


if __name__ == "__main__":
    main()
