#!/usr/bin/env python
"""Prepare the sequential 11–20 queue and audit default local stabilization.

This creates discovery configurations and reserves release seeds. It never
promotes a count. Existing configurations are checked rather than overwritten.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from gcartpole.config import dump_json, load_config, save_config
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.lqr_design import checked_discrete_lqr
import numpy as np

try:
    from scripts.make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics
except ModuleNotFoundError:
    from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics


WEIGHTS = dict(cart_position=0.1, absolute_angle=100.0, cart_velocity=0.1,
               absolute_angular_velocity=1.0, relative_angle=1.0,
               relative_angular_velocity=0.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/swingup11_uniform.yaml")
    parser.add_argument("--target-links", type=int, default=20)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.target_links < 11:
        raise ValueError("target-links must be at least eleven")
    path = Path(args.out)
    if path.exists():
        raise FileExistsError(path)
    base = load_config(args.base_config)
    records = []
    for n in range(11, args.target_links + 1):
        config_path = Path(f"configs/swingup{n}_uniform.yaml")
        cfg = copy.deepcopy(base)
        cfg["env"]["n_links"] = n
        cfg["experiment"].update(name=f"swingup{n}_uniform", seed=n * 10 + 1,
                                 out_dir=f"runs/swingup{n}_uniform")
        if config_path.exists():
            if load_config(config_path) != cfg:
                raise ValueError(f"existing configuration differs: {config_path}")
        else:
            save_config(cfg, config_path)
        seed_base = 200000 + n * 1000
        record = dict(n_links=n, status="active" if n == 11 else "queued",
                      config=file_metadata(config_path),
                      held_out_seed_plan=dict(gate20=seed_base, gate100=seed_base + 100,
                                              video=seed_base + 500),
                      release_verified=False)
        try:
            a, b = finite_difference_dynamics(cfg, 1.0, 1e-7)
            gain, _, diagnostics = checked_discrete_lqr(
                a, b, absolute_angle_cost(n, WEIGHTS), np.array([[1000.0]])
            )
            record["default_local_lqr"] = dict(valid=True, gain_norm=float(np.linalg.norm(gain)),
                                                diagnostics=diagnostics)
        except (ValueError, np.linalg.LinAlgError) as error:
            record["default_local_lqr"] = dict(valid=False, error=f"{type(error).__name__}: {error}")
        records.append(record)
        print(f"n={n}: default local LQR valid={record['default_local_lqr']['valid']}", flush=True)
    dump_json(dict(schema_version=1, generated_at=utc_timestamp(), target_links=args.target_links,
                   active_frontier=11, latest_released_frontier=10, status="active",
                   not_solution=True, counts=records, base_config=file_metadata(Path(args.base_config)),
                   runtime=runtime_metadata(), git=git_metadata(Path.cwd()),
                   method="hanging contraction -> exact target trajectory feedback -> capture/maintenance",
                   promotion_contract="ROADMAP.md: sequential canonical 20/100 gates and complete release bundle",
                   note="Local linear diagnostics do not establish nonlinear capture or hanging-start reachability. Reserved seeds must not be used for tuning."), path)


if __name__ == "__main__":
    main()
