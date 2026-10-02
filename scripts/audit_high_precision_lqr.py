#!/usr/bin/env python
"""Separate Riccati arithmetic failures from exact local capture performance."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from gcartpole.lqr_design import checked_discrete_lqr, high_precision_discrete_lqr

try:
    from scripts.audit_link_count_frontier import WEIGHTS, exact_capture
    from scripts.make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics
except ModuleNotFoundError:
    from audit_link_count_frontier import WEIGHTS, exact_capture
    from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup11_uniform.yaml")
    parser.add_argument("--counts", type=int, nargs="+", default=[11, 12, 13])
    parser.add_argument("--decimal-digits", type=int, default=80)
    parser.add_argument("--fd-eps", type=float, default=1e-7)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[1e-9, 1e-11])
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--design-json", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    if args.directions < 1 or min(args.counts) < 1 or min(args.amplitudes) <= 0:
        raise ValueError("positive counts, directions, and amplitudes required")
    if not np.isfinite(args.fd_eps) or args.fd_eps <= 0:
        raise ValueError("finite positive dynamics identification step required")
    base = load_config(args.config)
    prior_designs = {}
    for path in args.design_json:
        for record in json.loads(Path(path).read_text())["records"]:
            if "gain" in record:
                prior_designs[record["n_links"]] = (record, path)
    records = []
    payload = dict(not_solution=True, generated_at=utc_timestamp(), runtime=runtime_metadata(),
                   parameters=vars(args), config=file_metadata(args.config), records=records,
                   note="Fixed upright-state components only. No hanging-start or nonlinear robustness certificate.")
    for n in args.counts:
        cfg = copy.deepcopy(base)
        cfg["env"]["n_links"] = n
        a, b = finite_difference_dynamics(cfg, 1.0, args.fd_eps)
        q = absolute_angle_cost(n, WEIGHTS)
        r = np.array([[1000.0]])
        record = dict(n_links=n, dynamics_fd_epsilon=args.fd_eps,
                      input_matrices=dict(a=a.tolist(), b=b.tolist(), q=q.tolist(), r=r.tolist()))
        try:
            _, _, record["default_diagnostics"] = checked_discrete_lqr(a, b, q, r)
        except (ValueError, np.linalg.LinAlgError) as error:
            record["default_error"] = str(error)
        started = time.monotonic()
        try:
            if n in prior_designs:
                previous, source = prior_designs[n]
                for key, matrix in zip(("a", "b", "q", "r"), (a, b, q, r)):
                    if not np.array_equal(matrix, np.asarray(previous["input_matrices"][key])):
                        raise ValueError("reused design matrices differ from the current plant/design")
                gain = np.asarray(previous["gain"])[None, :]
                p = np.asarray(previous["riccati_matrix"])
                record["high_precision_diagnostics"] = previous["high_precision_diagnostics"]
                record["gain_source"] = file_metadata(source)
            else:
                gain, p, record["high_precision_diagnostics"] = high_precision_discrete_lqr(
                    a, b, q, r, decimal_digits=args.decimal_digits)
            record["gain"] = gain.ravel().tolist()
            record["riccati_matrix"] = p.tolist()
            record["design_wall_time_seconds"] = time.monotonic() - started
            directions = np.random.default_rng(args.seed + n).uniform(-1, 1, (args.directions, 2 * n))
            record["directions"] = directions.tolist()
            record["local_capture"] = {}
            for amplitude in args.amplitudes:
                results = [exact_capture(cfg, dict(
                    qpos=np.r_[0.0, amplitude * v[:n]].tolist(),
                    qvel=np.r_[0.0, amplitude * v[n:]].tolist()), gain.ravel()) for v in directions]
                record["local_capture"][str(amplitude)] = dict(
                    successes=sum(result["success"] for result in results), episodes=results)
        except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as error:
            record["high_precision_error"] = f"{type(error).__name__}: {error}"
            record["design_wall_time_seconds"] = time.monotonic() - started
        records.append(record)
        dump_json(payload, str(out) + ".progress.json")
        print(f"n={n}: {record.get('high_precision_diagnostics', record.get('high_precision_error'))}", flush=True)
    dump_json(payload, out)


if __name__ == "__main__":
    main()
