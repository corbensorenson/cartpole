#!/usr/bin/env python
"""Compare sampled control rates on a fixed physics grid; component evidence only."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from gcartpole.generalized_modes import chain_normal_modes
from gcartpole.lqr_design import checked_discrete_lqr, high_precision_discrete_lqr
from audit_link_count_frontier import WEIGHTS, exact_capture
from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics


def cadence_config(base, n, frame_skip):
    if n < 1 or frame_skip < 1:
        raise ValueError("positive link count and physics steps per control required")
    cfg = copy.deepcopy(base)
    cfg["env"]["n_links"] = n
    cfg["env"]["frame_skip"] = frame_skip
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup11_uniform.yaml")
    parser.add_argument("--counts", type=int, nargs="+", default=[10, 11, 12, 14, 20])
    parser.add_argument("--frame-skips", type=int, nargs="+", default=[4, 2, 1])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[1e-5, 1e-9, 1e-13])
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261010)
    parser.add_argument("--decimal-digits", type=int, default=100)
    parser.add_argument("--design-json", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    if args.directions < 1 or min(args.amplitudes) <= 0:
        raise ValueError("positive direction count and amplitude required")
    base = load_config(args.config)
    prior = {}
    for source in args.design_json:
        data = json.loads(Path(source).read_text())
        for record in data["records"]:
            if "gain" in record:
                prior[(record["n_links"], record.get("frame_skip", base["env"]["frame_skip"]))] = (record, source)
    records = []
    payload = dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(),
                   config=file_metadata(args.config), parameters=vars(args), runtime=runtime_metadata(), records=records,
                   note="Fixed physics timestep and integrator; redesigned gains at each control rate; identical initial directions by count. Upright components only. Changed rates are diagnostic benchmark variants, not canonical releases.")
    for n in args.counts:
        directions = np.random.default_rng(args.seed + n).uniform(-1, 1, (args.directions, 2 * n))
        for frame_skip in args.frame_skips:
            cfg = cadence_config(base, n, frame_skip)
            env = NLinkCartPoleEnv(cfg, progress=1., seed=0)
            dt = env.dt
            modes = chain_normal_modes(env)
            record = dict(n_links=n, frame_skip=frame_skip, physics_timestep=env.model.opt.timestep,
                          control_dt=dt, control_frequency_hz=1./dt,
                          canonical_cadence=frame_skip == base["env"]["frame_skip"],
                          hanging_frequencies_hz=(modes.angular_frequencies / (2 * np.pi)).tolist(),
                          fastest_hanging_frequency_hz=float(np.max(modes.angular_frequencies)/(2*np.pi)),
                          joint_mass_condition=float(np.linalg.cond(modes.joint_mass_matrix)),
                          directions=directions.tolist())
            env.close()
            a, b = finite_difference_dynamics(cfg, 1., 1e-7)
            q, r = absolute_angle_cost(n, WEIGHTS), np.array([[1000.]])
            record["input_matrices"] = dict(a=a.tolist(), b=b.tolist(), q=q.tolist(), r=r.tolist())
            values = np.linalg.eigvals(a)
            record["open_loop_spectral_radius"] = float(np.max(np.abs(values)))
            record["fastest_unstable_growth_rate_per_second"] = float(np.log(np.max(np.abs(values)))/dt)
            record["fastest_mode_growth_per_control_interval"] = float(np.max(np.abs(values)))
            try:
                _, _, record["default_diagnostics"] = checked_discrete_lqr(a, b, q, r)
            except (ValueError, np.linalg.LinAlgError) as error:
                record["default_error"] = str(error)
            started = time.monotonic()
            try:
                previous = prior.get((n, frame_skip))
                if previous and all(np.array_equal(matrix, np.asarray(previous[0]["input_matrices"][key]))
                                    for key, matrix in zip(("a", "b", "q", "r"), (a, b, q, r))):
                    item, source = previous
                    gain = np.asarray(item["gain"])[None, :]
                    p = np.asarray(item["riccati_matrix"])
                    record["high_precision_diagnostics"] = item["high_precision_diagnostics"]
                    record["gain_source"] = file_metadata(source)
                else:
                    gain, p, record["high_precision_diagnostics"] = high_precision_discrete_lqr(
                        a, b, q, r, decimal_digits=args.decimal_digits)
                record.update(gain=gain.ravel().tolist(), riccati_matrix=p.tolist(),
                              design_wall_time_seconds=time.monotonic()-started, local_capture={})
                for amplitude in args.amplitudes:
                    results = [exact_capture(cfg, dict(qpos=np.r_[0., amplitude*v[:n]].tolist(),
                                                       qvel=np.r_[0., amplitude*v[n:]].tolist()), gain.ravel())
                               for v in directions]
                    record["local_capture"][str(amplitude)] = dict(successes=sum(x["success"] for x in results), episodes=results)
            except (ValueError, np.linalg.LinAlgError, ZeroDivisionError) as error:
                record["design_error"] = f"{type(error).__name__}: {error}"
                record["design_wall_time_seconds"] = time.monotonic()-started
            records.append(record)
            dump_json(payload, str(out)+".progress.json")
            print(json.dumps(dict(n_links=n, control_frequency_hz=1./dt,
                                  gain_norm=record.get("high_precision_diagnostics", {}).get("gain_norm"),
                                  successes={k:v["successes"] for k,v in record.get("local_capture", {}).items()},
                                  error=record.get("design_error"))), flush=True)
    dump_json(payload, out)


if __name__ == "__main__":
    main()
