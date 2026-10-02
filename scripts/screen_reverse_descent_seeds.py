#!/usr/bin/env python
"""Replay amplitude/timing branches of reverse-descent proposals on the canonical target."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from gcartpole.ilqr import data_state

try:
    from scripts.search_capture_sequence import fixed_state_cfg
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg


def balanced_waveform(controls):
    """Remove the discrete impulse and first moment, before actuator clipping."""
    controls = np.asarray(controls, dtype=float)
    basis = np.vstack([np.ones(len(controls)), np.linspace(-1, 1, len(controls))])
    return controls - basis.T @ np.linalg.solve(basis @ basis.T, basis @ controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed-directory", required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=[1, 2, 4, 6, 8, 10])
    parser.add_argument("--retimes", type=float, nargs="+", default=[0.75, 1.0])
    parser.add_argument("--balance", action="store_true")
    parser.add_argument("--rail-feedback", action="store_true")
    parser.add_argument("--keep", type=int, default=4)
    parser.add_argument("--out-directory", required=True)
    args = parser.parse_args()
    if min(args.scales + args.retimes) <= 0 or args.keep < 1:
        raise ValueError("positive scales, retimings, and retained count required")
    directory = Path(args.out_directory)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    d = cfg["env"]["n_links"] + 1
    initial = dict(qpos=[0., float(np.pi)] + [0.] * (d - 2), qvel=[0.] * d)
    records, best = [], []
    started = time.monotonic()
    policy_steps = 0
    for source in sorted(Path(args.seed_directory).glob("seed_*.json")):
        payload = json.loads(source.read_text())
        original = np.asarray(payload["controller"]["controls"], dtype=float)
        transform = np.asarray(payload["controller"]["coordinate_transform"])
        if args.balance:
            original = balanced_waveform(original)
        for retime in args.retimes:
            controls = np.interp(np.linspace(0, 1, max(2, int(round(len(original) * retime)))),
                                 np.linspace(0, 1, len(original)), original)
            for scale in args.scales:
                env = NLinkCartPoleEnv(fixed_state_cfg(cfg, initial, 30.0), progress=1.0, seed=0)
                env.reset(seed=0)
                states, applied = [data_state(env.data)], []
                minimum_cost, best_step, best_state = np.inf, None, None
                info = {}
                for step, action in enumerate(controls):
                    correction = 0.0
                    if args.rail_feedback:
                        blend = np.clip((abs(env.data.qpos[0]) - 1.5) / 0.8, 0, 1)
                        correction = blend * (-20 * env.data.qpos[0] - 12 * env.data.qvel[0]) / env.force_limit
                    _, _, terminated, truncated, info = env.step([np.clip(scale * action + correction, -1, 1)])
                    state = data_state(env.data)
                    states.append(state)
                    applied.append(float(info["applied_action_norm"]))
                    policy_steps += 1
                    angle = serial_absolute_angles(state[1:d])
                    absolute_rates = np.cumsum(state[d + 1:])
                    cost = (np.max(np.abs(angle)) / 0.15)**2 + 0.3 * np.mean((absolute_rates / 0.75)**2)
                    cost += 0.3 * np.mean((state[d + 1:] / 0.75)**2) + (state[d] / 0.5)**2
                    if (step + 1) * env.dt >= 2 and not terminated and cost < minimum_cost:
                        minimum_cost, best_step, best_state = cost, step + 1, state.copy()
                    if terminated or truncated:
                        break
                record = dict(source=file_metadata(source), scale=scale, retime=retime,
                              minimum_cost=minimum_cost if np.isfinite(minimum_cost) else None,
                              best_step=best_step, steps=len(applied),
                              best_max_abs_angle=None if best_state is None else float(np.max(np.abs(serial_absolute_angles(best_state[1:d])))),
                              max_cart_excursion=env.max_cart_excursion,
                              max_hold_seconds=info.get("max_upright_streak_seconds", 0),
                              termination_reason=info.get("termination_reason"), simulation_error=env.simulation_error)
                records.append(record)
                if best_step is not None:
                    best.append((minimum_cost, len(records) - 1, np.asarray(states[:best_step + 1]), np.asarray(applied[:best_step]), transform))
                    best.sort(key=lambda row: row[0]); best = best[:args.keep]
                print(f"branch={len(records)} cost={record['minimum_cost']} angle={record['best_max_abs_angle']} hold={record['max_hold_seconds']} cart={record['max_cart_excursion']:.3f}", flush=True)
                env.close()
                dump_json(dict(not_solution=True, records=records, policy_steps=policy_steps), directory / "progress.json")
    artifacts = []
    for rank, (cost, index, states, controls, transform) in enumerate(best):
        path = directory / f"branch_{rank}.json"
        dump_json(dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(), selected_state=initial,
                       controller=dict(type="exact_reverse_descent_branch", controls=controls.tolist(),
                                       feedback_gains=np.zeros_like(states[:-1]).tolist(), continuous_angles=True,
                                       coordinate_transform=transform.tolist(), horizon_steps=len(controls),
                                       horizon_seconds=len(controls) * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]),
                       search=dict(nominal_coordinate_states=(states @ transform.T).tolist(), cost=cost,
                                   actual_serial_route=True, source_record=records[index]),
                       config=file_metadata(args.config), runtime=runtime_metadata()), path)
        artifacts.append(file_metadata(path))
    dump_json(dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(), parameters=vars(args),
                   records=records, artifacts=artifacts, policy_steps=policy_steps,
                   wall_time_seconds=time.monotonic() - started, runtime=runtime_metadata()), directory / "summary.json")


if __name__ == "__main__":
    main()
