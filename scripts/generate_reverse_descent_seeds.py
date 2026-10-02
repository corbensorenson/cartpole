#!/usr/bin/env python
"""Generate target-plant reverse-descent warm starts; never claim a swing solve."""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path
import time

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform

try:
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_fddp_capture import warm_start_diagnostics
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg
    from search_fddp_capture import warm_start_diagnostics


def reverse_descent(states, controls, hanging_state):
    """Reverse velocity and action timing, align branches/cart, retain mismatch."""
    states = np.asarray(states, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    hanging_state = np.asarray(hanging_state, dtype=np.float64)
    d = states.shape[1] // 2
    if states.shape != (len(controls) + 1, hanging_state.size):
        raise ValueError("descent states and controls have inconsistent dimensions")
    reverse = states[::-1].copy()
    reverse[:, d:] *= -1
    reverse[:, 0] -= reverse[0, 0]
    branch_shift = 2 * np.pi * np.round((reverse[0, 1:d] - hanging_state[1:d]) / (2 * np.pi))
    reverse[:, 1:d] -= branch_shift
    mismatch = reverse[0] - hanging_state
    return reverse, controls[::-1].copy(), mismatch


def hanging_quality(state):
    d = len(state) // 2
    angles = wrap_angle(np.cumsum(state[1:d]) - np.pi)
    absolute_rates = np.cumsum(state[d + 1:])
    return float(np.mean((angles / 0.15)**2)
                 + 0.2 * np.mean((absolute_rates / 0.75)**2)
                 + 0.2 * np.mean((state[d + 1:] / 0.75)**2)
                 + (state[d] / 0.5)**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-directory", required=True)
    parser.add_argument("--seconds", type=float, default=12.0)
    parser.add_argument("--minimum-end-time", type=float, default=6.0)
    parser.add_argument("--kick-amplitudes", type=float, nargs="+", default=[0.0001, 0.001, 0.01])
    parser.add_argument("--kick-seconds", type=float, nargs="+", default=[0.02, 0.1, 0.5])
    parser.add_argument("--velocity-gains", type=float, nargs="+", default=[2.0, 10.0, 40.0])
    parser.add_argument("--position-gains", type=float, nargs="+", default=[0.0, 0.5])
    parser.add_argument("--keep", type=int, default=4)
    args = parser.parse_args()
    if not 0 < args.minimum_end_time <= args.seconds or args.keep < 1:
        raise ValueError("positive coherent time interval and retained count required")
    directory = Path(args.out_directory)
    directory.mkdir(parents=True, exist_ok=False)
    cfg = load_config(args.config)
    n = cfg["env"]["n_links"]
    d = n + 1
    upright = dict(qpos=np.zeros(d).tolist(), qvel=np.zeros(d).tolist())
    hanging_physical = np.r_[0.0, np.pi, np.zeros(n - 1), np.zeros(d)]
    hanging = dict(qpos=hanging_physical[:d].tolist(), qvel=hanging_physical[d:].tolist())
    spec = load_config("benchmarks/p1_capture_envelope.yaml")["distribution"]
    transform = dimensionless_absolute_transform(n, StateScales(
        spec["cart_position_abs_max"], spec["absolute_link_angle_abs_max"],
        spec["cart_velocity_abs_max"], spec["hinge_velocity_rms_max"]))
    records, retained = [], []
    started = time.monotonic()
    total_policy_steps = 0
    for index, parameters in enumerate(itertools.product(
        args.kick_amplitudes, args.kick_seconds, args.velocity_gains, args.position_gains)):
        kick, duration, kv, kp = parameters
        env = NLinkCartPoleEnv(fixed_state_cfg(cfg, upright, args.seconds), progress=1.0, seed=0)
        env.reset(seed=0)
        states, controls = [data_state(env.data)], []
        best_score, best_step = np.inf, None
        info = {}
        for step in range(env.max_steps):
            time_seconds = step * env.dt
            action = kick if time_seconds < duration else (-kv * env.data.qvel[0] - kp * env.data.qpos[0]) / env.force_limit
            _, _, terminated, truncated, info = env.step([np.clip(action, -1, 1)])
            controls.append(float(info["applied_action_norm"]))
            states.append(data_state(env.data))
            total_policy_steps += 1
            if (step + 1) * env.dt >= args.minimum_end_time and not terminated:
                score = hanging_quality(states[-1])
                if score < best_score:
                    best_score, best_step = score, step + 1
            if terminated or truncated:
                break
        record = dict(index=index, kick=kick, kick_seconds=duration, velocity_gain=kv,
                      position_gain=kp, steps=len(controls), best_step=best_step,
                      best_hanging_quality=best_score if np.isfinite(best_score) else None,
                      max_cart_excursion=env.max_cart_excursion,
                      termination_reason=info.get("termination_reason"), simulation_error=env.simulation_error)
        env.close()
        records.append(record)
        if best_step is not None:
            prefix_states = np.asarray(states[:best_step + 1])
            reverse, reverse_controls, mismatch = reverse_descent(prefix_states, controls[:best_step], hanging_physical)
            reverse_max_cart = float(np.max(np.abs(reverse[:, 0])))
            if reverse_max_cart <= cfg["env"]["rail_limit"]:
                retained.append((best_score, index, reverse, reverse_controls, mismatch, prefix_states))
                retained.sort(key=lambda value: value[0])
                retained = retained[:args.keep]
        print(f"candidate={index} best_down={record['best_hanging_quality']} steps={len(controls)} cart={record['max_cart_excursion']:.3f}", flush=True)
        dump_json(dict(not_solution=True, records=records, policy_steps=total_policy_steps), directory / "progress.json")
    seed_records = []
    for rank, (score, index, reverse, controls, mismatch, descent_states) in enumerate(retained):
        env = NLinkCartPoleEnv(fixed_state_cfg(cfg, hanging, args.seconds), progress=1.0, seed=0)
        env.reset(seed=0)
        transition = MujocoTransition(env, transform, continuous_angles=True)
        nominal = reverse @ transform.T
        nominal[0] = hanging_physical @ transform.T
        diagnostics = warm_start_diagnostics(transition, nominal[0], controls, nominal)
        path = directory / f"seed_{rank}.json"
        payload = dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(),
                       selected_state=hanging, source_candidate=records[index],
                       reversal_initial_physical_mismatch=mismatch.tolist(),
                       actual_descent_states=descent_states.tolist(),
                       controller=dict(type="reversed_dissipative_descent_warm_start", controls=controls.tolist(),
                                       feedback_gains=np.zeros_like(nominal[:-1]).tolist(),
                                       continuous_angles=True, coordinate_transform=transform.tolist(),
                                       horizon_steps=len(controls), horizon_seconds=len(controls) * env.dt),
                       search=dict(nominal_coordinate_states=nominal.tolist(), source_hanging_quality=score,
                                   warm_start_diagnostics=diagnostics, is_feasible=diagnostics["is_feasible"]),
                       note="Reversal is not exact with damping/RK4 and its terminal mismatch. First nominal node is an optimizer reference, not a runtime state reset.",
                       config=file_metadata(args.config), runtime=runtime_metadata())
        dump_json(payload, path)
        seed_records.append(file_metadata(path))
        env.close()
    dump_json(dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(), parameters=vars(args),
                   records=records, seeds=seed_records, policy_steps=total_policy_steps,
                   physics_steps_upper_bound=total_policy_steps * cfg["env"]["frame_skip"],
                   wall_time_seconds=time.monotonic() - started, runtime=runtime_metadata()), directory / "summary.json")


if __name__ == "__main__":
    main()
