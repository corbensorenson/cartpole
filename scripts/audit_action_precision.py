#!/usr/bin/env python
"""Isolate runtime action/dot precision on saved local control-rate designs."""
import argparse
import copy
import json
from pathlib import Path

import mpmath
import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from gcartpole.simulation import SimulationError, advance_checked
from audit_control_frequency import cadence_config
from search_capture_sequence import fixed_state_cfg


def precision_step(env, action, *, float32):
    """Diagnostic counterpart of env.step, changing only the initial cast."""
    action_array = np.asarray(action, dtype=np.float32 if float32 else np.float64).reshape(-1)
    policy_action = float(np.clip(action_array[0], -1., 1.))
    applied = env._applied_action_norm(policy_action)
    env.last_policy_action_norm[0] = policy_action
    env.last_action_norm[0] = applied
    env.data.ctrl[0] = applied * env.force_limit
    try:
        advance_checked(env.model, env.data, env.frame_skip)
    except SimulationError as error:
        env.simulation_error = str(error)
    env.step_count += 1
    env.max_cart_excursion = max(env.max_cart_excursion, abs(float(env.data.qpos[0])))
    if env.simulation_error is None:
        env._update_upright_tracking()
    else:
        env.upright_streak_steps = env.centered_upright_streak_steps = env.low_momentum_upright_streak_steps = 0
    reason = env._termination_reason()
    truncated = env.step_count >= env.max_steps
    return reason is not None, truncated


def precision_capture(cfg, state, gain, mode, seconds=8.):
    env = NLinkCartPoleEnv(fixed_state_cfg(cfg, state, seconds), progress=1., seed=0)
    env.reset(seed=0)
    ctx = mpmath.mp.clone()
    ctx.dps = 80
    mp_gain = [ctx.mpf(float(x)) for x in gain] if mode == "action64_dot80" else None
    raw_max = rounding_max = 0.
    saturated = 0
    for _ in range(env.max_steps):
        physical = np.r_[env.data.qpos.copy(), env.data.qvel.copy()]
        physical[1:env.n+1] = wrap_angle(physical[1:env.n+1])
        raw = (-float(ctx.fsum(k * ctx.mpf(float(x)) for k,x in zip(mp_gain, physical)))
               if mp_gain is not None else float(-gain @ physical))
        raw_max = max(raw_max, abs(raw))
        saturated += abs(raw) > 1.
        action = float(np.clip(raw, -1., 1.))
        rounding_max = max(rounding_max, abs(action-float(np.float32(action))))
        terminated, truncated = precision_step(env, [action], float32=mode == "action32_dot64")
        if terminated or truncated:
            break
    clean = env.simulation_error is None
    result = dict(success=bool(truncated and not terminated and env._success()),
                  max_hold_seconds=env.max_upright_streak_steps*env.dt,
                  final_hold_seconds=env.upright_streak_steps*env.dt,
                  steps=env.step_count, max_cart_excursion=env.max_cart_excursion,
                  saturated_steps=int(saturated), raw_action_abs_max=raw_max,
                  max_float32_rounding_difference=rounding_max, trajectory_clean=clean,
                  termination_reason=env._termination_reason(), simulation_error=env.simulation_error)
    env.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup11_uniform.yaml")
    parser.add_argument("--design-json", required=True)
    parser.add_argument("--counts", type=int, nargs="+", default=[11, 14, 20])
    parser.add_argument("--frame-skips", type=int, nargs="+", default=[4, 1])
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[1e-9, 1e-13, 1e-15, 1e-17, 1e-19])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    if not np.all(np.isfinite(args.amplitudes)) or min(args.amplitudes) <= 0:
        raise ValueError("finite positive initial amplitudes required")
    base = load_config(args.config)
    source = json.loads(Path(args.design_json).read_text())
    records = []
    payload = dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(),
                   source=file_metadata(args.design_json), config=file_metadata(args.config),
                   runtime=runtime_metadata(), parameters=vars(args), records=records,
                   note="Matched initial directions and saved binary64 gains. Physics remains binary64 RK4 at 0.005 s. Diagnostic precision variants cannot count as canonical release evidence; 80-digit dot products cannot restore model or gain input precision.")
    for design in source["records"]:
        n, frame_skip = design["n_links"], design["frame_skip"]
        if n not in args.counts or frame_skip not in args.frame_skips:
            continue
        cfg = cadence_config(base, n, frame_skip)
        gain = np.asarray(design["gain"])
        directions = np.asarray(design["directions"])
        for mode in ("action32_dot64", "action64_dot64", "action64_dot80"):
            record = dict(n_links=n, frame_skip=frame_skip, control_frequency_hz=design["control_frequency_hz"],
                          mode=mode, local_capture={})
            for amplitude in args.amplitudes:
                results = [precision_capture(cfg, dict(qpos=np.r_[0., amplitude*v[:n]].tolist(),
                                                       qvel=np.r_[0., amplitude*v[n:]].tolist()), gain, mode)
                           for v in directions]
                record["local_capture"][str(amplitude)] = dict(successes=sum(x["success"] for x in results), episodes=results)
            records.append(record)
            dump_json(payload, args.out+".progress.json")
            print(json.dumps(dict(n_links=n, control_frequency_hz=record["control_frequency_hz"], mode=mode,
                                  successes={k:v["successes"] for k,v in record["local_capture"].items()})), flush=True)
    dump_json(payload, args.out)


if __name__ == "__main__":
    main()
