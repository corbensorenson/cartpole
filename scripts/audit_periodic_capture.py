#!/usr/bin/env python
"""Test near-upright periodic tracking as a component, never a chain release."""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np

from gcartpole.config import load_config, dump_json
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, runtime_metadata
from gcartpole.ilqr import MujocoTransition, QuadraticTrajectoryCost
from gcartpole.modal import StateScales, dimensionless_absolute_transform
from gcartpole.periodic_capture import linear_antiperiodic_nodes, refine_antiperiodic_orbit
from gcartpole.sqrt_ilqr import square_root_tracking_gains
from search_capture_sequence import fixed_state_cfg


def execute_component(cfg, nodes, controls, gains, perturbation, seed):
    nx, n = nodes.shape[1], int(cfg['env']['n_links'])
    start = nodes[0]+perturbation
    state = dict(qpos=start[:nx//2].tolist(), qvel=start[nx//2:].tolist())
    env = NLinkCartPoleEnv(fixed_state_cfg(cfg, state, 30.), progress=1., seed=seed)
    env.reset(seed=seed)
    trajectory, clean, raw_max, saturated, info = [], True, 0., 0, {}
    streak_steps = max_streak_steps = 0
    for step in range(env.max_steps):
        phase = step % len(controls)
        current = np.r_[env.data.qpos.copy(), env.data.qvel.copy()]
        raw = controls[phase]+float(gains[phase]@(current-nodes[phase]))
        raw_max, saturated = max(raw_max, abs(raw)), saturated+int(abs(raw) > 1)
        before_time = float(env.data.time)
        action = float(np.float32(np.clip(raw, -1., 1.)))
        _, _, terminated, truncated, info = env.step([action])
        clean = bool(clean and np.isclose(env.data.time-before_time, env.dt, rtol=0., atol=1e-9)
                     and not any(w.number for w in env.data.warning)
                     and np.all(np.isfinite(env.data.qpos)) and np.all(np.isfinite(env.data.qvel)))
        max_angle = float(np.max(np.abs(np.cumsum(env.data.qpos[1:n+1]))))
        streak_steps = streak_steps+1 if max_angle < cfg['env']['success_upright_threshold'] else 0
        max_streak_steps = max(max_streak_steps, streak_steps)
        trajectory.append(dict(time=float(env.data.time), action=action, qpos=env.data.qpos.tolist(),
                               qvel=env.data.qvel.tolist(), reference_phase=phase,
                               max_absolute_angle=max_angle))
        if not clean or terminated or truncated:
            break
    result = dict(success=bool(info.get('success')) and clean and info.get('termination_reason') == 'time_limit',
                  length=len(trajectory), termination_reason=info.get('termination_reason'),
                  physical_hold_seconds=max_streak_steps*env.dt,
                  environment_reported_hold_seconds=float(info.get('max_upright_streak_seconds', 0.)),
                  max_cart_excursion=float(env.max_cart_excursion), raw_action_abs_max=raw_max,
                  saturated_steps=saturated, trajectory_clean=clean, perturbation=perturbation.tolist(),
                  selected_state=state, trajectory=trajectory)
    env.close()
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--period-steps', type=int, required=True)
    p.add_argument('--angle-amplitude', type=float, default=.03)
    p.add_argument('--control-cost', type=float, default=.01)
    p.add_argument('--backward-seconds', type=float, default=60.)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    if (args.period_steps < 4 or args.period_steps % 2 or not 0 <= args.angle_amplitude < .15
            or args.control_cost <= 0 or args.backward_seconds <= 0):
        raise ValueError('even positive period, subthreshold amplitude and positive feedback costs/budget required')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    started = time.monotonic()
    cfg = load_config(args.config)
    n, nx = int(cfg['env']['n_links']), 2*(int(cfg['env']['n_links'])+1)
    env = NLinkCartPoleEnv(cfg, progress=1., seed=0)
    env.reset(seed=0)
    dt = float(env.dt)
    transition = MujocoTransition(env, coordinate_transform=np.eye(nx), continuous_angles=True)
    zero = np.zeros(nx)
    a, b = transition.linearize(zero, 0., state_epsilon=1e-5, action_epsilon=1e-4)
    phase = 2*np.pi*np.arange(args.period_steps//2)/args.period_steps
    unit = linear_antiperiodic_nodes(a, b, np.cos(phase))
    unit_angle = np.max(np.abs(np.cumsum(unit[:, 1:n+1], axis=1)))
    amplitude = min(args.angle_amplitude/unit_angle, .05, 1./max(1e-12, np.max(np.abs(unit[:, 0]))))
    half_controls = (amplitude*np.cos(phase)).astype(np.float32).astype(float)
    linear_nodes = linear_antiperiodic_nodes(a, b, half_controls)
    nodes, controls, diagnostics = refine_antiperiodic_orbit(transition, linear_nodes, half_controls,
                                                           state_epsilon=1e-5, action_epsilon=1e-4)
    max_angle = float(np.max(np.abs(np.cumsum(nodes[:, 1:n+1], axis=1))))
    if not diagnostics['converged'] or max_angle >= .15 or np.max(np.abs(nodes[:, 0])) > 3:
        raise ValueError('periodic boundary solve did not produce an admissible near-upright initializer')
    jacobians = [transition.linearize(x, float(u), state_epsilon=1e-5, action_epsilon=1e-4)
                 for x,u in zip(nodes[:-1], controls)]
    cycles = max(3, int(np.ceil(args.backward_seconds/(len(controls)*env.dt))))
    # Reuse only derivatives measured at the identical periodic nodes/actions.
    class CachedCycle:
        env = SimpleNamespace(n=n)
        def __init__(self):
            self.cursor = cycles*len(controls)-1
        def linearize(self, x, u, **kwargs):
            index = self.cursor % len(controls)
            assert np.array_equal(x, nodes[index]) and u == controls[index]
            self.cursor -= 1
            return jacobians[index]
    transform = dimensionless_absolute_transform(n, StateScales(1.25, .15, .5, .75))
    cost = QuadraticTrajectoryCost(transform.T@transform, transform.T@transform,
                                   args.control_cost, 2.85, 3., 0., wrap_angles=False,
                                   stage_factor=transform, terminal_factor=transform)
    repeated_nodes = np.vstack([np.tile(nodes[:-1], (cycles, 1)), nodes[:1]])
    repeated_controls = np.tile(controls, cycles)
    all_gains = square_root_tracking_gains(CachedCycle(), repeated_nodes, repeated_controls, cost)
    gains = all_gains[:len(controls)]
    change = float(np.linalg.norm(gains-all_gains[len(controls):2*len(controls)])/max(1., np.linalg.norm(gains)))
    env.close()
    episodes = []
    directions = np.random.default_rng(20261900).uniform(-1., 1., (4, 2*n))
    for error in (0., 1e-11, 1e-9, 1e-6, 1e-3):
        cohort = directions[:1] if error == 0 else directions
        for direction in cohort:
            perturbation = np.zeros(nx)
            perturbation[1:n+1] = np.diff(np.r_[0., error*direction[:n]])
            perturbation[n+2:] = np.diff(np.r_[0., error*direction[n:]])
            episode = execute_component(cfg, nodes, controls, gains, perturbation, 20261900)
            episode['perturbation_absolute_angle_and_rate_scale'] = error
            episodes.append(episode)
        print(f'period={args.period_steps} amplitude={args.angle_amplitude} error={error}: '+
              str([(e['success'], e['physical_hold_seconds']) for e in episodes[-len(cohort):]]), flush=True)
    payload = dict(not_solution=True, hanging_start_evidence=False, reserved_twelve_seeds_used=False,
                   hypothesis='A small allowed upright motion may improve controllability; this study tests physical periodic tracking, not hanging-start reachability.',
                   config=file_metadata(Path(args.config)), runtime=runtime_metadata(), n_links=n,
                   period_steps=args.period_steps, period_seconds=dt*args.period_steps,
                   requested_angle_amplitude=args.angle_amplitude, actual_maximum_absolute_angle=max_angle,
                   nominal_maximum_cart_excursion=float(np.max(np.abs(nodes[:, 0]))),
                   force_amplitude_normalized=float(amplitude), orbit_diagnostics=diagnostics,
                   controller=dict(coordinate_transform=np.eye(nx).tolist(), controls=controls.tolist(),
                                   feedback_gains=gains.tolist(), feedback_control_cost=args.control_cost,
                                   backwards_cycles=cycles, backwards_seconds=cycles*len(controls)*dt,
                                   first_two_cycle_gain_relative_change=change,
                                   maximum_gain_norm=float(np.max(np.linalg.norm(gains, axis=1))),
                                   implementation='pure_ltv_square_root_qr'),
                   search=dict(nominal_coordinate_states=nodes.tolist()), episode_results=episodes,
                   arithmetic='binary64 nonlinear MuJoCo and binary64 sparse/QR calculations; no high precision dynamics claim',
                   source=file_metadata(Path(__file__)), wall_time_seconds=time.monotonic()-started)
    dump_json(payload, out)


if __name__ == '__main__':
    main()
