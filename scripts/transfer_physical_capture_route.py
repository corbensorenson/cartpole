#!/usr/bin/env python
"""Transfer a passed physical route to adjacent material coordinates offline.

The target nodes generally violate target dynamics. Zero feedback placeholders
are explicit; this artifact is only an initializer for full-horizon synthesis.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, utc_timestamp
from scripts.synthesize_inverse_increment import exact_candidate_passed
from scripts.transfer_inverse_spline import transfer_coefficients


def transfer_physical_nodes(states, target_links):
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] % 2:
        raise ValueError('physical states must contain equal position and velocity blocks')
    d = states.shape[1]//2
    return np.column_stack([transfer_coefficients(states[:, :d], target_links),
                            transfer_coefficients(states[:, d:], target_links)])


def retime_physical_curve(states, controls, dt, route_seconds=None):
    """Offline time dilation with velocities equal to the interpolant derivative.

    Controls are interpolated as a warm start; neither controls nor states
    inherit feasibility under the changed timing. The default is bit preserving.
    """
    states, controls = np.asarray(states, dtype=float), np.asarray(controls, dtype=float)
    if (controls.ndim != 1 or not len(controls) or states.ndim != 2
            or states.shape[0] != len(controls)+1 or states.shape[1] % 2
            or not np.isfinite(dt) or dt <= 0
            or not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls))):
        raise ValueError('a finite physical curve, controls and positive timestep are required')
    duration = len(controls)*dt
    if route_seconds is None:
        return states.copy(), controls.copy()
    if (not np.isfinite(route_seconds) or route_seconds <= 0
            or not np.isclose(route_seconds/dt, round(route_seconds/dt), rtol=0., atol=1e-10)):
        raise ValueError('route duration must be positive and tick aligned')
    steps = round(route_seconds/dt)
    if steps == len(controls):
        return states.copy(), controls.copy()
    from scipy.interpolate import CubicHermiteSpline
    dimension = states.shape[1]//2
    scale = duration/(steps*dt)
    original_times = np.arange(len(states))*dt
    query = np.arange(steps+1)*dt*scale
    curve = CubicHermiteSpline(original_times, states[:, :dimension], states[:, dimension:])
    positions = curve(query)
    positions[0], positions[-1] = states[0, :dimension], states[-1, :dimension]
    velocities = scale*curve(query, 1)
    actions = np.interp(query[:-1], original_times[:-1], controls)
    return np.column_stack([positions, velocities]), actions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-controller', required=True)
    p.add_argument('--source-config', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--capture-tail-seconds', type=float, default=5.)
    p.add_argument('--route-seconds', type=float, default=None,
                   help='Optional offline time dilation; physical feasibility must be rebuilt.')
    p.add_argument('--seed', type=int, default=20263100)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    source = json.loads(Path(args.source_controller).read_text())
    source_cfg, cfg = load_config(args.source_config), load_config(args.config)
    n, sn = cfg['env']['n_links'], source_cfg['env']['n_links']
    if n != sn+1 or not exact_candidate_passed(source, source_cfg):
        raise ValueError('requires a fully passed feasible source and an adjacent target count')
    dt = cfg['env']['timestep']*cfg['env']['frame_skip']
    if (dt != source_cfg['env']['timestep']*source_cfg['env']['frame_skip']
            or not np.isfinite(args.capture_tail_seconds) or args.capture_tail_seconds <= 0
            or not np.isclose(args.capture_tail_seconds/dt, round(args.capture_tail_seconds/dt), rtol=0., atol=1e-10)):
        raise ValueError('matched control cadence and a positive tick-aligned tail required')
    controls = np.asarray(source['controller']['controls'], dtype=np.float32).astype(float)
    transform = np.asarray(source['controller']['coordinate_transform'], dtype=float)
    initial = np.linalg.solve(transform, np.asarray(source['search']['nominal_coordinate_states'])[0])
    rows = source['result']['trajectory'][:len(controls)]
    if len(rows) != len(controls):
        raise ValueError('source physical route prefix must be complete')
    physical = np.vstack([initial, [row['qpos']+row['qvel'] for row in rows]])
    original_duration = len(controls)*dt
    physical, controls = retime_physical_curve(physical, controls, dt, args.route_seconds)
    nodes = transfer_physical_nodes(physical, n)
    tail_steps = round(args.capture_tail_seconds/dt)
    target = np.zeros(2*(n+1))
    target[1:n+1] = 2*np.pi*np.round(nodes[-1, 1:n+1]/(2*np.pi))
    nodes = np.vstack([nodes, np.repeat(target[None], tail_steps, axis=0)])
    controls = np.r_[controls, np.zeros(tail_steps)]
    selected = dict(qpos=nodes[0, :n+1].tolist(), qvel=nodes[0, n+1:].tolist())
    dump_json(dict(schema_version=1, generated_at=utc_timestamp(), not_solution=True,
        initialization_only=True, selected_state=selected, seed=args.seed,
        source=file_metadata(args.source_controller), source_config=file_metadata(args.source_config),
        config=file_metadata(args.config), source_count=sn, n_links=n,
        timing=dict(source_route_seconds=original_duration, requested_route_seconds=args.route_seconds,
                    target_route_seconds=(len(controls)-tail_steps)*dt,
                    velocity_scale=original_duration/((len(controls)-tail_steps)*dt),
                    method='cubic_hermite_position_derivative_and_interpolated_controls'
                    if args.route_seconds is not None and args.route_seconds != original_duration
                    else 'unchanged'),
        controller=dict(type='adjacent_material_physical_curve_with_virtual_capture_tail',
            continuous_angles=True, physical_shooting=True, coordinate_transform=np.eye(2*(n+1)).tolist(),
            controls=controls.tolist(), feedback_gains=np.zeros((len(controls), 2*(n+1))).tolist(),
            feedback_placeholder=True, capture_start_seconds=(len(controls)-tail_steps)*dt,
            virtual_tail_seconds=args.capture_tail_seconds, angle_branch_terminal_target=target.tolist()),
        search=dict(nominal_coordinate_states=nodes.tolist()),
        note='Positions and velocities are interpolated in material absolute-angle coordinates. Target dynamics feasibility is not inherited. Tail and nodes are offline optimizer variables; they must never replace live plant states.'), out)


if __name__ == '__main__':
    main()
