#!/usr/bin/env python
"""Discover inverse-dynamics spline routes, then independently replay cart forces."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import BSpline

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.inverse_collocation import SplineInverseProblem, solve_inverse_spline
from gcartpole.modal import StateScales, dimensionless_absolute_transform
from search_capture_sequence import fixed_state_cfg
from search_fddp_capture import warm_start_diagnostics


def refine_saved_spline(source, rounds):
    """Insert interval midpoints exactly; no interpolation of saved states."""
    if rounds < 0:
        raise ValueError("nonnegative knot refinement rounds required")
    spline = BSpline(np.asarray(source["search"]["spline_knots"]),
                     np.asarray(source["search"]["spline_control_points"]), 5)
    for _ in range(rounds):
        unique = np.unique(spline.t)
        for midpoint in (unique[:-1] + unique[1:]) / 2:
            spline = spline.insert_knot(float(midpoint))
    return spline


def inherited_hanging_state(source, n_links):
    """Preserve a saved hanging winding without accepting a different start."""
    d = n_links+1
    selected = source.get('selected_state', {})
    qpos = np.asarray(selected.get('qpos', []), dtype=float)
    qvel = np.asarray(selected.get('qvel', []), dtype=float)
    points = np.asarray(source.get('search', {}).get('spline_control_points', []), dtype=float)
    reference = np.r_[np.pi, np.zeros(n_links-1)]
    if (qpos.shape != (d,) or qvel.shape != (d,) or points.ndim != 2 or points.shape[1] != d
            or not len(points) or not np.all(np.isfinite(np.r_[qpos, qvel, points.ravel()]))
            or qpos[0] != 0. or np.any(qvel != 0.) or not np.array_equal(points[0], qpos)
            or not np.allclose((qpos[1:]-reference)/(2*np.pi),
                               np.round((qpos[1:]-reference)/(2*np.pi)), rtol=0., atol=1e-12)):
        raise ValueError('saved spline must start at stationary canonical hanging, allowing integer angle windings')
    return qpos.copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-links", type=int)
    parser.add_argument("--seconds", type=float, default=8.)
    parser.add_argument("--control-points", type=int)
    parser.add_argument("--initial-spline")
    parser.add_argument("--refine-knots", type=int, default=0)
    parser.add_argument("--sample-dt", type=float, default=.04)
    parser.add_argument("--max-evaluations", type=int, default=50)
    parser.add_argument("--dynamics-weight", type=float, default=1.)
    parser.add_argument("--force-weight", type=float, default=100.)
    parser.add_argument("--effort-weight", type=float, default=1e-6)
    parser.add_argument("--reference-weight", type=float, default=1e-5)
    parser.add_argument("--dynamics-residual", choices=["torque", "acceleration"], default="torque")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    if args.sample_dt <= 0 or args.max_evaluations < 1:
        raise ValueError("positive sample interval and evaluation budget required")
    cfg = load_config(args.config)
    if args.n_links is not None:
        if args.n_links < 1:
            raise ValueError("positive link count required")
        cfg["env"]["n_links"] = args.n_links
    n, d = cfg["env"]["n_links"], cfg["env"]["n_links"] + 1
    source = json.loads(Path(args.initial_spline).read_text()) if args.initial_spline else None
    initial = np.r_[0., np.pi, np.zeros(n - 1)] if source is None else inherited_hanging_state(source, n)
    selected = dict(qpos=initial.tolist(), qvel=np.zeros(d).tolist())
    cfg = fixed_state_cfg(cfg, selected, 30.)
    env = NLinkCartPoleEnv(cfg, progress=1., seed=0)
    env.reset(seed=0)
    duration = round(args.seconds / env.dt) * env.dt
    if not 0 < duration <= 25.:
        raise ValueError("positive duration no longer than 25 seconds required")
    samples = np.linspace(0., duration, int(np.ceil(duration / args.sample_dt)) + 1)
    points = knots = source_metadata = None
    if args.initial_spline:
        if source["effective_config"]["env"] != cfg["env"] or source["controller"]["horizon_seconds"] != duration:
            raise ValueError("saved spline must use the same physical config, initial state and duration")
        refined = refine_saved_spline(source, args.refine_knots)
        points, knots = refined.c, refined.t
        if args.control_points is not None and args.control_points != len(points):
            raise ValueError("control point count disagrees with refined source spline")
        source_metadata = file_metadata(args.initial_spline)
    elif args.refine_knots:
        raise ValueError("knot refinement requires a saved spline")
    point_count = len(points) if points is not None else (args.control_points or 20)
    problem = SplineInverseProblem(env.model, initial, np.zeros(d), duration,
                                  point_count, samples, env.force_limit,
                                  dynamics_weight=args.dynamics_weight, force_weight=args.force_weight,
                                  effort_weight=args.effort_weight, reference_weight=args.reference_weight,
                                  dynamics_residual=args.dynamics_residual,
                                  knots=knots, initial_control_points=points,
                                  progress_callback=lambda row: print(json.dumps(row), flush=True))
    started = time.monotonic()
    result = solve_inverse_spline(problem, rail_limit=env.rail_limit, max_evaluations=args.max_evaluations)
    solve_seconds = time.monotonic() - started
    points = problem.unpack(result.x)
    knots = problem.knots
    spline = BSpline(knots, points, 5)
    times = np.arange(round(duration / env.dt) + 1) * env.dt
    q, v, a = [spline.derivative(order)(times) for order in range(3)]
    native_forces = np.array([problem.inverse(qi, vi, ai) for qi, vi, ai in zip(q, v, a)])
    controls = np.clip(native_forces[:-1, 0] / env.force_limit, -1., 1.).astype(np.float32).astype(float)
    spec = load_config("benchmarks/p1_capture_envelope.yaml")["distribution"]
    transform = dimensionless_absolute_transform(n, StateScales(
        spec["cart_position_abs_max"], spec["absolute_link_angle_abs_max"],
        spec["cart_velocity_abs_max"], spec["hinge_velocity_rms_max"]))
    nominal = np.column_stack([q, v]) @ transform.T
    transition = MujocoTransition(env, transform, continuous_angles=True)
    diagnostics = warm_start_diagnostics(transition, nominal[0], controls, nominal)
    info = {}
    env.reset(seed=0)
    states = [data_state(env.data)]
    for action in controls:
        _, _, terminated, truncated, info = env.step([action])
        states.append(data_state(env.data))
        if terminated or truncated:
            break
    dense_times = np.linspace(0., duration, 4 * (len(samples) - 1) + 1)
    dense_forces = np.array([problem.inverse(qi, vi, ai) for qi, vi, ai in zip(
        spline(dense_times), spline.derivative(1)(dense_times), spline.derivative(2)(dense_times))])
    dense_acceleration_errors = np.array([problem.acceleration_residual(qi, vi, ai) for qi, vi, ai in zip(
        spline(dense_times), spline.derivative(1)(dense_times), spline.derivative(2)(dense_times))])
    dump_json(dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(),
                   selected_state=selected, config=file_metadata(args.config), effective_config=cfg,
                   runtime=runtime_metadata(), parameters=vars(args),
                   initial_spline=source_metadata,
                   controller=dict(type="inverse_dynamics_spline_warm_start", controls=controls.tolist(),
                                   coordinate_transform=transform.tolist(), continuous_angles=True,
                                   feedback_gains=np.zeros_like(nominal[:-1]).tolist(),
                                   horizon_steps=len(controls), horizon_seconds=duration),
                   search=dict(cost=float(result.cost), evaluations=int(result.nfev), status=int(result.status),
                               optimizer_converged=bool(result.success), message=str(result.message),
                               is_feasible=diagnostics["is_feasible"], warm_start_diagnostics=diagnostics,
                               nominal_coordinate_states=nominal.tolist(), spline_control_points=points.tolist(),
                               spline_control_point_count=point_count,
                               spline_knots=knots.tolist(), inverse_sample_times=samples.tolist(),
                               joint_torque_scales=problem.torque_scales.tolist(),
                               dense_max_joint_torque=float(np.max(np.abs(dense_forces[:, 1:]))),
                               dense_max_normalized_joint_residual=float(np.max(np.abs(dense_forces[:, 1:] / problem.torque_scales))),
                               dense_max_cart_force=float(np.max(np.abs(dense_forces[:, 0]))),
                               dense_max_missing_force_acceleration=float(np.max(np.abs(dense_acceleration_errors))),
                               exact_serial_physical_states=np.asarray(states).tolist(),
                               exact_replay=dict(steps=len(states)-1, max_cart_excursion=env.max_cart_excursion,
                                                 max_hold_seconds=env.max_upright_streak_steps * env.dt,
                                                 termination_reason=info.get("termination_reason"),
                                                 simulation_error=env.simulation_error),
                               solve_wall_time_seconds=solve_seconds, wall_time_seconds=time.monotonic()-started,
                               inverse_calls=problem.inverse_calls, derivative_calls=problem.derivative_calls,
                               transition_evaluations=transition.evaluations, history=problem.history),
                   note="Unactuated torques are residuals only. Independent replay applies cart force exclusively; no joint forces or runtime state projection."), out)
    env.close()


if __name__ == "__main__":
    main()
