#!/usr/bin/env python
"""Fit a successful exact route to the spline warm-start format; feasibility is not inherited."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import lsq_linear

from gcartpole.config import dump_json
from gcartpole.evidence import data_sha256, file_metadata, runtime_metadata, utc_timestamp
try:
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.synthesize_inverse_increment import exact_candidate_passed
except ModuleNotFoundError:
    from search_capture_sequence import fixed_state_cfg
    from synthesize_inverse_increment import exact_candidate_passed


def fit_route_spline(template_points, knots, physical_states, times, rail_limit, velocity_weight):
    points = np.asarray(template_points, dtype=float).copy()
    states = np.asarray(physical_states, dtype=float).copy()
    times = np.asarray(times, dtype=float)
    if points.ndim != 2 or len(points) < 8:
        raise ValueError("finite quintic coefficient matrix required")
    d = points.shape[1]
    if states.shape != (len(times), 2*d) or not np.all(np.isfinite(np.r_[points.ravel(), states.ravel(), times])):
        raise ValueError("finite source states must match coefficient dimensions and sample times")
    if not np.isfinite(velocity_weight) or velocity_weight <= 0 or not np.isfinite(rail_limit) or rail_limit <= 0:
        raise ValueError("positive finite fitting weights and rail required")
    if (not np.allclose(points[:3], points[0], atol=1e-12, rtol=0)
            or not np.allclose(points[-3:], points[-1], atol=1e-12, rtol=0)):
        raise ValueError("template must fix endpoint position, velocity, and acceleration")
    states[:, 1:d] += 2*np.pi*np.round((points[0, 1:]-states[0, 1:d])/(2*np.pi))
    if not np.allclose(states[0, :d], points[0], atol=1e-8, rtol=0) or np.max(abs(states[0, d:])) > 1e-8:
        raise ValueError("route must begin at the same stationary hanging state")
    if np.any(np.round((states[-1, 1:d]-points[-1, 1:])/(2*np.pi)) != 0):
        raise ValueError("route terminal winding is incompatible with the template")
    basis_spline = BSpline(np.asarray(knots, dtype=float), np.eye(len(points)), 5)
    basis = basis_spline(times)
    derivative = basis_spline.derivative(1)(times)
    fixed = points.copy(); fixed[3:-3] = 0
    design = np.vstack([basis[:, 3:-3], velocity_weight*derivative[:, 3:-3]])
    rhs = np.vstack([states[:, :d]-basis@fixed,
                     velocity_weight*(states[:, d:]-derivative@fixed)])
    solution = np.linalg.lstsq(design, rhs, rcond=None)[0]
    limits = np.r_[rail_limit, np.full(d-1, 4*np.pi)]
    bounded_columns = []
    for column, limit in enumerate(limits):
        if np.any(abs(solution[:, column]) > limit):
            fitted = lsq_linear(design, rhs[:, column], bounds=(-limit, limit),
                                tol=1e-10, max_iter=300)
            if not fitted.success:
                raise ValueError(f"bounded spline fit failed at coordinate {column}: {fitted.message}")
            solution[:, column] = fitted.x
            bounded_columns.append(column)
    points[3:-3] = solution
    q_error = basis@points-states[:, :d]
    v_error = derivative@points-states[:, d:]
    return points, dict(max_cart_position_error=float(np.max(abs(q_error[:, 0]))),
                        max_absolute_link_angle_error=float(np.max(abs(np.cumsum(q_error[:, 1:], axis=1)))),
                        relative_joint_velocity_rms_error=float(np.sqrt(np.mean(v_error[:, 1:]**2))),
                        bounded_fit_columns=bounded_columns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-spline", required=True)
    parser.add_argument("--source-controller", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--velocity-weight", type=float)
    parser.add_argument("--curve", choices=["physical", "nominal"], default="physical",
                        help="Fit the executed physical prefix, including an early handoff, or the optimizer nodes.")
    args = parser.parse_args()
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    template = json.loads(Path(args.template_spline).read_text())
    source = json.loads(Path(args.source_controller).read_text())
    cfg = template["effective_config"]
    if not exact_candidate_passed(source, cfg):
        raise ValueError("source requires a feasible nominal route and a successful full physical replay")
    source_cfg = fixed_state_cfg(cfg, source["selected_state"], cfg["env"]["episode_seconds"])
    source_cfg["env"].setdefault("action_lqr_residual", {})["enabled"] = False
    evidence = source.get("evidence", {}).get("config", {})
    if evidence.get("progress") != 1.0 or evidence.get("resolved_sha256") != data_sha256(source_cfg):
        raise ValueError("source configuration must match the template's canonical full-progress plant")
    controls = np.asarray(source["controller"]["controls"], dtype=float)
    transform = np.asarray(source["controller"]["coordinate_transform"], dtype=float)
    states = np.linalg.solve(transform, np.asarray(source["search"]["nominal_coordinate_states"]).T).T
    duration = template["controller"]["horizon_seconds"]
    dt = cfg["env"]["timestep"]*cfg["env"]["frame_skip"]
    if len(controls) != round(duration/dt) or states.shape[1] != 2*(cfg["env"]["n_links"]+1):
        raise ValueError("source must have the same morphology and route duration as the template")
    if args.curve == "physical":
        rows = source["result"]["trajectory"][:len(controls)]
        if len(rows) != len(controls) or not np.allclose(
            [row["time_seconds"] for row in rows], np.arange(1, len(controls)+1)*dt,
            rtol=0, atol=1e-9
        ):
            raise ValueError("physical source prefix must have complete canonical control cadence")
        states = np.vstack([states[0], [np.r_[row["qpos"], row["qvel"]] for row in rows]])
    points, metrics = fit_route_spline(template["search"]["spline_control_points"],
        template["search"]["spline_knots"], states, np.arange(len(controls)+1)*dt,
        cfg["env"]["rail_limit"], dt if args.velocity_weight is None else args.velocity_weight)
    dump_json(dict(schema_version=1, not_solution=True, initialization_only=True,
        generated_at=utc_timestamp(), runtime=runtime_metadata(), parameters=vars(args),
        source=file_metadata(args.source_controller), template=file_metadata(args.template_spline),
        effective_config=cfg, selected_state=template["selected_state"],
        controller=dict(horizon_seconds=duration),
        search=dict(spline_control_points=points.tolist(), spline_knots=template["search"]["spline_knots"],
                    spline_control_point_count=len(points), fit_metrics=metrics),
        note="Least-squares position/velocity fit of the declared source curve with unchanged clamped endpoints. Physical fitting includes executed capture switches. Fitting and target transfer do not inherit exact dynamics feasibility; no runtime state projection."), args.out)
    print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
