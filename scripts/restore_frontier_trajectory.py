#!/usr/bin/env python
"""Restore a transferred trajectory's exact discrete dynamics before FDDP."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.dynamics_restoration import RestorationProblem, restore_trajectory
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform
from gcartpole.simulation import SimulationError

try:
    from scripts.refine_ilqr_capture_chain import source_trajectory
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_fddp_capture import lift_nominal_trajectory, warm_start_diagnostics
except ModuleNotFoundError:
    from refine_ilqr_capture_chain import source_trajectory
    from search_capture_sequence import fixed_state_cfg
    from search_fddp_capture import lift_nominal_trajectory, warm_start_diagnostics


def source_coordinate_transform(source, cfg, spec_path):
    saved = source["controller"].get("coordinate_transform")
    if saved is not None:
        transform = np.asarray(saved, dtype=float)
        provenance = "artifact"
    else:
        metadata = file_metadata(spec_path)
        prior = source.get("lyapunov", {}).get("coordinate_source", {})
        if prior.get("sha256") != metadata["sha256"]:
            raise ValueError("missing transform requires the original verified benchmark coordinate spec")
        spec = load_config(spec_path)["distribution"]
        transform = dimensionless_absolute_transform(cfg["env"]["n_links"], StateScales(
            spec["cart_position_abs_max"], spec["absolute_link_angle_abs_max"],
            spec["cart_velocity_abs_max"], spec["hinge_velocity_rms_max"]))
        provenance = "verified_legacy_benchmark_spec"
    dimension = 2 * (cfg["env"]["n_links"]+1)
    if transform.shape != (dimension, dimension) or not np.all(np.isfinite(transform)):
        raise ValueError("coordinate transform must be finite and match the target state dimension")
    return transform, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--max-evaluations", type=int, default=15)
    parser.add_argument("--defect-weight", type=float, default=1000.0)
    parser.add_argument("--terminal-weight", type=float, default=1000.0)
    parser.add_argument("--reference-weight", type=float, default=1e-4)
    parser.add_argument("--control-weight", type=float, default=1e-6)
    parser.add_argument("--lsmr-max-iterations", type=int, default=300)
    parser.add_argument("--solver", choices=["lsmr", "l1-osqp"], default="lsmr")
    parser.add_argument("--inexact-qp-tolerance", type=float, default=0.0)
    parser.add_argument("--monotone-defects", action="store_true")
    parser.add_argument("--capture-tail-seconds", type=float, default=0.0)
    parser.add_argument("--capture-angle-limit", type=float, default=0.14)
    args = parser.parse_args()
    if args.capture_tail_seconds < 0 or not 0 < args.capture_angle_limit < 0.15:
        raise ValueError("capture tail must be nonnegative and angle limit strictly inside 0.15 radians")
    output = Path(args.out)
    if output.exists():
        raise FileExistsError(output)
    source_path = Path(args.controller)
    source = json.loads(source_path.read_text())
    controls, reference, _ = source_trajectory(source)
    cfg = fixed_state_cfg(load_config(args.config), source["selected_state"], 30.0)
    transform, transform_source = source_coordinate_transform(source, cfg, args.spec)
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    transition = MujocoTransition(env, transform, continuous_angles=True)
    start = transition.to_coordinates(data_state(env.data))
    reference = lift_nominal_trajectory(transition, reference, start)
    reference[0] = start
    physical_target = np.zeros(start.size)
    physical_target[1:env.n + 1] = 2 * np.pi * np.round(
        transition.to_physical(reference[-1])[1:env.n + 1] / (2 * np.pi)
    )
    target = transform @ physical_target
    source_steps = len(controls)
    tail_steps = int(round(args.capture_tail_seconds / env.dt))
    lower_states = upper_states = None
    if tail_steps:
        controls = np.r_[controls, np.zeros(tail_steps)]
        reference = np.vstack([reference, np.tile(target, (tail_steps, 1))])
        lower_states = np.full((len(controls), start.size), -np.inf)
        upper_states = np.full_like(lower_states, np.inf)
        angle_scale = transform[1, 1]
        angle_bounds = args.capture_angle_limit * angle_scale
        # Decision node index source_steps-1 is the original swing endpoint.
        lower_states[source_steps - 1:, 1:env.n + 1] = target[1:env.n + 1] - angle_bounds
        upper_states[source_steps - 1:, 1:env.n + 1] = target[1:env.n + 1] + angle_bounds
    factor = np.eye(start.size)
    factor[1:env.n + 1, 1:env.n + 1] *= np.sqrt(20.0)
    factor[env.n + 2:, env.n + 2:] *= 2.0
    problem = RestorationProblem(
        transition, start, reference, controls, target, factor,
        args.defect_weight, args.terminal_weight, args.reference_weight, args.control_weight,
    )
    initial = warm_start_diagnostics(transition, start, controls, reference)
    started = time.monotonic()
    result, states, controls = restore_trajectory(
        problem, rail_limit=env.rail_limit * transform[0, 0],
        max_evaluations=args.max_evaluations, lsmr_max_iterations=args.lsmr_max_iterations,
        solver=args.solver,
        progress_callback=lambda row: print(json.dumps(row), flush=True),
        inexact_qp_tolerance=args.inexact_qp_tolerance,
        monotone_defects=args.monotone_defects,
        state_lower_bounds=lower_states, state_upper_bounds=upper_states,
    )
    elapsed = time.monotonic() - started
    final = warm_start_diagnostics(transition, start, controls, states)
    exact = [start.copy()]
    error = None
    try:
        for action in controls:
            exact.append(transition(exact[-1], float(action)))
    except SimulationError as exception:
        error = str(exception)
    exact = np.asarray(exact)
    controller = dict(source["controller"])
    controller.update(type="bounded_sparse_dynamics_restoration_warm_start",
                      controls=controls.tolist(), feedback_gains=np.zeros_like(states[:-1]).tolist(),
                      continuous_angles=True, coordinate_transform=transform.tolist(),
                      angle_branch_terminal_target=target.tolist())
    controller.update(horizon_steps=len(controls), horizon_seconds=len(controls) * env.dt,
                      capture_tail_seconds=tail_steps * env.dt,
                      capture_angle_limit=args.capture_angle_limit)
    controller.pop("solver_feedback_gains", None)
    dump_json(dict(schema_version=1, not_solution=True, generated_at=utc_timestamp(),
                   selected_state=source["selected_state"], source=file_metadata(source_path),
                   config=file_metadata(Path(args.config)), runtime=runtime_metadata(),
                   controller=controller,
                   search=dict(method="bounded_sparse_dynamics_restoration", cost=float(result.cost),
                               evaluations=int(result.nfev), status=int(result.status), message=str(result.message),
                               optimizer_converged=bool(result.success) if args.solver == "lsmr" else None,
                               is_feasible=final["is_feasible"],
                               initial_diagnostics=initial, final_diagnostics=final,
                               wall_time_seconds=elapsed, transition_evaluations=transition.evaluations,
                               physics_steps_upper_bound=transition.evaluations * env.frame_skip,
                               parameters=vars(args), history=problem.history,
                               coordinate_transform_source=transform_source,
                               qp_history=getattr(result, "qp_history", None),
                               nominal_coordinate_states=states.tolist(),
                               exact_open_loop_coordinate_states=exact.tolist(),
                               exact_rollout_error=error,
                               exact_terminal_error=float(np.linalg.norm(factor @ (exact[-1] - target))),
                               exact_max_cart_excursion=float(np.max(np.abs(exact[:, 0])) / transform[0, 0]))),
              output)
    env.close()
    print(f"maximum defect {initial['maximum_dynamics_defect']:.6g} -> {final['maximum_dynamics_defect']:.6g}; "
          f"exact terminal error={np.linalg.norm(factor @ (exact[-1] - target)):.6g}; "
          f"wall={elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
