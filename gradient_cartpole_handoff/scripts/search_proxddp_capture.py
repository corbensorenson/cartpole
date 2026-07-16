#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.capture_constraints import exact_constraint_margins, terminal_bounds
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.fddp import MujocoActionModel, rollout_controls
from gcartpole.ilqr import (
    MujocoTransition,
    QuadraticTrajectoryCost,
    add_terminal_cart_weights,
    data_state,
    optimize_ilqr,
)
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
)

try:
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.refine_ilqr_capture_chain import source_trajectory
    from scripts.search_capture_sequence import fixed_state_cfg, load_state
    from scripts.search_ilqr_capture import execute_controller
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from refine_ilqr_capture_chain import source_trajectory
    from search_capture_sequence import fixed_state_cfg, load_state
    from search_ilqr_capture import execute_controller
    from search_swingup_capture import lqr_gain


def main() -> None:
    try:
        import aligator
        import crocoddyl
    except ImportError as error:
        raise SystemExit(
            "Aligator is required; run `make setup-aligator` and invoke this "
            "script with `.conda-aligator/bin/python`"
        ) from error

    parser = argparse.ArgumentParser(
        description="Search one hard-constrained exact-MuJoCo capture trajectory with ProxDDP"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--state-json", default="runs/p1_capture_envelope/validation.json"
    )
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--initial-controller", required=True)
    parser.add_argument("--seed", type=int, default=69001)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--mu-init", type=float, default=0.2)
    parser.add_argument("--mu-lower-bound", type=float, default=1e-11)
    parser.add_argument("--max-al-iterations", type=int, default=100)
    parser.add_argument(
        "--acceptance-strategy",
        choices=("filter", "nonmonotone", "armijo"),
        default="filter",
    )
    parser.add_argument(
        "--rollout-type", choices=("linear", "nonlinear"), default="linear"
    )
    parser.add_argument(
        "--retain-best-exact", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--optimize-last-steps", type=int, default=30)
    parser.add_argument("--control-trust-region", type=float, default=0.1)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--control-weight", type=float, default=0.1)
    parser.add_argument("--stage-weight", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=100_000.0)
    parser.add_argument("--terminal-state-weight", type=float, default=100_000.0)
    parser.add_argument("--terminal-cart-weight", type=float, default=1_000_000.0)
    parser.add_argument(
        "--terminal-cart-velocity-weight", type=float, default=100_000.0
    )
    parser.add_argument("--rail-soft-limit", type=float, default=2.4)
    parser.add_argument("--rail-weight", type=float, default=100_000_000.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.25)
    parser.add_argument("--handoff-angle-abs", type=float, default=0.15)
    parser.add_argument("--handoff-cart-velocity-abs", type=float, default=0.5)
    parser.add_argument("--handoff-hinge-velocity-abs", type=float, default=0.75)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if (
        min(
            args.iterations,
            args.tolerance,
            args.mu_init,
            args.mu_lower_bound,
            args.max_al_iterations,
            args.optimize_last_steps,
            args.control_trust_region,
            args.lqr_scale,
            args.control_weight,
            args.stage_weight,
            args.terminal_weight,
            args.rail_soft_limit,
            args.rail_weight,
            args.handoff_lyapunov,
            args.handoff_cart_abs,
            args.handoff_angle_abs,
            args.handoff_cart_velocity_abs,
            args.handoff_hinge_velocity_abs,
        )
        <= 0.0
        or min(
            args.terminal_state_weight,
            args.terminal_cart_weight,
            args.terminal_cart_velocity_weight,
        )
        < 0.0
    ):
        raise ValueError("weights, constraints, and iteration settings are invalid")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    selected_state, state_index = load_state(args.state_json, args.state_index)
    cfg = fixed_state_cfg(
        base_cfg, selected_state, float(base_cfg["env"]["episode_seconds"])
    )
    n_links = int(cfg["env"]["n_links"])
    distribution = load_config(args.spec)["distribution"]
    scales = StateScales(
        float(distribution["cart_position_abs_max"]),
        float(distribution["absolute_link_angle_abs_max"]),
        float(distribution["cart_velocity_abs_max"]),
        float(distribution["hinge_velocity_rms_max"]),
    )
    transform = dimensionless_absolute_transform(n_links, scales)
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix, input_matrix, gain, transform, feedback_scale=args.lqr_scale
    )
    terminal_metric = (
        args.terminal_weight * lyapunov / args.handoff_lyapunov
        + args.terminal_state_weight * np.eye(transform.shape[0], dtype=np.float64)
    )
    terminal_metric = add_terminal_cart_weights(
        terminal_metric,
        n_links,
        cart_weight=args.terminal_cart_weight,
        cart_velocity_weight=args.terminal_cart_velocity_weight,
    )
    trajectory_cost = QuadraticTrajectoryCost(
        stage_state=args.stage_weight * np.eye(transform.shape[0], dtype=np.float64),
        terminal_state=terminal_metric,
        control=args.control_weight,
        rail_soft_limit=float(args.rail_soft_limit * transform[0, 0]),
        rail_limit=float(cfg["env"]["rail_limit"] * transform[0, 0]),
        rail_weight=args.rail_weight,
        wrap_angles=False,
    )

    initial_path = Path(args.initial_controller)
    initial_payload = json.loads(initial_path.read_text(encoding="utf-8"))
    initial_controls, initial_states, _ = source_trajectory(initial_payload)
    if initial_states.shape != (initial_controls.size + 1, transform.shape[0]):
        raise ValueError("initial controller state/control horizon is inconsistent")
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    env.reset(seed=args.seed)
    transition = MujocoTransition(env, coordinate_transform=transform)
    start_state = transition.to_coordinates(data_state(env.data))
    # Rebuild the warm start with this process's exact transition. This avoids
    # presenting ProxDDP with a nominal multiple-shooting defect due to serialization.
    initial_states = rollout_controls(transition, start_state, initial_controls)
    optimized_start = max(0, initial_controls.size - args.optimize_last_steps)
    frozen_controls = initial_controls[:optimized_start]
    solver_initial_controls = initial_controls[optimized_start:]
    solver_initial_states = initial_states[optimized_start:]
    solver_start_state = solver_initial_states[0]
    running_model = MujocoActionModel(transition, trajectory_cost)
    terminal_model = MujocoActionModel(transition, trajectory_cost, terminal=True)
    croc_problem = crocoddyl.ShootingProblem(
        solver_start_state,
        [running_model] * int(solver_initial_controls.size),
        terminal_model,
    )
    problem = aligator.croc.convertCrocoddylProblem(croc_problem)
    nx = transform.shape[0]
    nu = 1
    rail_limit = float(env.rail_limit * transform[0, 0])
    rail_matrix = np.zeros((1, nx), dtype=np.float64)
    rail_matrix[0, 0] = 1.0
    control_matrix = np.ones((1, nu), dtype=np.float64)
    for step, stage in enumerate(problem.stages):
        control_lower = max(
            -1.0, float(solver_initial_controls[step]) - args.control_trust_region
        )
        control_upper = min(
            1.0, float(solver_initial_controls[step]) + args.control_trust_region
        )
        stage.addConstraint(
            aligator.LinearFunction(
                rail_matrix, np.zeros((1, nu), dtype=np.float64), np.zeros(1)
            ),
            aligator.constraints.BoxConstraint(
                np.asarray([-rail_limit]), np.asarray([rail_limit])
            ),
        )
        stage.addConstraint(
            aligator.LinearFunction(
                np.zeros((1, nx), dtype=np.float64),
                control_matrix,
                np.zeros(1),
            ),
            aligator.constraints.BoxConstraint(
                np.asarray([control_lower]), np.asarray([control_upper])
            ),
        )
    lower, upper = terminal_bounds(
        n_links,
        scales,
        cart_abs=args.handoff_cart_abs,
        angle_abs=args.handoff_angle_abs,
        cart_velocity_abs=args.handoff_cart_velocity_abs,
        hinge_velocity_abs=args.handoff_hinge_velocity_abs,
    )
    problem.addTerminalConstraint(
        aligator.LinearFunction(
            np.eye(nx, dtype=np.float64),
            np.zeros((nx, nu), dtype=np.float64),
            np.zeros(nx),
        ),
        aligator.constraints.BoxConstraint(lower, upper),
    )
    if not problem.checkIntegrity():
        raise RuntimeError("converted constrained problem failed integrity checks")

    solver = aligator.SolverProxDDP(
        args.tolerance,
        args.mu_init,
        args.iterations,
        aligator.VerboseLevel.QUIET,
    )
    acceptance_strategies = {
        "filter": aligator.SA_FILTER,
        "nonmonotone": aligator.SA_LINESEARCH_NONMONOTONE,
        "armijo": aligator.SA_LINESEARCH_ARMIJO,
    }
    rollout_types = {
        "linear": aligator.ROLLOUT_LINEAR,
        "nonlinear": aligator.ROLLOUT_NONLINEAR,
    }
    solver.sa_strategy = acceptance_strategies[args.acceptance_strategy]
    solver.rollout_type = rollout_types[args.rollout_type]
    solver.max_al_iters = args.max_al_iterations
    solver.bcl_params.mu_lower_bound = args.mu_lower_bound

    class IterateCallback(aligator.BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.controls: list[np.ndarray] = []

        def call(self, workspace: Any, results: Any) -> None:
            del workspace
            self.controls.append(
                np.asarray([float(row[0]) for row in results.us], dtype=np.float64)
            )

    history = IterateCallback()
    solver.registerCallback("history", history)
    solver.setup(problem)
    initial_us = [
        np.asarray([action], dtype=np.float64) for action in solver_initial_controls
    ]
    started = time.time()
    converged = bool(
        solver.run(problem, [row.copy() for row in solver_initial_states], initial_us)
    )
    search_seconds = time.time() - started
    final_tail_controls = np.asarray([float(row[0]) for row in solver.results.us])
    final_controls = np.concatenate((frozen_controls, final_tail_controls))
    solver_states = np.asarray(solver.results.xs, dtype=np.float64)
    final_exact_states = rollout_controls(transition, start_state, final_controls)
    defects = np.asarray(
        [
            transition.difference(
                transition(solver_states[step], float(final_tail_controls[step])),
                solver_states[step + 1],
            )
            for step in range(final_tail_controls.size)
        ],
        dtype=np.float64,
    )
    candidates = [("initial", initial_controls, initial_states)]
    candidates.extend(
        (f"history_{index}", np.concatenate((frozen_controls, rows)), None)
        for index, rows in enumerate(history.controls)
    )
    candidates.append(("final", final_controls, final_exact_states))
    candidate_audit = []
    for source, candidate_controls, candidate_states in candidates:
        if candidate_states is None:
            candidate_states = rollout_controls(
                transition, start_state, candidate_controls
            )
        if not np.all(np.isfinite(candidate_states)):
            continue
        candidate_margins = exact_constraint_margins(
            candidate_states,
            lyapunov,
            lower,
            upper,
            rail_limit=rail_limit,
            handoff_lyapunov=args.handoff_lyapunov,
        )
        candidate_audit.append(
            {
                "source": source,
                "controls": candidate_controls,
                "states": candidate_states,
                "margins": candidate_margins,
                "score": (
                    max(0.0, -candidate_margins["minimum_hard"]),
                    max(0.0, -candidate_margins["lyapunov"]),
                ),
            }
        )
    selected_candidate = (
        min(candidate_audit, key=lambda candidate: candidate["score"])
        if args.retain_best_exact
        else candidate_audit[-1]
    )
    controls = selected_candidate["controls"]
    exact_states = selected_candidate["states"]
    margins = selected_candidate["margins"]
    tracking = optimize_ilqr(
        transition,
        start_state,
        controls,
        trajectory_cost,
        max_iterations=0,
    )
    nominal_values = np.maximum(
        0.0, np.einsum("ij,jk,ik->i", exact_states, lyapunov, exact_states)
    )
    env.close()
    result = execute_controller(
        cfg,
        seed=args.seed,
        controls=controls,
        nominal_states=exact_states,
        feedback_gains=tracking.feedback_gains,
        gain=gain,
        lqr_scale=args.lqr_scale,
        transform=transform,
        lyapunov=lyapunov,
        handoff_lyapunov=args.handoff_lyapunov,
        handoff_cart_abs=args.handoff_cart_abs,
        handoff_angle_abs=args.handoff_angle_abs,
        handoff_cart_velocity_abs=args.handoff_cart_velocity_abs,
        handoff_hinge_velocity_rms=args.handoff_hinge_velocity_abs,
        tracking_mode="aligator_proxddp_tracking",
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Single-state hard-constrained exact-MuJoCo ProxDDP diagnostic; not P1 evidence.",
        "state_index": int(state_index),
        "selected_state": selected_state,
        "seed": int(args.seed),
        "controller": {
            "type": "aligator_proxddp_exact_mujoco_then_lqr",
            "initial_controller": file_metadata(initial_path),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(
                controls.size * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]
            ),
            "iterations": int(args.iterations),
            "tolerance": float(args.tolerance),
            "mu_init": float(args.mu_init),
            "mu_lower_bound": float(args.mu_lower_bound),
            "max_al_iterations": int(args.max_al_iterations),
            "acceptance_strategy": args.acceptance_strategy,
            "rollout_type": args.rollout_type,
            "retain_best_exact": bool(args.retain_best_exact),
            "optimize_last_steps": int(args.optimize_last_steps),
            "control_trust_region": float(args.control_trust_region),
            "lqr_scale": float(args.lqr_scale),
            "control_weight": float(args.control_weight),
            "stage_weight": float(args.stage_weight),
            "terminal_weight": float(args.terminal_weight),
            "terminal_state_weight": float(args.terminal_state_weight),
            "terminal_cart_weight": float(args.terminal_cart_weight),
            "terminal_cart_velocity_weight": float(args.terminal_cart_velocity_weight),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "handoff_angle_abs": float(args.handoff_angle_abs),
            "handoff_cart_velocity_abs": float(args.handoff_cart_velocity_abs),
            "handoff_hinge_velocity_abs": float(args.handoff_hinge_velocity_abs),
            "controls": controls.astype(float).tolist(),
            "feedback_gains": tracking.feedback_gains.astype(float).tolist(),
        },
        "search": {
            "converged": converged,
            "iterations": int(solver.results.num_iters),
            "al_iterations": int(solver.results.al_iter),
            "cost": float(solver.results.traj_cost),
            "primal_infeasibility": float(solver.results.primal_infeas),
            "dual_infeasibility": float(solver.results.dual_infeas),
            "wall_time_seconds": float(search_seconds),
            "selected_candidate": selected_candidate["source"],
            "exact_candidate_audit": [
                {
                    "source": candidate["source"],
                    "exact_constraint_margins": candidate["margins"],
                }
                for candidate in candidate_audit
            ],
            "solver_defect_rms": float(np.sqrt(np.mean(defects**2))),
            "solver_defect_abs_max": float(np.max(np.abs(defects))),
            "exact_constraint_margins": margins,
            "exact_minimum_lyapunov": float(np.min(nominal_values)),
            "exact_terminal_lyapunov": float(nominal_values[-1]),
            "solver_coordinate_states": solver_states.astype(float).tolist(),
            "nominal_coordinate_states": exact_states.astype(float).tolist(),
        },
        "result": result,
        "lyapunov": {
            "coordinate_source": file_metadata(args.spec),
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "evidence": {
            "config": {
                "path": str(Path(args.config)),
                "resolved_sha256": data_sha256(cfg),
            },
            "state_source": file_metadata(args.state_json),
            "solver_versions": {
                "aligator": aligator.__version__,
                "crocoddyl": crocoddyl.__version__,
            },
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"converged={converged} iterations={solver.results.num_iters} "
        f"primal={solver.results.primal_infeas:.3e} "
        f"defect={payload['search']['solver_defect_abs_max']:.3e} "
        f"hard_margin={margins['minimum_hard']:.3e} "
        f"terminal_v={nominal_values[-1]:.2f} wall={search_seconds:.1f}s"
    )
    print(
        f"success={result['success']} latched={result['latched']} "
        f"live_min_v={result['minimum_lyapunov']:.2f} "
        f"hold={result['max_upright_streak_seconds']:.3f}s "
        f"cart={result['max_cart_excursion']:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
