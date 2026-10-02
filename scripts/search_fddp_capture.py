#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

try:
    import crocoddyl
except ImportError as error:
    raise SystemExit(
        "Crocoddyl is required; run `.venv/bin/pip install -r requirements-fddp.txt`"
    ) from error

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.lqr_design import high_precision_lyapunov_factor
from gcartpole.fddp import MujocoActionModel, rollout_controls
from gcartpole.ilqr import (
    MujocoTransition,
    QuadraticTrajectoryCost,
    add_terminal_cart_weights,
    data_state,
    wrapped_state,
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
    from scripts.search_ilqr_capture import (
        execute_controller,
        interpolate_initial_state,
    )
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from refine_ilqr_capture_chain import source_trajectory
    from search_capture_sequence import fixed_state_cfg, load_state
    from search_ilqr_capture import execute_controller, interpolate_initial_state
    from search_swingup_capture import lqr_gain


def load_solver_feedback(payload: dict[str, Any], expected_shape: tuple[int, int]) -> np.ndarray:
    """Select unscaled saved solver gains explicitly, preserving replay provenance."""
    gains = np.asarray(payload["controller"]["solver_feedback_gains"], dtype=np.float64)
    if gains.shape != expected_shape or not np.all(np.isfinite(gains)):
        raise ValueError("saved solver feedback gains must have the expected finite shape")
    return gains.copy()


def rebuild_feedback_warm_start(
    transition: Any,
    start_state: np.ndarray,
    controls: np.ndarray,
    nominal_states: np.ndarray,
    feedback_gains: np.ndarray,
    *,
    feedback_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the applied controls and states from exact feedback replay."""
    controls = np.asarray(controls, dtype=np.float64)
    nominal_states = np.asarray(nominal_states, dtype=np.float64)
    feedback_gains = np.asarray(feedback_gains, dtype=np.float64)
    start_state = np.asarray(start_state, dtype=np.float64)
    if controls.ndim != 1:
        raise ValueError("warm-start controls must be one-dimensional")
    if nominal_states.shape != (controls.size + 1, start_state.size):
        raise ValueError("warm-start nominal states have inconsistent dimensions")
    if feedback_gains.shape != (controls.size, start_state.size):
        raise ValueError("warm-start feedback gains have inconsistent dimensions")
    if feedback_scale < 0.0:
        raise ValueError("warm-start feedback scale must be nonnegative")

    states = [start_state.copy()]
    applied_controls = []
    for step, control in enumerate(controls):
        error = (
            transition.difference(states[-1], nominal_states[step])
            if hasattr(transition, "difference")
            else states[-1] - nominal_states[step]
        )
        action = float(
            np.clip(
                control + feedback_scale * feedback_gains[step] @ error,
                -1.0,
                1.0,
            )
        )
        applied_controls.append(action)
        states.append(transition(states[-1], action))
    return (
        np.asarray(applied_controls, dtype=np.float64),
        np.asarray(states, dtype=np.float64),
    )


def warm_start_diagnostics(
    transition: Any,
    start_state: np.ndarray,
    controls: np.ndarray,
    states: np.ndarray,
    *,
    tolerance: float = 1e-8,
    require_feasible: bool = False,
) -> dict[str, Any]:
    """Measure feasibility in the Euclidean state used by Crocoddyl.

    Periodically equivalent states on different branches are not feasible
    for StateVector: its actual gap is their Euclidean difference.
    """
    controls = np.asarray(controls, dtype=np.float64)
    states = np.asarray(states, dtype=np.float64)
    start_state = np.asarray(start_state, dtype=np.float64)
    if controls.ndim != 1 or states.shape != (controls.size + 1, start_state.size):
        raise ValueError("warm-start state/control dimensions are inconsistent")
    if not (np.all(np.isfinite(controls)) and np.all(np.isfinite(states))):
        raise ValueError("warm start contains non-finite values")
    if np.max(np.abs(controls), initial=0.0) > 1.0:
        raise ValueError("warm-start controls exceed the normalized force bound")
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("feasibility tolerance must be finite and positive")
    initial_gap = float(np.max(np.abs(states[0] - start_state), initial=0.0))
    defects = []
    for step, action in enumerate(controls):
        predicted = np.asarray(transition(states[step], float(action)), dtype=np.float64)
        if not np.all(np.isfinite(predicted)):
            raise ValueError("warm-start transition produced a non-finite state")
        defects.append(float(np.max(np.abs(predicted - states[step + 1]), initial=0.0)))
    max_gap = max([initial_gap, *defects])
    feasible = max_gap <= tolerance
    exactly_feasible = max_gap == 0.0
    if require_feasible and not exactly_feasible:
        raise ValueError(
            f"initial-feasible requested, but maximum state/control defect is "
            f"{max_gap:.6g}; the solver's flag requires zero defects, "
            f"not just the reporting tolerance {tolerance:.6g}; rebuild the trajectory "
            "or let FDDP start infeasibly"
        )
    return {
        "is_feasible": bool(feasible),
        "is_exactly_feasible": bool(exactly_feasible),
        "tolerance": float(tolerance),
        "initial_state_gap": initial_gap,
        "maximum_dynamics_defect": max(defects, default=0.0),
        "median_dynamics_defect": float(np.median(defects)) if defects else 0.0,
        "max_abs_control": float(np.max(np.abs(controls), initial=0.0)),
        "coordinate_gap": "euclidean_crocoddyl_state_vector",
        "gap_norm": "linf",
    }


def load_terminal_target(
    path: str,
    index: str,
    state_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one saved coordinate-space handoff state for terminal targeting."""

    source_path = Path(path)
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if isinstance(payload.get("terminal_target"), list):
        states = np.asarray([payload["terminal_target"]], dtype=np.float64)
        selected_index = 0
    else:
        search = payload.get("search")
        if not isinstance(search, dict) or not isinstance(
            search.get("nominal_coordinate_states"), list
        ):
            raise ValueError(
                "terminal target artifact must contain terminal_target or "
                "search.nominal_coordinate_states"
            )
        states = np.asarray(search["nominal_coordinate_states"], dtype=np.float64)
        if index == "last":
            selected_index = states.shape[0] - 1
        else:
            selected_index = int(index)
            if selected_index < 0:
                selected_index += states.shape[0]
    if states.ndim != 2 or states.shape[1] != state_size:
        raise ValueError("terminal target states do not match the target coordinate dimension")
    if not 0 <= selected_index < states.shape[0]:
        raise ValueError("terminal target index is out of range")
    return states[selected_index].copy(), {
        "path": str(source_path),
        "sha256": data_sha256(payload),
        "index": int(selected_index),
        "state_count": int(states.shape[0]),
    }


def lift_nominal_trajectory(
    transition: MujocoTransition, states: np.ndarray, start_state: np.ndarray
) -> np.ndarray:
    """Choose a continuous joint-angle branch aligned with the launch state."""
    physical = np.asarray([transition.to_physical(row) for row in states])
    initial = transition.to_physical(start_state)
    angles = slice(1, transition.env.n + 1)
    original_angles = physical[:, angles].copy()
    physical[:, angles] = np.unwrap(physical[:, angles], axis=0)
    physical[:, angles] += 2 * np.pi * np.round(
        (initial[angles] - physical[0, angles]) / (2 * np.pi)
    )
    # Apply only the angle-branch correction. A round trip through physical
    # coordinates perturbs even an already aligned, exactly feasible route,
    # forcing FDDP to restore defects introduced by this loader itself.
    correction = np.zeros_like(physical)
    correction[:, angles] = physical[:, angles] - original_angles
    if transition.coordinate_transform is None:
        return np.asarray(states, dtype=np.float64).copy() + correction
    return np.asarray(states, dtype=np.float64).copy() + (
        correction @ transition.coordinate_transform.T
    )


def upright_branch_target(transition, reference_state):
    """Choose the equilibrium winding from the inherited route, not a free tail."""
    physical = transition.to_physical(reference_state)
    target = np.zeros_like(physical)
    angles = slice(1, transition.env.n + 1)
    target[angles] = 2*np.pi*np.round(physical[angles]/(2*np.pi))
    return transition.to_coordinates(target)


def capture_interval_cost(base_cost, identity, mapping, stage_weight,
                          value_factor=None, value_weight=0., handoff_value=1.):
    """Retain both pose/rate and directional value residuals during capture."""
    factor = np.sqrt(stage_weight)*np.diag(np.sqrt(np.diag(identity))) @ mapping
    if value_weight > 0:
        if value_factor is None:
            raise ValueError("capture value running costs require a factored Lyapunov design")
        factor = np.vstack([factor, np.sqrt(value_weight/handoff_value)*value_factor])
    return replace(base_cost, stage_state=factor.T@factor, stage_factor=factor,
                   stage_target=base_cost.terminal_target)


def feedback_capture_tail(transition, start, gain, steps, scale=1.):
    """Roll a declared LQR tail on the actual step map; never project its nodes."""
    gain = np.asarray(gain, dtype=float).reshape(-1)
    if (steps < 1 or gain.shape != np.asarray(start).shape
            or not np.all(np.isfinite(gain)) or not np.isfinite(scale) or scale < 0):
        raise ValueError("feedback tail requires positive steps, matching finite gain and nonnegative scale")
    states = np.empty((steps+1, gain.size))
    controls = np.empty(steps)
    gains = np.empty((steps, gain.size))
    states[0] = start
    inverse = transition.inverse_transform
    coordinate_gain = gain if inverse is None else gain @ inverse
    for step in range(steps):
        physical = transition.to_physical(states[step])
        error = wrapped_state(physical, transition.env.n)
        proposed = -scale*float(gain @ error)
        controls[step] = float(np.float32(np.clip(proposed, -1., 1.)))
        gains[step] = -scale*coordinate_gain if -1. < proposed < 1. else 0.
        states[step+1] = transition(states[step], controls[step])
    return controls, states, gains


def capture_node_constraints(transition, target, start_step, steps, limit):
    """Absolute-angle constraints about the declared upright winding branch."""
    if not np.isfinite(limit) or not 0 < limit < .15 or not 0 <= start_step <= steps:
        raise ValueError('capture node bounds require a valid start and an angle margin below .15')
    n, nx = transition.env.n, len(target)
    matrix = np.zeros((n, nx))
    matrix[:, 1:n+1] = np.tril(np.ones((n, n)))
    if transition.inverse_transform is not None:
        matrix = matrix @ transition.inverse_transform
    center = matrix @ target
    lower, upper = np.full((steps, n), -np.inf), np.full((steps, n), np.inf)
    lower[max(0, start_step-1):] = center-limit
    upper[max(0, start_step-1):] = center+limit
    return matrix, lower, upper


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search one exact-MuJoCo capture trajectory with Box-FDDP"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument(
        "--progress",
        type=float,
        default=1.0,
        help="morphology schedule progress for this exact replay; 1.0 is the endpoint plant",
    )
    parser.add_argument(
        "--lqr-progress",
        type=float,
        default=None,
        help="morphology progress used to linearize the post-handoff LQR; defaults to --progress",
    )
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--state-json", default="runs/p1_capture_envelope/validation.json"
    )
    parser.add_argument("--state-index", required=True)
    parser.add_argument("--interpolate-from-state-index", default=None)
    parser.add_argument("--interpolation-alpha", type=float, default=1.0)
    parser.add_argument("--initial-controller", default=None)
    parser.add_argument("--initial-solver-feedback", action="store_true",
                        help="Select unscaled solver_feedback_gains from the source instead of its applied feedback_gains.")
    parser.add_argument(
        "--terminal-target-json",
        default=None,
        help="artifact containing search.nominal_coordinate_states for a measured handoff target",
    )
    parser.add_argument(
        "--terminal-target-index",
        default="last",
        help="row index in the terminal target artifact, or 'last'",
    )
    parser.add_argument(
        "--replay-only",
        action="store_true",
        help="Skip Box-FDDP and replay the inherited controller directly on the target plant.",
    )
    parser.add_argument("--horizon-seconds", type=float, default=4.5)
    parser.add_argument(
        "--append-tail-seconds",
        type=float,
        default=0.0,
        help="append a zero-control settling tail to an inherited controller warm start",
    )
    parser.add_argument("--initial-feasible", action="store_true")
    parser.add_argument("--tail-initializer", choices=("zero", "lqr"), default="zero",
                        help="Initialize an appended tail with zero actions or physical LQR feedback; neither implies successful capture.")
    parser.add_argument("--rebuild-initial-states", action="store_true")
    parser.add_argument(
        "--continuous-angles",
        action="store_true",
        help="use a continuous joint-angle lift consistent with Euclidean FDDP state gaps",
    )
    parser.add_argument(
        "--rebuild-initial-feedback",
        action="store_true",
        help=(
            "rebuild an inherited warm start by applying its saved feedback "
            "gains while rolling it through the target plant"
        ),
    )
    parser.add_argument(
        "--initial-feedback-scale",
        type=float,
        default=1.0,
        help="scale inherited feedback only while rebuilding a target-plant warm start",
    )
    parser.add_argument("--seed", type=int, default=67001)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--optimizer", choices=("box-fddp", "sqrt-ilqr", "sqrt-fddp", "sparse-scvx"), default="box-fddp")
    parser.add_argument("--defect-penalty", type=float, default=1e8)
    parser.add_argument("--derivative-order", choices=(2, 4), type=int, default=2)
    parser.add_argument("--state-epsilon", type=float, default=1e-5)
    parser.add_argument("--action-epsilon", type=float, default=1e-4)
    parser.add_argument("--physical-shooting", action="store_true",
                        help="Integrate physical state coordinates, retaining the normalized cost via transformed factors.")
    parser.add_argument("--enforce-rail-during-search", action="store_true",
                        help="Reject rail-violating nominal trials in QR optimizers; requires an in-rail initializer and does not certify physical replay.")
    parser.add_argument("--optimize-suffix-start-seconds", type=float, default=0.,
                        help="Freeze an exactly feasible inherited prefix and optimize only the remaining QR horizon; deployment still starts from hanging.")
    parser.add_argument("--sparse-trust-policy", choices=("legacy", "agreement"), default="legacy")
    parser.add_argument("--sparse-qp-solver", choices=("osqp", "clarabel"), default="osqp")
    parser.add_argument("--sparse-state-trust", type=float, default=.05)
    parser.add_argument("--sparse-control-trust", type=float, default=.1)
    parser.add_argument("--sparse-qp-max-iterations", type=int, default=10000)
    parser.add_argument("--sparse-qp-tolerance", type=float, default=1e-8)
    parser.add_argument("--sparse-qp-initial-tolerance", type=float, default=None,
                        help="Optional initial QP accuracy cap, tightened against measured scaled dynamics gaps; physical bounds and replay gates stay fixed.")
    parser.add_argument("--sparse-qp-inexact-dual-tolerance", type=float, default=0.,
                        help="Optional inexact QP candidate cap; requires primal residual below 10 percent of current scaled gaps, then exact nonlinear merit and hard-bound checks.")
    parser.add_argument("--sparse-capture-angle-limit", type=float, default=None,
                        help="Optional hard absolute-angle bounds on virtual capture nodes in sparse SCvx; physical replay remains authoritative.")
    parser.add_argument("--solver-verbose", action="store_true",
                        help="Record Crocoddyl iteration diagnostics in the experiment log.")
    parser.add_argument("--recompute-tracking-feedback", action="store_true",
                        help="Rebuild pure LTV perturbation gains about the final route, independently of optimization descent clipping and regularization.")
    parser.add_argument("--tracking-feedback-regularization", type=float, default=0.)
    parser.add_argument("--tracking-feedback-decimal-digits", type=int, default=0,
                        help="Experimental MP finite-horizon Riccati feedback; zero preserves native QR design.")
    parser.add_argument("--export-tracking-value-factors", default=None,
                        help="Save MP-designed finite-horizon value roots for nonlinear tracking probes.")
    parser.add_argument("--fddp-ascent-acceptance", type=float,
                        help="Override native uphill-step acceptance threshold; zero requires cost decrease in its ascent branch.")
    parser.add_argument("--initial-regularization", type=float, default=1e-6)
    parser.add_argument("--tracking-gain-scale", type=float, default=0.0)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--lqr-decimal-digits", type=int, default=0,
                        help="Zero retains standard design; >=30 uses promoted-input gain and factored value design.")
    parser.add_argument(
        "--lqr-control-cost",
        type=float,
        default=1000.0,
        help="R term used to compute the post-handoff LQR gain",
    )
    parser.add_argument("--lqr-cart-position-cost", type=float, default=0.1)
    parser.add_argument("--lqr-absolute-angle-cost", type=float, default=100.0)
    parser.add_argument("--lqr-cart-velocity-cost", type=float, default=0.1)
    parser.add_argument("--lqr-absolute-angular-velocity-cost", type=float, default=1.0)
    parser.add_argument("--lqr-relative-angle-cost", type=float, default=1.0)
    parser.add_argument("--lqr-relative-angular-velocity-cost", type=float, default=0.01)
    parser.add_argument("--control-cost", type=float, default=0.1)
    parser.add_argument("--stage-weight", type=float, default=0.1)
    parser.add_argument("--capture-start-seconds", type=float, default=None,
                        help="Start a separately weighted upright interval within the optimized route.")
    parser.add_argument("--capture-stage-weight", type=float, default=1000.0)
    parser.add_argument("--capture-value-weight", type=float, default=0.0,
                        help="Directional capture value coefficient in each capture stage; requires a factored design.")
    parser.add_argument(
        "--split-link",
        type=int,
        default=None,
        help=(
            "one-based source link whose new target joint should remain phase-aligned "
            "during the optimized route"
        ),
    )
    parser.add_argument(
        "--split-angle-stage-weight",
        type=float,
        default=0.0,
        help="running weight on the inserted relative angle in scaled coordinates",
    )
    parser.add_argument(
        "--split-rate-stage-weight",
        type=float,
        default=0.0,
        help="running weight on the inserted hinge rate in scaled coordinates",
    )
    parser.add_argument("--terminal-weight", type=float, default=10_000.0)
    parser.add_argument("--terminal-state-weight", type=float, default=100_000.0)
    parser.add_argument("--terminal-cart-weight", type=float, default=0.0)
    parser.add_argument("--terminal-cart-velocity-weight", type=float, default=0.0)
    parser.add_argument(
        "--terminal-cart-velocity-factor",
        type=float,
        default=1.0,
        help="multiply the terminal cart-velocity block in the full-state objective",
    )
    parser.add_argument(
        "--terminal-hinge-velocity-factor",
        type=float,
        default=1.0,
        help="multiply the terminal hinge-velocity block in the full-state objective",
    )
    parser.add_argument(
        "--terminal-angle-factor",
        type=float,
        default=1.0,
        help="multiply the terminal absolute-angle block in the full-state objective",
    )
    parser.add_argument("--rail-soft-limit", type=float, default=2.4)
    parser.add_argument("--rail-weight", type=float, default=100_000_000.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--switch-lyapunov", type=float, default=None)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--handoff-angle-abs", type=float, default=0.15)
    parser.add_argument("--handoff-cart-velocity-abs", type=float, default=0.5)
    parser.add_argument("--handoff-hinge-velocity-rms", type=float, default=0.75)
    parser.add_argument(
        "--defer-handoff-until-horizon",
        action="store_true",
        help="Keep replaying the optimized feedback trajectory until its horizon before switching to LQR.",
    )
    parser.add_argument(
        "--phase-adaptive",
        action="store_true",
        help="Select the nearest forward nominal route phase within a bounded window during replay.",
    )
    parser.add_argument(
        "--phase-window",
        type=int,
        default=12,
        help="Maximum forward route steps considered by phase-adaptive replay.",
    )
    parser.add_argument(
        "--prefix-control-scale",
        type=float,
        default=1.0,
        help="multiply inherited feedforward controls during an initial prefix screen",
    )
    parser.add_argument(
        "--prefix-control-seconds",
        type=float,
        default=0.0,
        help="duration of the inherited-control prefix to scale; zero disables the screen",
    )
    parser.add_argument(
        "--allow-unstable-lyapunov",
        action="store_true",
        help="Use an identity terminal metric when the seven-link linear LQR is not asymptotically stable.",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if not np.isfinite(args.state_epsilon) or not np.isfinite(args.action_epsilon) or min(args.state_epsilon, args.action_epsilon) <= 0:
        raise ValueError("finite positive derivative increments required")
    if not np.isfinite(args.defect_penalty) or args.defect_penalty <= 0:
        raise ValueError("finite positive defect penalty required")
    if args.fddp_ascent_acceptance is not None and (
        not np.isfinite(args.fddp_ascent_acceptance) or args.fddp_ascent_acceptance < 0
    ):
        raise ValueError("FDDP ascent acceptance must be finite and nonnegative")
    if (
        min(
            args.iterations,
            args.horizon_seconds,
            args.initial_regularization,
            args.lqr_scale,
            args.lqr_control_cost,
            args.lqr_cart_position_cost,
            args.lqr_absolute_angle_cost,
            args.lqr_cart_velocity_cost,
            args.lqr_absolute_angular_velocity_cost,
            args.lqr_relative_angle_cost,
            args.lqr_relative_angular_velocity_cost,
            args.control_cost,
            args.stage_weight,
            args.rail_soft_limit,
            args.rail_weight,
            args.handoff_lyapunov,
            args.handoff_cart_abs,
            args.handoff_angle_abs,
            args.handoff_cart_velocity_abs,
            args.handoff_hinge_velocity_rms,
            args.terminal_cart_velocity_factor,
            args.terminal_hinge_velocity_factor,
            args.terminal_angle_factor,
        )
        <= 0.0
        or min(
            args.terminal_state_weight,
            args.terminal_cart_weight,
            args.terminal_cart_velocity_weight,
        )
        < 0.0
        or min(args.split_angle_stage_weight, args.split_rate_stage_weight) < 0.0
    ):
        raise ValueError("required counts, weights, and thresholds must be positive; optional split weights must be nonnegative")
    if not np.isfinite(args.terminal_weight) or args.terminal_weight < 0:
        raise ValueError("terminal capture value weight must be finite and nonnegative")
    if args.switch_lyapunov is not None and args.switch_lyapunov <= 0.0:
        raise ValueError("switch Lyapunov threshold must be positive")
    if args.capture_stage_weight <= 0 or (args.capture_start_seconds is not None and args.capture_start_seconds < 0):
        raise ValueError("capture stage weight must be positive and start time nonnegative")
    if not np.isfinite(args.capture_value_weight) or args.capture_value_weight < 0:
        raise ValueError("capture value running weight must be finite and nonnegative")
    if args.capture_value_weight > 0 and args.capture_start_seconds is None:
        raise ValueError("capture value running weight requires a capture interval")
    if args.enforce_rail_during_search and (args.optimizer == "box-fddp" or args.replay_only):
        raise ValueError("hard rail enforcement requires a square-root optimization run")
    if not np.isfinite(args.optimize_suffix_start_seconds) or args.optimize_suffix_start_seconds < 0:
        raise ValueError("suffix start must be finite and nonnegative")
    if args.optimize_suffix_start_seconds > 0 and (
        args.optimizer == "box-fddp" or args.replay_only or args.initial_controller is None
    ):
        raise ValueError("suffix optimization requires an inherited square-root optimization run")
    if args.optimizer == 'sparse-scvx' and (not args.continuous_angles or args.initial_controller is None or args.replay_only):
        raise ValueError('sparse SCvx requires an inherited continuous-angle optimization run')
    if args.sparse_capture_angle_limit is not None and (
        args.optimizer != 'sparse-scvx' or args.capture_start_seconds is None
        or not np.isfinite(args.sparse_capture_angle_limit) or not 0 < args.sparse_capture_angle_limit < .15
    ):
        raise ValueError('sparse angle bounds require a capture interval and an angle margin below .15')
    if args.tracking_gain_scale < 0.0:
        raise ValueError("tracking gain scale must be nonnegative")
    if args.lqr_decimal_digits != 0 and args.lqr_decimal_digits < 30:
        raise ValueError("precision design requires zero or at least thirty digits")
    if not 0.0 <= args.progress <= 1.0:
        raise ValueError("progress must be in [0, 1]")
    lqr_progress = args.progress if args.lqr_progress is None else args.lqr_progress
    if not 0.0 <= lqr_progress <= 1.0:
        raise ValueError("lqr-progress must be in [0, 1]")
    if args.append_tail_seconds < 0.0:
        raise ValueError("append-tail-seconds must be nonnegative")
    if args.tail_initializer != "zero" and args.append_tail_seconds <= 0:
        raise ValueError("feedback tail initializer requires append-tail-seconds")
    if args.prefix_control_seconds < 0.0:
        raise ValueError("prefix-control-seconds must be nonnegative")
    if args.prefix_control_seconds > 0.0 and args.initial_controller is None:
        raise ValueError("prefix-control-seconds requires an initial controller")
    if args.append_tail_seconds > 0.0 and args.initial_controller is None:
        raise ValueError("append-tail-seconds requires an initial controller")
    if args.rebuild_initial_feedback and args.initial_controller is None:
        raise ValueError("rebuild-initial-feedback requires an initial controller")
    if args.replay_only and args.initial_controller is None:
        raise ValueError("replay-only requires an initial controller")
    if args.initial_feedback_scale < 0.0:
        raise ValueError("initial feedback scale must be nonnegative")
    if args.phase_window < 0:
        raise ValueError("phase window must be nonnegative")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    if args.split_link is not None and not 1 <= args.split_link < int(
        base_cfg["env"]["n_links"]
    ):
        raise ValueError("split-link must identify an internal source link")
    if (
        args.split_link is None
        and (args.split_angle_stage_weight > 0.0 or args.split_rate_stage_weight > 0.0)
    ):
        raise ValueError("split-link is required for split-mode stage weights")
    base_cfg["env"].setdefault("action_lqr_residual", {})["enabled"] = False
    state, state_index = load_state(args.state_json, args.state_index)
    interpolation = None
    if args.interpolate_from_state_index is not None:
        source_state, source_index = load_state(
            args.state_json, args.interpolate_from_state_index
        )
        state = interpolate_initial_state(source_state, state, args.interpolation_alpha)
        interpolation = {
            "source_index": int(source_index),
            "target_index": int(state_index),
            "alpha": float(args.interpolation_alpha),
        }
    cfg = fixed_state_cfg(base_cfg, state, float(base_cfg["env"]["episode_seconds"]))

    lqr_weights = {
        "cart_position": args.lqr_cart_position_cost,
        "absolute_angle": args.lqr_absolute_angle_cost,
        "cart_velocity": args.lqr_cart_velocity_cost,
        "absolute_angular_velocity": args.lqr_absolute_angular_velocity_cost,
        "relative_angle": args.lqr_relative_angle_cost,
        "relative_angular_velocity": args.lqr_relative_angular_velocity_cost,
    }
    lqr_fallback_reason: str | None = None
    try:
        gain = lqr_gain(
            cfg,
            progress=lqr_progress,
            fd_eps=1e-7,
            control_cost=args.lqr_control_cost,
            q_weights=lqr_weights,
            decimal_digits=args.lqr_decimal_digits,
        )
    except (np.linalg.LinAlgError, ValueError) as error:
        if not args.replay_only and not args.allow_unstable_lyapunov:
            raise
        # Temporarily constrained or nearly singular continuation plants can
        # lack a finite upright Riccati solution. Keep the exact FDDP search
        # available when the caller explicitly permits an uncertified
        # Lyapunov metric; the artifact records why the fallback was used.
        gain = np.zeros(2 * (int(cfg["env"]["n_links"]) + 1), dtype=np.float64)
        lqr_fallback_reason = f"{type(error).__name__}: {error}"
    spec = load_config(args.spec)
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        int(cfg["env"]["n_links"]),
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    state_matrix, input_matrix = finite_difference_dynamics(cfg, args.progress, 1e-7)
    lyapunov_factor = None
    lyapunov_precision_diagnostics = None
    try:
        if args.lqr_decimal_digits:
            lyapunov, lyapunov_factor, lyapunov_precision_diagnostics = high_precision_lyapunov_factor(
                state_matrix, input_matrix, gain, transform, feedback_scale=args.lqr_scale,
                decimal_digits=args.lqr_decimal_digits)
            spectral_radius = float(lyapunov_precision_diagnostics["spectral_radius"])
            lyapunov_source = "lqr_high_precision_discrete_lyapunov"
        else:
            lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
                state_matrix, input_matrix, gain, transform, feedback_scale=args.lqr_scale
            )
            lyapunov_source = "lqr_discrete_lyapunov"
    except ValueError:
        if not args.allow_unstable_lyapunov:
            raise
        # The uniform seven-link linearization has nearly uncontrollable
        # internal modes, so its LQR closed loop is not a valid Lyapunov
        # certificate.  Keep the exact nonlinear FDDP search available with a
        # positive-definite target metric, while recording the missing
        # certificate explicitly in the artifact.
        lyapunov = np.eye(transform.shape[0], dtype=np.float64)
        spectral_radius = float(
            np.max(
                np.abs(
                    np.linalg.eigvals(
                        state_matrix
                        - input_matrix
                        @ (args.lqr_scale * gain).reshape(1, -1)
                    )
                )
            )
        )
        lyapunov_source = "identity_fallback_unstable_lqr"

    cost_mapping = np.eye(transform.shape[0])
    if args.physical_shooting:
        cost_mapping = transform.copy()
        lyapunov = transform.T @ lyapunov @ transform
        if lyapunov_factor is not None:
            lyapunov_factor = lyapunov_factor @ transform
            lyapunov = lyapunov_factor.T @ lyapunov_factor
        transform = np.eye(transform.shape[0])
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    env.reset(seed=args.seed)
    transition_type = MujocoTransition
    if args.derivative_order == 4:
        from gcartpole.high_order_dynamics import FourthOrderMujocoTransition
        transition_type = FourthOrderMujocoTransition
    transition = transition_type(
        env, coordinate_transform=transform, continuous_angles=args.continuous_angles
    )
    start_state = transition.to_coordinates(data_state(env.data))
    initial_path = (
        None if args.initial_controller is None else Path(args.initial_controller)
    )
    source_feedback_gains: np.ndarray | None = None
    if initial_path is None:
        if args.initial_solver_feedback:
            raise ValueError("selecting saved solver feedback requires an initial controller")
        horizon_steps = max(2, int(round(args.horizon_seconds / env.dt)))
        initial_controls = np.zeros(horizon_steps, dtype=np.float64)
        initial_states = rollout_controls(transition, start_state, initial_controls)
        branch_reference_state = initial_states[-1].copy()
    else:
        initial_payload = json.loads(initial_path.read_text(encoding="utf-8"))
        initial_controls, initial_states, source_feedback_gains = source_trajectory(
            initial_payload
        )
        if args.initial_solver_feedback:
            source_feedback_gains = load_solver_feedback(initial_payload, source_feedback_gains.shape)
        if args.physical_shooting:
            source_transform = np.asarray(initial_payload["controller"]["coordinate_transform"])
            to_source = source_transform @ np.linalg.inv(transform)
            initial_states = initial_states @ np.linalg.inv(to_source).T
            source_feedback_gains = source_feedback_gains @ to_source
        if args.continuous_angles:
            initial_states = lift_nominal_trajectory(transition, initial_states, start_state)
        branch_reference_state = initial_states[-1].copy()
        if args.append_tail_seconds > 0.0:
            tail_steps = max(1, int(round(args.append_tail_seconds / env.dt)))
            if args.tail_initializer == "lqr":
                tail_controls, tail_states, tail_gains = feedback_capture_tail(
                    transition, initial_states[-1], gain, tail_steps, args.lqr_scale)
            else:
                tail_controls = np.zeros(tail_steps, dtype=np.float64)
                tail_states = rollout_controls(transition, initial_states[-1], tail_controls)
                tail_gains = np.zeros((tail_steps, transform.shape[0]), dtype=np.float64)
            initial_controls = np.concatenate([initial_controls, tail_controls])
            initial_states = np.vstack([initial_states, tail_states[1:]])
            if source_feedback_gains is not None:
                source_feedback_gains = np.vstack(
                    [
                        source_feedback_gains,
                        tail_gains,
                    ]
                )
        if args.prefix_control_seconds > 0.0:
            prefix_steps = min(
                initial_controls.size,
                max(1, int(round(args.prefix_control_seconds / env.dt))),
            )
            initial_controls[:prefix_steps] = np.clip(
                initial_controls[:prefix_steps] * args.prefix_control_scale,
                -1.0,
                1.0,
            )
    if initial_states.shape != (initial_controls.size + 1, transform.shape[0]):
        raise ValueError("initial controller state/control horizon is inconsistent")
    initial_states = initial_states.copy()
    initial_states[0] = start_state
    replay_nominal_states = initial_states.copy()
    branch_terminal_target = None
    if args.continuous_angles:
        branch_terminal_target = upright_branch_target(transition, branch_reference_state)
    if args.rebuild_initial_feedback:
        assert source_feedback_gains is not None
        initial_controls, initial_states = rebuild_feedback_warm_start(
            transition,
            start_state,
            initial_controls,
            initial_states,
            source_feedback_gains,
            feedback_scale=args.initial_feedback_scale,
        )
    elif args.rebuild_initial_states:
        initial_states = rollout_controls(transition, start_state, initial_controls)
    initial_diagnostics = warm_start_diagnostics(
        transition,
        start_state,
        initial_controls,
        initial_states,
        require_feasible=args.initial_feasible and not args.replay_only,
    )
    warm_start_transition_evaluations = transition.evaluations
    terminal_identity = np.eye(transform.shape[0], dtype=np.float64)
    state_half = transform.shape[0] // 2
    terminal_identity[1:state_half, 1:state_half] *= args.terminal_angle_factor
    terminal_identity[state_half, state_half] *= args.terminal_cart_velocity_factor
    terminal_identity[state_half + 1 :, state_half + 1 :] *= args.terminal_hinge_velocity_factor
    terminal_target = branch_terminal_target
    terminal_target_source = None
    if args.terminal_target_json is not None:
        terminal_target, terminal_target_source = load_terminal_target(
            args.terminal_target_json,
            args.terminal_target_index,
            transform.shape[0],
        )
    terminal_physical_cost = add_terminal_cart_weights(
        args.terminal_state_weight * terminal_identity,
        int(cfg["env"]["n_links"]),
        cart_weight=args.terminal_cart_weight,
        cart_velocity_weight=args.terminal_cart_velocity_weight,
    )
    terminal_metric = (args.terminal_weight * lyapunov / args.handoff_lyapunov
                       + cost_mapping.T @ terminal_physical_cost @ cost_mapping)
    terminal_factor = None
    if lyapunov_factor is not None:
        diagonal = args.terminal_state_weight*np.diag(terminal_identity).copy()
        diagonal[0] += args.terminal_cart_weight
        diagonal[state_half] += args.terminal_cart_velocity_weight
        terminal_factor = np.vstack([
            np.sqrt(args.terminal_weight/args.handoff_lyapunov)*lyapunov_factor,
            np.diag(np.sqrt(diagonal)) @ cost_mapping])
        terminal_metric = terminal_factor.T@terminal_factor
    stage_state = args.stage_weight * np.eye(transform.shape[0], dtype=np.float64)
    if args.split_link is not None:
        split_angle = np.zeros(transform.shape[0], dtype=np.float64)
        split_angle[args.split_link] = 1.0
        split_angle[args.split_link + 1] = -1.0
        stage_state += args.split_angle_stage_weight * np.outer(
            split_angle, split_angle
        )
        split_rate = np.zeros(transform.shape[0], dtype=np.float64)
        split_rate[int(cfg["env"]["n_links"]) + 1 + args.split_link] = 1.0
        stage_state += args.split_rate_stage_weight * np.outer(
            split_rate, split_rate
        )
    stage_state = cost_mapping.T @ stage_state @ cost_mapping
    trajectory_cost = QuadraticTrajectoryCost(
        stage_state=stage_state,
        terminal_state=terminal_metric,
        control=float(args.control_cost),
        rail_soft_limit=float(args.rail_soft_limit * transform[0, 0]),
        rail_limit=float(env.rail_limit * transform[0, 0]),
        rail_weight=float(args.rail_weight),
        wrap_angles=False,
        terminal_target=terminal_target,
        stage_target=terminal_target if args.continuous_angles else None,
        terminal_factor=terminal_factor,
    )
    running_costs = [trajectory_cost] * int(initial_controls.size)
    capture_cost = None
    capture_step = None
    if args.capture_start_seconds is not None:
        capture_step = int(round(args.capture_start_seconds / env.dt))
        if capture_step >= initial_controls.size:
            raise ValueError("capture interval starts outside the optimized horizon")
        capture_cost = capture_interval_cost(
            trajectory_cost, terminal_identity, cost_mapping, args.capture_stage_weight,
            lyapunov_factor, args.capture_value_weight, args.handoff_lyapunov)
        running_costs[capture_step:] = [capture_cost] * (initial_controls.size - capture_step)
    optimizer_history = None
    optimization_start_step = int(round(args.optimize_suffix_start_seconds / env.dt))
    if optimization_start_step >= initial_controls.size:
        raise ValueError("suffix optimization starts outside the control horizon")
    if optimization_start_step > 0:
        warm_start_diagnostics(transition, start_state, initial_controls[:optimization_start_step],
                               initial_states[:optimization_start_step+1], require_feasible=True)
        if args.enforce_rail_during_search and np.any(
            np.abs(initial_states[:optimization_start_step+1, 0]) > trajectory_cost.rail_limit
        ):
            raise ValueError("frozen prefix violates the hard rail")
    warm_start_transition_evaluations = transition.evaluations
    if args.replay_only:
        assert source_feedback_gains is not None
        controls = initial_controls.copy()
        nominal_states = replay_nominal_states
        solver_feedback_gains = source_feedback_gains.copy()
        feedback_gains = args.tracking_gain_scale * source_feedback_gains.copy()
        converged = False
        search_seconds = 0.0
        search_iterations = 0
        search_cost = 0.0
        search_stop = 0.0
        search_is_feasible = initial_diagnostics["is_feasible"]
        invalid_transition_count = 0
    elif args.optimizer in {"sqrt-ilqr", "sqrt-fddp", "sparse-scvx"}:
        from gcartpole.sqrt_ilqr import optimize_sqrt_ilqr
        optimize_function = optimize_sqrt_ilqr
        options = dict(enforce_rail=args.enforce_rail_during_search)
        if args.optimizer == "sqrt-fddp":
            from gcartpole.sqrt_fddp import optimize_sqrt_fddp
            optimize_function = optimize_sqrt_fddp
            options.update(initial_states=initial_states[optimization_start_step:], defect_penalty=args.defect_penalty)
        elif args.optimizer == 'sparse-scvx':
            from gcartpole.constrained_shooting import optimize_constrained_shooting
            optimize_function = optimize_constrained_shooting
            options = dict(initial_states=initial_states[optimization_start_step:], defect_penalty=args.defect_penalty,
                           defect_factor=cost_mapping, qp_solver=args.sparse_qp_solver, trust_policy=args.sparse_trust_policy, state_trust=args.sparse_state_trust,
                           control_trust=args.sparse_control_trust, qp_max_iterations=args.sparse_qp_max_iterations,
                           qp_tolerance=args.sparse_qp_tolerance, qp_initial_tolerance=args.sparse_qp_initial_tolerance,
                           qp_inexact_dual_tolerance=args.sparse_qp_inexact_dual_tolerance)
            if args.sparse_capture_angle_limit is not None:
                matrix, lower, upper = capture_node_constraints(
                    transition, terminal_target, capture_step, initial_controls.size, args.sparse_capture_angle_limit)
                options.update(node_constraint_matrix=matrix, node_lower=lower[optimization_start_step:],
                               node_upper=upper[optimization_start_step:])
        started = time.time()
        optimized = optimize_function(
            transition, initial_states[optimization_start_step], initial_controls[optimization_start_step:], cost=trajectory_cost,
            **options,
            max_iterations=args.iterations,
            initial_regularization=args.initial_regularization,
            state_epsilon=args.state_epsilon,
            action_epsilon=args.action_epsilon,
            running_costs=running_costs[optimization_start_step:],
            callback=(lambda row: print(json.dumps(row), flush=True)) if args.solver_verbose else None,
        )
        search_seconds = time.time() - started
        controls, nominal_states = optimized.controls, optimized.states
        solver_feedback_gains = optimized.feedback_gains
        if optimization_start_step > 0:
            assert source_feedback_gains is not None
            from gcartpole.ilqr import stitch_feedback_trajectories
            controls, nominal_states, solver_feedback_gains = stitch_feedback_trajectories(
                initial_controls[:optimization_start_step], initial_states[:optimization_start_step+1],
                source_feedback_gains[:optimization_start_step], controls, nominal_states, solver_feedback_gains,
                boundary_tolerance=0.)
        feedback_gains = args.tracking_gain_scale * solver_feedback_gains
        converged = optimized.converged
        search_iterations = optimized.iterations
        search_cost = optimized.cost
        search_stop = None
        search_is_feasible = warm_start_diagnostics(
            transition, start_state, controls, nominal_states
        )["is_exactly_feasible"]
        optimizer_history = optimized.history
        invalid_transition_count = sum(row.get("invalid_transition_trials", 0) for row in optimizer_history)
    else:
        running_model = MujocoActionModel(transition, trajectory_cost,
                                         state_epsilon=args.state_epsilon,
                                         action_epsilon=args.action_epsilon)
        running_models = [running_model] * int(initial_controls.size)
        capture_model = None
        if capture_cost is not None:
            capture_model = MujocoActionModel(transition, capture_cost,
                                             state_epsilon=args.state_epsilon,
                                             action_epsilon=args.action_epsilon)
            running_models[capture_step:] = [capture_model] * (initial_controls.size - capture_step)
        terminal_model = MujocoActionModel(transition, trajectory_cost, terminal=True)
        problem = crocoddyl.ShootingProblem(
            start_state,
            running_models,
            terminal_model,
        )
        solver = crocoddyl.SolverBoxFDDP(problem)
        if args.fddp_ascent_acceptance is not None:
            solver.th_acceptNegStep = args.fddp_ascent_acceptance
        if args.solver_verbose:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])
        initial_xs = [row.copy() for row in initial_states]
        initial_us = [np.asarray([action], dtype=np.float64) for action in initial_controls]
        started = time.time()
        converged = bool(
            solver.solve(
                initial_xs,
                initial_us,
                args.iterations,
                # Nonzero defects must reach FDDP's gap restoration, even if
                # they pass a reporting tolerance. Unstable chains can amplify
                # a tiny omitted defect across the entire swing horizon.
                initial_diagnostics["is_exactly_feasible"],
                args.initial_regularization,
            )
        )
        search_seconds = time.time() - started
        controls = np.asarray([float(row[0]) for row in solver.us], dtype=np.float64)
        nominal_states = np.asarray(solver.xs, dtype=np.float64)
        solver_feedback_gains = -np.asarray(solver.K, dtype=np.float64).reshape(
            controls.size, transform.shape[0]
        )
        feedback_gains = args.tracking_gain_scale * solver_feedback_gains
        search_iterations = int(solver.iter)
        search_cost = float(solver.cost)
        search_stop = float(solver.stop)
        search_is_feasible = bool(solver.isFeasible)
        invalid_transition_count = int(running_model.invalid_transition_count)
        if capture_model is not None:
            invalid_transition_count += int(capture_model.invalid_transition_count)
    optimization_transition_evaluations = transition.evaluations - warm_start_transition_evaluations
    tracking_feedback_evaluations = 0
    tracking_value_export = None
    if args.recompute_tracking_feedback:
        from gcartpole.sqrt_ilqr import square_root_tracking_gains
        value_factors = None
        if args.export_tracking_value_factors:
            if args.tracking_feedback_decimal_digits < 40:
                raise ValueError('value factor export requires high-precision feedback design')
            if Path(args.export_tracking_value_factors).exists():
                raise FileExistsError(args.export_tracking_value_factors)
            value_factors = [None] * (len(controls)+1)
        before_feedback = transition.evaluations
        solver_feedback_gains = square_root_tracking_gains(
            transition, nominal_states, controls, trajectory_cost,
            running_costs=running_costs, regularization=args.tracking_feedback_regularization,
            state_epsilon=args.state_epsilon, action_epsilon=args.action_epsilon,
            decimal_digits=args.tracking_feedback_decimal_digits,
            callback=(lambda row: print(json.dumps(row), flush=True)) if args.solver_verbose else None,
            value_factor_callback=(lambda step,root:value_factors.__setitem__(step,root)) if value_factors is not None else None)
        feedback_gains = args.tracking_gain_scale * solver_feedback_gains
        tracking_feedback_evaluations = transition.evaluations - before_feedback
        if value_factors is not None:
            if any(factor is None for factor in value_factors):
                raise ValueError('tracking value design omitted a node')
            dump_json(dict(not_solution=True, benchmark_evidence=False,
                generated_at=utc_timestamp(), config=file_metadata(args.config),
                source_controller=file_metadata(args.initial_controller),
                controls_sha256=data_sha256(controls.tolist()),
                nominal_states_sha256=data_sha256(nominal_states.tolist()),
                coordinate_transform=transform.tolist(),
                decimal_digits=args.tracking_feedback_decimal_digits,
                regularization=args.tracking_feedback_regularization,
                control_cost=args.control_cost,
                derivative_order=args.derivative_order,
                state_epsilon=args.state_epsilon, action_epsilon=args.action_epsilon,
                value_factors=[factor.tolist() for factor in value_factors],
                scope='MP finite-horizon perturbation value roots rounded to binary64. Same native derivative inputs. These roots guide experiments; they do not certify nonlinear capture or feasibility.'),
                args.export_tracking_value_factors)
            tracking_value_export = file_metadata(args.export_tracking_value_factors)
    elif args.export_tracking_value_factors:
        raise ValueError('value export requires recomputed tracking feedback')
    final_diagnostics = warm_start_diagnostics(
        transition, start_state, controls, nominal_states
    )
    if args.optimizer.startswith("sqrt-") and not args.replay_only:
        checkpoint_path = Path(args.out).with_name("optimization_checkpoint.json")
        dump_json(dict(not_solution=True, controls=controls.tolist(),
                       nominal_states=nominal_states.tolist(), feedback_gains=feedback_gains.tolist(),
                       optimizer_history=optimizer_history, final_diagnostics=final_diagnostics,
                       cost=search_cost, selected_state=state,
                       coordinate_transform=transform.tolist()), checkpoint_path)
    env.close()

    metric_states = nominal_states
    if args.continuous_angles:
        metric_states = nominal_states - terminal_target
    nominal_values = (np.sum((metric_states@lyapunov_factor.T)**2, axis=1)
                      if lyapunov_factor is not None else np.maximum(
                          0., np.einsum("ij,jk,ik->i", metric_states, lyapunov, metric_states)))
    result = execute_controller(
        cfg,
        progress=args.progress,
        seed=args.seed,
        controls=controls,
        nominal_states=nominal_states,
        feedback_gains=feedback_gains,
        gain=gain,
        lqr_scale=args.lqr_scale,
        transform=transform,
        lyapunov=lyapunov,
        handoff_lyapunov=(
            args.handoff_lyapunov
            if args.switch_lyapunov is None
            else args.switch_lyapunov
        ),
        handoff_cart_abs=args.handoff_cart_abs,
        handoff_angle_abs=args.handoff_angle_abs,
        handoff_cart_velocity_abs=args.handoff_cart_velocity_abs,
        handoff_hinge_velocity_rms=args.handoff_hinge_velocity_rms,
        tracking_mode=args.optimizer.replace("-", "_") + "_tracking",
        defer_handoff_until_horizon=args.defer_handoff_until_horizon,
        phase_adaptive=args.phase_adaptive,
        phase_window=args.phase_window,
        continuous_angles=args.continuous_angles,
        lyapunov_factor=lyapunov_factor,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": f"Single-state exact-MuJoCo {args.optimizer} diagnostic; not P1 evidence.",
        "state_index": int(state_index),
        "selected_state": state,
        "state_interpolation": interpolation,
        "seed": int(args.seed),
        "controller": {
            "type": (
                "exact_mujoco_feedback_replay_then_lqr"
                if args.replay_only
                else ("square_root_ilqr_exact_mujoco_then_lqr" if args.optimizer == "sqrt-ilqr"
                      else ("square_root_fddp_exact_mujoco_then_lqr" if args.optimizer == "sqrt-fddp"
                            else ("sparse_residual_scvx_exact_mujoco_then_lqr" if args.optimizer == 'sparse-scvx'
                                  else "crocoddyl_box_fddp_exact_mujoco_then_lqr")))
            ),
            "initial_controller": (
                None if initial_path is None else file_metadata(initial_path)
            ),
            "terminal_target": terminal_target_source,
            "initial_feasible": initial_diagnostics["is_exactly_feasible"],
            "initial_feasible_within_tolerance": initial_diagnostics["is_feasible"],
            "initial_feasible_requested": bool(args.initial_feasible),
            "initial_solver_feedback_selected": bool(args.initial_solver_feedback),
            "warm_start_diagnostics": initial_diagnostics,
            "final_trajectory_diagnostics": final_diagnostics,
            "invalid_optimizer_transition_trials": invalid_transition_count,
            "transition_evaluation_budget": {
                "warm_start": int(warm_start_transition_evaluations),
                "optimization": int(optimization_transition_evaluations),
                "final_defect_validation": int(controls.size),
                "physics_steps_upper_bound": int(transition.evaluations * cfg["env"]["frame_skip"]),
                "scope": "trajectory transition calls, including derivative and rejected trial evaluations; excludes LQR design and full controller evaluation",
            },
            "replay_only": bool(args.replay_only),
            "rebuilt_initial_states": bool(args.rebuild_initial_states),
            "rebuilt_initial_feedback": bool(args.rebuild_initial_feedback),
            "initial_feedback_scale": float(args.initial_feedback_scale),
            "continuous_angles": bool(args.continuous_angles),
            "coordinate_transform": transform.tolist(),
            "angle_branch_terminal_target": (
                None if branch_terminal_target is None else branch_terminal_target.tolist()
            ),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(
                controls.size * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]
            ),
            "iterations": int(args.iterations),
            "solver_verbose": bool(args.solver_verbose),
            "optimizer": args.optimizer,
            "defect_penalty": args.defect_penalty if args.optimizer == "sqrt-fddp" else None,
            "derivative_order": args.derivative_order,
            "state_epsilon": args.state_epsilon,
            "action_epsilon": args.action_epsilon,
            "physical_shooting": args.physical_shooting,
            "normalized_cost_mapping": cost_mapping.tolist(),
            "fddp_ascent_acceptance": args.fddp_ascent_acceptance,
            "fddp_ascent_acceptance_effective": (
                None if args.replay_only or args.optimizer != "box-fddp"
                else float(solver.th_acceptNegStep)
            ),
            "append_tail_seconds": float(args.append_tail_seconds),
            "initial_regularization": float(args.initial_regularization),
            "tracking_gain_scale": float(args.tracking_gain_scale),
            "tracking_feedback_design": None if not args.recompute_tracking_feedback else dict(
                implementation="pure_ltv_mp_riccati" if args.tracking_feedback_decimal_digits else "pure_ltv_square_root_qr",
                decimal_digits=args.tracking_feedback_decimal_digits,
                derivative_arithmetic="native_binary64_finite_difference",
                replay_gain_arithmetic="binary64",
                regularization=args.tracking_feedback_regularization,
                transition_evaluations=tracking_feedback_evaluations,
                value_factor_export=tracking_value_export,
                nominal_controls_unchanged=True, nominal_states_unchanged=True,
                action_saturation="physical controller clips delivered actions"),
            "lqr_scale": float(args.lqr_scale),
            "lqr_progress": float(lqr_progress),
            "lqr_control_cost": float(args.lqr_control_cost),
            "lqr_decimal_digits": int(args.lqr_decimal_digits),
            "lqr_weights": {key: float(value) for key, value in lqr_weights.items()},
            "control_cost": float(args.control_cost),
            "stage_weight": float(args.stage_weight),
            "split_link": (
                None if args.split_link is None else int(args.split_link)
            ),
            "split_angle_stage_weight": float(args.split_angle_stage_weight),
            "split_rate_stage_weight": float(args.split_rate_stage_weight),
            "terminal_weight": float(args.terminal_weight),
            "terminal_state_weight": float(args.terminal_state_weight),
            "terminal_cart_weight": float(args.terminal_cart_weight),
            "terminal_cart_velocity_weight": float(args.terminal_cart_velocity_weight),
            "terminal_cart_velocity_factor": float(args.terminal_cart_velocity_factor),
            "terminal_hinge_velocity_factor": float(args.terminal_hinge_velocity_factor),
            "terminal_angle_factor": float(args.terminal_angle_factor),
            "rail_soft_limit": float(args.rail_soft_limit),
            "rail_weight": float(args.rail_weight),
            "enforce_rail_during_search": bool(args.enforce_rail_during_search or args.optimizer == 'sparse-scvx'),
            "sparse_qp_parameters": None if args.optimizer != 'sparse-scvx' else dict(
                solver=args.sparse_qp_solver, trust_policy=args.sparse_trust_policy,
                formulation='explicit_cost_residuals_and_l1_virtual_control',
                state_trust=args.sparse_state_trust, control_trust=args.sparse_control_trust,
                qp_max_iterations=args.sparse_qp_max_iterations, qp_tolerance=args.sparse_qp_tolerance,
                qp_initial_tolerance=args.sparse_qp_initial_tolerance,
                qp_inexact_dual_tolerance=args.sparse_qp_inexact_dual_tolerance,
                virtual_capture_angle_limit=args.sparse_capture_angle_limit,
                defect_factor_sha256=data_sha256(cost_mapping.tolist())),
            "optimize_suffix_start_seconds": args.optimize_suffix_start_seconds,
            "frozen_prefix_steps": optimization_start_step,
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "switch_lyapunov": (
                float(args.handoff_lyapunov)
                if args.switch_lyapunov is None
                else float(args.switch_lyapunov)
            ),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "handoff_angle_abs": float(args.handoff_angle_abs),
            "handoff_cart_velocity_abs": float(args.handoff_cart_velocity_abs),
            "handoff_hinge_velocity_rms": float(args.handoff_hinge_velocity_rms),
            "defer_handoff_until_horizon": bool(args.defer_handoff_until_horizon),
            "phase_adaptive": bool(args.phase_adaptive),
            "phase_window": int(args.phase_window),
            "capture_start_seconds": args.capture_start_seconds,
            "tail_initializer": args.tail_initializer,
            "capture_stage_weight": args.capture_stage_weight,
            "capture_value_weight": args.capture_value_weight,
            "capture_interval_steps": None if capture_step is None else int(initial_controls.size-capture_step),
            "capture_stage_factor_sha256": None if capture_cost is None else data_sha256(capture_cost.stage_factor.tolist()),
            "prefix_control_scale": float(args.prefix_control_scale),
            "prefix_control_seconds": float(args.prefix_control_seconds),
            "allow_unstable_lyapunov": bool(args.allow_unstable_lyapunov),
            "lqr_fallback_reason": lqr_fallback_reason,
            "progress": float(args.progress),
            "controls": controls.astype(float).tolist(),
            "feedback_gains": feedback_gains.astype(float).tolist(),
            "solver_feedback_gains": solver_feedback_gains.astype(float).tolist(),
        },
        "search": {
            "converged": converged,
            "progress": float(args.progress),
            "iterations": search_iterations,
            "optimizer_history": optimizer_history,
            "cost": search_cost,
            "stopping_criterion": search_stop,
            "is_feasible": search_is_feasible,
            "wall_time_seconds": float(search_seconds),
            "optimized_horizon_steps": int(initial_controls.size-optimization_start_step),
            "cost_scope": "optimized_suffix" if optimization_start_step else "full_trajectory",
            "initial_lyapunov": float(nominal_values[0]),
            "minimum_lyapunov": float(np.min(nominal_values)),
            "terminal_lyapunov": float(nominal_values[-1]),
            "nominal_coordinate_states": nominal_states.astype(float).tolist(),
            "nominal_max_cart_excursion": float(np.max(np.abs(nominal_states[:, 0] / transform[0, 0]))),
            "nominal_rail_admissible": bool(np.all(np.abs(nominal_states[:, 0] / transform[0, 0]) <= env.rail_limit)),
        },
        "lyapunov": {
            "coordinate_source": file_metadata(args.spec),
            "source": lyapunov_source,
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
            "factor": None if lyapunov_factor is None else lyapunov_factor.tolist(),
            "factor_sha256": None if lyapunov_factor is None else data_sha256(lyapunov_factor.tolist()),
            "precision_diagnostics": lyapunov_precision_diagnostics,
        },
        "result": result,
        "evidence": {
            "config": {
                "path": str(Path(args.config)),
                "resolved_sha256": data_sha256(cfg),
                "progress": float(args.progress),
            },
            "state_source": file_metadata(args.state_json),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"converged={converged} feasible={search_is_feasible} "
        f"iterations={search_iterations} min_v={np.min(nominal_values):.2f} "
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
