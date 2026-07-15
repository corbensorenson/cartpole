#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import (
    MujocoTransition,
    QuadraticTrajectoryCost,
    optimize_ilqr,
    stitch_feedback_trajectories,
)
from gcartpole.modal import (
    StateScales,
    closed_loop_lyapunov_matrix,
    dimensionless_absolute_transform,
)

try:
    from scripts.evaluate_feedback_mpc_capture import (
        feedback_action,
        schedule_value,
        search_feedback_plan,
    )
    from scripts.make_lqr_checkpoint import finite_difference_dynamics
    from scripts.search_capture_sequence import fixed_state_cfg
    from scripts.search_ilqr_capture import execute_controller, initial_controls
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from make_lqr_checkpoint import finite_difference_dynamics
    from evaluate_feedback_mpc_capture import (
        feedback_action,
        schedule_value,
        search_feedback_plan,
    )
    from search_capture_sequence import fixed_state_cfg
    from search_ilqr_capture import execute_controller, initial_controls
    from search_swingup_capture import lqr_gain


def source_trajectory(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    controls = np.asarray(payload["controller"]["controls"], dtype=np.float64)
    states = np.asarray(payload["search"]["nominal_coordinate_states"], dtype=np.float64)
    gains = np.asarray(payload["controller"]["feedback_gains"], dtype=np.float64)
    if states.ndim != 2:
        raise ValueError("source controller states must be two-dimensional")
    # Reuse the stitch validator so malformed source evidence cannot enter a chain.
    empty_controls = np.empty(0, dtype=np.float64)
    empty_gains = np.empty((0, states.shape[1]), dtype=np.float64)
    validated_controls, validated_states, validated_gains = stitch_feedback_trajectories(
        controls,
        states,
        gains,
        empty_controls,
        states[-1:],
        empty_gains,
    )
    return validated_controls, validated_states, validated_gains


def warm_start_controls(
    path: str | Path,
    *,
    offset_steps: int,
    horizon_steps: int,
) -> np.ndarray:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    saved_controls = payload.get("controller", {}).get("controls")
    if saved_controls is None:
        saved_controls = [row["action"] for row in payload.get("result", {}).get("trajectory", [])]
    controls = np.asarray(saved_controls, dtype=np.float64)
    if controls.ndim != 1 or controls.size <= offset_steps:
        raise ValueError("tail warm start has no scalar controls at the requested offset")
    selected = controls[offset_steps : offset_steps + horizon_steps]
    if selected.size < horizon_steps:
        selected = np.pad(selected, (0, horizon_steps - selected.size))
    return np.clip(selected, -1.0, 1.0)


def controls_from_knots(knots: np.ndarray, horizon_steps: int) -> np.ndarray:
    source_time = np.linspace(0.0, 1.0, len(knots), dtype=np.float64)
    target_time = np.linspace(0.0, 1.0, horizon_steps, dtype=np.float64)
    return np.clip(np.interp(target_time, source_time, knots), -1.0, 1.0)


def search_action_tail(
    transition: MujocoTransition,
    initial_state: np.ndarray,
    lyapunov: np.ndarray,
    *,
    horizon_steps: int,
    knot_count: int,
    iterations: int,
    population: int,
    elites: int,
    seed: int,
    handoff_lyapunov: float,
    handoff_cart_abs: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    center = np.zeros(knot_count, dtype=np.float64)
    sigma = np.full(knot_count, 0.7, dtype=np.float64)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for iteration in range(iterations):
        candidates = [center.copy(), np.zeros_like(center)]
        candidates.extend(center + rng.normal(0.0, sigma) for _ in range(population - 2))
        records: list[dict[str, Any]] = []
        for candidate in candidates:
            knots = np.clip(candidate, -1.0, 1.0)
            controls = controls_from_knots(knots, horizon_steps)
            state = initial_state.copy()
            values = [float(max(0.0, state @ lyapunov @ state))]
            max_cart_abs = abs(float(transition.to_physical(state)[0]))
            latched = values[0] <= handoff_lyapunov and max_cart_abs <= handoff_cart_abs
            rail_hit = False
            action_energy = 0.0
            for action in controls:
                action_energy += float(action * action)
                state = transition(state, float(action))
                if not np.all(np.isfinite(state)):
                    rail_hit = True
                    break
                physical = transition.to_physical(state)
                cart_abs = abs(float(physical[0]))
                max_cart_abs = max(max_cart_abs, cart_abs)
                rail_hit = rail_hit or cart_abs > transition.env.rail_limit
                value = float(max(0.0, state @ lyapunov @ state))
                values.append(value)
                latched = latched or (value <= handoff_lyapunov and cart_abs <= handoff_cart_abs)
                if rail_hit:
                    break
            minimum_value = min(values)
            terminal_value = values[-1]
            score = (
                10_000_000.0 * float(rail_hit)
                + 3_000.0 * float(np.log1p(terminal_value))
                + 300.0 * float(np.log1p(minimum_value))
                + 2.0 * float(np.sum(np.log1p(values)))
                + 2.0 * action_energy
                - 50_000.0 * float(latched and not rail_hit)
            )
            records.append(
                {
                    "score": float(score),
                    "knots": knots.astype(float).tolist(),
                    "minimum_lyapunov": float(minimum_value),
                    "terminal_lyapunov": float(terminal_value),
                    "latched": bool(latched),
                    "rail_hit": bool(rail_hit),
                    "max_cart_abs": float(max_cart_abs),
                }
            )
        records.sort(key=lambda row: float(row["score"]))
        if best is None or float(records[0]["score"]) < float(best["score"]):
            best = records[0]
        history.append(
            {
                "iteration": iteration + 1,
                **{
                    key: records[0][key]
                    for key in (
                        "score",
                        "minimum_lyapunov",
                        "terminal_lyapunov",
                        "latched",
                        "rail_hit",
                        "max_cart_abs",
                    )
                },
            }
        )
        elite = np.asarray([row["knots"] for row in records[:elites]], dtype=np.float64)
        center = elite.mean(axis=0)
        sigma = np.maximum(0.90 * elite.std(axis=0), 0.03)
    assert best is not None
    return {"best": best, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append a constrained nonlinear settling tail to an existing iLQR approach"
    )
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--source-controller", required=True)
    parser.add_argument("--seed", type=int, default=62601)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tail-seconds", type=float, default=1.5)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--initial-lqr-scale", type=float, default=0.0)
    parser.add_argument("--initial-controller", default=None)
    parser.add_argument("--initial-controller-offset-steps", type=int, default=0)
    parser.add_argument("--initial-feedback-cem", action="store_true")
    parser.add_argument("--initial-action-cem", action="store_true")
    parser.add_argument("--action-knots", type=int, default=24)
    parser.add_argument("--cem-iterations", type=int, default=8)
    parser.add_argument("--cem-population", type=int, default=256)
    parser.add_argument("--cem-elites", type=int, default=24)
    parser.add_argument("--planning-lqr-scale", type=float, default=1.30)
    parser.add_argument("--lqr-scale", type=float, default=1.30)
    parser.add_argument("--control-cost", type=float, default=0.1)
    parser.add_argument("--stage-weight", type=float, default=0.1)
    parser.add_argument("--terminal-weight", type=float, default=10_000.0)
    parser.add_argument("--terminal-state-weight", type=float, default=100_000.0)
    parser.add_argument("--rail-soft-limit", type=float, default=2.4)
    parser.add_argument("--rail-weight", type=float, default=100_000_000.0)
    parser.add_argument("--handoff-lyapunov", type=float, default=1800.0)
    parser.add_argument("--handoff-cart-abs", type=float, default=1.5)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    positive = (
        args.tail_seconds,
        args.control_cost,
        args.stage_weight,
        args.terminal_weight,
        args.terminal_state_weight,
        args.rail_soft_limit,
        args.rail_weight,
        args.handoff_lyapunov,
        args.handoff_cart_abs,
    )
    if min(positive) <= 0.0 or args.iterations < 0 or args.initial_controller_offset_steps < 0:
        raise ValueError(
            "durations, weights, and thresholds must be positive; iterations nonnegative"
        )
    initialization_count = sum(
        (
            args.initial_feedback_cem,
            args.initial_action_cem,
            args.initial_controller is not None,
        )
    )
    if initialization_count > 1:
        raise ValueError("choose only one CEM or initial-controller warm start")
    if (
        args.cem_iterations < 1
        or args.cem_population < 2
        or not 1 <= args.cem_elites <= args.cem_population
    ):
        raise ValueError("invalid CEM iterations, population, or elite count")
    if args.action_knots < 2:
        raise ValueError("action CEM requires at least two knots")

    source_path = Path(args.source_controller)
    source_payload = json.loads(source_path.read_text(encoding="utf-8"))
    source_controls, source_states, source_gains = source_trajectory(source_payload)
    if "selected_state" not in source_payload:
        raise ValueError("source controller does not identify its selected initial state")

    base_cfg = apply_overrides(load_config(args.config), args.override)
    base_cfg["env"]["action_lqr_residual"]["enabled"] = False
    cfg = fixed_state_cfg(
        base_cfg,
        source_payload["selected_state"],
        float(base_cfg["env"]["episode_seconds"]),
    )
    source_config_hash = source_payload.get("evidence", {}).get("config", {}).get("resolved_sha256")
    resolved_config_hash = data_sha256(cfg)
    if source_config_hash is not None and source_config_hash != resolved_config_hash:
        raise ValueError(
            "source controller resolved config hash differs from the requested refinement config"
        )

    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
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
    if source_states.shape[1] != transform.shape[0]:
        raise ValueError("source controller state dimension differs from the requested plant")
    state_matrix, input_matrix = finite_difference_dynamics(cfg, 1.0, 1e-7)
    lyapunov, spectral_radius = closed_loop_lyapunov_matrix(
        state_matrix,
        input_matrix,
        gain,
        transform,
        feedback_scale=args.lqr_scale,
    )

    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    env.reset(seed=args.seed)
    transition = MujocoTransition(env, coordinate_transform=transform)
    policy_dt = float(env.dt)
    source_seconds = source_controls.size * policy_dt
    declared_source_seconds = float(source_payload["controller"]["horizon_seconds"])
    if not np.isclose(source_seconds, declared_source_seconds, atol=1e-9):
        raise ValueError("source controller action frequency differs from the requested plant")
    tail_steps = max(2, int(round(args.tail_seconds / policy_dt)))
    tail_initial_state = source_states[-1].copy()
    cem_search: dict[str, Any] | None = None
    initialization_started = time.time()
    if args.initial_action_cem:
        cem_search = search_action_tail(
            transition,
            tail_initial_state,
            lyapunov,
            horizon_steps=tail_steps,
            knot_count=args.action_knots,
            iterations=args.cem_iterations,
            population=args.cem_population,
            elites=args.cem_elites,
            seed=args.seed + 1,
            handoff_lyapunov=args.handoff_lyapunov,
            handoff_cart_abs=args.handoff_cart_abs,
        )
        tail_initial_controls = controls_from_knots(
            np.asarray(cem_search["best"]["knots"], dtype=np.float64), tail_steps
        )
    elif args.initial_feedback_cem:
        target_count = 6
        residual_count = 8
        scratch = mujoco.MjData(env.model)
        tail_physical_state = transition.to_physical(tail_initial_state)
        nq = int(env.model.nq)
        cem_search = search_feedback_plan(
            env,
            scratch,
            start_qpos=tail_physical_state[:nq],
            start_qvel=tail_physical_state[nq:],
            gain=gain,
            lqr_scale=args.lqr_scale,
            planning_lqr_scale=args.planning_lqr_scale,
            transform=transform,
            lyapunov=lyapunov,
            initial_controller={
                "target_knots": np.zeros(target_count, dtype=np.float64).tolist(),
                "residual_knots": np.zeros(residual_count, dtype=np.float64).tolist(),
            },
            rng=np.random.default_rng(args.seed + 1),
            horizon_steps=tail_steps,
            horizon_seconds=tail_steps * policy_dt,
            handoff_lyapunov=args.handoff_lyapunov,
            handoff_cart_abs=args.handoff_cart_abs,
            iterations=args.cem_iterations,
            population=args.cem_population,
            elites=args.cem_elites,
            target_count=target_count,
            residual_count=residual_count,
            target_limit=1.5,
            residual_limit=0.4,
            target_sigma=0.5,
            residual_sigma=0.15,
        )
        best_controller = cem_search["best"]["controller"]
        target_knots = np.asarray(best_controller["target_knots"], dtype=np.float64)
        residual_knots = np.asarray(best_controller["residual_knots"], dtype=np.float64)
        tail_initial_controls = np.empty(tail_steps, dtype=np.float64)
        current = tail_initial_state.copy()
        for step in range(tail_steps):
            physical = transition.to_physical(current)
            target = schedule_value(step * policy_dt, target_knots, tail_steps * policy_dt)
            residual = schedule_value(step * policy_dt, residual_knots, tail_steps * policy_dt)
            tail_initial_controls[step] = feedback_action(
                physical[:nq],
                physical[nq:],
                gain,
                n_links=env.n,
                scale=args.planning_lqr_scale,
                cart_target=target,
                residual=residual,
            )
            current = transition(current, float(tail_initial_controls[step]))
    elif args.initial_controller is None:
        tail_initial_controls = initial_controls(
            transition,
            tail_initial_state,
            gain,
            horizon_steps=tail_steps,
            lqr_scale=args.initial_lqr_scale,
        )
    else:
        tail_initial_controls = warm_start_controls(
            args.initial_controller,
            offset_steps=args.initial_controller_offset_steps,
            horizon_steps=tail_steps,
        )
    trajectory_cost = QuadraticTrajectoryCost(
        stage_state=args.stage_weight * np.eye(transform.shape[0], dtype=np.float64),
        terminal_state=(
            args.terminal_weight * lyapunov / args.handoff_lyapunov
            + args.terminal_state_weight * np.eye(transform.shape[0], dtype=np.float64)
        ),
        control=float(args.control_cost),
        rail_soft_limit=float(args.rail_soft_limit * transform[0, 0]),
        rail_limit=float(env.rail_limit * transform[0, 0]),
        rail_weight=float(args.rail_weight),
        wrap_angles=False,
    )
    started = time.time()
    tail = optimize_ilqr(
        transition,
        tail_initial_state,
        tail_initial_controls,
        trajectory_cost,
        max_iterations=args.iterations,
    )
    search_seconds = time.time() - started
    env.close()

    controls, nominal_states, feedback_gains = stitch_feedback_trajectories(
        source_controls,
        source_states,
        source_gains,
        tail.controls,
        tail.states,
        tail.feedback_gains,
    )
    nominal_values = np.maximum(
        0.0, np.einsum("ij,jk,ik->i", nominal_states, lyapunov, nominal_states)
    )
    tail_values = nominal_values[source_controls.size :]
    nominal_physical_states = np.asarray(
        [transition.to_physical(row) for row in nominal_states], dtype=np.float64
    )
    result = execute_controller(
        cfg,
        seed=args.seed,
        controls=controls,
        nominal_states=nominal_states,
        feedback_gains=feedback_gains,
        gain=gain,
        lqr_scale=args.lqr_scale,
        transform=transform,
        lyapunov=lyapunov,
        handoff_lyapunov=args.handoff_lyapunov,
        handoff_cart_abs=args.handoff_cart_abs,
        tracking_mode="ilqr_chain_tracking",
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": (
            "Single-state reset-free constrained iLQR approach-plus-settling diagnostic; "
            "not P1 evidence."
        ),
        "state_index": source_payload.get("state_index"),
        "selected_state": source_payload["selected_state"],
        "seed": int(args.seed),
        "controller": {
            "type": "exact_mujoco_stitched_box_constrained_ilqr_tracking_then_lqr",
            "reset_at_boundary": False,
            "source_controller": file_metadata(source_path),
            "source_horizon_steps": int(source_controls.size),
            "tail_horizon_steps": int(tail_steps),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(controls.size * policy_dt),
            "iterations": int(args.iterations),
            "initial_lqr_scale": float(args.initial_lqr_scale),
            "initial_controller": (
                None if args.initial_controller is None else file_metadata(args.initial_controller)
            ),
            "initial_controller_offset_steps": int(args.initial_controller_offset_steps),
            "initial_feedback_cem": bool(args.initial_feedback_cem),
            "initial_action_cem": bool(args.initial_action_cem),
            "action_knots": int(args.action_knots),
            "planning_lqr_scale": float(args.planning_lqr_scale),
            "cem_iterations": int(args.cem_iterations),
            "cem_population": int(args.cem_population),
            "cem_elites": int(args.cem_elites),
            "lqr_scale": float(args.lqr_scale),
            "control_cost": float(args.control_cost),
            "stage_weight": float(args.stage_weight),
            "terminal_weight": float(args.terminal_weight),
            "terminal_state_weight": float(args.terminal_state_weight),
            "rail_soft_limit": float(args.rail_soft_limit),
            "rail_weight": float(args.rail_weight),
            "handoff_lyapunov": float(args.handoff_lyapunov),
            "handoff_cart_abs": float(args.handoff_cart_abs),
            "controls": controls.astype(float).tolist(),
            "feedback_gains": feedback_gains.astype(float).tolist(),
        },
        "search": {
            "tail_cost": float(tail.cost),
            "tail_iterations": int(tail.iterations),
            "tail_converged": bool(tail.converged),
            "tail_active_control_steps": int(tail.active_control_steps),
            "wall_time_seconds": float(search_seconds),
            "initialization_wall_time_seconds": float(started - initialization_started),
            "boundary_lyapunov": float(tail_values[0]),
            "tail_minimum_lyapunov": float(np.min(tail_values)),
            "tail_terminal_lyapunov": float(tail_values[-1]),
            "tail_terminal_dimensionless_state_norm": float(np.linalg.norm(tail.states[-1])),
            "max_cart_excursion": float(np.max(np.abs(nominal_physical_states[:, 0]))),
            "tail_history": tail.history,
            "cem_search": cem_search,
            "nominal_coordinate_states": nominal_states.astype(float).tolist(),
            "nominal_physical_states": nominal_physical_states.astype(float).tolist(),
        },
        "lyapunov": {
            "coordinate_source": file_metadata(args.spec),
            "closed_loop_spectral_radius": float(spectral_radius),
            "matrix_sha256": data_sha256(lyapunov.astype(float).tolist()),
        },
        "result": result,
        "evidence": {
            "config": {
                "path": str(Path(args.config)),
                "resolved_sha256": resolved_config_hash,
            },
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"boundary_v={tail_values[0]:.2f} tail_min_v={np.min(tail_values):.2f} "
        f"tail_terminal_v={tail_values[-1]:.2f} wall={search_seconds:.1f}s"
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
