#!/usr/bin/env python
"""Search a canonical hanging-start route into a measured handoff-state bank.

The search is intentionally separate from the generic upright CEM.  It uses
the exact environment step loop, but ranks late states by distance to real
saved handoff states so the swing expert is optimized for the capture expert's
measured basin rather than for a one-frame upright crossing.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.generalized_solver import periodic_coordinate_error
from gcartpole.modal import (
    StateScales,
    dimensionless_absolute_transform,
    dimensionless_wrapped_state,
)

try:
    from scripts.search_swingup_global_exact_cem import (
        interpolation_matrix,
        load_initial_center,
    )
except ModuleNotFoundError:
    from search_swingup_global_exact_cem import interpolation_matrix, load_initial_center


ROOT = Path(__file__).resolve().parents[1]


def load_handoff_targets(path: str, n_links: int) -> tuple[np.ndarray, dict[str, Any]]:
    """Load bank states as [cart, cart-rate, absolute-angles, absolute-rates]."""

    bank_path = Path(path)
    payload = json.loads(bank_path.read_text(encoding="utf-8"))
    states = payload.get("states")
    if not isinstance(states, list) or not states:
        raise ValueError("handoff bank must contain a nonempty states list")
    targets: list[np.ndarray] = []
    for state in states:
        qpos = np.asarray(state.get("qpos", []), dtype=np.float64)
        qvel = np.asarray(state.get("qvel", []), dtype=np.float64)
        if qpos.shape != (n_links + 1,) or qvel.shape != (n_links + 1,):
            raise ValueError("handoff bank state dimension does not match the target plant")
        absolute_angles = np.asarray(
            state.get("absolute_angles", serial_absolute_angles(qpos[1:])),
            dtype=np.float64,
        )
        absolute_rates = np.cumsum(qvel[1:])
        if absolute_angles.shape != (n_links,) or not np.all(np.isfinite(absolute_angles)):
            raise ValueError("handoff bank absolute angles are invalid")
        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(qvel)):
            raise ValueError("handoff bank contains non-finite state values")
        targets.append(
            np.concatenate(
                ([qpos[0], qvel[0]], absolute_angles, absolute_rates)
            )
        )
    target_array = np.asarray(targets, dtype=np.float64)
    return target_array, {
        "path": str(bank_path),
        "sha256": data_sha256(payload),
        "file": file_metadata(bank_path),
        "state_count": int(target_array.shape[0]),
    }


def load_full_resolution_center(
    path: str | None,
    step_count: int,
    seconds: float,
    env: NLinkCartPoleEnv,
) -> np.ndarray:
    """Prefer an uninterrupted trace over a truncated swing-only control field."""

    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        result = payload.get("result") if isinstance(payload, dict) else None
        trace = result.get("trajectory") if isinstance(result, dict) else None
        if isinstance(trace, list) and trace and all("action" in row for row in trace):
            actions = np.asarray([float(row["action"]) for row in trace], dtype=np.float64)
            times = np.asarray(
                [
                    float(row.get("time_seconds", (index + 1) * env.dt))
                    for index, row in enumerate(trace)
                ],
                dtype=np.float64,
            )
            finite = np.isfinite(actions) & np.isfinite(times)
            actions = actions[finite]
            times = times[finite]
            if actions.size and np.all(np.diff(times) >= 0.0):
                times = np.concatenate(([0.0], times))
                actions = np.concatenate(([actions[0]], actions))
                target_times = np.arange(step_count, dtype=np.float64) * env.dt
                return np.clip(
                    np.interp(
                        target_times,
                        times,
                        actions,
                        left=actions[0],
                        right=actions[-1],
                    ),
                    -1.0,
                    1.0,
                )
    return load_initial_center(path, step_count, seconds, env)


def state_features(env: NLinkCartPoleEnv) -> np.ndarray:
    """Return the same physical feature ordering used for bank targets."""

    absolute_angles = env._angles()[1]
    absolute_rates = env._absolute_angular_velocity()
    return np.concatenate(
        (
            [float(env.data.qpos[0]), float(env.data.qvel[0])],
            absolute_angles,
            absolute_rates,
        )
    )


def handoff_distance(
    features: np.ndarray,
    targets: np.ndarray,
    scales: np.ndarray,
) -> float:
    """Return the nearest normalized RMS distance to any measured handoff."""

    normalized = (targets - np.asarray(features, dtype=np.float64)[None, :]) / scales
    distances = np.sqrt(np.mean(normalized * normalized, axis=1))
    return float(np.min(distances))


def load_feedback_route(path: str | Path, n_links: int) -> dict[str, Any]:
    """Load a time-varying feedback route for residual search."""

    route_path = Path(path)
    payload = json.loads(route_path.read_text(encoding="utf-8"))
    controller = payload.get("controller") if isinstance(payload, dict) else None
    search = payload.get("search") if isinstance(payload, dict) else None
    if not isinstance(controller, dict) or not isinstance(search, dict):
        raise ValueError("feedback route must contain controller and search objects")
    controls = np.asarray(controller.get("controls", []), dtype=np.float64)
    feedback = np.asarray(controller.get("feedback_gains", []), dtype=np.float64)
    nominal = np.asarray(search.get("nominal_coordinate_states", []), dtype=np.float64)
    state_dim = 2 * (n_links + 1)
    if controls.ndim != 1 or controls.size < 2:
        raise ValueError("feedback route controls must be a nonempty vector")
    if feedback.shape != (controls.size, state_dim):
        raise ValueError("feedback route gains do not match the target state dimension")
    if nominal.shape != (controls.size + 1, state_dim):
        raise ValueError("feedback route nominal states do not match the controls")
    transform = dimensionless_absolute_transform(
        n_links,
        StateScales(
            cart_position=1.25,
            absolute_angle=0.15,
            cart_velocity=0.50,
            hinge_velocity=0.75,
        ),
    )
    return {
        "path": str(route_path),
        "sha256": data_sha256(payload),
        "controls": np.clip(controls, -1.0, 1.0),
        "feedback_gains": feedback,
        "nominal_coordinate_states": nominal,
        "transform": transform,
        "periodic_coordinate_errors": bool(controller.get("periodic_coordinate_errors", False)),
        "phase_cursor": 0,
    }


def reset_feedback_route(route: dict[str, Any]) -> None:
    route["phase_cursor"] = 0


def feedback_route_action(
    env: NLinkCartPoleEnv,
    route: dict[str, Any],
    *,
    tracking_gain_scale: float,
    phase_window: int,
) -> tuple[float, int | None]:
    """Apply route feedback at the current live state and advance its phase."""

    controls = route["controls"]
    cursor = int(route["phase_cursor"])
    if cursor >= controls.size:
        return 0.0, None
    coordinate_state = dimensionless_wrapped_state(
        np.asarray(env.data.qpos, dtype=np.float64),
        np.asarray(env.data.qvel, dtype=np.float64),
        route["transform"],
    )
    nominal = route["nominal_coordinate_states"]
    upper = min(controls.size, cursor + max(0, int(phase_window)) + 1)
    candidates = np.arange(cursor, upper, dtype=np.int64)
    candidate_errors = np.asarray(
        [
            periodic_coordinate_error(
                coordinate_state,
                nominal[index],
                route["transform"],
            )
            if route["periodic_coordinate_errors"]
            else coordinate_state - nominal[index]
            for index in candidates
        ],
        dtype=np.float64,
    )
    route_index = int(candidates[int(np.argmin(np.einsum("ij,ij->i", candidate_errors, candidate_errors)))])
    route["phase_cursor"] = route_index + 1
    error = (
        periodic_coordinate_error(
            coordinate_state,
            nominal[route_index],
            route["transform"],
        )
        if route["periodic_coordinate_errors"]
        else coordinate_state - nominal[route_index]
    )
    action = controls[route_index] + float(tracking_gain_scale) * float(
        route["feedback_gains"][route_index] @ error
    )
    return float(np.clip(action, -1.0, 1.0)), route_index


def evaluate_candidate(
    env: NLinkCartPoleEnv,
    actions: np.ndarray,
    *,
    target_bank: np.ndarray,
    target_scales: np.ndarray,
    handoff_start_step: int,
    rolling_window: int,
    target_weight: float,
    terminal_weight: float,
    energy_weight: float,
    rail_penalty_weight: float,
    cart_position_weight: float = 0.0,
    feedback_route: dict[str, Any] | None = None,
    tracking_gain_scale: float = 1.0,
    phase_window: int = 12,
) -> tuple[float, dict[str, Any]]:
    env.reset(seed=0)
    if feedback_route is not None:
        reset_feedback_route(feedback_route)
    distances: list[float] = []
    energy_scores: list[float] = []
    states: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    termination_reason: str | None = None
    previous_action = 0.0
    action_smooth_cost = 0.0
    max_cart = 0.0
    applied_actions: list[float] = []
    route_indices: list[int | None] = []
    cart_position_cost = 0.0
    for step, action in enumerate(actions):
        residual = float(np.clip(action, -1.0, 1.0))
        if feedback_route is None:
            route_action = 0.0
            route_index = None
            action = residual
        else:
            route_action, route_index = feedback_route_action(
                env,
                feedback_route,
                tracking_gain_scale=tracking_gain_scale,
                phase_window=phase_window,
            )
            action = float(np.clip(route_action + residual, -1.0, 1.0))
        action_smooth_cost += (action - previous_action) ** 2
        previous_action = action
        applied_actions.append(action)
        route_indices.append(route_index)
        _, _, terminated, truncated, info = env.step([action])
        features = state_features(env)
        absolute_angles = env._angles()[1]
        distance = handoff_distance(features, target_bank, target_scales)
        # This is only a discovery guide before the handoff window. It rewards
        # upward energy without making an early angle crossing admissible.
        energy_score = float(np.mean(1.0 + np.cos(absolute_angles)))
        distances.append(distance)
        energy_scores.append(energy_score)
        max_cart = max(max_cart, abs(float(env.data.qpos[0])))
        cart_position_cost += (float(env.data.qpos[0]) / max(1e-9, env.rail_limit)) ** 2
        states.append(
            {
                "step": int(step + 1),
                "time_seconds": float((step + 1) * env.dt),
                "distance": float(distance),
                "energy_score": energy_score,
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "cart": float(env.data.qpos[0]),
                "cart_velocity": float(env.data.qvel[0]),
            }
        )
        if terminated or truncated:
            termination_reason = str(info.get("termination_reason"))
            break

    if not distances:
        return float("inf"), {"invalid": True, "termination_reason": termination_reason}

    distances_array = np.asarray(distances, dtype=np.float64)
    energy_array = np.asarray(energy_scores, dtype=np.float64)
    late = distances_array[handoff_start_step:]
    if late.size == 0:
        return float("inf"), {"invalid": True, "termination_reason": "short_horizon"}
    window = min(max(1, rolling_window), late.size)
    rolling = np.asarray(
        [
            np.mean(late[index : index + window])
            for index in range(late.size - window + 1)
        ],
        dtype=np.float64,
    )
    best_offset = int(np.argmin(rolling))
    best_index = handoff_start_step + best_offset + window - 1
    best_distance = float(rolling[best_offset])
    terminal_distance = float(distances_array[-1])
    peak_energy = float(np.max(energy_array[: max(1, handoff_start_step)]))
    rail_excess = max(0.0, max_cart - float(env.rail_limit))
    termination_penalty = 0.0
    if terminated and not truncated:
        termination_penalty = 1000.0 + 1000.0 * rail_excess
    # A late target distance is the main objective. The early energy term is
    # deliberately weak: it only breaks the zero-input/hanging attractor.
    cost = (
        float(target_weight) * best_distance
        + float(terminal_weight) * terminal_distance
        - float(energy_weight) * peak_energy
        + float(rail_penalty_weight) * rail_excess * rail_excess
        + float(cart_position_weight) * cart_position_cost / max(1, len(states))
        + termination_penalty
        + 0.01 * action_smooth_cost
    )
    best_state = states[best_index]
    return cost, {
        "invalid": bool(terminated and not truncated),
        "termination_reason": termination_reason or "time_limit",
        "steps": len(states),
        "best_index": int(best_index),
        "best_time_seconds": float((best_index + 1) * env.dt),
        "best_distance": best_distance,
        "terminal_distance": terminal_distance,
        "peak_pre_handoff_energy_score": peak_energy,
        "max_cart": float(max_cart),
        "action_smooth_cost": float(action_smooth_cost),
        "mean_cart_position_cost": float(cart_position_cost / max(1, len(states))),
        "applied_controls": applied_actions,
        "route_indices": route_indices,
        "best_state": best_state,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup11_uniform.yaml")
    parser.add_argument("--handoff-bank-json", required=True)
    parser.add_argument("--seconds", type=float, default=16.0)
    parser.add_argument("--knot-count", type=int, default=48)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--handoff-start-time", type=float, default=8.0)
    parser.add_argument("--rolling-window", type=int, default=12)
    parser.add_argument("--action-sigma", type=float, default=0.35)
    parser.add_argument("--sigma-decay", type=float, default=0.92)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--target-weight", type=float, default=100.0)
    parser.add_argument("--terminal-weight", type=float, default=40.0)
    parser.add_argument("--energy-weight", type=float, default=3.0)
    parser.add_argument("--rail-penalty-weight", type=float, default=20000.0)
    parser.add_argument("--cart-position-weight", type=float, default=0.0)
    parser.add_argument("--init-controller-json", default=None)
    parser.add_argument(
        "--residual-around-controller",
        action="store_true",
        help="search additive low-dimensional corrections around the supplied controller",
    )
    parser.add_argument("--residual-start-time", type=float, default=0.0)
    parser.add_argument(
        "--feedback-route-json",
        default=None,
        help="keep a saved time-varying feedback route in the loop and search an additive residual",
    )
    parser.add_argument("--route-tracking-gain-scale", type=float, default=1.0)
    parser.add_argument("--route-phase-window", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.knot_count < 2 or args.population < 2 or not 1 <= args.elites <= args.population:
        raise ValueError("invalid CEM dimensions")
    if min(args.seconds, args.handoff_start_time, args.rolling_window) <= 0.0:
        raise ValueError("horizon, handoff start, and rolling window must be positive")
    if args.handoff_start_time >= args.seconds:
        raise ValueError("handoff start must be before the search horizon")
    if args.residual_around_controller and args.init_controller_json is None:
        raise ValueError("--residual-around-controller requires --init-controller-json")
    if args.residual_around_controller and args.feedback_route_json is not None:
        raise ValueError("open-loop and feedback-route residual modes are mutually exclusive")
    if args.route_tracking_gain_scale < 0.0 or args.route_phase_window < 0:
        raise ValueError("route tracking scale and phase window must be nonnegative")
    if not 0.0 <= args.residual_start_time <= args.seconds:
        raise ValueError("residual start time must be within the search horizon")

    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {**cfg["env"], "init_mode": "hanging"}
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][key] = 0.0
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    target_bank, target_metadata = load_handoff_targets(
        args.handoff_bank_json, env.n
    )
    target_scales = np.concatenate(
        (
            [1.25, 0.50],
            np.full(env.n, 0.15, dtype=np.float64),
            np.full(env.n, 0.75, dtype=np.float64),
        )
    )
    step_count = max(2, round(args.seconds / env.dt))
    interpolation = interpolation_matrix(args.knot_count, step_count)
    feedback_route = (
        load_feedback_route(args.feedback_route_json, env.n)
        if args.feedback_route_json is not None
        else None
    )
    if feedback_route is not None:
        base_actions = np.zeros(step_count, dtype=np.float64)
        center = np.zeros(args.knot_count, dtype=np.float64)
        residual_start_step = min(
            step_count, max(0, round(args.residual_start_time / env.dt))
        )
        residual_interpolation = interpolation.copy()
        residual_interpolation[:residual_start_step, :] = 0.0
    elif args.residual_around_controller:
        base_actions = load_full_resolution_center(
            args.init_controller_json, step_count, args.seconds, env
        )
        center = np.zeros(args.knot_count, dtype=np.float64)
        residual_start_step = min(
            step_count, max(0, round(args.residual_start_time / env.dt))
        )
        residual_interpolation = interpolation.copy()
        residual_interpolation[:residual_start_step, :] = 0.0
    else:
        base_actions = np.zeros(step_count, dtype=np.float64)
        center = load_initial_center(
            args.init_controller_json, args.knot_count, args.seconds, env
        )
        residual_start_step = 0
        residual_interpolation = interpolation

    rng = np.random.default_rng(args.seed)
    sigma = np.full(args.knot_count, args.action_sigma, dtype=np.float64)
    handoff_start_step = max(1, round(args.handoff_start_time / env.dt))
    best_record: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    for iteration in range(args.iterations):
        knots = np.clip(
            center[None, :] + rng.normal(0.0, sigma, size=(args.population, args.knot_count)),
            -1.0,
            1.0,
        )
        knots[0] = center
        records: list[dict[str, Any]] = []
        for candidate_index, candidate in enumerate(knots):
            actions = np.clip(
                base_actions + candidate @ residual_interpolation.T,
                -1.0,
                1.0,
            )
            cost, metrics = evaluate_candidate(
                env,
                actions,
                target_bank=target_bank,
                target_scales=target_scales,
                handoff_start_step=handoff_start_step,
                rolling_window=max(1, args.rolling_window),
                target_weight=args.target_weight,
                terminal_weight=args.terminal_weight,
                energy_weight=args.energy_weight,
                rail_penalty_weight=args.rail_penalty_weight,
                cart_position_weight=args.cart_position_weight,
                feedback_route=feedback_route,
                tracking_gain_scale=args.route_tracking_gain_scale,
                phase_window=args.route_phase_window,
            )
            records.append({"index": candidate_index, "cost": cost, "metrics": metrics})
        records.sort(key=lambda record: float(record["cost"]))
        elite_records = records[: args.elites]
        elite_knots = knots[[int(record["index"]) for record in elite_records]]
        center = np.mean(elite_knots, axis=0)
        sigma = np.maximum(np.std(elite_knots, axis=0) * args.sigma_decay, args.sigma_floor)
        top = records[0]
        metrics = top["metrics"]
        history.append(
            {
                "iteration": int(iteration + 1),
                "cost": float(top["cost"]),
                "best_distance": metrics.get("best_distance"),
                "terminal_distance": metrics.get("terminal_distance"),
                "best_time_seconds": metrics.get("best_time_seconds"),
                "max_cart": metrics.get("max_cart"),
                "invalid": metrics.get("invalid"),
            }
        )
        if best_record is None or float(top["cost"]) < float(best_record["cost"]):
            best_actions = np.clip(
                base_actions + knots[int(top["index"])] @ residual_interpolation.T,
                -1.0,
                1.0,
            )
            best_metrics = dict(metrics)
            best_metrics["residual_knots"] = knots[int(top["index"])].astype(float).tolist()
            best_record = {
                "iteration": int(iteration + 1),
                "cost": float(top["cost"]),
                "controls": best_metrics.pop("applied_controls", best_actions.astype(np.float32).astype(float).tolist()),
                **best_metrics,
            }
        print(
            f"iter={iteration + 1:03d} cost={float(top['cost']):.3f} "
            f"distance={metrics.get('best_distance')} terminal={metrics.get('terminal_distance')} "
            f"time={metrics.get('best_time_seconds')} cart={metrics.get('max_cart')} "
            f"invalid={metrics.get('invalid')}",
            flush=True,
        )

    assert best_record is not None
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "canonical_11_handoff_bank_search_not_solution",
        "not_solution": True,
        "summary": "Exact canonical hanging-start CEM scored against real saved handoff states; this artifact does not prove swing-up or hold.",
        "config": file_metadata(Path(args.config)),
        "config_sha256": data_sha256(cfg),
        "handoff_bank": target_metadata,
        "search": {
            "seconds": float(args.seconds),
            "knot_count": int(args.knot_count),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "handoff_start_time": float(args.handoff_start_time),
            "handoff_start_step": int(handoff_start_step),
            "rolling_window": int(args.rolling_window),
            "target_scales": target_scales.astype(float).tolist(),
            "target_weight": float(args.target_weight),
            "terminal_weight": float(args.terminal_weight),
            "energy_weight": float(args.energy_weight),
            "rail_penalty_weight": float(args.rail_penalty_weight),
            "cart_position_weight": float(args.cart_position_weight),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "seed": int(args.seed),
            "residual_around_controller": bool(args.residual_around_controller),
            "residual_start_time": float(args.residual_start_time),
            "residual_start_step": int(residual_start_step),
            "feedback_route_json": None if args.feedback_route_json is None else str(args.feedback_route_json),
            "feedback_route": None if feedback_route is None else {
                "path": feedback_route["path"],
                "sha256": feedback_route["sha256"],
                "tracking_gain_scale": float(args.route_tracking_gain_scale),
                "phase_window": int(args.route_phase_window),
            },
            "wall_time_seconds": float(time.time() - started),
        },
        "best": best_record,
        "history": history,
        "runtime": runtime_metadata(),
        "git": git_metadata(ROOT),
    }
    env.close()
    dump_json(payload, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
