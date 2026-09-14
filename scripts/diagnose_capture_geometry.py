#!/usr/bin/env python
"""Audit a proposed capture state under saturated nonlinear feedback.

This is a diagnostic, not a capture certificate.  It reports dimensionless
linear conditioning and then executes the resulting LQR on the exact MuJoCo
plant with the real normalized action bound.
"""

from __future__ import annotations

import argparse
import copy
import json
import warnings
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.capture_terminal import feedback_horizon_metric
from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles, wrap_angle
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.modal import (
    StateScales,
    dimensionless_absolute_transform,
    real_schur_decomposition,
    transform_dynamics,
)
from scripts.make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics

DEFAULT_Q_WEIGHTS = {
    "cart_position": 0.1,
    "absolute_angle": 100.0,
    "cart_velocity": 0.1,
    "absolute_angular_velocity": 1.0,
    "relative_angle": 1.0,
    "relative_angular_velocity": 0.01,
}


def dimensionless_transform(
    cfg: dict[str, Any], progress: float
) -> tuple[np.ndarray, float, float]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=0)
    chain_length = float(np.sum(env.morphology.lengths))
    natural_time = float(np.sqrt(chain_length / 9.81))
    transform = dimensionless_absolute_transform(
        env.n,
        StateScales(
            cart_position=chain_length,
            absolute_angle=1.0,
            cart_velocity=chain_length / natural_time,
            hinge_velocity=1.0 / natural_time,
        ),
    )
    env.close()
    return transform, chain_length, natural_time


def load_state(
    path: Path,
    index: int,
    *,
    cfg: dict[str, Any],
    progress: float,
    coordinate_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload.get("states"), list):
        record = payload["states"][index]
    elif (
        isinstance(payload.get("episode_results"), list)
        and payload["episode_results"]
        and isinstance(
            payload["episode_results"][0].get("route_state"), dict
        )
    ):
        record = payload["episode_results"][0]["route_state"]
    elif (
        isinstance(payload.get("episode_results"), list)
        and payload["episode_results"]
        and isinstance(payload["episode_results"][0].get("trajectory"), list)
    ):
        record = payload["episode_results"][0]["trajectory"][index]
    elif isinstance(payload.get("selected_state"), dict):
        record = payload["selected_state"]
    elif isinstance(payload.get("route_state"), dict):
        record = payload["route_state"]
    elif isinstance(payload.get("search", {}).get("nominal_coordinate_states"), list):
        coordinate = np.asarray(
            payload["search"]["nominal_coordinate_states"][index],
            dtype=np.float64,
        )
        transform = coordinate_transform
        if transform is None:
            transform, _, _ = dimensionless_transform(cfg, progress)
        physical = np.linalg.solve(transform, coordinate)
        dimension = physical.size // 2
        record = {
            "qpos": physical[:dimension].astype(float).tolist(),
            "qvel": physical[dimension:].astype(float).tolist(),
            "source_coordinate_state": coordinate.astype(float).tolist(),
            "source_coordinate_state_index": int(index),
            "conversion": "inverse_dimensionless_absolute_transform",
        }
    else:
        record = payload
    qpos = np.asarray(record["qpos"], dtype=np.float64)
    qvel = np.asarray(record["qvel"], dtype=np.float64)
    if qpos.ndim != 1 or qvel.shape != qpos.shape:
        raise ValueError("capture state qpos/qvel must be equal-length vectors")
    return qpos, qvel, record, payload


def config_provenance_candidates(
    cfg: dict[str, Any], state_payload: dict[str, Any]
) -> dict[str, str]:
    """Hash raw and known deterministic wrappers without changing the plant."""

    candidates = {"raw_config": data_sha256(cfg)}
    evaluation = copy.deepcopy(cfg)
    evaluation["env"]["init_mode"] = "hanging"
    evaluation["env"]["action_lqr_residual"] = {"enabled": False}
    evaluation["env"].setdefault("action_lqr_switch", {"enabled": False})[
        "enabled"
    ] = False
    candidates["hanging_evaluation_config"] = data_sha256(evaluation)

    source_run = state_payload.get("source_run")
    if isinstance(source_run, str) and Path(source_run).is_file():
        source_payload = json.loads(Path(source_run).read_text(encoding="utf-8"))
        seconds = source_payload.get("search", {}).get("seconds")
        if seconds is not None:
            search = copy.deepcopy(evaluation)
            search["env"]["episode_seconds"] = float(seconds) + 1.0
            for key in (
                "init_angle_noise",
                "init_vel_noise",
                "init_cart_noise",
                "init_cart_vel_noise",
            ):
                search["env"][key] = 0.0
                search["env"][f"{key}_start"] = 0.0
                search["env"][f"{key}_end"] = 0.0
            candidates["deterministic_hanging_search_config"] = data_sha256(
                search
            )
    return candidates


def controllability_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
    """Return deterministic finite-horizon controllability SVD diagnostics."""

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("state matrix must be square")
    if b.shape != (a.shape[0], 1):
        raise ValueError("single-input matrix has the wrong shape")
    blocks = []
    block = b.copy()
    for _ in range(a.shape[0]):
        blocks.append(block)
        block = a @ block
    matrix = np.concatenate(blocks, axis=1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    tolerance = (
        max(matrix.shape)
        * np.finfo(np.float64).eps
        * float(singular_values[0])
    )
    rank = int(np.sum(singular_values > tolerance))
    smallest = float(singular_values[-1])
    condition = (
        float(singular_values[0] / smallest) if smallest > 0.0 else float("inf")
    )
    return {
        "rank": rank,
        "state_dimension": int(a.shape[0]),
        "singular_values": singular_values.astype(float).tolist(),
        "condition_number": condition,
        "log10_condition_number": (
            float(np.log10(condition)) if np.isfinite(condition) else float("inf")
        ),
        "rank_tolerance": float(tolerance),
    }


def gain_line_feasibility(
    a: np.ndarray,
    b: np.ndarray,
    gain: np.ndarray,
    state: np.ndarray,
    *,
    grid_points: int = 1001,
) -> dict[str, Any]:
    """Compare linear stability with the action bound along ``alpha * gain``."""

    if grid_points < 2:
        raise ValueError("grid_points must be at least two")
    gain = np.asarray(gain, dtype=np.float64).reshape(1, -1)
    state = np.asarray(state, dtype=np.float64)
    if a.shape[0] != gain.shape[1] or state.shape != (a.shape[0],):
        raise ValueError("gain or state shape does not match the dynamics")

    def spectral_radius(scale: float) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(a - b @ (scale * gain)))))

    scales = np.linspace(0.0, 1.0, grid_points)
    radii = np.asarray([spectral_radius(float(scale)) for scale in scales])
    stable = np.flatnonzero(radii < 1.0)
    minimum_stabilizing_scale: float | None = None
    if stable.size:
        first = int(stable[0])
        if first == 0:
            minimum_stabilizing_scale = 0.0
        else:
            lower = float(scales[first - 1])
            upper = float(scales[first])
            for _ in range(60):
                midpoint = 0.5 * (lower + upper)
                if spectral_radius(midpoint) < 1.0:
                    upper = midpoint
                else:
                    lower = midpoint
            minimum_stabilizing_scale = upper

    unit_scale_action = abs(float((gain @ state).item()))
    maximum_nonsaturating_scale = (
        None if unit_scale_action == 0.0 else float(1.0 / unit_scale_action)
    )
    feasible = (
        minimum_stabilizing_scale is not None
        and (
            maximum_nonsaturating_scale is None
            or minimum_stabilizing_scale <= maximum_nonsaturating_scale
        )
    )
    gap_ratio = None
    if (
        minimum_stabilizing_scale is not None
        and maximum_nonsaturating_scale not in (None, 0.0)
    ):
        gap_ratio = float(
            minimum_stabilizing_scale / maximum_nonsaturating_scale
        )
    return {
        "minimum_stabilizing_feedback_scale_on_gain_line": minimum_stabilizing_scale,
        "maximum_initially_nonsaturating_feedback_scale": maximum_nonsaturating_scale,
        "stabilizing_and_initially_nonsaturating_scale_exists": bool(feasible),
        "stability_to_action_scale_gap_ratio": gap_ratio,
        "grid_points": int(grid_points),
        "scope": "fixed LQR gain direction and proposed state only",
    }


def wrapped_physical_state(
    qpos: np.ndarray, qvel: np.ndarray, *, cart_target: float = 0.0
) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64).copy()
    qvel = np.asarray(qvel, dtype=np.float64)
    qpos[0] -= float(cart_target)
    qpos[1:] = wrap_angle(qpos[1:])
    return np.r_[qpos, qvel]


def warning_counts(data: mujoco.MjData) -> list[int]:
    return [int(item.number) for item in data.warning]


def saturated_feedback_rollout(
    cfg: dict[str, Any],
    *,
    progress: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
    gain: np.ndarray,
    feedback_scale: float,
    seconds: float,
    cart_target: float,
    qacc_warmstart: np.ndarray | None,
) -> dict[str, Any]:
    """Run exact clipped LQR and expose saturation and simulator recovery."""

    exact_cfg = copy.deepcopy(cfg)
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        exact_cfg["env"][key] = 0.0
        exact_cfg["env"][f"{key}_start"] = 0.0
        exact_cfg["env"][f"{key}_end"] = 0.0
    for key in ("init_qpos_scale", "init_qvel_scale"):
        exact_cfg["env"][key] = 1.0
        exact_cfg["env"][f"{key}_start"] = 1.0
        exact_cfg["env"][f"{key}_end"] = 1.0
    env = NLinkCartPoleEnv(exact_cfg, progress=progress, seed=0)
    _, initial_info = env.reset(options={"qpos": qpos, "qvel": qvel})
    if qacc_warmstart is not None:
        if qacc_warmstart.shape != env.data.qacc_warmstart.shape:
            raise ValueError("qacc_warmstart shape does not match the plant")
        env.data.qacc_warmstart[:] = qacc_warmstart
    initial_warnings = warning_counts(env.data)
    max_cart = abs(float(env.data.qpos[0]))
    max_raw_action = 0.0
    saturation_steps = 0
    current_saturation_run = 0
    longest_saturation_run = 0
    steps = min(env.max_steps, max(1, int(np.ceil(seconds / env.dt))))
    terminated = False
    truncated = False
    info = dict(initial_info)
    simulated_steps = 0
    for simulated_steps in range(1, steps + 1):
        state = wrapped_physical_state(
            env.data.qpos, env.data.qvel, cart_target=cart_target
        )
        raw_action = float(-feedback_scale * (np.asarray(gain) @ state))
        action = float(np.clip(raw_action, -1.0, 1.0))
        saturated = abs(raw_action) > 1.0
        max_raw_action = max(max_raw_action, abs(raw_action))
        saturation_steps += int(saturated)
        current_saturation_run = current_saturation_run + 1 if saturated else 0
        longest_saturation_run = max(longest_saturation_run, current_saturation_run)
        _, _, terminated, truncated, info = env.step([action])
        max_cart = max(max_cart, abs(float(env.data.qpos[0])))
        if terminated or truncated:
            break
    final_warnings = warning_counts(env.data)
    warning_deltas = [
        max(0, final - initial)
        for initial, final in zip(initial_warnings, final_warnings, strict=True)
    ]
    result = {
        "requested_seconds": float(seconds),
        "simulated_steps": int(simulated_steps),
        "simulated_seconds": float(simulated_steps * env.dt),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": info.get("termination_reason"),
        "success": bool(info.get("success", False)),
        "max_raw_normalized_action": float(max_raw_action),
        "saturation_steps": int(saturation_steps),
        "saturation_fraction": float(saturation_steps / max(1, simulated_steps)),
        "saturation_seconds": float(saturation_steps * env.dt),
        "longest_saturation_seconds": float(longest_saturation_run * env.dt),
        "max_cart_excursion": float(max_cart),
        "minimum_rail_clearance": float(env.rail_limit - max_cart),
        "max_upright_streak_seconds": float(
            info.get("max_upright_streak_seconds", 0.0)
        ),
        "final_max_abs_angle": float(info.get("max_abs_angle", np.nan)),
        "final_absolute_angular_velocity_rms": float(
            info.get("absolute_angular_velocity_rms", np.nan)
        ),
        "mujoco_warning_counts_before": initial_warnings,
        "mujoco_warning_counts_after": final_warnings,
        "mujoco_warning_count_deltas": warning_deltas,
        "mujoco_recovery_detected": bool(any(warning_deltas)),
        "initialization": (
            "qpos_qvel_qacc_warmstart"
            if qacc_warmstart is not None
            else "qpos_qvel_cold_start"
        ),
        "integration_state_complete": bool(qacc_warmstart is not None),
    }
    result["rollout_completed"] = bool(
        simulated_steps == steps and not terminated and not truncated
    )
    result["requested_upright_hold_completed"] = bool(
        result["rollout_completed"]
        and result["max_upright_streak_seconds"] >= seconds
    )
    env.close()
    return result


def scaled_capture_state(
    qpos: np.ndarray,
    qvel: np.ndarray,
    scale: float,
    *,
    cart_target: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Scale one physical error-state ray toward the upright equilibrium.

    Joint positions are wrapped before scaling so an equivalent ``2*pi``
    branch does not create a fictitious large error. This is a directional
    diagnostic, not an estimate of the complete nonlinear capture basin.
    """

    if not np.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("capture-ray scale must be finite and in [0, 1]")
    position_error = np.asarray(qpos, dtype=np.float64).copy()
    velocity_error = np.asarray(qvel, dtype=np.float64)
    if position_error.ndim != 1 or velocity_error.shape != position_error.shape:
        raise ValueError("qpos and qvel must be equal-length vectors")
    position_error[0] -= float(cart_target)
    position_error[1:] = wrap_angle(position_error[1:])
    scaled_qpos = float(scale) * position_error
    scaled_qpos[0] += float(cart_target)
    return scaled_qpos, float(scale) * velocity_error


def capture_ray_boundary(
    cfg: dict[str, Any],
    *,
    progress: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
    gain: np.ndarray,
    feedback_scale: float,
    seconds: float,
    cart_target: float,
    minimum_scale: float = 1.0e-8,
    grid_points: int = 33,
    bisection_steps: int = 20,
) -> dict[str, Any]:
    """Bracket the origin-connected exact capture interval on one state ray."""

    if not np.isfinite(minimum_scale) or not 0.0 < minimum_scale < 1.0:
        raise ValueError("minimum capture-ray scale must be in (0, 1)")
    if grid_points < 2 or bisection_steps < 0:
        raise ValueError(
            "capture-ray grid must have two points and nonnegative refinement"
        )

    cache: dict[float, dict[str, Any]] = {}

    def evaluate(scale: float) -> dict[str, Any]:
        key = float(scale)
        if key not in cache:
            scaled_qpos, scaled_qvel = scaled_capture_state(
                qpos,
                qvel,
                key,
                cart_target=cart_target,
            )
            rollout = saturated_feedback_rollout(
                cfg,
                progress=progress,
                qpos=scaled_qpos,
                qvel=scaled_qvel,
                gain=gain,
                feedback_scale=feedback_scale,
                seconds=seconds,
                cart_target=cart_target,
                # A warm-start acceleration belongs only to the original
                # state. Recomputing it keeps scaled points comparable.
                qacc_warmstart=None,
            )
            cache[key] = {
                "scale": key,
                "passed": bool(rollout["requested_upright_hold_completed"]),
                "termination_reason": rollout["termination_reason"],
                "max_upright_streak_seconds": float(
                    rollout["max_upright_streak_seconds"]
                ),
                "max_raw_normalized_action": float(
                    rollout["max_raw_normalized_action"]
                ),
                "saturation_fraction": float(rollout["saturation_fraction"]),
                "max_cart_excursion": float(rollout["max_cart_excursion"]),
            }
        return cache[key]

    full = evaluate(1.0)
    if full["passed"]:
        return {
            "scope": "origin-connected radial lower bound, not a global basin certificate",
            "rollout_seconds": float(seconds),
            "full_state_passed": True,
            "maximum_origin_connected_pass_scale": 1.0,
            "required_radial_contraction_factor": 1.0,
            "boundary_is_lower_bound": True,
            "nonmonotonic_pass_detected": False,
            "grid_minimum_scale": float(minimum_scale),
            "grid_points": int(grid_points),
            "bisection_steps": int(bisection_steps),
            "samples": [full],
        }

    scales = np.r_[0.0, np.geomspace(minimum_scale, 1.0, grid_points)]
    rows = [evaluate(float(scale)) for scale in scales]
    if not rows[0]["passed"]:
        return {
            "scope": "origin-connected radial diagnostic, not a global basin certificate",
            "rollout_seconds": float(seconds),
            "full_state_passed": False,
            "maximum_origin_connected_pass_scale": None,
            "required_radial_contraction_factor": None,
            "boundary_is_lower_bound": False,
            "nonmonotonic_pass_detected": any(row["passed"] for row in rows[1:]),
            "grid_minimum_scale": float(minimum_scale),
            "grid_points": int(grid_points),
            "bisection_steps": int(bisection_steps),
            "samples": rows,
            "error": "upright equilibrium did not complete the requested rollout",
        }

    first_failure = next(
        index for index, row in enumerate(rows[1:], start=1) if not row["passed"]
    )
    lower = float(rows[first_failure - 1]["scale"])
    upper = float(rows[first_failure]["scale"])
    for _ in range(bisection_steps):
        midpoint = 0.5 * (lower + upper)
        if evaluate(midpoint)["passed"]:
            lower = midpoint
        else:
            upper = midpoint
    later_pass = any(row["passed"] for row in rows[first_failure + 1 :])
    ordered_samples = [cache[key] for key in sorted(cache)]
    return {
        "scope": "origin-connected radial diagnostic, not a global basin certificate",
        "rollout_seconds": float(seconds),
        "full_state_passed": False,
        "maximum_origin_connected_pass_scale": lower,
        "first_failing_scale": upper,
        "required_radial_contraction_factor": (
            float(1.0 / lower) if lower > 0.0 else None
        ),
        "boundary_is_lower_bound": False,
        "nonmonotonic_pass_detected": bool(later_pass),
        "grid_minimum_scale": float(minimum_scale),
        "grid_points": int(grid_points),
        "bisection_steps": int(bisection_steps),
        "samples": ordered_samples,
    }


def diagnose(
    cfg: dict[str, Any],
    *,
    progress: float,
    qpos: np.ndarray,
    qvel: np.ndarray,
    fd_epsilon: float,
    control_cost: float,
    feedback_scale: float,
    rollout_seconds: float,
    cart_target: float,
    qacc_warmstart: np.ndarray | None = None,
    scan_capture_ray: bool = False,
    capture_ray_minimum_scale: float = 1.0e-8,
    capture_ray_grid_points: int = 33,
    capture_ray_bisection_steps: int = 20,
    linear_feedback_horizon_natural_times: float = 1.0,
) -> dict[str, Any]:
    n_links = int(cfg["env"]["n_links"])
    expected = n_links + 1
    if qpos.shape != (expected,) or qvel.shape != (expected,):
        raise ValueError(f"capture state must have {expected} positions and velocities")
    a, b = finite_difference_dynamics(cfg, progress, fd_epsilon)
    transform, chain_length, natural_time = dimensionless_transform(
        cfg, progress
    )
    scaled_a, scaled_b = transform_dynamics(a, b, transform)
    q = absolute_angle_cost(n_links, DEFAULT_Q_WEIGHTS)
    r = np.array([[float(control_cost)]], dtype=np.float64)
    caught: list[warnings.WarningMessage]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(b.T @ p @ b + r, b.T @ p @ a).reshape(-1)
    closed_loop_eigenvalues = np.linalg.eigvals(
        a - b @ (feedback_scale * gain).reshape(1, -1)
    )
    state = wrapped_physical_state(qpos, qvel, cart_target=cart_target)
    absolute_angles = serial_absolute_angles(state[1 : 1 + n_links])
    absolute_rates = np.cumsum(qvel[1:])
    dimensionless_state = transform @ state
    schur = real_schur_decomposition(scaled_a, scaled_b)
    schur_amplitudes = schur.grouped_amplitudes(dimensionless_state)
    raw_action = float(-feedback_scale * gain @ state)
    policy_dt = float(cfg["env"]["timestep"]) * int(
        cfg["env"].get("frame_skip", 1)
    )
    feedback_horizon_steps = max(
        1,
        int(
            np.ceil(
                float(linear_feedback_horizon_natural_times)
                * natural_time
                / policy_dt
            )
        ),
    )
    feedback_metric = feedback_horizon_metric(
        a,
        b,
        gain,
        transform,
        horizon_steps=feedback_horizon_steps,
        feedback_scale=feedback_scale,
    )
    feedback_metric_record = feedback_metric.to_dict()
    feedback_metric_record.update(
        {
            "horizon_natural_times": float(
                feedback_horizon_steps * policy_dt / natural_time
            ),
            "policy_dt": policy_dt,
            "proposed_state": feedback_metric.evaluate(state),
        }
    )
    result = {
        "schema_version": 1,
        "claim_status": "capture_diagnostic_not_certificate",
        "n_links": n_links,
        "dimensionless_scales": {
            "chain_length": chain_length,
            "natural_time": natural_time,
            "cart_velocity": chain_length / natural_time,
            "hinge_velocity": 1.0 / natural_time,
        },
        "linearization": {
            "progress": float(progress),
            "finite_difference_epsilon": float(fd_epsilon),
            "control_cost": float(control_cost),
            "feedback_scale": float(feedback_scale),
            "cart_target": float(cart_target),
            "gain": gain.astype(float).tolist(),
            "gain_norm": float(np.linalg.norm(gain)),
            "closed_loop_spectral_radius": float(
                np.max(np.abs(closed_loop_eigenvalues))
            ),
            "warnings": [str(item.message) for item in caught],
            "dimensionless_controllability": controllability_metrics(
                scaled_a, scaled_b
            ),
            "actuator_feasible_gain_line": gain_line_feasibility(
                a, b, gain, state
            ),
            "feedback_horizon_terminal_metric": feedback_metric_record,
        },
        "proposed_state": {
            "qpos": qpos.astype(float).tolist(),
            "qvel": qvel.astype(float).tolist(),
            "dimensionless_state": dimensionless_state.astype(float).tolist(),
            "dimensionless_state_norm": float(np.linalg.norm(dimensionless_state)),
            "cart_abs": abs(float(qpos[0])),
            "cart_error_abs": abs(float(qpos[0]) - cart_target),
            "cart_velocity_abs": abs(float(qvel[0])),
            "max_abs_angle": float(np.max(np.abs(absolute_angles))),
            "absolute_angular_velocity_rms": float(
                np.sqrt(np.mean(absolute_rates**2))
            ),
            "raw_normalized_feedback_action": raw_action,
            "would_saturate": bool(abs(raw_action) > 1.0),
            "schur_group_amplitudes": schur_amplitudes.astype(float).tolist(),
            "schur_group_input_coupling": schur.group_input_coupling.astype(float).tolist(),
        },
        "nonlinear_saturated_rollout": saturated_feedback_rollout(
            cfg,
            progress=progress,
            qpos=qpos,
            qvel=qvel,
            gain=gain,
            feedback_scale=feedback_scale,
            seconds=rollout_seconds,
            cart_target=cart_target,
            qacc_warmstart=qacc_warmstart,
        ),
    }
    if scan_capture_ray:
        result["empirical_capture_ray"] = capture_ray_boundary(
            cfg,
            progress=progress,
            qpos=qpos,
            qvel=qvel,
            gain=gain,
            feedback_scale=feedback_scale,
            seconds=rollout_seconds,
            cart_target=cart_target,
            minimum_scale=capture_ray_minimum_scale,
            grid_points=capture_ray_grid_points,
            bisection_steps=capture_ray_bisection_steps,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-json", required=True)
    parser.add_argument("--state-index", type=int, default=0)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-7)
    parser.add_argument("--control-cost", type=float, default=1000.0)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--cart-target", type=float, default=0.0)
    parser.add_argument(
        "--coordinate-spec",
        help="capture-envelope spec defining saved route coordinate scales",
    )
    parser.add_argument("--rollout-seconds", type=float, default=5.0)
    parser.add_argument(
        "--scan-capture-ray",
        action="store_true",
        help="measure the exact origin-connected capture boundary along the supplied state ray",
    )
    parser.add_argument("--capture-ray-minimum-scale", type=float, default=1.0e-8)
    parser.add_argument("--capture-ray-grid-points", type=int, default=33)
    parser.add_argument("--capture-ray-bisection-steps", type=int, default=20)
    parser.add_argument(
        "--linear-feedback-horizon-natural-times",
        type=float,
        default=1.0,
        help="natural-time duration used to derive the optimizer-facing feedback metric",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if not 0.0 <= args.progress <= 1.0:
        parser.error("--progress must be in [0, 1]")
    if min(
        args.fd_epsilon,
        args.control_cost,
        args.feedback_scale,
        args.rollout_seconds,
        args.linear_feedback_horizon_natural_times,
    ) <= 0.0:
        parser.error("diagnostic scales and rollout duration must be positive")
    if not 0.0 < args.capture_ray_minimum_scale < 1.0:
        parser.error("--capture-ray-minimum-scale must be in (0, 1)")
    if args.capture_ray_grid_points < 2 or args.capture_ray_bisection_steps < 0:
        parser.error("capture-ray grid points must be >=2 and bisection steps >=0")
    config_path = Path(args.config)
    state_path = Path(args.state_json)
    cfg = load_config(config_path)
    coordinate_transform = None
    coordinate_spec = None
    if args.coordinate_spec:
        coordinate_spec = Path(args.coordinate_spec)
        distribution = load_config(coordinate_spec)["distribution"]
        coordinate_transform = dimensionless_absolute_transform(
            int(cfg["env"]["n_links"]),
            StateScales(
                float(distribution["cart_position_abs_max"]),
                float(distribution["absolute_link_angle_abs_max"]),
                float(distribution["cart_velocity_abs_max"]),
                float(distribution["hinge_velocity_rms_max"]),
            ),
        )
    qpos, qvel, state_record, state_payload = load_state(
        state_path,
        args.state_index,
        cfg=cfg,
        progress=args.progress,
        coordinate_transform=coordinate_transform,
    )
    qacc_warmstart = None
    if state_record.get("qacc_warmstart") is not None:
        qacc_warmstart = np.asarray(
            state_record["qacc_warmstart"], dtype=np.float64
        )
    provenance_candidates = config_provenance_candidates(cfg, state_payload)
    resolved_config_sha256 = provenance_candidates["raw_config"]
    recorded_config_sha256 = (
        state_payload.get("config_sha256")
        or state_payload.get("resolved_config_sha256")
        or state_payload.get("config", {}).get("resolved_sha256")
    )
    matching_modes = [
        mode
        for mode, digest in provenance_candidates.items()
        if digest == recorded_config_sha256
    ]
    result = diagnose(
        cfg,
        progress=args.progress,
        qpos=qpos,
        qvel=qvel,
        fd_epsilon=args.fd_epsilon,
        control_cost=args.control_cost,
        feedback_scale=args.feedback_scale,
        rollout_seconds=args.rollout_seconds,
        cart_target=args.cart_target,
        qacc_warmstart=qacc_warmstart,
        scan_capture_ray=args.scan_capture_ray,
        capture_ray_minimum_scale=args.capture_ray_minimum_scale,
        capture_ray_grid_points=args.capture_ray_grid_points,
        capture_ray_bisection_steps=args.capture_ray_bisection_steps,
        linear_feedback_horizon_natural_times=(
            args.linear_feedback_horizon_natural_times
        ),
    )
    result.update(
        {
            "generated_at": utc_timestamp(),
            "config": file_metadata(config_path),
            "config_resolved_sha256": resolved_config_sha256,
            "config_provenance_candidates": provenance_candidates,
            "state_source": file_metadata(state_path),
            "coordinate_spec": (
                file_metadata(coordinate_spec)
                if coordinate_spec is not None
                else None
            ),
            "state_source_recorded_config_sha256": recorded_config_sha256,
            "state_source_config_matches": (
                recorded_config_sha256 is None or bool(matching_modes)
            ),
            "state_source_config_match_mode": (
                matching_modes[0] if matching_modes else None
            ),
            "provenance_warnings": (
                []
                if recorded_config_sha256 is None or matching_modes
                else [
                    "state source config digest does not match the resolved diagnostic config"
                ]
            ),
            "state_index": int(args.state_index),
            "source_state_metadata": state_record,
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        }
    )
    dump_json(result, Path(args.out))
    print(
        f"gain_norm={result['linearization']['gain_norm']:.6g} "
        f"raw_action={result['proposed_state']['raw_normalized_feedback_action']:.6g} "
        f"saturation_fraction={result['nonlinear_saturated_rollout']['saturation_fraction']:.6f}"
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
