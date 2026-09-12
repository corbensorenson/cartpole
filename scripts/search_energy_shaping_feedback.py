#!/usr/bin/env python
"""Search a low-dimensional energy-shaping cart-acceleration controller.

The controller treats cart acceleration as the virtual input and uses the
MuJoCo mass matrix to convert that virtual input to the available cart force.
Its features are deliberately interpretable: pendulum energy error, relative
horizontal momentum, a weighted angular phase, phase velocity, cart position,
cart velocity, and a small deterministic kick so the exact hanging state is
not an equilibrium of the search.

This is a discovery controller.  It is not final evidence until a separate
reset-free replay passes the canonical seven-link gates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp


def state_features(env: NLinkCartPoleEnv, t: float) -> dict[str, float]:
    qrel = wrap_angle(np.asarray(env.data.qpos[1 : 1 + env.n], dtype=np.float64))
    angles = np.unwrap(np.cumsum(qrel))
    omegas = np.cumsum(np.asarray(env.data.qvel[1 : 1 + env.n], dtype=np.float64))
    lengths = np.asarray(env.morphology.lengths, dtype=np.float64)
    masses = np.asarray(env.morphology.masses, dtype=np.float64)
    gravity = 9.81

    # Link COM kinematics in the cart frame.  This gives the momentum that
    # couples most directly to cart acceleration in the energy derivative.
    potential = 0.0
    kinetic = 0.0
    horizontal_momentum = 0.0
    joint_x = 0.0
    joint_z = 0.0
    joint_vx = 0.0
    joint_vz = 0.0
    for index, (length, mass) in enumerate(zip(lengths, masses)):
        c = np.cos(angles[index])
        s = np.sin(angles[index])
        com_x = joint_x + 0.5 * length * s
        com_z = joint_z + 0.5 * length * c
        com_vx = joint_vx + 0.5 * length * c * omegas[index]
        com_vz = joint_vz - 0.5 * length * s * omegas[index]
        potential += mass * gravity * com_z
        kinetic += 0.5 * mass * (com_vx * com_vx + com_vz * com_vz)
        kinetic += 0.5 * (mass * length * length / 12.0) * omegas[index] ** 2
        horizontal_momentum += mass * com_vx
        joint_x += length * s
        joint_z += length * c
        joint_vx += length * c * omegas[index]
        joint_vz -= length * s * omegas[index]

    upright_potential = float(np.sum(masses * gravity * np.cumsum(lengths - 0.5 * lengths)))
    # The direct expression above is not used because the link COM heights
    # depend on all downstream lengths.  Compute the two reference poses with
    # the same kinematics used for the live state.
    def reference_potential(angle: float) -> float:
        joint_height = 0.0
        total = 0.0
        for length, mass in zip(lengths, masses):
            joint_height += 0.5 * length * np.cos(angle)
            total += mass * gravity * joint_height
            joint_height += 0.5 * length * np.cos(angle)
        return float(total)

    upright_potential = reference_potential(0.0)
    hanging_potential = reference_potential(np.pi)
    energy_gap = max(1e-9, upright_potential - hanging_potential)
    energy_fraction = (potential + kinetic - hanging_potential) / energy_gap
    energy_error = float(energy_fraction - 1.0)
    scale = max(1e-9, float(np.sum(masses * lengths)))
    phase = float(np.sum(masses * lengths * np.sin(angles)) / scale)
    phase_velocity = float(np.sum(masses * lengths * np.cos(angles) * omegas) / scale)
    return {
        "energy_fraction": float(energy_fraction),
        "energy_error": energy_error,
        "horizontal_momentum": float(horizontal_momentum / scale),
        "phase": phase,
        "phase_velocity": phase_velocity,
        "cart_position": float(env.data.qpos[0]),
        "cart_velocity": float(env.data.qvel[0]),
        "kick_sin": float(np.sin(2.0 * np.pi * t)),
        "kick_cos": float(np.cos(2.0 * np.pi * t)),
    }


def virtual_acceleration(params: np.ndarray, features: dict[str, float], t: float) -> float:
    # Parameterization is intentionally low dimensional and signed.  The CEM
    # bounds below keep the search in a physically readable range.
    energy_gain, phase_gain, phase_velocity_gain, cart_kp, cart_kd, kick_amp, kick_frequency, kick_phase = params
    kick = kick_amp * np.sin(2.0 * np.pi * kick_frequency * t + kick_phase)
    return float(
        energy_gain * features["energy_error"] * features["horizontal_momentum"]
        + phase_gain * features["phase"]
        + phase_velocity_gain * features["phase_velocity"]
        - cart_kp * features["cart_position"]
        - cart_kd * features["cart_velocity"]
        + kick
    )


def force_for_cart_acceleration(env: NLinkCartPoleEnv, acceleration: float) -> float:
    mass_matrix = np.zeros((env.model.nv, env.model.nv), dtype=np.float64)
    mujoco.mj_fullM(env.model, mass_matrix, env.data.qM)
    bias = np.asarray(env.data.qfrc_bias, dtype=np.float64)
    zero_acceleration = np.linalg.solve(mass_matrix, -bias)
    force_response = np.linalg.solve(mass_matrix, np.eye(env.model.nv)[:, 0])
    coefficient = float(force_response[0])
    if abs(coefficient) < 1e-9:
        return 0.0
    return float((float(acceleration) - zero_acceleration[0]) / coefficient)


def rollout(
    cfg: dict[str, Any],
    params: np.ndarray,
    *,
    progress: float,
    seconds: float,
    rail_limit: float | None = None,
    return_trace: bool = False,
) -> dict[str, Any]:
    env_cfg = {
        **cfg["env"],
        "init_mode": "hanging",
        "episode_seconds": float(seconds),
        "terminate_abs_angle": None,
        "obs_include_capture_features": False,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
    }
    if rail_limit is not None:
        env_cfg["rail_limit"] = float(rail_limit)
        env_cfg["rail_limit_start"] = float(rail_limit)
        env_cfg["rail_limit_end"] = float(rail_limit)
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        env_cfg[f"{key}_start"] = 0.0
        env_cfg[f"{key}_end"] = 0.0
    env = NLinkCartPoleEnv({**cfg, "env": env_cfg}, progress=progress, seed=0)
    env.reset(seed=0)
    rows: list[dict[str, Any]] = []
    max_cart = 0.0
    action_sq = 0.0
    action_slew = 0.0
    previous_action = 0.0
    final_info: dict[str, Any] = {}
    best_row: dict[str, Any] | None = None
    best_late_row: dict[str, Any] | None = None
    best_local_cost = float("inf")
    best_late_cost = float("inf")
    steps = min(env.max_steps, int(seconds / env.dt))
    for step in range(steps):
        t = step * env.dt
        features = state_features(env, t)
        desired_acceleration = virtual_acceleration(params, features, t)
        force = force_for_cart_acceleration(env, desired_acceleration)
        action = float(np.clip(force / env.force_limit, -1.0, 1.0))
        _, _, terminated, truncated, info = env.step([action])
        _, absolute = env._angles()
        row = {
            "step": int(step + 1),
            "time_seconds": float((step + 1) * env.dt),
            "action": action,
            "desired_cart_acceleration": float(desired_acceleration),
            "force": float(force),
            **features,
            "max_abs_angle": float(np.max(np.abs(absolute))),
            "hinge_velocity_rms": float(np.sqrt(np.mean(np.asarray(env.data.qvel[1 : 1 + env.n]) ** 2))),
            "x": float(env.data.qpos[0]),
            "cart_velocity": float(env.data.qvel[0]),
            "absolute_angles": absolute.astype(float).tolist(),
            "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
            "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
            "is_upright": bool(info.get("is_upright", False)),
            "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
            "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
        }
        local_cost = (
            30.0 * (row["max_abs_angle"] / 0.15) ** 2
            + 8.0 * (row["hinge_velocity_rms"] / 0.75) ** 2
            + 3.0 * (abs(row["x"]) / 1.25) ** 2
            + 3.0 * (abs(row["cart_velocity"]) / 0.50) ** 2
        )
        if local_cost < best_local_cost:
            best_local_cost = local_cost
            best_row = dict(row)
        if t >= 0.40 * seconds and local_cost < best_late_cost:
            best_late_cost = local_cost
            best_late_row = dict(row)
        if return_trace and (step % 4 == 0 or row["is_upright"]):
            rows.append(row)
        max_cart = max(max_cart, abs(row["x"]))
        action_sq += action * action
        action_slew += (action - previous_action) ** 2
        previous_action = action
        final_info = dict(info)
        if terminated or truncated:
            break
    best = best_late_row or best_row
    if not np.isfinite(best_late_cost):
        best_late_cost = best_local_cost
    # Include all steps in the score even when trace recording is disabled.
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    angle = float(final_info.get("max_abs_angle", np.inf))
    hinge = float(final_info.get("hinge_velocity_rms", np.inf))
    cart = abs(float(final_info.get("x", np.inf)))
    cart_velocity = abs(float(env.data.qvel[0]))
    rail_over = max(0.0, max_cart - float(env.rail_limit))
    score = (
        0.45 * best_late_cost
        + 0.25 * best_local_cost
        + 20.0 * (angle / 0.15) ** 2
        + 5.0 * (hinge / 0.75) ** 2
        + 2.0 * (cart / 1.25) ** 2
        + 2.0 * (cart_velocity / 0.50) ** 2
        + 100.0 * rail_over * rail_over
        - 12000.0 * max_streak
        - 5000.0 * float(final_info.get("max_centered_upright_streak_seconds", 0.0))
        + 0.01 * action_sq
        + 0.02 * action_slew
    )
    result = {
        "score": float(score),
        "success": bool(final_info.get("success", False)),
        "max_upright_streak_seconds": max_streak,
        "max_centered_upright_streak_seconds": float(final_info.get("max_centered_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "max_cart_abs": float(max_cart),
        "termination_reason": final_info.get("termination_reason"),
        "best_row": best,
        "best_local_cost": float(best_local_cost),
        "best_late_cost": float(best_late_cost),
        "final_info": final_info,
        "trace": rows,
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="CEM search for seven-link energy-shaping cart acceleration")
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument("--progress", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--rail-limit", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=48)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.elites < 1 or args.elites > args.population:
        raise ValueError("--elites must be between one and population")
    cfg = apply_overrides(load_config(args.config), args.override)
    rng = np.random.default_rng(args.seed)
    # energy_gain, phase_gain, phase_velocity_gain, cart_kp, cart_kd,
    # kick_amp, kick_frequency, kick_phase
    center = np.asarray([12.0, -4.0, -1.0, 1.0, 1.0, 2.0, 0.20, 0.0], dtype=np.float64)
    sigma = np.full(center.shape, float(args.sigma), dtype=np.float64)
    sigma[6] = 0.08
    sigma[7] = 0.50
    lower = np.asarray([-40.0, -40.0, -20.0, 0.0, 0.0, -30.0, 0.02, -np.pi], dtype=np.float64)
    upper = np.asarray([40.0, 40.0, 20.0, 20.0, 20.0, 30.0, 2.00, np.pi], dtype=np.float64)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()
    for iteration in range(args.iterations + 1):
        if iteration == 0:
            candidates = [center.copy()]
        else:
            candidates = [center + rng.normal(0.0, sigma) for _ in range(args.population)]
        records: list[dict[str, Any]] = []
        for vector in candidates:
            vector = np.clip(np.asarray(vector, dtype=np.float64), lower, upper)
            result = rollout(
                cfg,
                vector,
                progress=args.progress,
                seconds=args.seconds,
                rail_limit=args.rail_limit,
            )
            records.append({"vector": vector, "result": result})
        records.sort(key=lambda row: float(row["result"]["score"]))
        top = records[0]
        if best is None or float(top["result"]["score"]) < float(best["score"]):
            best = {
                "score": float(top["result"]["score"]),
                "vector": top["vector"].astype(float).tolist(),
                "result": top["result"],
            }
        elite = np.asarray([row["vector"] for row in records[: args.elites]], dtype=np.float64)
        center = np.mean(elite, axis=0)
        sigma = np.maximum(np.std(elite, axis=0) * args.sigma_decay, args.sigma_floor)
        top_result = top["result"]
        history.append(
            {
                "iteration": int(iteration),
                "score": float(top_result["score"]),
                "best_score": float(best["score"]),
                "streak": float(top_result["max_upright_streak_seconds"]),
                "centered_streak": float(top_result["max_centered_upright_streak_seconds"]),
                "best_angle": float((top_result.get("best_row") or {}).get("max_abs_angle", np.inf)),
                "max_cart_abs": float(top_result["max_cart_abs"]),
            }
        )
        print(
            f"iter={iteration:03d} score={top_result['score']:.2f} "
            f"streak={top_result['max_upright_streak_seconds']:.3f}s "
            f"angle={(top_result.get('best_row') or {}).get('max_abs_angle', np.nan):.4f} "
            f"max_cart={top_result['max_cart_abs']:.2f}",
            flush=True,
        )
    assert best is not None
    final = rollout(
        cfg,
        np.asarray(best["vector"], dtype=np.float64),
        progress=args.progress,
        seconds=args.seconds,
        rail_limit=args.rail_limit,
        return_trace=True,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Low-dimensional MuJoCo mass-matrix energy-shaping feedback search; discovery only.",
        "config_path": str(Path(args.config)),
        "progress": float(args.progress),
        "seconds": float(args.seconds),
        "rail_limit": args.rail_limit,
        "search": {
            "seed": int(args.seed),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "sigma": float(args.sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "wall_time_seconds": float(time.time() - started),
        },
        "parameter_names": ["energy_gain", "phase_gain", "phase_velocity_gain", "cart_kp", "cart_kd", "kick_amp", "kick_frequency", "kick_phase"],
        "best": best,
        "final_eval": final,
        "history": history,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
