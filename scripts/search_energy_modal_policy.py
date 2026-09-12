#!/usr/bin/env python
"""Search a low-dimensional energy-shaping/modal two-expert controller.

This is a discovery controller, not final benchmark evidence.  It follows the
underactuated-control pattern of pumping the passive-chain energy toward the
upright energy, then handing the state to a modal capture law.  The search is
deliberately low-dimensional so every lever remains inspectable and can be
transferred to a learned expert later.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config, save_config
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles


def modal_basis(n_links: int) -> np.ndarray:
    """Four interpretable link-weighting modes, column-major."""
    s = np.linspace(-1.0, 1.0, n_links, dtype=np.float64)
    basis = np.column_stack(
        [
            np.ones(n_links, dtype=np.float64),
            s,
            np.exp(2.0 * s),
            np.where(np.arange(n_links) % 2 == 0, 1.0, -1.0),
        ]
    )
    norms = np.linalg.norm(basis, axis=0)
    return basis / np.maximum(norms, 1e-12)


def feature_state(env: NLinkCartPoleEnv, mass_matrix: np.ndarray) -> dict[str, Any]:
    """Compute controller features from the physical state, without observation shortcuts."""
    qpos = np.asarray(env.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    # Preserve the exact hanging boundary at +pi until after the serial sum;
    # wrapping relative joints first makes hanging look upright.
    absolute = serial_absolute_angles(qpos[1 : 1 + env.n])
    omega = np.cumsum(qvel[1 : 1 + env.n])

    lengths = np.asarray(env.morphology.lengths, dtype=np.float64)
    masses = np.asarray(env.morphology.masses, dtype=np.float64)
    com_x = 0.0
    com_v = 0.0
    tip_x = 0.0
    tip_v = 0.0
    for index in range(env.n):
        x = float(np.sum(lengths[:index] * np.sin(absolute[:index]))) if index else 0.0
        v = float(np.sum(lengths[:index] * np.cos(absolute[:index]) * omega[:index])) if index else 0.0
        x += 0.5 * lengths[index] * np.sin(absolute[index])
        v += 0.5 * lengths[index] * np.cos(absolute[index]) * omega[index]
        com_x += masses[index] * x
        com_v += masses[index] * v
        tip_x += lengths[index] * np.sin(absolute[index])
        tip_v += lengths[index] * np.cos(absolute[index]) * omega[index]
    total_mass = max(float(np.sum(masses)), 1e-12)
    com_x /= total_mass
    com_v /= total_mass

    # The joint block of the generalized mass matrix gives the kinetic energy
    # of the serial chain with the cart translation held fixed.  This is the
    # multi-link analogue of the simple-pendulum energy used by PFL swing-up.
    joint_vel = qvel[1 : 1 + env.n]
    joint_kinetic = 0.5 * float(joint_vel @ mass_matrix[1 : 1 + env.n, 1 : 1 + env.n] @ joint_vel)
    potential = float(env._potential_energy())
    passive_energy = potential + joint_kinetic
    energy_error = (passive_energy - float(env._upright_potential_energy)) / float(env._energy_gap)

    return {
        "absolute": absolute,
        "omega": omega,
        "sin_angle": np.sin(absolute),
        "coupling": np.cos(absolute) * omega,
        "energy_error": float(np.clip(energy_error, -4.0, 4.0)),
        "energy_fraction": float(np.clip((passive_energy - env._down_potential_energy) / env._energy_gap, -4.0, 4.0)),
        "com_x": float(com_x),
        "com_v": float(com_v),
        "tip_x": float(tip_x),
        "tip_v": float(tip_v),
        "cart_x": float(qpos[0]),
        "cart_v": float(qvel[0]),
    }


def unpack(vector: np.ndarray, n_links: int) -> dict[str, Any]:
    """Decode the 32-parameter controller vector."""
    if vector.shape != (32,):
        raise ValueError(f"expected 32 controller parameters, got {vector.shape}")
    return {
        "swing_pump": vector[0:4].copy(),
        "swing_angle": vector[4:8].copy(),
        "swing_omega": vector[8:12].copy(),
        "swing_cart": float(vector[12]),
        "swing_cart_velocity": float(vector[13]),
        "kick_gain": float(vector[14]),
        "kick_frequency": float(vector[15]),
        "swing_com_energy": float(vector[16]),
        "swing_tip_energy": float(vector[17]),
        "capture_angle": vector[18:22].copy(),
        "capture_omega": vector[22:26].copy(),
        "capture_cart": float(vector[26]),
        "capture_cart_velocity": float(vector[27]),
        "capture_energy": float(vector[28]),
        "capture_com_velocity": float(vector[29]),
        "gate_energy": float(vector[30]),
        "gate_angle": float(vector[31]),
        "n_links": int(n_links),
    }


def controller_action(
    env: NLinkCartPoleEnv,
    features: dict[str, Any],
    params: dict[str, Any],
    basis: np.ndarray,
    *,
    capture_blend: float,
    time_seconds: float,
) -> tuple[float, str, dict[str, float]]:
    def modal(values: np.ndarray, coefficients: np.ndarray) -> float:
        return float(values @ basis @ coefficients)

    energy_error = float(features["energy_error"])
    swing_pump = energy_error * (
        modal(features["coupling"], params["swing_pump"])
        + params["swing_com_energy"] * float(features["com_v"])
        + params["swing_tip_energy"] * float(features["tip_v"])
    )
    swing_shape = modal(features["sin_angle"], params["swing_angle"]) + modal(
        features["omega"], params["swing_omega"]
    )
    swing_signal = (
        swing_pump
        + swing_shape
        + params["swing_cart"] * float(features["cart_x"])
        + params["swing_cart_velocity"] * float(features["cart_v"])
    )
    # Exact hanging is an equilibrium, so a reproducible finite-duration kick
    # is needed for deterministic zero-noise probes.  It is a bounded search
    # lever, not a hidden reset or injected state.
    if time_seconds < 1.50:
        swing_signal += params["kick_gain"] * np.sin(
            2.0 * np.pi * np.clip(abs(params["kick_frequency"]), 0.10, 3.0) * time_seconds
        )

    capture_signal = (
        modal(features["sin_angle"], params["capture_angle"])
        + modal(features["omega"], params["capture_omega"])
        + params["capture_cart"] * float(features["cart_x"])
        + params["capture_cart_velocity"] * float(features["cart_v"])
        + params["capture_energy"] * energy_error
        + params["capture_com_velocity"] * float(features["com_v"])
    )
    # The capture law is a negative-feedback law in the usual convention;
    # CEM is free to choose either sign for each coefficient.
    swing_action = float(np.tanh(swing_signal))
    capture_action = float(np.tanh(capture_signal))
    action = float(np.clip((1.0 - capture_blend) * swing_action + capture_blend * capture_action, -1.0, 1.0))
    mode = "capture" if capture_blend >= 0.5 else "swing"
    return action, mode, {
        "swing_signal": float(swing_signal),
        "capture_signal": float(capture_signal),
        "swing_action": swing_action,
        "capture_action": capture_action,
    }


def rollout(
    cfg: dict[str, Any],
    *,
    seed: int,
    vector: np.ndarray,
    seconds: float,
    n_links: int,
    progress: float,
    rail_limit: float,
    return_trace: bool = False,
) -> dict[str, Any]:
    local_cfg = {**cfg, "env": {**cfg["env"], "episode_seconds": float(seconds)}}
    # This controller must still start from the actual hanging state at the
    # requested morphology progress; zero progress is only a curriculum probe.
    local_cfg["env"]["init_mode"] = "hanging"
    local_cfg["env"]["init_angle_noise"] = 0.0
    local_cfg["env"]["init_vel_noise"] = 0.0
    local_cfg["env"]["init_cart_vel_noise"] = 0.0
    # This branch is the controller under test; curriculum residual/switch
    # teachers would make the result impossible to attribute.
    local_cfg["env"]["obs_include_capture_features"] = False
    local_cfg["env"]["action_lqr_residual"] = {"enabled": False}
    local_cfg["env"]["action_lqr_switch"] = {"enabled": False}
    env = NLinkCartPoleEnv(local_cfg, progress=progress, seed=seed)
    env.reset(seed=seed)
    params = unpack(vector, n_links)
    basis = modal_basis(n_links)
    mass_matrix = np.zeros((env.model.nv, env.model.nv), dtype=np.float64)
    trace: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    max_cart = 0.0
    best_angle = float("inf")
    best_handoff_score = float("inf")
    best_handoff: dict[str, Any] | None = None
    max_capture_quality = 0.0
    action_sq = 0.0
    action_smooth = 0.0
    previous_action = 0.0
    steps = min(env.max_steps, int(seconds / env.dt))

    for step in range(steps):
        mujoco.mj_fullM(env.model, mass_matrix, env.data.qM)
        features = feature_state(env, mass_matrix)
        max_abs_angle = float(np.max(np.abs(features["absolute"])))
        hinge_rms = float(np.sqrt(np.mean(features["omega"] ** 2)))
        # Blend only when the energy is close enough and the chain is entering
        # the top region.  The angle gate prevents premature capture takeover.
        energy_gate = 1.0 / (1.0 + np.exp(-params["gate_energy"] * (0.80 - abs(float(features["energy_error"])))) )
        angle_gate = 1.0 / (1.0 + np.exp(params["gate_angle"] * (max_abs_angle - 0.45)))
        capture_blend = float(np.clip(energy_gate * angle_gate, 0.0, 1.0))
        action, mode, action_metrics = controller_action(
            env,
            features,
            params,
            basis,
            capture_blend=capture_blend,
            time_seconds=float(step * env.dt),
        )
        _, reward, terminated, truncated, info = env.step([action])
        final_info = dict(info)
        max_cart = max(max_cart, abs(float(info["x"])))
        best_angle = min(best_angle, float(info["max_abs_angle"]))
        handoff_score = (
            30.0 * (float(info["max_abs_angle"]) / 0.15) ** 2
            + 10.0 * (float(info["hinge_velocity_rms"]) / 0.75) ** 2
            + 3.0 * (abs(float(info["x"])) / 1.25) ** 2
            + 3.0 * (abs(float(env.data.qvel[0])) / 0.50) ** 2
        )
        if handoff_score < best_handoff_score:
            best_handoff_score = handoff_score
            best_handoff = {
                "time_seconds": float((step + 1) * env.dt),
                "score": float(handoff_score),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "x": float(info["x"]),
                "cart_velocity": float(env.data.qvel[0]),
                "energy_error": float(features["energy_error"]),
                "energy_fraction": float(features["energy_fraction"]),
                "capture_blend": float(capture_blend),
            }
        max_capture_quality = max(max_capture_quality, float(info.get("capture_quality", 0.0)))
        action_sq += action * action
        action_smooth += (action - previous_action) ** 2
        previous_action = action
        if return_trace and (step % 4 == 0 or info["is_upright"]):
            trace.append(
                {
                    "step": int(step + 1),
                    "time_seconds": float((step + 1) * env.dt),
                    "action": float(action),
                    "mode": mode,
                    "capture_blend": float(capture_blend),
                    "max_abs_angle": float(info["max_abs_angle"]),
                    "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                    "x": float(info["x"]),
                    "cart_velocity": float(env.data.qvel[0]),
                    "energy_error": float(features["energy_error"]),
                    "energy_fraction": float(features["energy_fraction"]),
                    "com_x": float(features["com_x"]),
                    "com_v": float(features["com_v"]),
                    **action_metrics,
                }
            )
        if terminated or truncated:
            break

    success = bool(final_info.get("success", False))
    ever_upright = final_info.get("time_to_first_upright") is not None
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    rail_over = max(0.0, max_cart - rail_limit)
    score = (
        -30000.0 * float(success)
        -7000.0 * max_streak
        -1500.0 * float(ever_upright)
        -1200.0 * max_capture_quality
        + 0.5 * best_angle
        + 0.35 * best_handoff_score
        + 100.0 * rail_over * rail_over
        + 0.01 * action_sq
        + 0.04 * action_smooth
    )
    result = {
        "score": float(score),
        "success": success,
        "ever_upright": bool(ever_upright),
        "best_max_abs_angle": float(best_angle),
        "best_handoff_score": float(best_handoff_score),
        "best_handoff": best_handoff,
        "max_cart_abs": float(max_cart),
        "max_capture_quality": float(max_capture_quality),
        "max_upright_streak_seconds": max_streak,
        "max_centered_upright_streak_seconds": float(final_info.get("max_centered_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "final_info": final_info,
        "trace": trace,
    }
    env.close()
    return result


def initial_sigma() -> np.ndarray:
    sigma = np.full(32, 0.35, dtype=np.float64)
    sigma[0:4] = 1.0
    sigma[4:12] = 0.35
    sigma[12:14] = 0.20
    sigma[14:16] = [0.35, 0.40]
    sigma[16:18] = 1.00
    sigma[18:30] = 0.50
    sigma[30:32] = 1.50
    return sigma


def evaluate(
    cfg: dict[str, Any],
    *,
    vector: np.ndarray,
    seed: int,
    episodes: int,
    seconds: float,
    n_links: int,
    progress: float,
    rail_limit: float,
) -> dict[str, Any]:
    rows = [
        rollout(
            cfg,
            seed=seed + index,
            vector=vector,
            seconds=seconds,
            n_links=n_links,
            progress=progress,
            rail_limit=rail_limit,
        )
        for index in range(episodes)
    ]
    return {
        "score": float(np.mean([row["score"] for row in rows])),
        "success_rate": float(np.mean([float(row["success"]) for row in rows])),
        "ever_upright_rate": float(np.mean([float(row["ever_upright"]) for row in rows])),
        "max_upright_streak_mean": float(np.mean([row["max_upright_streak_seconds"] for row in rows])),
        "max_upright_streak_max": float(np.max([row["max_upright_streak_seconds"] for row in rows])),
        "best_max_abs_angle_min": float(np.min([row["best_max_abs_angle"] for row in rows])),
        "best_handoff_score_min": float(np.min([row["best_handoff_score"] for row in rows])),
        "max_capture_quality_max": float(np.max([row["max_capture_quality"] for row in rows])),
        "episodes": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Search low-dimensional energy/modal 7-link swing-up controller")
    parser.add_argument("--config", default="configs/swingup7_swing_curriculum.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--progress", type=float, default=0.0)
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=32)
    parser.add_argument("--elites", type=int, default=6)
    parser.add_argument("--sigma-decay", type=float, default=0.88)
    parser.add_argument("--sigma-floor", type=float, default=0.01)
    parser.add_argument("--init-json", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")

    cfg = apply_overrides(load_config(args.config), args.override)
    probe = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed)
    n_links = int(probe.n)
    rail_limit = float(probe.rail_limit)
    probe.close()

    rng = np.random.default_rng(args.seed)
    center = np.zeros(32, dtype=np.float64)
    # Start from the recognizable PFL sign convention: energy error times
    # positive horizontal passive-chain velocity, with light cart regulation.
    center[0:4] = np.asarray([8.0, 0.0, 0.0, 0.0])
    center[4:8] = 0.0
    center[8:12] = 0.0
    center[12:14] = [-0.08, -0.12]
    center[14:16] = [1.0, 0.55]
    center[16:18] = [8.0, 0.0]
    center[18:22] = [1.5, 0.0, 0.0, 0.0]
    center[22:26] = [0.5, 0.0, 0.0, 0.0]
    center[26:28] = [-0.6, -0.3]
    center[28:30] = [-0.2, -0.1]
    center[30:32] = [8.0, 8.0]
    if args.init_json:
        prior = json.loads(Path(args.init_json).read_text(encoding="utf-8"))
        prior_vector = np.asarray(prior.get("best_vector", []), dtype=np.float64)
        if prior_vector.shape != (32,):
            raise ValueError(f"{args.init_json} best_vector has shape {prior_vector.shape}; expected (32,)")
        center = prior_vector.copy()
    sigma = initial_sigma()
    best_vector = center.copy()
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()

    for iteration in range(args.iterations + 1):
        candidates = [best_vector.copy()]
        if iteration > 0:
            candidates.extend(center + rng.normal(0.0, sigma, size=center.shape) for _ in range(args.population - 1))
        records: list[dict[str, Any]] = []
        for vector in candidates:
            metrics = evaluate(
                cfg,
                vector=np.asarray(vector, dtype=np.float64),
                seed=args.seed,
                episodes=args.episodes,
                seconds=args.seconds,
                n_links=n_links,
                progress=args.progress,
                rail_limit=rail_limit,
            )
            records.append({"vector": np.asarray(vector, dtype=np.float64), "metrics": metrics})
        records.sort(key=lambda row: float(row["metrics"]["score"]))
        top = records[0]
        if best is None or float(top["metrics"]["score"]) < float(best["score"]):
            best = {
                "score": float(top["metrics"]["score"]),
                "metrics": top["metrics"],
                "vector": top["vector"].astype(float).tolist(),
            }
            best_vector = top["vector"].copy()
        top_metrics = top["metrics"]
        history.append(
            {
                "iteration": iteration,
                "score": float(top_metrics["score"]),
                "best_score": float(best["score"]),
                "ever_upright_rate": float(top_metrics["ever_upright_rate"]),
                "max_upright_streak_max": float(top_metrics["max_upright_streak_max"]),
                "best_max_abs_angle_min": float(top_metrics["best_max_abs_angle_min"]),
                "best_handoff_score_min": float(top_metrics["best_handoff_score_min"]),
            }
        )
        print(
            f"iter={iteration:03d} score={top_metrics['score']:.2f} "
            f"ever={top_metrics['ever_upright_rate']:.2f} "
            f"streak={top_metrics['max_upright_streak_max']:.3f}s "
            f"angle={top_metrics['best_max_abs_angle_min']:.3f} "
            f"handoff={top_metrics['best_handoff_score_min']:.2f}",
            flush=True,
        )
        if iteration > 0:
            elite = np.asarray([row["vector"] for row in records[: args.elites]])
            center = best_vector.copy()
            sigma = np.maximum(elite.std(axis=0), args.sigma_floor) * args.sigma_decay

    assert best is not None
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_path.with_name(out_path.stem + ".config.resolved.yaml"))
    final_eval = evaluate(
        cfg,
        vector=best_vector,
        seed=args.seed + 4444,
        episodes=args.eval_episodes,
        seconds=args.seconds,
        n_links=n_links,
        progress=args.progress,
        rail_limit=rail_limit,
    )
    trace = rollout(
        cfg,
        seed=args.seed + 8888,
        vector=best_vector,
        seconds=args.seconds,
        n_links=n_links,
        progress=args.progress,
        rail_limit=rail_limit,
        return_trace=True,
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Low-dimensional energy-shaping plus modal capture CEM proposal; not canonical 7-link evidence.",
        "config_path": str(Path(args.config)),
        "progress": float(args.progress),
        "seconds": float(args.seconds),
        "parameter_count": 32,
        "modal_basis": modal_basis(n_links).astype(float).tolist(),
        "search": {
            "seed": int(args.seed),
            "episodes": int(args.episodes),
            "eval_episodes": int(args.eval_episodes),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "wall_time_seconds": float(time.time() - started),
        },
        "best_search": best,
        "best_vector": best_vector.astype(float).tolist(),
        "best_parameters": {
            key: (value.astype(float).tolist() if isinstance(value, np.ndarray) else value)
            for key, value in unpack(best_vector, n_links).items()
        },
        "eval": final_eval,
        "trace_rollout": trace,
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
