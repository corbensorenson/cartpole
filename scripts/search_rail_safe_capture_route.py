#!/usr/bin/env python
"""Search a local rail-safe correction around a successful n-link route.

This is a narrow continuation search, not a new swing-up method.  It keeps the
saved route as the incumbent, perturbs only a selected action window, and
evaluates the complete route followed by the exact upright LQR.  A candidate
is useful only if it preserves the downstream hold while reducing the peak
cart excursion toward the canonical rail.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.generalized_energy import upright_lqr_gain
try:
    from scripts.search_swingup_capture import lqr_action
except ModuleNotFoundError:
    from search_swingup_capture import lqr_action


def load_controls(path: Path) -> tuple[np.ndarray, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    record = payload.get("best", payload)
    if not isinstance(record, dict) or "controls" not in record:
        raise ValueError(f"{path} has no saved best controls")
    controls = np.asarray(record["controls"], dtype=np.float64)
    seconds = float(record.get("horizon_seconds", 0.0))
    if controls.ndim != 1 or controls.size < 2 or seconds <= 0.0:
        raise ValueError(f"{path} has invalid route controls")
    return np.clip(controls, -1.0, 1.0), seconds


def deterministic_config(config_path: str, *, episode_seconds: float) -> dict[str, Any]:
    cfg = copy.deepcopy(load_config(config_path))
    cfg["env"].update(
        {
            "init_mode": "hanging",
            "episode_seconds": float(episode_seconds),
            "terminate_abs_angle": None,
            "init_angle_noise": 0.0,
            "init_angle_noise_start": 0.0,
            "init_angle_noise_end": 0.0,
            "init_vel_noise": 0.0,
            "init_vel_noise_start": 0.0,
            "init_vel_noise_end": 0.0,
            "init_cart_noise": 0.0,
            "init_cart_noise_start": 0.0,
            "init_cart_noise_end": 0.0,
            "init_cart_vel_noise": 0.0,
            "init_cart_vel_noise_start": 0.0,
            "init_cart_vel_noise_end": 0.0,
            "action_lqr_residual": {"enabled": False},
            "action_lqr_switch": {"enabled": False},
            # Discovery rail: the canonical rail is scored, not used as an
            # early termination condition during local optimization.
            "rail_limit": 12.0,
            "rail_limit_start": 12.0,
            "rail_limit_end": 12.0,
        }
    )
    return cfg


def interpolation_matrix(
    knot_count: int, step_count: int, start: int, end: int
) -> np.ndarray:
    if not 0 <= start < end <= step_count:
        raise ValueError("correction window must be inside the route")
    source = np.linspace(float(start), float(end - 1), knot_count)
    target = np.arange(step_count, dtype=np.float64)
    matrix = np.zeros((step_count, knot_count), dtype=np.float64)
    inside = (target >= float(start)) & (target <= float(end - 1))
    local = target[inside]
    right = np.searchsorted(source, local, side="right").clip(1, knot_count - 1)
    left = right - 1
    fraction = (local - source[left]) / (source[right] - source[left])
    rows = np.flatnonzero(inside)
    matrix[rows, left] = 1.0 - fraction
    matrix[rows, right] += fraction
    return matrix


def evaluate(
    cfg: dict[str, Any],
    base_controls: np.ndarray,
    correction: np.ndarray,
    interpolation: np.ndarray,
    gain: np.ndarray,
    *,
    capture_steps: int,
    canonical_rail: float,
    seed: int,
    keep_trace: bool = False,
) -> dict[str, Any]:
    controls = np.clip(base_controls + interpolation @ correction, -1.0, 1.0)
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
    env.reset(seed=seed)
    max_cart = 0.0
    route_peak = 0.0
    trace: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    route_steps = controls.size
    for step in range(route_steps + capture_steps):
        action = (
            float(controls[step])
            if step < route_steps
            else lqr_action(env, gain, scale=1.0, cart_target=0.0)
        )
        _, _, terminated, truncated, info = env.step([action])
        x = float(env.data.qpos[0])
        max_cart = max(max_cart, abs(x))
        if step < route_steps:
            route_peak = max(route_peak, abs(x))
        final_info = dict(info)
        if keep_trace:
            trace.append(
                {
                    "step": int(step + 1),
                    "time_seconds": float((step + 1) * env.dt),
                    "phase": "swing" if step < route_steps else "capture_lqr",
                    "action": float(action),
                    "x": x,
                    "qpos": np.asarray(env.data.qpos).astype(float).tolist(),
                    "qvel": np.asarray(env.data.qvel).astype(float).tolist(),
                    "max_abs_angle": float(info["max_abs_angle"]),
                    "absolute_angular_velocity_rms": float(
                        info["absolute_angular_velocity_rms"]
                    ),
                    "max_upright_streak_seconds": float(
                        info.get("max_upright_streak_seconds", 0.0)
                    ),
                }
            )
        if terminated or truncated:
            break
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    success = bool(final_info.get("success", False))
    rail_excess = max(0.0, max_cart - float(canonical_rail))
    hold_shortfall = max(0.0, 5.0 - max_streak)
    # The hold is a hard prerequisite.  Among candidates that preserve it,
    # minimize the canonical-rail excess and then the size of the correction.
    score = (
        1.0e7 * hold_shortfall**2
        + 1.0e6 * rail_excess**2
        + 0.01 * float(correction @ correction)
        + 0.001 * max_cart
    )
    env.close()
    return {
        "score": float(score),
        "success": success,
        "route_peak": float(route_peak),
        "max_cart_excursion": float(max_cart),
        "canonical_rail_excess": float(rail_excess),
        "max_upright_streak_seconds": max_streak,
        "termination_reason": final_info.get("termination_reason"),
        "simulated_steps": int(step + 1),
        "trace": trace,
        "controls": controls.astype(float).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup8_uniform.yaml")
    parser.add_argument(
        "--controller", default="runs/generalized_solver/n8_gn_capture_preimage10.json"
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--capture-seconds", type=float, default=6.0)
    parser.add_argument("--window-start", type=int, default=105)
    parser.add_argument("--window-end", type=int, default=180)
    parser.add_argument("--knots", type=int, default=24)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=8)
    parser.add_argument("--action-sigma", type=float, default=0.08)
    parser.add_argument("--sigma-decay", type=float, default=0.90)
    parser.add_argument("--sigma-floor", type=float, default=0.002)
    parser.add_argument("--canonical-rail", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    if min(
        args.capture_seconds,
        args.knots,
        args.iterations,
        args.population,
        args.elites,
        args.action_sigma,
        args.sigma_decay,
        args.sigma_floor,
        args.canonical_rail,
    ) <= 0.0:
        raise ValueError("search values must be positive")
    if not 0 < args.elites < args.population:
        raise ValueError("elites must be between one and population minus one")

    source_path = Path(args.controller)
    base_controls, route_seconds = load_controls(source_path)
    cfg = deterministic_config(
        args.config,
        episode_seconds=route_seconds + args.capture_seconds + 0.10,
    )
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    if not np.isclose(route_seconds / base_controls.size, env.dt, atol=1e-10, rtol=0.0):
        raise ValueError("source route period does not match target environment dt")
    gain = upright_lqr_gain(env, control_cost=1000.0)
    interpolation = interpolation_matrix(
        args.knots, base_controls.size, args.window_start, args.window_end
    )
    env.close()

    rng = np.random.default_rng(args.seed)
    mean = np.zeros(args.knots, dtype=np.float64)
    sigma = float(args.action_sigma)
    best: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    for iteration in range(args.iterations):
        corrections = rng.normal(
            mean,
            sigma,
            size=(args.population - 1, args.knots),
        )
        corrections = np.clip(corrections, -0.40, 0.40)
        corrections = np.vstack([mean, corrections])
        results = [
            evaluate(
                cfg,
                base_controls,
                correction,
                interpolation,
                gain,
                capture_steps=round(args.capture_seconds / env.dt),
                canonical_rail=args.canonical_rail,
                seed=args.seed,
                keep_trace=False,
            )
            for correction in corrections
        ]
        order = np.argsort([result["score"] for result in results])
        elite_indices = order[: args.elites]
        mean = np.mean(corrections[elite_indices], axis=0)
        elite_sigma = np.std(corrections[elite_indices], axis=0)
        sigma = max(float(args.sigma_floor), min(sigma * args.sigma_decay, float(np.mean(elite_sigma))))
        incumbent = results[int(order[0])]
        if best is None or incumbent["score"] < best["score"]:
            best = dict(incumbent)
            best["correction"] = corrections[int(order[0])].astype(float).tolist()
        history.append(
            {
                "iteration": iteration + 1,
                "score": float(incumbent["score"]),
                "route_peak": float(incumbent["route_peak"]),
                "max_cart_excursion": float(incumbent["max_cart_excursion"]),
                "max_upright_streak_seconds": float(
                    incumbent["max_upright_streak_seconds"]
                ),
                "sigma": float(sigma),
            }
        )
        print(
            f"iter={iteration + 1:03d} score={incumbent['score']:.4f} "
            f"route_peak={incumbent['route_peak']:.4f} "
            f"max_x={incumbent['max_cart_excursion']:.4f} "
            f"hold={incumbent['max_upright_streak_seconds']:.2f}s sigma={sigma:.4f}",
            flush=True,
        )

    assert best is not None
    # Re-evaluate the incumbent once with a complete trace for the artifact.
    correction = np.asarray(best["correction"], dtype=np.float64)
    traced = evaluate(
        cfg,
        base_controls,
        correction,
        interpolation,
        gain,
        capture_steps=round(args.capture_seconds / env.dt),
        canonical_rail=args.canonical_rail,
        seed=args.seed,
        keep_trace=True,
    )
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "local_rail_safe_capture_search_not_solution",
        "not_solution": True,
        "summary": "Local route correction scored on complete swing-up, LQR capture, and hold.",
        "source_controller": file_metadata(source_path),
        "source_controller_sha256": data_sha256(base_controls.tolist()),
        "config_sha256": data_sha256(cfg),
        "search": {
            "algorithm": "local_cem_route_correction_with_stitched_lqr_hold",
            "route_seconds": float(route_seconds),
            "capture_seconds": float(args.capture_seconds),
            "window_start": int(args.window_start),
            "window_end": int(args.window_end),
            "knots": int(args.knots),
            "iterations": int(args.iterations),
            "population": int(args.population),
            "elites": int(args.elites),
            "seed": int(args.seed),
            "canonical_rail": float(args.canonical_rail),
            "history": history,
        },
        "best": {
            "controls": traced["controls"],
            "correction": best["correction"],
            "route_peak": traced["route_peak"],
            "max_cart_excursion": traced["max_cart_excursion"],
            "canonical_rail_excess": traced["canonical_rail_excess"],
            "max_upright_streak_seconds": traced["max_upright_streak_seconds"],
            "success": traced["success"],
            "termination_reason": traced["termination_reason"],
            "trace": traced["trace"],
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, args.out)
    print(
        f"best route_peak={traced['route_peak']:.4f} "
        f"max_x={traced['max_cart_excursion']:.4f} "
        f"hold={traced['max_upright_streak_seconds']:.2f}s "
        f"success={traced['success']}"
    )


if __name__ == "__main__":
    main()
