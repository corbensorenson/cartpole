#!/usr/bin/env python
"""Refine the final action tail of an exact MuJoCo route by endpoint SVD.

The route prefix is held fixed.  A finite-difference endpoint map is built for
only the last action intervals, and a damped minimum-norm correction is applied
with an exact serial replay and line search.  This is a discovery artifact: the
result still needs feedback replay, noisy evaluation, and a sustained hold.
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles
from gcartpole.evidence import data_sha256, file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.ilqr import MujocoTransition, data_state

try:
    from scripts.search_swingup_ilqr_terminal import absolute_rate_transform
except ModuleNotFoundError:
    from search_swingup_ilqr_terminal import absolute_rate_transform


def state_residual(qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
    qpos = np.asarray(qpos, dtype=np.float64)
    qvel = np.asarray(qvel, dtype=np.float64)
    return np.r_[
        qpos[0] / 1.25,
        serial_absolute_angles(qpos[1:]) / 0.15,
        qvel[0] / 0.50,
        np.cumsum(qvel[1:]) / 0.75,
    ].astype(np.float64)


def endpoint_residual(env: NLinkCartPoleEnv) -> np.ndarray:
    return state_residual(env.data.qpos, env.data.qvel)


def deterministic_config(cfg: dict[str, Any], seconds: float) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    result["env"] = {
        **result["env"],
        "init_mode": "hanging",
        "episode_seconds": float(seconds),
        "terminate_abs_angle": None,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        result["env"][key] = 0.0
        result["env"][f"{key}_start"] = 0.0
        result["env"][f"{key}_end"] = 0.0
    return result


def load_controller(path: Path) -> tuple[dict[str, Any], np.ndarray, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    controller = payload.get("controller")
    if not isinstance(controller, dict):
        controller = payload.get("best")
    if not isinstance(controller, dict):
        raise ValueError(f"{path} has no controller record")
    controls = np.asarray(controller.get("controls"), dtype=np.float64)
    if controls.ndim != 1 or controls.size < 2:
        raise ValueError("source controller must contain scalar interval controls")
    seconds = float(
        controller.get(
            "horizon_seconds",
            payload.get("search", {}).get("seconds", controls.size * 0.02),
        )
    )
    if seconds <= 0.0:
        raise ValueError("source controller has no positive horizon")
    return payload, np.clip(controls, -1.0, 1.0), seconds


def damped_step(jacobian: np.ndarray, residual: np.ndarray, regularization: float) -> np.ndarray:
    left, singular_values, right = np.linalg.svd(jacobian, full_matrices=False)
    factors = singular_values / (singular_values * singular_values + regularization)
    return -(right.T @ (factors * (left.T @ residual)))


def replay(
    env: NLinkCartPoleEnv,
    actions: np.ndarray,
    *,
    transition: MujocoTransition | None = None,
    collect_states: bool = False,
    window_steps: int = 1,
) -> dict[str, Any]:
    env.reset(seed=0)
    states = [
        transition.to_coordinates(data_state(env.data)).astype(float).tolist()
    ] if collect_states and transition is not None else []
    trace: list[dict[str, Any]] = []
    terminated = False
    truncated = False
    final_info: dict[str, Any] = {}
    for step, action in enumerate(actions):
        _, _, terminated, truncated, info = env.step([float(np.clip(action, -1.0, 1.0))])
        final_info = dict(info)
        trace.append(
            {
                "step": int(step + 1),
                "time_seconds": float((step + 1) * env.dt),
                "action": float(info["applied_action_norm"]),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "absolute_angular_velocity_rms": float(info["absolute_angular_velocity_rms"]),
            }
        )
        if collect_states and transition is not None:
            states.append(transition.to_coordinates(data_state(env.data)).astype(float).tolist())
        if terminated or truncated:
            break
    if not trace:
        raise RuntimeError("replay produced no states")
    effective_window = min(int(window_steps), len(trace))
    window = trace[-effective_window:]
    residual = np.concatenate(
        [state_residual(row["qpos"], row["qvel"]) for row in window]
    )
    terminal_residual = state_residual(window[-1]["qpos"], window[-1]["qvel"])
    max_cart = max((abs(float(row["qpos"][0])) for row in trace), default=0.0)
    # A fixed-horizon episode reports ``truncated`` on its final step.  That is
    # a normal completion here; only an early termination makes the replay
    # unusable for the endpoint Jacobian.
    complete = len(trace) == len(actions) and not terminated
    env_result = {
        "cost": float(residual @ residual),
        "residual": residual,
        "residual_norm": float(np.linalg.norm(residual)),
        "terminal_residual_norm": float(np.linalg.norm(terminal_residual)),
        "complete": bool(complete),
        "termination_reason": final_info.get("termination_reason"),
        "max_cart_excursion": float(max_cart),
        "trace": trace,
        "states": states,
    }
    return env_result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--tail-steps", type=int, default=50)
    parser.add_argument(
        "--window-steps",
        type=int,
        default=1,
        help="number of final exact states included in the endpoint objective",
    )
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--fd-epsilon", type=float, default=1.0e-3)
    parser.add_argument("--regularization", type=float, default=1.0e-5)
    parser.add_argument("--trust-radius", type=float, default=0.03)
    parser.add_argument("--minimum-step", type=float, default=1.0 / 64.0)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if min(args.tail_steps, args.window_steps, args.iterations, args.fd_epsilon, args.regularization, args.trust_radius, args.minimum_step) <= 0:
        raise ValueError("tail and solver parameters must be positive")

    source_path = Path(args.controller)
    source_payload, source_controls, source_seconds = load_controller(source_path)
    seconds = float(args.seconds if args.seconds is not None else source_seconds)
    cfg = deterministic_config(apply_overrides(load_config(args.config), args.override), seconds)
    env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=0)
    if not np.isclose(seconds / source_controls.size, env.dt, rtol=0.0, atol=1.0e-10):
        raise ValueError("source policy period does not match the target environment dt")
    step_count = max(2, round(seconds / env.dt))
    if source_controls.size < step_count:
        raise ValueError("source controller is shorter than the requested horizon")
    if not 2 <= args.tail_steps <= step_count:
        raise ValueError("tail-steps must be between two and the route horizon")
    if args.window_steps > step_count:
        raise ValueError("window-steps cannot exceed the route horizon")
    actions = source_controls[:step_count].copy()
    correction = np.zeros(args.tail_steps, dtype=np.float64)
    started = time.time()
    current = replay(env, actions, window_steps=args.window_steps)
    best = current
    best_actions = actions.copy()
    history: list[dict[str, Any]] = []

    for iteration in range(args.iterations):
        residual = np.asarray(current["residual"], dtype=np.float64)
        jacobian = np.empty((residual.size, args.tail_steps), dtype=np.float64)
        for column in range(args.tail_steps):
            plus = actions.copy()
            minus = actions.copy()
            plus[-args.tail_steps + column] += args.fd_epsilon
            minus[-args.tail_steps + column] -= args.fd_epsilon
            plus_result = replay(env, plus, window_steps=args.window_steps)
            minus_result = replay(env, minus, window_steps=args.window_steps)
            if not plus_result["complete"] or not minus_result["complete"]:
                jacobian[:, column] = 0.0
            else:
                jacobian[:, column] = (
                    plus_result["residual"] - minus_result["residual"]
                ) / (2.0 * args.fd_epsilon)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        step = damped_step(jacobian, residual, args.regularization)
        maximum = float(np.max(np.abs(step), initial=0.0))
        if maximum > args.trust_radius:
            step *= args.trust_radius / maximum
        accepted = False
        scale = 1.0
        trial = current
        while scale >= args.minimum_step:
            candidate = actions.copy()
            candidate[-args.tail_steps:] = np.clip(
                candidate[-args.tail_steps:] + scale * step, -1.0, 1.0
            )
            candidate_result = replay(env, candidate, window_steps=args.window_steps)
            if candidate_result["complete"] and candidate_result["cost"] < current["cost"]:
                actions = candidate
                current = candidate_result
                trial = candidate_result
                accepted = True
                break
            scale *= 0.5
        if accepted and current["cost"] < best["cost"]:
            best = current
            best_actions = actions.copy()
        history.append(
            {
                "iteration": int(iteration + 1),
                "accepted": bool(accepted),
                "cost": float(current["cost"]),
                "residual_norm": float(current["residual_norm"]),
                "terminal_residual_norm": float(current["terminal_residual_norm"]),
                "max_cart_excursion": float(current["max_cart_excursion"]),
                "jacobian_rank": int(np.linalg.matrix_rank(jacobian)),
                "jacobian_condition": float(singular_values[0] / max(singular_values[-1], 1.0e-30)),
                "minimum_singular_value": float(singular_values[-1]),
                "regularization": float(args.regularization),
                "maximum_step": maximum,
                "step_scale": float(scale),
            }
        )
        print(
            f"iter={iteration + 1:03d} accepted={accepted} cost={current['cost']:.10g} "
            f"residual={current['residual_norm']:.6g} rail={current['max_cart_excursion']:.4f} "
            f"rank={history[-1]['jacobian_rank']} cond={history[-1]['jacobian_condition']:.3g}",
            flush=True,
        )

    transition = None
    source_controller = source_payload.get("controller")
    # The iLQR route stores this fixed scaling in its producer rather than in
    # every controller payload.  Recreate it so the refined route's nominal
    # states correspond to the modified controls, not the stale source route.
    transition = MujocoTransition(
        env,
        coordinate_transform=absolute_rate_transform(
            int(cfg["env"]["n_links"]),
            cart_position=1.25,
            angle=0.15,
            cart_velocity=0.50,
            rate=0.75,
        ),
    )
    verified = replay(
        env,
        best_actions,
        transition=transition,
        collect_states=transition is not None,
        window_steps=args.window_steps,
    )
    controller = copy.deepcopy(source_controller) if isinstance(source_controller, dict) else {}
    controller["type"] = "exact_endpoint_tail_svd_refinement"
    controller["controls"] = best_actions.astype(float).tolist()
    controller["horizon_steps"] = int(step_count)
    controller["horizon_seconds"] = float(step_count * env.dt)
    controller["tail_refinement_steps"] = int(args.tail_steps)
    search = copy.deepcopy(source_payload.get("search", {}))
    if transition is not None and verified["states"]:
        search["nominal_coordinate_states"] = verified["states"]
    search.update(
        {
            "algorithm": "exact_serial_endpoint_tail_svd",
            "source_controller": file_metadata(source_path),
            "tail_steps": int(args.tail_steps),
            "window_steps": int(args.window_steps),
            "iterations": int(args.iterations),
            "fd_epsilon": float(args.fd_epsilon),
            "regularization": float(args.regularization),
            "trust_radius": float(args.trust_radius),
            "initial_cost": float(
                replay(env, source_controls[:step_count], window_steps=args.window_steps)["cost"]
            ),
            "final_cost": float(best["cost"]),
            "final_residual_norm": float(best["residual_norm"]),
            "final_terminal_residual_norm": float(best["terminal_residual_norm"]),
            "history": history,
            "wall_time_seconds": float(time.time() - started),
        }
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_exact_endpoint_tail_not_solution",
        "not_solution": True,
        "summary": "Exact MuJoCo tail endpoint refinement with SVD-damped minimum-norm corrections; feedback and held-out verification required.",
        "config_path": str(Path(args.config)),
        "config_sha256": data_sha256(cfg),
        "source_controller": file_metadata(source_path),
        "controller": controller,
        "search": search,
        "result": {
            "cost": float(best["cost"]),
            "residual_norm": float(best["residual_norm"]),
            "max_cart_excursion": float(best["max_cart_excursion"]),
            "termination_reason": best["termination_reason"],
            "complete": bool(best["complete"]),
            "endpoint": best["trace"][-1] if best["trace"] else None,
        },
        "trace": verified["trace"],
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(payload, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
