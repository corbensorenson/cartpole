#!/usr/bin/env python
"""Evaluate a feedforward swing trajectory with time-varying local feedback.

This is a reset-free diagnostic for the seven-link problem.  The source cart
trajectory and CEM tail define a nominal state/action route from the exact
hanging state.  A finite-horizon Riccati recursion supplies local feedback
around that route, and an upright linear controller takes over after the tail.

The controller is intentionally kept separate from PPO and the environment's
optional LQR hooks so that a result can be attributed to this two-degree-of-
freedom trajectory-following design.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from search_swingup_tail_action_cem import interpolation_matrix, load_controller, replay_to_tail
from probe_swingup_trajectory import trajectory_action


def load_tail_record(path: str, key: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    record = payload.get(key)
    if not isinstance(record, dict) or "knots" not in record:
        raise ValueError(f"{path} does not contain tail record {key!r} with knots")
    return record


def state_from_data(data: mujoco.MjData) -> np.ndarray:
    return np.r_[np.asarray(data.qpos, dtype=np.float64), np.asarray(data.qvel, dtype=np.float64)].copy()


def set_state(model: mujoco.MjModel, data: mujoco.MjData, state: np.ndarray) -> None:
    nq = int(model.nq)
    data.qpos[:] = state[:nq]
    data.qvel[:] = state[nq:]
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)


def step_map(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: np.ndarray,
    action: float,
    *,
    frame_skip: int,
    force_limit: float,
) -> np.ndarray:
    set_state(model, data, state)
    data.ctrl[0] = float(np.clip(action, -1.0, 1.0)) * force_limit
    for _ in range(frame_skip):
        mujoco.mj_step(model, data)
    return state_from_data(data)


def local_linearization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    state: np.ndarray,
    action: float,
    *,
    frame_skip: int,
    force_limit: float,
    fd_eps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference the discrete MuJoCo map at a nominal state/action."""
    state_dim = state.size
    A = np.empty((state_dim, state_dim), dtype=np.float64)
    B = np.empty((state_dim, 1), dtype=np.float64)
    for index in range(state_dim):
        delta = np.zeros(state_dim, dtype=np.float64)
        delta[index] = fd_eps
        plus = step_map(
            model,
            data,
            state + delta,
            action,
            frame_skip=frame_skip,
            force_limit=force_limit,
        )
        minus = step_map(
            model,
            data,
            state - delta,
            action,
            frame_skip=frame_skip,
            force_limit=force_limit,
        )
        A[:, index] = (plus - minus) / (2.0 * fd_eps)
    plus = step_map(
        model,
        data,
        state,
        action + fd_eps,
        frame_skip=frame_skip,
        force_limit=force_limit,
    )
    minus = step_map(
        model,
        data,
        state,
        action - fd_eps,
        frame_skip=frame_skip,
        force_limit=force_limit,
    )
    B[:, 0] = (plus - minus) / (2.0 * fd_eps)
    return A, B


def error_state(actual: np.ndarray, nominal: np.ndarray, n_links: int) -> np.ndarray:
    """State difference in the local relative-angle coordinates."""
    d = n_links + 1
    error = actual - nominal
    error[:d] = actual[:d] - nominal[:d]
    error[1:d] = wrap_angle(actual[1:d] - nominal[1:d])
    return error


def absolute_error_cost(n_links: int, weights: dict[str, float]) -> np.ndarray:
    """Build Q from cart, absolute-angle, and velocity penalties."""
    d = n_links + 1
    state_dim = 2 * d
    transform = np.zeros((state_dim, state_dim), dtype=np.float64)
    transform[0, 0] = 1.0
    transform[d, d] = 1.0
    for index in range(n_links):
        transform[1 + index, 1 : 2 + index] = 1.0
        transform[d + 1 + index, d + 1 : d + 2 + index] = 1.0
    diag = np.asarray(
        [weights["cart_position"]]
        + [weights["absolute_angle"]] * n_links
        + [weights["cart_velocity"]]
        + [weights["absolute_angular_velocity"]] * n_links,
        dtype=np.float64,
    )
    q = transform.T @ np.diag(diag) @ transform
    q += np.diag(
        [0.0]
        + [weights["relative_angle"]] * n_links
        + [0.0]
        + [weights["relative_angular_velocity"]] * n_links
    )
    return q


def finite_horizon_gains(
    linearizations: list[tuple[np.ndarray, np.ndarray]],
    q: np.ndarray,
    r: float,
    terminal_scale: float,
) -> list[np.ndarray]:
    """Return K_t for u_t = u_nominal_t - K_t delta_x_t."""
    gains: list[np.ndarray] = [np.zeros((1, q.shape[0]), dtype=np.float64) for _ in linearizations]
    p = float(terminal_scale) * q
    for index in range(len(linearizations) - 1, -1, -1):
        a, b = linearizations[index]
        s = np.asarray([[float(r)]], dtype=np.float64) + b.T @ p @ b
        rhs = b.T @ p @ a
        try:
            gain = np.linalg.solve(s, rhs)
        except np.linalg.LinAlgError:
            gain = np.linalg.pinv(s) @ rhs
        gains[index] = gain
        p = q + a.T @ p @ (a - b @ gain)
        p = 0.5 * (p + p.T)
        if not np.all(np.isfinite(p)):
            raise FloatingPointError(f"non-finite Riccati matrix at step {index}")
    return gains


def upright_gain(
    env: NLinkCartPoleEnv,
    q: np.ndarray,
    r: float,
    fd_eps: float,
) -> tuple[np.ndarray, float]:
    """Compute a stationary upright gain, with an explicit stability report."""
    state = np.zeros(2 * (env.n + 1), dtype=np.float64)
    data = mujoco.MjData(env.model)
    a, b = local_linearization(
        env.model,
        data,
        state,
        0.0,
        frame_skip=env.frame_skip,
        force_limit=env.force_limit,
        fd_eps=fd_eps,
    )
    p = solve_discrete_are(a, b, q, np.asarray([[float(r)]], dtype=np.float64))
    gain = np.linalg.solve(b.T @ p @ b + float(r), b.T @ p @ a)
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a - b @ gain))))
    return gain, spectral_radius


def metrics(env: NLinkCartPoleEnv, info: dict[str, Any]) -> dict[str, Any]:
    _, absolute = env._angles()
    return {
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
        "max_centered_upright_streak_seconds": float(info.get("max_centered_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(info.get("max_low_momentum_upright_streak_seconds", 0.0)),
    }


def build_nominal_tail(
    env: NLinkCartPoleEnv,
    tail_start_state: np.ndarray,
    tail_actions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    states = np.empty((len(tail_actions) + 1, tail_start_state.size), dtype=np.float64)
    states[0] = tail_start_state
    scratch = mujoco.MjData(env.model)
    for index, action in enumerate(tail_actions):
        states[index + 1] = step_map(
            env.model,
            scratch,
            states[index],
            float(action),
            frame_skip=env.frame_skip,
            force_limit=env.force_limit,
        )
    return states, tail_actions


def evaluate(
    cfg: dict[str, Any],
    *,
    source_controller: dict[str, Any],
    tail_record: dict[str, Any],
    tail_start_seconds: float,
    tail_seconds: float,
    post_seconds: float,
    feedback_scale: float,
    upright_scale: float,
    fd_eps: float,
    control_cost: float,
    terminal_scale: float,
    q_weights: dict[str, float],
    return_trace: bool,
) -> dict[str, Any]:
    cfg = json.loads(json.dumps(cfg))
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        # The source swing is replayed before the tail starts, so the MuJoCo
        # episode budget must include that prefix as well as the evaluation.
        "episode_seconds": float(tail_start_seconds + tail_seconds + post_seconds + 1.0),
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
        "terminate_abs_angle": None,
        "obs_include_capture_features": False,
        "action_lqr_residual": {"enabled": False},
        "action_lqr_switch": {"enabled": False},
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0

    env = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
    replay_state = replay_to_tail(env, source_controller, tail_start_seconds)
    nq = int(env.model.nq)
    tail_steps = max(2, int(round(tail_seconds / env.dt)))
    interpolation = interpolation_matrix(len(tail_record["knots"]), tail_steps)
    tail_actions = np.clip(
        np.asarray(tail_record["knots"], dtype=np.float64) @ interpolation.T,
        -1.0,
        1.0,
    )
    nominal_states, nominal_actions = build_nominal_tail(env, replay_state[1 : 1 + nq + env.model.nv], tail_actions)
    q = absolute_error_cost(env.n, q_weights)
    scratch = mujoco.MjData(env.model)
    linearizations: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(tail_steps):
        linearizations.append(
            local_linearization(
                env.model,
                scratch,
                nominal_states[index],
                float(nominal_actions[index]),
                frame_skip=env.frame_skip,
                force_limit=env.force_limit,
                fd_eps=fd_eps,
            )
        )
    gains = finite_horizon_gains(linearizations, q, control_cost, terminal_scale)
    upright_k, upright_radius = upright_gain(env, q, control_cost, fd_eps)

    env.reset(options={"qpos": replay_state[1 : 1 + nq], "qvel": replay_state[1 + nq : 1 + nq + env.model.nv]})
    trace: list[dict[str, Any]] = []
    action_abs_max = 0.0
    post_steps = max(1, int(round(post_seconds / env.dt)))
    final_info: dict[str, Any] = {}
    stopped_at: str | None = None
    all_rows: list[dict[str, Any]] = []
    total_steps = tail_steps + post_steps
    for step in range(total_steps):
        actual = state_from_data(env.data)
        if step < tail_steps:
            nominal = nominal_states[step]
            delta = error_state(actual, nominal, env.n)
            residual = -float(feedback_scale) * (gains[step] @ delta)[0]
            action = float(np.clip(nominal_actions[step] + residual, -1.0, 1.0))
            mode = "trajectory_feedback"
            gain_norm = float(np.linalg.norm(gains[step]))
            nominal_action = float(nominal_actions[step])
        else:
            delta = error_state(actual, np.zeros_like(actual), env.n)
            residual = -float(upright_scale) * (upright_k @ delta)[0]
            action = float(np.clip(residual, -1.0, 1.0))
            mode = "upright_feedback"
            gain_norm = float(np.linalg.norm(upright_k))
            nominal_action = 0.0
        _, _, terminated, truncated, info = env.step([action])
        row = metrics(env, info)
        row.update(
            {
                "step": int(step + 1),
                "time_seconds": float(tail_start_seconds + (step + 1) * env.dt),
                "action": action,
                "nominal_action": nominal_action,
                "feedback_residual": residual,
                "mode": mode,
                "gain_norm": gain_norm,
            }
        )
        all_rows.append(row)
        if return_trace and (step % 4 == 0 or row["is_upright"] or mode == "upright_feedback"):
            trace.append(row)
        action_abs_max = max(action_abs_max, abs(action))
        final_info = dict(info)
        if terminated or truncated:
            stopped_at = str(info.get("termination_reason"))
            break

    best = min(all_rows, key=lambda row: row["max_abs_angle"]) if all_rows else {}
    result = {
        "success": bool(final_info.get("success", False)),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "max_centered_upright_streak_seconds": float(final_info.get("max_centered_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(final_info.get("max_low_momentum_upright_streak_seconds", 0.0)),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "max_cart_abs": float(max(abs(row["x"]) for row in all_rows)) if all_rows else 0.0,
        "action_abs_max": float(action_abs_max),
        "best_angle_row": best,
        "final_info": final_info,
        "termination_reason": stopped_at,
        "simulated_steps": len(all_rows),
        "trace": trace,
        "nominal_tail_endpoint": {
            "qpos": nominal_states[-1, :nq].astype(float).tolist(),
            "qvel": nominal_states[-1, nq:].astype(float).tolist(),
        },
        "tail_start_state": {
            "qpos": replay_state[1 : 1 + nq].astype(float).tolist(),
            "qvel": replay_state[1 + nq : 1 + nq + env.model.nv].astype(float).tolist(),
        },
        "upright_closed_loop_spectral_radius": float(upright_radius),
        "tail_gain_norm_max": float(max(np.linalg.norm(gain) for gain in gains)),
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate time-varying trajectory feedback for seven-link capture")
    parser.add_argument("--config", default="configs/swingup7_capture_handoff.yaml")
    parser.add_argument("--swing-controller-json", default="runs/swingup7_cart_trajectory_longrail24.json")
    parser.add_argument("--tail-json", default="runs/swingup7_cart_crossing_tail_cem_hingeheavy_feasible.json")
    parser.add_argument("--tail-key", default="best_feasible")
    parser.add_argument("--tail-start-seconds", type=float, default=None)
    parser.add_argument("--tail-seconds", type=float, default=None)
    parser.add_argument("--post-seconds", type=float, default=5.0)
    parser.add_argument("--feedback-scale", type=float, default=1.0)
    parser.add_argument("--upright-scale", type=float, default=1.0)
    parser.add_argument("--fd-eps", type=float, default=1e-6)
    parser.add_argument("--control-cost", type=float, default=100.0)
    parser.add_argument("--terminal-scale", type=float, default=10.0)
    parser.add_argument("--cart-position-cost", type=float, default=0.2)
    parser.add_argument("--absolute-angle-cost", type=float, default=100.0)
    parser.add_argument("--cart-velocity-cost", type=float, default=0.2)
    parser.add_argument("--absolute-angular-velocity-cost", type=float, default=2.0)
    parser.add_argument("--relative-angle-cost", type=float, default=0.5)
    parser.add_argument("--relative-angular-velocity-cost", type=float, default=0.05)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    tail_payload = json.loads(Path(args.tail_json).read_text(encoding="utf-8"))
    tail_record = load_tail_record(args.tail_json, args.tail_key)
    tail_start = float(args.tail_start_seconds if args.tail_start_seconds is not None else tail_payload["tail_start_seconds"])
    tail_seconds = float(
        args.tail_seconds
        if args.tail_seconds is not None
        else tail_payload.get("tail_horizon_seconds", 5.0)
    )
    source_controller = load_controller(args.swing_controller_json, None)
    q_weights = {
        "cart_position": float(args.cart_position_cost),
        "absolute_angle": float(args.absolute_angle_cost),
        "cart_velocity": float(args.cart_velocity_cost),
        "absolute_angular_velocity": float(args.absolute_angular_velocity_cost),
        "relative_angle": float(args.relative_angle_cost),
        "relative_angular_velocity": float(args.relative_angular_velocity_cost),
    }
    result = evaluate(
        cfg,
        source_controller=source_controller,
        tail_record=tail_record,
        tail_start_seconds=tail_start,
        tail_seconds=tail_seconds,
        post_seconds=float(args.post_seconds),
        feedback_scale=float(args.feedback_scale),
        upright_scale=float(args.upright_scale),
        fd_eps=float(args.fd_eps),
        control_cost=float(args.control_cost),
        terminal_scale=float(args.terminal_scale),
        q_weights=q_weights,
        return_trace=bool(args.trace),
    )
    payload = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Reset-free seven-link feedforward tail with finite-horizon time-varying feedback and an upright takeover.",
        "config_path": str(Path(args.config)),
        "swing_controller_json": str(Path(args.swing_controller_json)),
        "tail_json": str(Path(args.tail_json)),
        "tail_key": args.tail_key,
        "tail_start_seconds": tail_start,
        "tail_seconds": tail_seconds,
        "post_seconds": float(args.post_seconds),
        "controller": {
            "method": "nominal_tail_plus_finite_horizon_riccati_feedback",
            "feedback_scale": float(args.feedback_scale),
            "upright_scale": float(args.upright_scale),
            "fd_eps": float(args.fd_eps),
            "control_cost": float(args.control_cost),
            "terminal_scale": float(args.terminal_scale),
            "q_weights": q_weights,
        },
        "result": result,
        "source_controller": source_controller,
        "tail_record_metrics": {key: tail_record.get(key) for key in ("cost", "max_angle", "hinge_rms", "max_hinge", "cart_abs", "cart_velocity", "rail")},
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(payload, Path(args.out))
    print(
        f"success={result['success']} streak={result['max_upright_streak_seconds']:.3f}s "
        f"centered={result['max_centered_upright_streak_seconds']:.3f}s "
        f"low_momentum={result['max_low_momentum_upright_streak_seconds']:.3f}s "
        f"best_angle={result['best_angle_row'].get('max_abs_angle', float('nan')):.4f} "
        f"upright_radius={result['upright_closed_loop_spectral_radius']:.4f}",
        flush=True,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
