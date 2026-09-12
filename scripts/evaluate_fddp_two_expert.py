#!/usr/bin/env python
"""Evaluate a saved exact-MuJoCo swing trajectory followed by LQR capture.

This evaluator keeps the route and capture stages in one uninterrupted
environment. It is intended for discovery and robustness checks; canonical
claim gates still require the repository's full evidence workflow.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import mujoco
from scipy.linalg import solve_discrete_are

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.ilqr import data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform, dimensionless_wrapped_state

try:
    from scripts.make_lqr_checkpoint import absolute_angle_cost
except ModuleNotFoundError:
    from make_lqr_checkpoint import absolute_angle_cost

try:
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from search_swingup_capture import lqr_action, lqr_gain


def load_controller(path: Path, n_links: int, spec: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    controller = payload.get("controller")
    search = payload.get("search")
    if not isinstance(controller, dict) or not isinstance(search, dict):
        raise ValueError("controller artifact must contain controller and search objects")
    controls = np.asarray(controller.get("controls"), dtype=np.float64)
    feedback_gains = np.asarray(controller.get("feedback_gains"), dtype=np.float64)
    nominal_states = np.asarray(search.get("nominal_coordinate_states"), dtype=np.float64)
    if controls.ndim != 1 or feedback_gains.ndim != 2 or nominal_states.ndim != 2:
        raise ValueError("controller arrays have invalid dimensions")
    state_dim = 2 * (n_links + 1)
    if (
        nominal_states.shape != (controls.size + 1, state_dim)
        or feedback_gains.shape != (controls.size, state_dim)
    ):
        raise ValueError("controller arrays have inconsistent horizons")
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        n_links,
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    return {
        "source": file_metadata(path),
        "controls": controls,
        "feedback_gains": feedback_gains,
        "nominal_states": nominal_states,
        "transform": transform,
        "lqr_scale": float(controller.get("lqr_scale", 1.0)),
        "horizon_steps": int(controls.size),
        "horizon_seconds": float(controls.size * 0.02),
        "payload_summary": payload.get("summary"),
    }


def hanging_lqr_gain(
    cfg: dict[str, Any],
    *,
    progress: float,
    fd_eps: float,
    control_cost: float,
) -> np.ndarray:
    """Linearize and stabilize the exact hanging equilibrium for the prelude."""
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=0)
    n = env.n
    d = n + 1
    state_dim = 2 * d
    reference = np.zeros(d, dtype=np.float64)
    reference[1] = np.pi
    x0 = np.r_[reference, np.zeros(d, dtype=np.float64)]

    def step_map(state: np.ndarray, action: float) -> np.ndarray:
        env.data.qpos[:] = state[:d]
        env.data.qvel[:] = state[d:]
        env.data.ctrl[:] = float(np.clip(action, -1.0, 1.0)) * env.force_limit
        mujoco.mj_forward(env.model, env.data)
        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, env.data)
        return np.r_[env.data.qpos.copy(), env.data.qvel.copy()]

    a = np.zeros((state_dim, state_dim), dtype=np.float64)
    b = np.zeros((state_dim, 1), dtype=np.float64)
    for index in range(state_dim):
        delta = np.zeros(state_dim, dtype=np.float64)
        delta[index] = fd_eps
        a[:, index] = (step_map(x0 + delta, 0.0) - step_map(x0 - delta, 0.0)) / (2.0 * fd_eps)
    b[:, 0] = (step_map(x0, fd_eps) - step_map(x0, -fd_eps)) / (2.0 * fd_eps)
    q = absolute_angle_cost(
        n,
        {
            "cart_position": 10.0,
            "absolute_angle": 25.0,
            "cart_velocity": 5.0,
            "absolute_angular_velocity": 10.0,
            "relative_angle": 1.0,
            "relative_angular_velocity": 1.0,
        },
    )
    r = np.array([[float(control_cost)]], dtype=np.float64)
    p = solve_discrete_are(a, b, q, r)
    gain = np.linalg.solve(b.T @ p @ b + r, b.T @ p @ a).reshape(-1)
    env.close()
    return gain


def hanging_lqr_action(env: NLinkCartPoleEnv, gain: np.ndarray, *, scale: float) -> float:
    n = env.n
    d = n + 1
    qpos = np.asarray(env.data.qpos, dtype=np.float64)
    qvel = np.asarray(env.data.qvel, dtype=np.float64)
    state = np.zeros(2 * d, dtype=np.float64)
    state[0] = qpos[0]
    state[1] = wrap_angle(qpos[1] - np.pi)
    state[2 : 1 + n] = wrap_angle(qpos[2 : 1 + n])
    state[d:] = qvel
    return float(np.clip(-float(scale) * float(gain @ state), -1.0, 1.0))


def run_episode(
    cfg: dict[str, Any],
    controller: dict[str, Any],
    gain: np.ndarray,
    *,
    seed: int,
    episode: int,
    tracking_gain_scale: float,
    prelude_steps: int,
    settle_mode: str,
    settle_gain: np.ndarray | None,
    settle_scale: float,
    phase_adaptive: bool,
    phase_window: int,
    shift_cart_nominal: bool,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
    _, reset_info = env.reset(seed=seed)
    controls = controller["controls"]
    nominal_states = controller["nominal_states"]
    feedback_gains = controller["feedback_gains"]
    transform = controller["transform"]
    route_nominal_states = nominal_states.copy()
    cart_nominal_shift: float | None = None
    trajectory: list[dict[str, Any]] = []
    final_info: dict[str, Any] = {}
    max_cart = abs(float(reset_info.get("x", env.data.qpos[0])))
    first_upright: float | None = None
    phase_cursor = 0
    terminated = False
    truncated = False
    for step in range(env.max_steps):
        coordinate_state = dimensionless_wrapped_state(
            env.data.qpos, env.data.qvel, transform
        )
        route_step = step - prelude_steps
        if step < prelude_steps:
            if settle_mode == "hanging_lqr":
                if settle_gain is None:
                    raise ValueError("hanging_lqr settle mode requires a gain")
                action = hanging_lqr_action(env, settle_gain, scale=settle_scale)
                mode = "hanging_lqr_settle"
            else:
                action = 0.0
                mode = "hanging_settle"
        elif phase_adaptive and phase_cursor < controls.size:
            if shift_cart_nominal and cart_nominal_shift is None:
                cart_nominal_shift = float(coordinate_state[0] - nominal_states[0, 0])
                route_nominal_states[:, 0] += cart_nominal_shift
            candidates = np.arange(
                phase_cursor,
                min(controls.size, phase_cursor + phase_window + 1),
                dtype=np.int64,
            )
            errors = route_nominal_states[candidates] - coordinate_state
            route_step = int(candidates[int(np.argmin(np.einsum("ij,ij->i", errors, errors)))])
            phase_cursor = route_step + 1
            action = float(
                np.clip(
                    controls[route_step]
                    + tracking_gain_scale
                    * feedback_gains[route_step]
                    @ (coordinate_state - route_nominal_states[route_step]),
                    -1.0,
                    1.0,
                )
            )
            mode = "phase_adaptive_feedback"
        elif not phase_adaptive and route_step < controls.size:
            if shift_cart_nominal and cart_nominal_shift is None:
                cart_nominal_shift = float(coordinate_state[0] - nominal_states[0, 0])
                route_nominal_states[:, 0] += cart_nominal_shift
            action = float(
                np.clip(
                    controls[route_step]
                    + tracking_gain_scale
                    * feedback_gains[route_step]
                    @ (coordinate_state - route_nominal_states[route_step]),
                    -1.0,
                    1.0,
                )
            )
            mode = "swing_feedback"
        else:
            action = lqr_action(
                env,
                gain,
                scale=controller["lqr_scale"],
                cart_target=0.0,
            )
            mode = "capture_lqr"
        _, reward, terminated, truncated, info = env.step([action])
        final_info = dict(info)
        if first_upright is None and bool(info.get("is_upright", False)):
            first_upright = float((step + 1) * env.dt)
        max_cart = max(max_cart, abs(float(info.get("x", env.data.qpos[0]))))
        trajectory.append(
            {
                "step": int(step + 1),
                "time_seconds": float((step + 1) * env.dt),
                "mode": mode,
                "action": float(action),
                "x": float(info["x"]),
                "max_abs_angle": float(info["max_abs_angle"]),
                "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                "is_upright": bool(info["is_upright"]),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
            }
        )
        if terminated or truncated:
            break
    env.close()
    return {
        "episode": int(episode),
        "seed": int(seed),
        "initial_qpos": np.asarray(reset_info.get("qpos", []), dtype=np.float64).astype(float).tolist(),
        "initial_qvel": np.asarray(reset_info.get("qvel", []), dtype=np.float64).astype(float).tolist(),
        "success": bool(final_info.get("success", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": final_info.get("termination_reason"),
        "first_upright_time": first_upright,
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(
            final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(max_cart),
        "length": int(len(trajectory)),
        "final_info": final_info,
        "trajectory": trajectory,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved FDDP swing plus LQR capture controller")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20732)
    parser.add_argument("--tracking-gain-scale", type=float, default=1.0)
    parser.add_argument("--prelude-seconds", type=float, default=0.0)
    parser.add_argument("--settle-mode", choices=("zero", "hanging_lqr"), default="zero")
    parser.add_argument("--settle-scale", type=float, default=1.0)
    parser.add_argument("--settle-control-cost", type=float, default=1000.0)
    parser.add_argument("--phase-adaptive", action="store_true")
    parser.add_argument("--phase-window", type=int, default=12)
    parser.add_argument(
        "--shift-cart-nominal",
        action="store_true",
        help="translate the nominal cart-position channel to the measured settled cart position",
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if (
        args.episodes < 1
        or args.tracking_gain_scale < 0.0
            or args.prelude_seconds < 0.0
            or args.settle_scale < 0.0
            or args.settle_control_cost <= 0.0
            or args.phase_window < 0
    ):
        raise ValueError("episodes must be positive and scales/durations nonnegative")

    cfg = copy.deepcopy(load_config(args.config))
    cfg["env"]["init_mode"] = "hanging"
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"].setdefault("action_lqr_switch", {"enabled": False})["enabled"] = False
    spec = load_config(args.spec)
    controller = load_controller(Path(args.controller), int(cfg["env"]["n_links"]), spec)
    gain = lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    settle_gain = (
        hanging_lqr_gain(
            cfg,
            progress=1.0,
            fd_eps=1e-7,
            control_cost=args.settle_control_cost,
        )
        if args.settle_mode == "hanging_lqr"
        else None
    )
    prelude_steps = int(round(args.prelude_seconds / float(cfg["env"]["timestep"] * cfg["env"]["frame_skip"])))
    episodes = [
        run_episode(
            cfg,
            controller,
            gain,
            seed=args.seed + episode,
            episode=episode,
            tracking_gain_scale=args.tracking_gain_scale,
            prelude_steps=prelude_steps,
            settle_mode=args.settle_mode,
            settle_gain=settle_gain,
            settle_scale=args.settle_scale,
            phase_adaptive=args.phase_adaptive,
            phase_window=args.phase_window,
            shift_cart_nominal=args.shift_cart_nominal,
        )
        for episode in range(args.episodes)
    ]
    success_rate = float(np.mean([episode["success"] for episode in episodes]))
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_claim": True,
        "summary": "Canonical noisy hanging-start robustness check for a saved two-expert controller.",
        "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(cfg)},
        "spec": file_metadata(Path(args.spec)),
        "controller": controller["source"],
        "controller_summary": controller["payload_summary"],
        "episodes": int(args.episodes),
        "seed_start": int(args.seed),
        "tracking_gain_scale": float(args.tracking_gain_scale),
        "prelude_seconds": float(prelude_steps * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]),
        "settle_mode": str(args.settle_mode),
        "settle_scale": float(args.settle_scale),
        "settle_control_cost": float(args.settle_control_cost),
        "settle_controller": (
            {
                "type": "hanging_equilibrium_lqr",
                "reference_qpos": [0.0, float(np.pi)] + [0.0] * int(cfg["env"]["n_links"] - 1),
                "fd_eps": 1e-7,
                "control_cost": float(args.settle_control_cost),
                "gain": settle_gain.astype(float).tolist() if settle_gain is not None else None,
                "gain_sha256": data_sha256(settle_gain.astype(float).tolist()) if settle_gain is not None else None,
            }
            if settle_gain is not None
            else None
        ),
        "phase_adaptive": bool(args.phase_adaptive),
        "phase_window": int(args.phase_window),
        "shift_cart_nominal": bool(args.shift_cart_nominal),
        "success_rate": success_rate,
        "ever_upright_rate": float(np.mean([episode["first_upright_time"] is not None for episode in episodes])),
        "max_upright_streak_mean": float(np.mean([episode["max_upright_streak_seconds"] for episode in episodes])),
        "max_upright_streak_max": float(np.max([episode["max_upright_streak_seconds"] for episode in episodes])),
        "max_cart_excursion_max": float(np.max([episode["max_cart_excursion"] for episode in episodes])),
        "termination_counts": dict(Counter(str(episode["termination_reason"]) for episode in episodes)),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
        "episode_results": episodes,
    }
    dump_json(output, Path(args.out))
    print(
        f"episodes={args.episodes} success_rate={success_rate:.3f} "
        f"ever_upright={output['ever_upright_rate']:.3f} "
        f"hold_mean={output['max_upright_streak_mean']:.3f} "
        f"hold_max={output['max_upright_streak_max']:.3f} "
        f"cart_max={output['max_cart_excursion_max']:.3f}"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
