#!/usr/bin/env python
"""Evaluate the released settled-launch seven-link hybrid controller.

The hanging LQR, Box-FDDP route, and upright LQR run in one uninterrupted
MuJoCo episode. Public evaluation files contain complete per-episode summaries
without bulky transition traces; pass ``--include-trajectories`` for a local
diagnostic trace.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.config import dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    text_sha256,
    utc_timestamp,
)
from gcartpole.generalized_solver import periodic_coordinate_error
from gcartpole.modal import (
    StateScales,
    dimensionless_absolute_transform,
    dimensionless_wrapped_state,
)

try:
    from scripts.make_lqr_checkpoint import absolute_angle_cost
except ModuleNotFoundError:
    from make_lqr_checkpoint import absolute_angle_cost

try:
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from search_swingup_capture import lqr_gain


def load_controller(path: Path, n_links: int, spec: dict[str, Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    controller = payload.get("controller")
    search = payload.get("search")
    if not isinstance(controller, dict) or not isinstance(search, dict):
        raise TypeError("controller artifact must contain controller and search objects")
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
        "lqr_control_cost": float(controller.get("lqr_control_cost", 1000.0)),
        "lqr_weights": {
            key: float(value)
            for key, value in controller.get("lqr_weights", {}).items()
        },
        "horizon_steps": int(controls.size),
        "horizon_seconds": float(controls.size * 0.02),
        "payload_summary": payload.get("summary"),
        "periodic_coordinate_errors": bool(
            controller.get("periodic_coordinate_errors", False)
        ),
    }


def coordinate_feedback_error(
    current: np.ndarray,
    reference: np.ndarray,
    transform: np.ndarray,
    *,
    periodic: bool,
) -> np.ndarray:
    """Return current-reference, optionally on the nearest joint-angle branch."""

    current_array = np.asarray(current, dtype=np.float64)
    reference_array = np.asarray(reference, dtype=np.float64)
    transform_array = np.asarray(transform, dtype=np.float64)
    error = current_array - reference_array
    if not periodic:
        return error
    state_dim = int(transform_array.shape[0])
    if transform_array.shape != (state_dim, state_dim) or state_dim % 2 != 0:
        raise ValueError("coordinate transform must be even-dimensional and square")
    return periodic_coordinate_error(current_array, reference_array, transform_array)


def validate_release_identity(
    manifest: dict[str, Any], cfg: dict[str, Any], controller: dict[str, Any], settings: dict[str, Any]
) -> None:
    """A release evidence label must describe the controller actually run."""
    experts = manifest["experts"]
    expected = {
        "config_sha256": manifest["benchmark"]["config_sha256"],
        "controller_sha256": manifest["controller"]["sha256"],
        "tracking_gain_scale": experts["swing"]["tracking_gain_scale"],
        "prelude_seconds": experts["conditioning"]["duration_seconds"],
        "settle_mode": "hanging_lqr",
        "settle_scale": experts["conditioning"]["scale"],
        "settle_control_cost": experts["conditioning"]["control_cost"],
        "shift_cart_nominal": experts["swing"]["shift_nominal_cart_position_to_settled_x"],
        "phase_adaptive": False,
        "lqr_scale": experts["capture"]["scale"],
    }
    actual = {
        **settings,
        "config_sha256": data_sha256(cfg),
        "controller_sha256": controller["source"]["sha256"],
        "lqr_scale": controller["lqr_scale"],
    }
    mismatches = [key for key, value in expected.items() if actual.get(key) != value]
    if mismatches:
        raise ValueError(
            "run differs from release manifest: " + ", ".join(mismatches)
            + "; use --diagnostic for transfer experiments"
        )


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


def hanging_lqr_action_from_state(
    qpos: np.ndarray,
    qvel: np.ndarray,
    gain: np.ndarray,
    *,
    scale: float,
) -> float:
    n = int(qpos.size - 1)
    d = n + 1
    state = np.zeros(2 * d, dtype=np.float64)
    state[0] = qpos[0]
    state[1] = wrap_angle(qpos[1] - np.pi)
    state[2 : 1 + n] = wrap_angle(qpos[2 : 1 + n])
    state[d:] = qvel
    return float(np.clip(-float(scale) * float(gain @ state), -1.0, 1.0))


def hanging_lqr_action(env: NLinkCartPoleEnv, gain: np.ndarray, *, scale: float) -> float:
    return hanging_lqr_action_from_state(
        np.asarray(env.data.qpos, dtype=np.float64),
        np.asarray(env.data.qvel, dtype=np.float64),
        gain,
        scale=scale,
    )


def upright_lqr_action_from_state(
    qpos: np.ndarray,
    qvel: np.ndarray,
    gain: np.ndarray,
    *,
    scale: float,
    cart_target: float,
) -> float:
    n = int(qpos.size - 1)
    d = n + 1
    state = np.zeros(2 * d, dtype=np.float64)
    state[0] = qpos[0] - cart_target
    state[1 : 1 + n] = wrap_angle(qpos[1 : 1 + n])
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
    cart_target: float = 0.0,
    include_trajectory: bool = False,
    measurement_noise_std: float = 0.0,
    control_delay_steps: int = 0,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
    _, reset_info = env.reset(seed=seed)
    initial_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    initial_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
    controls = controller["controls"]
    nominal_states = controller["nominal_states"]
    feedback_gains = controller["feedback_gains"]
    transform = controller["transform"]
    route_nominal_states = nominal_states.copy()
    cart_nominal_shift: float | None = None
    trajectory: list[dict[str, Any]] = []
    route_state: dict[str, Any] | None = None
    final_info: dict[str, Any] = {}
    max_cart = abs(float(reset_info.get("x", env.data.qpos[0])))
    first_upright: float | None = None
    phase_cursor = 0
    terminated = False
    truncated = False
    episode_return = 0.0
    measurement_rng = np.random.default_rng(int(seed) ^ 0x5E11507)
    delayed_actions = [0.0] * int(control_delay_steps)
    completed_steps = 0
    for step in range(env.max_steps):
        measured_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
        measured_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
        if measurement_noise_std > 0.0:
            measured_qpos += measurement_rng.normal(0.0, measurement_noise_std, size=measured_qpos.shape)
            measured_qvel += measurement_rng.normal(0.0, measurement_noise_std, size=measured_qvel.shape)
        coordinate_state = dimensionless_wrapped_state(measured_qpos, measured_qvel, transform)
        route_step = step - prelude_steps
        if step < prelude_steps:
            if settle_mode == "hanging_lqr":
                if settle_gain is None:
                    raise ValueError("hanging_lqr settle mode requires a gain")
                action = hanging_lqr_action_from_state(
                    measured_qpos, measured_qvel, settle_gain, scale=settle_scale
                )
                mode = "hanging_lqr_settle"
            else:
                action = 0.0
                mode = "hanging_settle"
        elif phase_adaptive and phase_cursor < controls.size:
            if shift_cart_nominal and cart_nominal_shift is None:
                cart_nominal_shift = float(coordinate_state[0] - nominal_states[0, 0])
                # The measured launch state already includes any parked cart target.
                route_nominal_states[:, 0] += cart_nominal_shift
            candidates = np.arange(
                phase_cursor,
                min(controls.size, phase_cursor + phase_window + 1),
                dtype=np.int64,
            )
            errors = coordinate_feedback_error(
                route_nominal_states[candidates],
                coordinate_state,
                transform,
                periodic=controller["periodic_coordinate_errors"],
            )
            route_step = int(candidates[int(np.argmin(np.einsum("ij,ij->i", errors, errors)))])
            phase_cursor = route_step + 1
            action = float(
                np.clip(
                    controls[route_step]
                    + tracking_gain_scale
                    * feedback_gains[route_step]
                    @ coordinate_feedback_error(
                        coordinate_state,
                        route_nominal_states[route_step],
                        transform,
                        periodic=controller["periodic_coordinate_errors"],
                    ),
                    -1.0,
                    1.0,
                )
            )
            mode = "phase_adaptive_feedback"
        elif not phase_adaptive and route_step < controls.size:
            if shift_cart_nominal and cart_nominal_shift is None:
                cart_nominal_shift = float(coordinate_state[0] - nominal_states[0, 0])
                # The measured launch state already includes any parked cart target.
                route_nominal_states[:, 0] += cart_nominal_shift
            action = float(
                np.clip(
                    controls[route_step]
                    + tracking_gain_scale
                    * feedback_gains[route_step]
                    @ coordinate_feedback_error(
                        coordinate_state,
                        route_nominal_states[route_step],
                        transform,
                        periodic=controller["periodic_coordinate_errors"],
                    ),
                    -1.0,
                    1.0,
                )
            )
            mode = "swing_feedback"
        else:
            action = upright_lqr_action_from_state(
                measured_qpos,
                measured_qvel,
                gain,
                scale=controller["lqr_scale"],
                cart_target=cart_target,
            )
            mode = "capture_lqr"
        commanded_action = float(action)
        if control_delay_steps:
            delayed_actions.append(commanded_action)
            action = float(delayed_actions.pop(0))
        _, reward, terminated, truncated, info = env.step([action])
        completed_steps = step + 1
        episode_return += float(reward)
        final_info = dict(info)
        if first_upright is None and bool(info.get("is_upright", False)):
            first_upright = float((step + 1) * env.dt)
        max_cart = max(max_cart, abs(float(info.get("x", env.data.qpos[0]))))
        if mode in {"swing_feedback", "phase_adaptive_feedback"} and (
            route_step == controls.size - 1
        ):
            route_state = {
                "step": int(step + 1),
                "time_seconds": float((step + 1) * env.dt),
                "qpos": np.asarray(env.data.qpos, dtype=np.float64)
                .astype(float)
                .tolist(),
                "qvel": np.asarray(env.data.qvel, dtype=np.float64)
                .astype(float)
                .tolist(),
                "qacc_warmstart": np.asarray(
                    env.data.qacc_warmstart, dtype=np.float64
                )
                .astype(float)
                .tolist(),
            }
        if include_trajectory:
            trajectory.append(
                {
                    "step": int(step + 1),
                    "time_seconds": float((step + 1) * env.dt),
                    "mode": mode,
                    "commanded_action": commanded_action,
                    "applied_action": float(action),
                    "x": float(info["x"]),
                    "max_abs_angle": float(info["max_abs_angle"]),
                    "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
                    "is_upright": bool(info["is_upright"]),
                    "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
                    "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
                    "qacc_warmstart": np.asarray(
                        env.data.qacc_warmstart, dtype=np.float64
                    )
                    .astype(float)
                    .tolist(),
                }
            )
        if terminated or truncated:
            break
    env.close()
    result = {
        "episode": int(episode),
        "seed": int(seed),
        "initial_qpos": initial_qpos.astype(float).tolist(),
        "initial_qvel": initial_qvel.astype(float).tolist(),
        "return": float(episode_return),
        "success": bool(final_info.get("success", False)),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": final_info.get("termination_reason"),
        "first_upright_time": first_upright,
        "time_to_first_upright": first_upright,
        "time_to_capture": final_info.get("time_to_capture"),
        "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
        "final_upright_streak_seconds": float(final_info.get("upright_streak_seconds", 0.0)),
        "max_low_momentum_upright_streak_seconds": float(
            final_info.get("max_low_momentum_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(max_cart),
        "length": int(completed_steps),
        "final_info": final_info,
        "route_state": route_state,
    }
    if include_trajectory:
        result["trajectory"] = trajectory
        result["length"] = len(trajectory)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved FDDP swing plus LQR capture controller")
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--controller", required=True)
    parser.add_argument("--manifest", default="runs/swingup7_uniform/seven_link_swingup_manifest.json")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=30732)
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
    parser.add_argument("--include-trajectories", action="store_true")
    parser.add_argument("--diagnostic", action="store_true", help="mark custom configurations and transfer runs as development diagnostics")
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

    repo_root = Path(__file__).resolve().parents[1]
    source_git = {
        key: value
        for key, value in git_metadata(repo_root, include_untracked=False).items()
        if key != "root"
    }
    source_cfg = load_config(args.config)
    cfg = copy.deepcopy(source_cfg)
    cfg["env"]["init_mode"] = "hanging"
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"].setdefault("action_lqr_switch", {"enabled": False})["enabled"] = False
    spec = load_config(args.spec)
    controller = load_controller(Path(args.controller), int(cfg["env"]["n_links"]), spec)
    manifest = None
    if not args.diagnostic:
        manifest_path = Path(args.manifest)
        validate_release_identity(
            json.loads(manifest_path.read_text(encoding="utf-8")), source_cfg,
            controller, vars(args),
        )
        manifest = file_metadata(manifest_path)
    probe = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    generated_xml_sha256 = text_sha256(probe.xml)
    observation_dim = int(probe.observation_space.shape[0])
    action_dim = int(probe.action_space.shape[0])
    action_frequency_hz = float(1.0 / probe.dt)
    probe.close()
    gain = lqr_gain(
        cfg,
        progress=1.0,
        fd_eps=1e-7,
        control_cost=controller["lqr_control_cost"],
        q_weights=controller["lqr_weights"],
    )
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
    prelude_steps = round(
        args.prelude_seconds
        / float(cfg["env"]["timestep"] * cfg["env"]["frame_skip"])
    )
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
            include_trajectory=args.include_trajectories,
        )
        for episode in range(args.episodes)
    ]
    success_rate = float(np.mean([episode["success"] for episode in episodes]))
    output = {
        "schema_version": 2,
        "generated_at": utc_timestamp(),
        "claim_status": "development_diagnostic" if args.diagnostic else "canonical_noisy_gate_evidence",
        "summary": "Development hybrid-controller replay." if args.diagnostic else "Released canonical noisy hanging-start evaluation of the settled-launch hybrid controller.",
        "config": {"path": str(Path(args.config)), "resolved_sha256": data_sha256(source_cfg)},
        "spec": file_metadata(Path(args.spec)),
        "controller": controller["source"],
        "controller_summary": "Settled-launch hybrid: hanging LQR, Box-FDDP trajectory feedback, then upright LQR.",
        "policy_manifest": manifest,
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
        "git": source_git,
        "evidence": {
            "deterministic_policy": True,
            "progress": 1.0,
            "plant_progress": 1.0,
            "config": {
                "path": str(Path(args.config)),
                "resolved_sha256": data_sha256(source_cfg),
                "overrides": [],
            },
            "controller": controller["source"],
            "policy_manifest": manifest,
            "generated_xml_sha256": generated_xml_sha256,
            "environment": {
                "n_links": int(cfg["env"]["n_links"]),
                "init_mode": "hanging",
                "force_limit": float(cfg["env"]["force_limit"]),
                "rail_limit": float(cfg["env"]["rail_limit"]),
                "observation_dim": observation_dim,
                "action_dim": action_dim,
                "action_frequency_hz": action_frequency_hz,
                "episode_seconds": float(cfg["env"]["episode_seconds"]),
                "init_angle_noise": float(cfg["env"]["init_angle_noise"]),
                "init_velocity_noise": float(cfg["env"]["init_vel_noise"]),
            },
            "git": source_git,
            "runtime": runtime_metadata(),
        },
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
