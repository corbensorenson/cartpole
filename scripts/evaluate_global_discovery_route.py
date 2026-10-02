#!/usr/bin/env python
"""Evaluate a saved exact-MuJoCo route as the six-link discovery gate.

This is deliberately separate from the P3 cohort evaluator.  It proves only
that a feedback route exists from the canonical hanging distribution.  Route
selection is done with isolated forward predictions, while the reported
episode is one live MuJoCo episode with a single initial reset and a complete
state/action trace.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    text_sha256,
    utc_timestamp,
)
from gcartpole.modal import dimensionless_wrapped_state

try:
    from scripts.evaluate_fddp_two_expert import (
        coordinate_feedback_error,
        hanging_lqr_action_from_state,
        hanging_lqr_gain,
        load_controller,
        run_episode,
        upright_lqr_action_from_state,
    )
    from scripts.evaluate_generalized_route_library import fixed_state_config
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from evaluate_fddp_two_expert import (
        coordinate_feedback_error,
        hanging_lqr_action_from_state,
        hanging_lqr_gain,
        load_controller,
        run_episode,
        upright_lqr_action_from_state,
    )
    from evaluate_generalized_route_library import fixed_state_config
    from search_swingup_capture import lqr_gain


DEFAULT_CONTROLLERS = (
    "runs/generalized_solver/n6_rail300_route.json",
    "runs/generalized_solver/n6_rail300_route_mirror.json",
)


def json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def trace_row(
    env: NLinkCartPoleEnv,
    *,
    step: int,
    phase: str,
    route_index: int | None,
    action: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "phase": phase,
        "route_index": None if route_index is None else int(route_index),
        "action": float(action),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "x": float(info.get("x", env.data.qpos[0])),
        "max_abs_angle": float(info.get("max_abs_angle", 0.0)),
        "hinge_velocity_rms": float(info.get("hinge_velocity_rms", 0.0)),
        "is_upright": bool(info.get("is_upright", False)),
        "upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
        "max_upright_streak_seconds": float(
            info.get("max_upright_streak_seconds", 0.0)
        ),
        "termination_reason": info.get("termination_reason"),
    }


def execute_live_route(
    env: NLinkCartPoleEnv,
    controller: dict[str, Any],
    upright_gain: np.ndarray,
    *,
    tracking_gain_scale: float,
    phase_window: int,
    trace: list[dict[str, Any]],
    step_offset: int,
    include_trace: bool,
) -> tuple[dict[str, Any], list[float], int, dict[str, Any] | None]:
    controls = controller["controls"]
    nominal = controller["nominal_states"].copy()
    feedback = controller["feedback_gains"]
    transform = controller["transform"]
    phase_cursor = 0
    cart_shift: float | None = None
    cart_positions = [float(env.data.qpos[0])]
    final_info: dict[str, Any] = env._info()
    handoff_state: dict[str, Any] | None = None
    while env.step_count < env.max_steps:
        qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
        if phase_cursor < controls.size:
            coordinates = dimensionless_wrapped_state(qpos, qvel, transform)
            if cart_shift is None:
                cart_shift = float(coordinates[0] - nominal[0, 0])
                nominal[:, 0] += cart_shift
            candidates = np.arange(
                phase_cursor,
                min(controls.size, phase_cursor + phase_window + 1),
                dtype=np.int64,
            )
            errors = coordinate_feedback_error(
                nominal[candidates],
                coordinates,
                transform,
                periodic=controller["periodic_coordinate_errors"],
            )
            route_index = int(
                candidates[int(np.argmin(np.einsum("ij,ij->i", errors, errors)))]
            )
            phase_cursor = route_index + 1
            action = float(
                np.clip(
                    controls[route_index]
                    + tracking_gain_scale
                    * feedback[route_index]
                    @ coordinate_feedback_error(
                        coordinates,
                        nominal[route_index],
                        transform,
                        periodic=controller["periodic_coordinate_errors"],
                    ),
                    -1.0,
                    1.0,
                )
            )
            phase = "swing_route_feedback"
        else:
            route_index = None
            if handoff_state is None:
                handoff_state = {
                    "step": int(step_offset + env.step_count),
                    "time_seconds": float(env.step_count * env.dt),
                    "qpos": qpos.astype(float).tolist(),
                    "qvel": qvel.astype(float).tolist(),
                    "route_steps": int(controls.size),
                }
            action = upright_lqr_action_from_state(
                qpos,
                qvel,
                upright_gain,
                scale=controller["lqr_scale"],
                cart_target=0.0,
            )
            phase = "capture_lqr"
        _, _, terminated, truncated, info = env.step([action])
        final_info = dict(info)
        cart_positions.append(float(env.data.qpos[0]))
        if include_trace:
            trace.append(
                trace_row(
                    env,
                    step=step_offset + int(env.step_count),
                    phase=phase,
                    route_index=route_index,
                    action=action,
                    info=info,
                )
            )
        if terminated or truncated:
            break
    return final_info, cart_positions, phase_cursor, handoff_state


def replay_capture_from_handoff(
    cfg: dict[str, Any],
    handoff_state: dict[str, Any],
    gain: np.ndarray,
    *,
    lqr_scale: float,
    seconds: float,
) -> dict[str, Any]:
    """Replay only the capture expert from a measured live-route state."""

    replay_cfg = fixed_state_config(
        cfg,
        np.asarray(handoff_state["qpos"], dtype=np.float64),
        np.asarray(handoff_state["qvel"], dtype=np.float64),
    )
    replay_cfg["env"]["episode_seconds"] = float(seconds)
    replay_cfg["env"]["action_lqr_residual"] = {"enabled": False}
    replay_cfg["env"]["action_lqr_switch"] = {"enabled": False}
    env = NLinkCartPoleEnv(replay_cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    max_cart = abs(float(env.data.qpos[0]))
    final_info: dict[str, Any] = env._info()
    for _ in range(env.max_steps):
        qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
        action = upright_lqr_action_from_state(
            qpos,
            qvel,
            gain,
            scale=lqr_scale,
            cart_target=0.0,
        )
        _, _, terminated, truncated, info = env.step([action])
        final_info = dict(info)
        max_cart = max(max_cart, abs(float(env.data.qpos[0])))
        if terminated or truncated:
            break
    env.close()
    return {
        "success": bool(final_info.get("success", False)),
        "termination_reason": final_info.get("termination_reason"),
        "max_upright_streak_seconds": float(
            final_info.get("max_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(max_cart),
    }


def benchmark_checks(cfg: dict[str, Any], *, source_init_mode: str) -> dict[str, bool]:
    env_cfg = cfg["env"]
    return {
        "n_links": int(env_cfg["n_links"]) == 6,
        "total_length": np.isclose(float(env_cfg["total_length"]), 3.0),
        "total_mass": np.isclose(float(env_cfg["total_mass"]), 1.0),
        "cart_mass": np.isclose(float(env_cfg["cart_mass"]), 1.0),
        "rail_limit": np.isclose(float(env_cfg["rail_limit"]), 3.0),
        "force_limit": np.isclose(float(env_cfg["force_limit"]), 80.0),
        "timestep": np.isclose(float(env_cfg["timestep"]), 0.005),
        "frame_skip": int(env_cfg["frame_skip"]) == 4,
        "cart_damping": np.isclose(float(env_cfg["cart_damping"]), 0.02),
        "joint_armature": np.isclose(float(env_cfg["joint_armature"]), 0.0005),
        "hanging_start": str(env_cfg["init_mode"]) == "hanging",
        "source_was_not_fixed_state": source_init_mode not in {"fixed_state", "state_list"},
        "nonzero_angle_noise": float(env_cfg["init_angle_noise"]) > 0.0,
        "nonzero_velocity_noise": float(env_cfg["init_vel_noise"]) > 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--controller", action="append", default=None)
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=90001)
    parser.add_argument("--conditioning-seconds", type=float, default=15.0)
    parser.add_argument("--tracking-gain-scale", type=float, default=1.0)
    parser.add_argument("--phase-window", type=int, default=12)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument(
        "--include-traces",
        action="store_true",
        help="include complete per-step qpos/qvel/action traces in the output",
    )
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()
    if args.episodes < 4:
        raise ValueError("the discovery gate requires at least four held-out starts")
    if args.conditioning_seconds < 0.0 or args.phase_window < 0:
        raise ValueError("conditioning duration and phase window must be nonnegative")
    controller_paths = tuple(args.controller or DEFAULT_CONTROLLERS)
    if len(controller_paths) < 1:
        raise ValueError("at least one route controller is required")

    source_cfg = load_config(args.config)
    source_init_mode = str(source_cfg["env"].get("init_mode", ""))
    cfg = apply_overrides(source_cfg, args.override)
    cfg["env"] = copy.deepcopy(cfg["env"])
    cfg["env"]["init_mode"] = "hanging"
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"]["action_lqr_switch"] = {"enabled": False}
    spec = load_config("benchmarks/p1_capture_envelope.yaml")
    n_links = int(cfg["env"]["n_links"])
    controllers = [
        load_controller(Path(path), n_links, spec) for path in controller_paths
    ]
    upright_gains = [
        lqr_gain(
            cfg,
            progress=1.0,
            fd_eps=1e-7,
            control_cost=controller["lqr_control_cost"],
            q_weights=controller["lqr_weights"],
        )
        for controller in controllers
    ]
    hanging_gain = hanging_lqr_gain(
        cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0
    )
    episodes: list[dict[str, Any]] = []
    for episode_index in range(args.episodes):
        seed = int(args.seed + episode_index)
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
        _, reset_info = env.reset(seed=seed)
        initial_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
        initial_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
        trace: list[dict[str, Any]] = []
        conditioning_steps = round(args.conditioning_seconds / env.dt)
        for _ in range(conditioning_steps):
            qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
            qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
            action = hanging_lqr_action_from_state(qpos, qvel, hanging_gain, scale=1.0)
            _, _, terminated, truncated, info = env.step([action])
            if args.include_traces:
                trace.append(
                    trace_row(
                        env,
                        step=int(env.step_count),
                        phase="hanging_lqr_conditioning",
                        route_index=None,
                        action=action,
                        info=info,
                    )
                )
            if terminated or truncated:
                raise RuntimeError("conditioning terminated before route selection")

        start_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
        start_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
        predictive_cfg = fixed_state_config(cfg, start_qpos, start_qvel)
        predictions = [
            run_episode(
                predictive_cfg,
                controller,
                upright_gains[index],
                seed=0,
                episode=episode_index,
                tracking_gain_scale=args.tracking_gain_scale,
                prelude_steps=0,
                settle_mode="zero",
                settle_gain=None,
                settle_scale=1.0,
                phase_adaptive=True,
                phase_window=args.phase_window,
                shift_cart_nominal=True,
                cart_target=0.0,
                include_trajectory=False,
            )
            for index, controller in enumerate(controllers)
        ]
        selected = min(
            range(len(predictions)),
            key=lambda index: (
                not bool(predictions[index]["success"]),
                -float(predictions[index]["max_upright_streak_seconds"]),
                float(predictions[index]["max_cart_excursion"]),
                index,
            ),
        )
        final_info, cart_positions, route_cursor, handoff_state = execute_live_route(
            env,
            controllers[selected],
            upright_gains[selected],
            tracking_gain_scale=args.tracking_gain_scale,
            phase_window=args.phase_window,
            trace=trace,
            step_offset=0,
            include_trace=args.include_traces,
        )
        capture_replay = (
            None
            if handoff_state is None
            else replay_capture_from_handoff(
                cfg,
                handoff_state,
                upright_gains[selected],
                lqr_scale=controllers[selected]["lqr_scale"],
                seconds=15.0,
            )
        )
        env.close()
        episodes.append(
            {
                "episode": episode_index,
                "seed": seed,
                "reset_count": 1,
                "post_launch_reset_count": 0,
                "initial_state": {
                    "qpos": initial_qpos.astype(float).tolist(),
                    "qvel": initial_qvel.astype(float).tolist(),
                },
                "conditioning_steps": conditioning_steps,
                "conditioning_seconds": float(conditioning_steps * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]),
                "selected_route": int(selected),
                "route_cursor_final": int(route_cursor),
                "handoff_state": handoff_state,
                "capture_replay": capture_replay,
                "predictions": [
                    {
                        "success": bool(row["success"]),
                        "max_upright_streak_seconds": float(row["max_upright_streak_seconds"]),
                        "max_cart_excursion": float(row["max_cart_excursion"]),
                    }
                    for row in predictions
                ],
                "success": bool(final_info.get("success", False)),
                "termination_reason": final_info.get("termination_reason"),
                "time_to_first_upright": final_info.get("time_to_first_upright"),
                "time_to_capture": final_info.get("time_to_capture"),
                "max_upright_streak_seconds": float(final_info.get("max_upright_streak_seconds", 0.0)),
                "max_cart_excursion": float(max(abs(value) for value in cart_positions)),
                "rail_hit": final_info.get("termination_reason") == "rail_violation",
                "trace": trace,
            }
        )
        print(
            f"episode={episode_index} seed={seed} route={selected} "
            f"success={episodes[-1]['success']}",
            flush=True,
        )

    checks = benchmark_checks(cfg, source_init_mode=source_init_mode)
    successes = sum(bool(row["success"]) for row in episodes)
    held = sum(
        bool(row["success"])
        and float(row["max_upright_streak_seconds"]) >= args.hold_seconds
        for row in episodes
    )
    gate = {
        "benchmark": {"checks": checks, "passed": bool(all(checks.values()))},
        "required_heldout_starts": 4,
        "heldout_starts": len(episodes),
        "heldout_starts_with_success": successes,
        "heldout_starts_with_five_second_hold": held,
        "passed": bool(all(checks.values()) and len(episodes) >= 4 and held >= 4),
    }
    probe = NLinkCartPoleEnv(cfg, progress=1.0, seed=args.seed)
    probe_xml_hash = text_sha256(probe.xml)
    probe.close()
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "canonical_global_discovery_gate_evidence",
        "not_solution": True,
        "summary": (
            "Exact uniform six-link hanging-start route feasibility evidence; "
            "this artifact does not replace P1, P2, or P3."
        ),
        "config": file_metadata(Path(args.config)),
        "resolved_config_sha256": data_sha256(cfg),
        "generated_xml_sha256": probe_xml_hash,
        "controllers": [file_metadata(Path(path)) for path in controller_paths],
        "controller_sha256": data_sha256([file_metadata(Path(path)) for path in controller_paths]),
        "selection": {
            "type": "exact_forward_model_argmin",
            "conditioning_seconds": float(args.conditioning_seconds),
            "tracking_gain_scale": float(args.tracking_gain_scale),
            "phase_window": int(args.phase_window),
        },
        "episodes": len(episodes),
        "success_rate": float(successes / len(episodes)),
        "hold_rate": float(held / len(episodes)),
        "termination_counts": dict(Counter(str(row["termination_reason"]) for row in episodes)),
        "gate": gate,
        "episodes_detail": episodes,
        "trace_sha256": data_sha256([row["trace"] for row in episodes]),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(json_safe(output), args.out)
    print(
        f"global_discovery_passed={gate['passed']} holds={held}/{len(episodes)} "
        f"successes={successes}/{len(episodes)}"
    )
    print(f"Wrote {args.out}")
    if args.fail_on_gate and not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
