#!/usr/bin/env python
"""Evaluate deterministic model-predictive selection over symmetric routes."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import (
    file_metadata,
    git_metadata,
    runtime_metadata,
    utc_timestamp,
)
from gcartpole.generalized_solver import rail_requirement, setup_from_config
from gcartpole.modal import dimensionless_wrapped_state

try:
    from scripts.evaluate_fddp_two_expert import (
        hanging_lqr_action_from_state,
        hanging_lqr_gain,
        load_controller,
        run_episode,
        upright_lqr_action_from_state,
    )
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from evaluate_fddp_two_expert import (
        hanging_lqr_action_from_state,
        hanging_lqr_gain,
        load_controller,
        run_episode,
        upright_lqr_action_from_state,
    )
    from search_swingup_capture import lqr_gain


def uniform_config(cfg: dict[str, Any], n_links: int) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    result["env"]["n_links"] = int(n_links)
    result["experiment"]["name"] = f"generalized_uniform_n{n_links}"
    for name in (
        "lengths",
        "masses",
        "damping",
        "frictionloss",
        "joint_stiffness",
        "joint_lock",
    ):
        result["morphology"].pop(f"{name}_start", None)
        result["morphology"].pop(f"{name}_end", None)
    for endpoint in ("start", "end"):
        result["morphology"][endpoint]["alpha_length"] = 0.0
        result["morphology"][endpoint]["alpha_mass"] = 0.0
        result["morphology"][endpoint]["alpha_damping"] = 0.0
    return result


def fixed_state_config(
    cfg: dict[str, Any], qpos: np.ndarray, qvel: np.ndarray
) -> dict[str, Any]:
    result = copy.deepcopy(cfg)
    result["env"]["init_mode"] = "fixed_state"
    result["env"]["init_qpos"] = (
        np.asarray(qpos, dtype=np.float64).astype(float).tolist()
    )
    result["env"]["init_qvel"] = (
        np.asarray(qvel, dtype=np.float64).astype(float).tolist()
    )
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        result["env"][key] = 0.0
        result["env"][f"{key}_start"] = 0.0
        result["env"][f"{key}_end"] = 0.0
    return result


def execute_route(
    env: NLinkCartPoleEnv,
    controller: dict[str, Any],
    upright_gain: np.ndarray,
    *,
    tracking_gain_scale: float,
    phase_window: int,
    cart_target: float,
) -> tuple[dict[str, Any], list[float]]:
    controls = controller["controls"]
    nominal = controller["nominal_states"].copy()
    feedback = controller["feedback_gains"]
    transform = controller["transform"]
    phase_cursor = 0
    cart_shift: float | None = None
    final_info: dict[str, Any] = env._info()
    cart_positions = [float(env.data.qpos[0])]
    while env.step_count < env.max_steps:
        qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
        qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
        coordinates = dimensionless_wrapped_state(qpos, qvel, transform)
        if phase_cursor < controls.size:
            if cart_shift is None:
                cart_shift = float(coordinates[0] - nominal[0, 0])
                # The measured launch state already includes any parked cart target.
                nominal[:, 0] += cart_shift
            candidates = np.arange(
                phase_cursor,
                min(controls.size, phase_cursor + phase_window + 1),
                dtype=np.int64,
            )
            errors = nominal[candidates] - coordinates
            route_step = int(
                candidates[int(np.argmin(np.einsum("ij,ij->i", errors, errors)))]
            )
            phase_cursor = route_step + 1
            action = float(
                np.clip(
                    controls[route_step]
                    + tracking_gain_scale
                    * feedback[route_step]
                    @ (coordinates - nominal[route_step]),
                    -1.0,
                    1.0,
                )
            )
        else:
            action = upright_lqr_action_from_state(
                qpos,
                qvel,
                upright_gain,
                scale=controller["lqr_scale"],
                cart_target=cart_target,
            )
        _, _, terminated, truncated, final_info = env.step([action])
        cart_positions.append(float(env.data.qpos[0]))
        if terminated or truncated:
            break
    return final_info, cart_positions


def route_cart_target(
    controller: dict[str, Any], *, target: float, mode: str
) -> float:
    if mode == "fixed":
        return float(target)
    if mode == "mirror_symmetric":
        return float(-target if controller.get("mirror_symmetry", False) else target)
    raise ValueError(f"unsupported route cart target mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-links", type=int)
    parser.add_argument("--controller", action="append", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=72021)
    parser.add_argument("--conditioning-seconds", type=float, default=15.0)
    parser.add_argument("--tracking-gain-scale", type=float, default=1.5)
    parser.add_argument("--phase-window", type=int, default=6)
    parser.add_argument(
        "--cart-target",
        type=float,
        default=0.0,
        help="cart target in metres used during conditioning, route translation, and capture",
    )
    parser.add_argument(
        "--route-target-mode",
        choices=("fixed", "mirror_symmetric"),
        default="fixed",
        help="use one target for every route or flip the target for the mirrored route",
    )
    parser.add_argument(
        "--park-seconds",
        type=float,
        default=0.0,
        help="reset-free hanging-LQR parking time after route selection",
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.park_seconds < 0.0:
        raise ValueError("park-seconds must be nonnegative")
    cfg = apply_overrides(load_config(args.config), args.override)
    if args.n_links is not None:
        if args.n_links < 1:
            raise ValueError("--n-links must be positive")
        cfg = uniform_config(cfg, args.n_links)
    spec = load_config("benchmarks/p1_capture_envelope.yaml")
    n_links = int(cfg["env"]["n_links"])
    controllers = [
        load_controller(Path(path), n_links, spec) for path in args.controller
    ]
    upright_gains = [
        lqr_gain(
            cfg,
            progress=1.0,
            fd_eps=1e-7,
            control_cost=float(controller["lqr_control_cost"]),
            q_weights=controller["lqr_weights"],
        )
        for controller in controllers
    ]
    settle_gain = hanging_lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    setup = setup_from_config(cfg)
    live_cfg = copy.deepcopy(cfg)
    conditioning_target = (
        float(args.cart_target)
        if args.route_target_mode == "fixed"
        else 0.0
    )
    live_cfg["env"]["episode_seconds"] = (
        float(cfg["env"]["episode_seconds"])
        + float(args.conditioning_seconds)
        + float(args.park_seconds)
    )
    results: list[dict[str, Any]] = []
    for episode_index in range(args.episodes):
        seed = args.seed + episode_index
        env = NLinkCartPoleEnv(live_cfg, progress=1.0, seed=seed)
        env.reset(seed=seed)
        conditioning_steps = round(args.conditioning_seconds / env.dt)
        live_cart = [float(env.data.qpos[0])]
        for _ in range(conditioning_steps):
            settle_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
            settle_qpos[0] -= conditioning_target
            action = hanging_lqr_action_from_state(
                settle_qpos,
                np.asarray(env.data.qvel, dtype=np.float64),
                settle_gain,
                scale=1.0,
            )
            _, _, terminated, truncated, _info = env.step([action])
            live_cart.append(float(env.data.qpos[0]))
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
                episode=index,
                tracking_gain_scale=args.tracking_gain_scale,
                prelude_steps=0,
                settle_mode="zero",
                settle_gain=None,
                settle_scale=1.0,
                phase_adaptive=True,
                phase_window=args.phase_window,
                shift_cart_nominal=True,
                cart_target=route_cart_target(
                    controller,
                    target=args.cart_target,
                    mode=args.route_target_mode,
                ),
                include_trajectory=False,
            )
            for index, controller in enumerate(controllers)
        ]
        selected = min(
            range(len(predictions)),
            key=lambda index: (
                not predictions[index]["success"],
                -float(predictions[index]["max_upright_streak_seconds"]),
                float(predictions[index]["max_cart_excursion"]),
                index,
            ),
        )
        selected_cart_target = route_cart_target(
            controllers[selected],
            target=args.cart_target,
            mode=args.route_target_mode,
        )
        parking_steps = round(float(args.park_seconds) / env.dt)
        for _ in range(parking_steps):
            settle_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
            settle_qpos[0] -= selected_cart_target
            action = hanging_lqr_action_from_state(
                settle_qpos,
                np.asarray(env.data.qvel, dtype=np.float64),
                settle_gain,
                scale=1.0,
            )
            _, _, terminated, truncated, _info = env.step([action])
            live_cart.append(float(env.data.qpos[0]))
            if terminated or truncated:
                raise RuntimeError("route-specific parking terminated before swing route")
        final_info, route_cart = execute_route(
            env,
            controllers[selected],
            upright_gains[selected],
            tracking_gain_scale=args.tracking_gain_scale,
            phase_window=args.phase_window,
            cart_target=selected_cart_target,
        )
        live_cart.extend(route_cart[1:])
        result = {
            "episode": episode_index,
            "seed": seed,
            "success": bool(final_info.get("success", False)),
            "termination_reason": final_info.get("termination_reason"),
            "selected_route": selected,
            "predictions": [
                {
                    "success": row["success"],
                    "max_upright_streak_seconds": row["max_upright_streak_seconds"],
                    "max_cart_excursion": row["max_cart_excursion"],
                }
                for row in predictions
            ],
            "max_upright_streak_seconds": float(
                final_info.get("max_upright_streak_seconds", 0.0)
            ),
            "max_cart_excursion": float(max(abs(value) for value in live_cart)),
            "rail_requirement": rail_requirement(np.asarray(live_cart), setup),
        }
        results.append(result)
        env.close()
        print(
            f"episode={episode_index} seed={seed} route={selected} "
            f"predicted={predictions[selected]['success']} actual={result['success']}",
            flush=True,
        )
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_deterministic_model_predictive_library",
        "summary": "Uninterrupted noisy evaluation with exact forward selection over mirror-symmetric deterministic routes.",
        "config": file_metadata(Path(args.config)),
        "controllers": [file_metadata(Path(path)) for path in args.controller],
        "selection": {
            "type": "exact_forward_model_argmin",
            "tracking_gain_scale": args.tracking_gain_scale,
            "phase_window": args.phase_window,
            "cart_target": args.cart_target,
            "route_target_mode": args.route_target_mode,
            "park_seconds": args.park_seconds,
            "conditioning_target": conditioning_target,
            "conditioning_seconds": args.conditioning_seconds,
            "route_lqr_control_costs": [
                float(controller["lqr_control_cost"]) for controller in controllers
            ],
        },
        "episodes": args.episodes,
        "seed_start": args.seed,
        "success_rate": float(np.mean([row["success"] for row in results])),
        "route_counts": dict(Counter(str(row["selected_route"]) for row in results)),
        "termination_counts": dict(
            Counter(str(row["termination_reason"]) for row in results)
        ),
        "max_required_rail_ratio": float(
            max(row["rail_requirement"]["required_rail_ratio"] for row in results)
        ),
        "episode_results": results,
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, args.out)
    print(
        f"episodes={args.episodes} success={output['success_rate']:.3f} "
        f"max_required_rho={output['max_required_rail_ratio']:.3f}"
    )


if __name__ == "__main__":
    main()
