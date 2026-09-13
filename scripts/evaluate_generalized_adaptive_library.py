#!/usr/bin/env python
"""Evaluate bounded natural-time conditioning and symmetric route selection."""

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
from gcartpole.generalized_solver import (
    dimensionless_setup,
    rail_requirement,
    setup_from_config,
)
from gcartpole.modal import dimensionless_wrapped_state

try:
    from scripts.evaluate_fddp_two_expert import (
        hanging_lqr_action,
        hanging_lqr_gain,
        load_controller,
        run_episode,
        upright_lqr_action_from_state,
    )
    from scripts.evaluate_generalized_route_library import (
        fixed_state_config,
        uniform_config,
    )
    from scripts.search_swingup_capture import lqr_gain
except ModuleNotFoundError:
    from evaluate_fddp_two_expert import (
        hanging_lqr_action,
        hanging_lqr_gain,
        load_controller,
        run_episode,
        upright_lqr_action_from_state,
    )
    from evaluate_generalized_route_library import fixed_state_config, uniform_config
    from search_swingup_capture import lqr_gain


def conditioning_schedule(
    natural_time: float, control_dt: float, ratios: list[float]
) -> tuple[tuple[float, float, int], ...]:
    """Return unique (ratio, quantized seconds, steps) in shortest-first order."""

    if natural_time <= 0.0 or control_dt <= 0.0:
        raise ValueError("natural time and control period must be positive")
    if not ratios or any(not np.isfinite(value) or value < 0.0 for value in ratios):
        raise ValueError("conditioning ratios must be finite and nonnegative")
    by_steps: dict[int, float] = {}
    for ratio in sorted(ratios):
        steps = round(float(ratio) * natural_time / control_dt)
        by_steps.setdefault(steps, float(ratio))
    return tuple(
        (ratio, float(steps * control_dt), steps)
        for steps, ratio in sorted(by_steps.items())
    )


def select_candidate(candidates: list[dict[str, Any]]) -> int:
    """Prefer success, then less conditioning, longer hold, and less rail travel."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    return min(
        range(len(candidates)),
        key=lambda index: (
            not bool(candidates[index]["result"]["success"]),
            float(candidates[index]["conditioning_seconds"]),
            -float(candidates[index]["result"]["max_upright_streak_seconds"]),
            float(candidates[index]["result"]["max_cart_excursion"]),
            int(candidates[index]["route_index"]),
        ),
    )


def execute_route(
    env: NLinkCartPoleEnv,
    controller: dict[str, Any],
    upright_gain: np.ndarray,
    *,
    tracking_gain_scale: float,
    phase_window: int,
) -> tuple[dict[str, Any], list[float]]:
    """Execute one selected route in the live, never-reset evaluation episode."""

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
                cart_target=0.0,
            )
        _, _, terminated, truncated, final_info = env.step([action])
        cart_positions.append(float(env.data.qpos[0]))
        if terminated or truncated:
            break
    return final_info, cart_positions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--n-links", type=int)
    parser.add_argument("--controller", action="append", required=True)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=72021)
    parser.add_argument(
        "--conditioning-time-ratio",
        type=float,
        action="append",
        dest="conditioning_ratios",
        help=(
            "candidate hanging-LQR duration divided by morphology natural time; "
            "repeat for a bounded grid (default: 0, 0.25, 0.5, 1, 2, 4)"
        ),
    )
    parser.add_argument("--tracking-gain-scale", type=float, default=1.0)
    parser.add_argument("--phase-window", type=int, default=0)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.phase_window < 0:
        raise ValueError("--phase-window must be nonnegative")

    cfg = apply_overrides(load_config(args.config), args.override)
    if args.n_links is not None:
        if args.n_links < 1:
            raise ValueError("--n-links must be positive")
        cfg = uniform_config(cfg, args.n_links)
    setup = setup_from_config(cfg)
    pi = dimensionless_setup(setup)
    control_dt = float(cfg["env"]["timestep"]) * int(cfg["env"]["frame_skip"])
    ratios = args.conditioning_ratios or [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    schedule = conditioning_schedule(pi.natural_time, control_dt, ratios)
    spec = load_config("benchmarks/p1_capture_envelope.yaml")
    controllers = [
        load_controller(Path(path), setup.n_links, spec) for path in args.controller
    ]
    upright_gains = [
        lqr_gain(
            cfg,
            progress=1.0,
            fd_eps=1e-7,
            control_cost=float(controller["lqr_control_cost"]),
        )
        for controller in controllers
    ]
    settle_gain = hanging_lqr_gain(cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0)
    results: list[dict[str, Any]] = []
    for episode_index in range(args.episodes):
        seed = args.seed + episode_index
        live_cfg = copy.deepcopy(cfg)
        live_cfg["env"]["episode_seconds"] = (
            float(cfg["env"]["episode_seconds"]) + schedule[-1][1]
        )
        env = NLinkCartPoleEnv(live_cfg, progress=1.0, seed=seed)
        env.reset(seed=seed)
        initial_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
        initial_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
        candidates: list[dict[str, Any]] = []
        for ratio, seconds, prelude_steps in schedule:
            candidate_cfg = fixed_state_config(cfg, initial_qpos, initial_qvel)
            candidate_cfg["env"]["episode_seconds"] = (
                float(cfg["env"]["episode_seconds"]) + seconds
            )
            rows = [
                {
                    "conditioning_time_ratio": ratio,
                    "conditioning_seconds": seconds,
                    "conditioning_steps": prelude_steps,
                    "route_index": route_index,
                    "result": run_episode(
                        candidate_cfg,
                        controller,
                        upright_gains[route_index],
                        seed=seed,
                        episode=episode_index,
                        tracking_gain_scale=args.tracking_gain_scale,
                        prelude_steps=prelude_steps,
                        settle_mode="hanging_lqr",
                        settle_gain=settle_gain,
                        settle_scale=1.0,
                        phase_adaptive=True,
                        phase_window=args.phase_window,
                        shift_cart_nominal=True,
                        include_trajectory=False,
                    ),
                }
                for route_index, controller in enumerate(controllers)
            ]
            candidates.extend(rows)
            if any(bool(row["result"]["success"]) for row in rows):
                break
        selected_index = select_candidate(candidates)
        selected = candidates[selected_index]
        selected_steps = int(selected["conditioning_steps"])
        env.max_steps = round(float(cfg["env"]["episode_seconds"]) / env.dt) + selected_steps
        live_cart = [float(env.data.qpos[0])]
        for _ in range(selected_steps):
            action = hanging_lqr_action(env, settle_gain, scale=1.0)
            _, _, terminated, truncated, _info = env.step([action])
            live_cart.append(float(env.data.qpos[0]))
            if terminated or truncated:
                raise RuntimeError("selected conditioning terminated before route")
        final_info, route_cart = execute_route(
            env,
            controllers[int(selected["route_index"])],
            upright_gains[int(selected["route_index"])],
            tracking_gain_scale=args.tracking_gain_scale,
            phase_window=args.phase_window,
        )
        live_cart.extend(route_cart[1:])
        env.close()
        prediction = selected["result"]
        actual_success = bool(final_info.get("success", False))
        maximum_cart = float(max(abs(value) for value in live_cart))
        result = {
            "episode": episode_index,
            "seed": seed,
            "initial_qpos": initial_qpos.astype(float).tolist(),
            "initial_qvel": initial_qvel.astype(float).tolist(),
            "resets_inside_episode": 0,
            "success": actual_success,
            "termination_reason": final_info.get("termination_reason"),
            "selected_route": int(selected["route_index"]),
            "selected_conditioning_time_ratio": float(
                selected["conditioning_time_ratio"]
            ),
            "selected_conditioning_seconds": float(selected["conditioning_seconds"]),
            "prediction_success": bool(prediction["success"]),
            "prediction_execution_agreement": bool(prediction["success"])
            == actual_success,
            "evaluated_candidates": len(candidates),
            "max_upright_streak_seconds": float(
                final_info.get("max_upright_streak_seconds", 0.0)
            ),
            "max_cart_excursion": maximum_cart,
            "rail_requirement": rail_requirement(
                np.asarray(live_cart, dtype=np.float64), setup
            ),
        }
        results.append(result)
        print(
            f"episode={episode_index} seed={seed} route={result['selected_route']} "
            f"tau={result['selected_conditioning_time_ratio']:.3f} "
            f"predicted={result['prediction_success']} actual={result['success']}",
            flush=True,
        )

    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_bounded_deterministic_adaptation_gate",
        "summary": (
            "Measured-start exact-model selection over analytic mirror routes and a "
            "bounded natural-time hanging-LQR conditioning grid."
        ),
        "config": file_metadata(Path(args.config)),
        "controllers": [file_metadata(Path(path)) for path in args.controller],
        "adaptation": {
            "type": "bounded_exact_model_natural_time_conditioning_and_route_selection",
            "learned_parameters": 0,
            "uses_measured_initial_state": True,
            "execution_resets_after_measurement": 0,
            "natural_time_seconds": pi.natural_time,
            "conditioning_schedule": [
                {"ratio": ratio, "seconds": seconds, "steps": steps}
                for ratio, seconds, steps in schedule
            ],
            "early_exit_after_first_successful_duration": True,
            "tracking_gain_scale": args.tracking_gain_scale,
            "phase_window": args.phase_window,
        },
        "episodes": args.episodes,
        "seed_start": args.seed,
        "success_rate": float(np.mean([row["success"] for row in results])),
        "prediction_execution_agreements": int(
            sum(row["prediction_execution_agreement"] for row in results)
        ),
        "route_counts": dict(Counter(str(row["selected_route"]) for row in results)),
        "conditioning_ratio_counts": dict(
            Counter(str(row["selected_conditioning_time_ratio"]) for row in results)
        ),
        "termination_counts": dict(Counter(str(row["termination_reason"]) for row in results)),
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
        f"agreements={output['prediction_execution_agreements']} "
        f"max_required_rho={output['max_required_rail_ratio']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
