#!/usr/bin/env python
"""Evaluate bounded online actuator calibration inside generalized swing-up.

The physical plant remains the exact configured MuJoCo morphology.  A hidden
affine actuator map is inserted between requested and delivered normalized
action.  The adaptive arm uses only exact one-step nominal predictions and the
local action Jacobian; it cannot change the route, learn a swing trajectory, or
apply more than its declared correction bound.  Every seed is run both with
and without adaptation for a paired diagnostic.
"""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from dataclasses import dataclass
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
from gcartpole.generalized_energy import (
    EnergySwingParameters,
    GeneralizedEnergyController,
)
from gcartpole.generalized_solver import (
    BoundedForceAdapter,
    dimensionless_setup,
    rail_requirement,
    setup_from_config,
)
from gcartpole.ilqr import MujocoTransition, data_state
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


def calibration_dither(step: int, dt: float, seconds: float, amplitude: float) -> float:
    """Balanced deterministic excitation on the natural controller clock."""

    if step * dt >= seconds or amplitude == 0.0:
        return 0.0
    # The eight-symbol pattern has zero mean, multiple transitions, and enough
    # dwell time for the cart acceleration to be observable above roundoff.
    pattern = np.asarray([1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0])
    dwell_steps = max(1, round(0.08 / dt))
    return float(amplitude * pattern[(step // dwell_steps) % pattern.size])


@dataclass
class AdaptiveStepper:
    """One exact-model prediction, one bounded compensation, one live step."""

    env: NLinkCartPoleEnv
    transform: np.ndarray
    actuator_gain: float
    actuator_bias: float
    adapt: bool
    action_epsilon: float = 0.01

    def __post_init__(self) -> None:
        self.transition = MujocoTransition(self.env, coordinate_transform=self.transform)
        self.adapter = BoundedForceAdapter()
        self.steps = 0
        self.saturated_deliveries = 0
        self.max_correction = 0.0
        self.reason_counts: Counter[str] = Counter()

    def step(
        self, nominal_action: float, *, learn: bool = True
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        nominal = float(np.clip(nominal_action, -1.0, 1.0))
        command = self.adapter.command(nominal) if self.adapt else nominal
        self.max_correction = max(self.max_correction, abs(command - nominal))

        if self.adapt and learn:
            state = self.transition.to_coordinates(data_state(self.env.data))
            predicted = self.transition(state, command)
            upper = min(1.0, command + self.action_epsilon)
            lower = max(-1.0, command - self.action_epsilon)
            jacobian = self.transition.difference(
                self.transition(state, upper), self.transition(state, lower)
            ) / (upper - lower)

        unclipped_delivery = self.actuator_gain * command + self.actuator_bias
        delivered = float(np.clip(unclipped_delivery, -1.0, 1.0))
        if delivered != unclipped_delivery:
            self.saturated_deliveries += 1
        result = self.env.step([delivered])

        if self.adapt and learn:
            observed = self.transition.to_coordinates(data_state(self.env.data))
            diagnostic = self.adapter.observe_diagnostic(
                command, predicted, observed, jacobian
            )
            self.reason_counts[str(diagnostic["reason"])] += 1
        self.steps += 1
        return result

    def summary(self) -> dict[str, Any]:
        estimate = self.adapter.to_dict()
        return {
            "enabled": self.adapt,
            "estimated_gain": estimate["gain"],
            "estimated_bias": estimate["bias"],
            "gain_absolute_error": abs(float(estimate["gain"]) - self.actuator_gain),
            "bias_absolute_error": abs(float(estimate["bias"]) - self.actuator_bias),
            "updates": estimate["updates"],
            "rejections": estimate["rejections"],
            "covariance_trace": estimate["covariance_trace"],
            "rejection_reasons": dict(self.reason_counts),
            "maximum_action_correction": self.max_correction,
            "saturated_deliveries": self.saturated_deliveries,
            "steps": self.steps,
        }


def route_action(
    env: NLinkCartPoleEnv,
    controller: dict[str, Any],
    upright_gain: np.ndarray,
    nominal: np.ndarray,
    phase_cursor: int,
    cart_shift: float | None,
    *,
    tracking_gain_scale: float,
    phase_window: int,
) -> tuple[float, int, float | None]:
    controls = controller["controls"]
    qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
    if phase_cursor >= controls.size:
        return (
            upright_lqr_action_from_state(
                qpos,
                qvel,
                upright_gain,
                scale=controller["lqr_scale"],
                cart_target=0.0,
            ),
            phase_cursor,
            cart_shift,
        )

    coordinates = dimensionless_wrapped_state(qpos, qvel, controller["transform"])
    if cart_shift is None:
        cart_shift = float(coordinates[0] - nominal[0, 0])
        nominal[:, 0] += cart_shift
    candidates = np.arange(
        phase_cursor,
        min(controls.size, phase_cursor + phase_window + 1),
        dtype=np.int64,
    )
    errors = nominal[candidates] - coordinates
    route_step = int(candidates[int(np.argmin(np.einsum("ij,ij->i", errors, errors)))])
    action = float(
        np.clip(
            controls[route_step]
            + tracking_gain_scale
            * controller["feedback_gains"][route_step]
            @ (coordinates - nominal[route_step]),
            -1.0,
            1.0,
        )
    )
    return action, route_step + 1, cart_shift


def select_route(
    cfg: dict[str, Any],
    controllers: list[dict[str, Any]],
    upright_gains: list[np.ndarray],
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    tracking_gain_scale: float,
    phase_window: int,
) -> tuple[int, list[dict[str, Any]]]:
    predictive_cfg = fixed_state_config(cfg, qpos, qvel)
    predictions = [
        run_episode(
            predictive_cfg,
            controller,
            upright_gains[index],
            seed=0,
            episode=index,
            tracking_gain_scale=tracking_gain_scale,
            prelude_steps=0,
            settle_mode="zero",
            settle_gain=None,
            settle_scale=1.0,
            phase_adaptive=True,
            phase_window=phase_window,
            shift_cart_nominal=True,
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
    return selected, predictions


def run_trial(
    cfg: dict[str, Any],
    controllers: list[dict[str, Any]],
    upright_gains: list[np.ndarray],
    settle_gain: np.ndarray,
    parameters: EnergySwingParameters,
    *,
    seed: int,
    adapt: bool,
    actuator_gain: float,
    actuator_bias: float,
    conditioning_seconds: float,
    calibration_seconds: float,
    calibration_amplitude: float,
    tracking_gain_scale: float,
    phase_window: int,
) -> dict[str, Any]:
    live_cfg = copy.deepcopy(cfg)
    live_cfg["env"]["episode_seconds"] = float(cfg["env"]["episode_seconds"]) + float(
        conditioning_seconds
    )
    env = NLinkCartPoleEnv(live_cfg, progress=1.0, seed=seed)
    env.reset(seed=seed)
    if controllers:
        transform = controllers[0]["transform"]
    else:
        # Energy control itself is morphology-normalized; use the same standard
        # dimensionless state coordinates as route tracking for identification.
        from gcartpole.modal import StateScales, dimensionless_absolute_transform

        transform = dimensionless_absolute_transform(
            env.n,
            StateScales(3.0, 0.15, 4.0, 15.0),
        )
    stepper = AdaptiveStepper(
        env,
        np.asarray(transform, dtype=np.float64),
        actuator_gain,
        actuator_bias,
        adapt,
    )
    cart_positions = [float(env.data.qpos[0])]
    final_info: dict[str, Any] = env._info()
    conditioning_steps = round(conditioning_seconds / env.dt)
    for step in range(conditioning_steps):
        nominal_action = hanging_lqr_action(env, settle_gain, scale=1.0)
        nominal_action += calibration_dither(
            step, env.dt, calibration_seconds, calibration_amplitude
        )
        _, _, terminated, truncated, final_info = stepper.step(
            nominal_action, learn=step * env.dt < calibration_seconds
        )
        cart_positions.append(float(env.data.qpos[0]))
        if terminated or truncated:
            break

    selected: int | None = None
    predictions: list[dict[str, Any]] = []
    if not final_info.get("termination_reason"):
        if controllers:
            selected, predictions = select_route(
                cfg,
                controllers,
                upright_gains,
                np.asarray(env.data.qpos, dtype=np.float64).copy(),
                np.asarray(env.data.qvel, dtype=np.float64).copy(),
                tracking_gain_scale=tracking_gain_scale,
                phase_window=phase_window,
            )
            controller = controllers[selected]
            upright_gain = upright_gains[selected]
            nominal = controller["nominal_states"].copy()
            phase_cursor = 0
            cart_shift: float | None = None
            while env.step_count < env.max_steps:
                action, phase_cursor, cart_shift = route_action(
                    env,
                    controller,
                    upright_gain,
                    nominal,
                    phase_cursor,
                    cart_shift,
                    tracking_gain_scale=tracking_gain_scale,
                    phase_window=phase_window,
                )
                _, _, terminated, truncated, final_info = stepper.step(
                    action, learn=False
                )
                cart_positions.append(float(env.data.qpos[0]))
                if terminated or truncated:
                    break
        else:
            energy = GeneralizedEnergyController(env, parameters)
            while env.step_count < env.max_steps:
                action, _diagnostic = energy.action(
                    env, (env.step_count - conditioning_steps) * env.dt
                )
                _, _, terminated, truncated, final_info = stepper.step(
                    action, learn=False
                )
                cart_positions.append(float(env.data.qpos[0]))
                if terminated or truncated:
                    break

    setup = setup_from_config(cfg)
    result = {
        "seed": seed,
        "mode": "adaptive" if adapt else "baseline",
        "success": bool(final_info.get("success", False)),
        "termination_reason": final_info.get("termination_reason"),
        "max_upright_streak_seconds": float(
            final_info.get("max_upright_streak_seconds", 0.0)
        ),
        "max_cart_excursion": float(max(abs(value) for value in cart_positions)),
        "rail_requirement": rail_requirement(np.asarray(cart_positions), setup),
        "selected_route": selected,
        "selected_prediction": (
            None
            if selected is None
            else {
                "success": bool(predictions[selected]["success"]),
                "max_upright_streak_seconds": float(
                    predictions[selected]["max_upright_streak_seconds"]
                ),
                "max_cart_excursion": float(
                    predictions[selected]["max_cart_excursion"]
                ),
            }
        ),
        "adaptation": stepper.summary(),
    }
    env.close()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--n-links", type=int, required=True)
    parser.add_argument("--controller", action="append", default=[])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=81001)
    parser.add_argument("--conditioning-seconds", type=float, default=15.0)
    parser.add_argument("--calibration-seconds", type=float, default=2.0)
    parser.add_argument("--calibration-amplitude", type=float, default=0.18)
    parser.add_argument("--actuator-gain", type=float, default=0.80)
    parser.add_argument("--actuator-bias", type=float, default=0.08)
    parser.add_argument("--tracking-gain-scale", type=float, default=1.0)
    parser.add_argument("--phase-window", type=int, default=0)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.n_links < 1 or args.episodes < 1:
        raise ValueError("link and episode counts must be positive")
    if args.n_links > 1 and not args.controller:
        raise ValueError("n>1 requires at least one deterministic route controller")
    if min(args.conditioning_seconds, args.calibration_seconds) < 0.0:
        raise ValueError("conditioning and calibration durations must be nonnegative")
    if not 0.0 <= args.calibration_amplitude <= 1.0:
        raise ValueError("calibration amplitude must lie in [0, 1]")
    if args.actuator_gain <= 0.0 or abs(args.actuator_bias) >= 1.0:
        raise ValueError("actuator gain must be positive and bias magnitude below one")

    cfg = uniform_config(
        apply_overrides(load_config(args.config), args.override), args.n_links
    )
    spec = load_config("benchmarks/p1_capture_envelope.yaml")
    controllers = [
        load_controller(Path(path), args.n_links, spec) for path in args.controller
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
    settle_gain = hanging_lqr_gain(
        cfg, progress=1.0, fd_eps=1e-7, control_cost=1000.0
    )
    parameters = EnergySwingParameters()
    results: list[dict[str, Any]] = []
    for episode in range(args.episodes):
        seed = args.seed + episode
        for adapt in (False, True):
            row = run_trial(
                cfg,
                controllers,
                upright_gains,
                settle_gain,
                parameters,
                seed=seed,
                adapt=adapt,
                actuator_gain=args.actuator_gain,
                actuator_bias=args.actuator_bias,
                conditioning_seconds=args.conditioning_seconds,
                calibration_seconds=args.calibration_seconds,
                calibration_amplitude=args.calibration_amplitude,
                tracking_gain_scale=args.tracking_gain_scale,
                phase_window=args.phase_window,
            )
            results.append(row)
            print(
                f"n={args.n_links} seed={seed} mode={row['mode']} "
                f"success={row['success']} gain={row['adaptation']['estimated_gain']:.4f} "
                f"bias={row['adaptation']['estimated_bias']:.4f}",
                flush=True,
            )

    by_mode = {
        mode: [row for row in results if row["mode"] == mode]
        for mode in ("baseline", "adaptive")
    }
    setup = setup_from_config(cfg)
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "development_bounded_adaptation_diagnostic",
        "summary": "Paired uninterrupted exact-MuJoCo swing-up under a hidden affine actuator mismatch.",
        "config": file_metadata(Path(args.config)),
        "controllers": [file_metadata(Path(path)) for path in args.controller],
        "n_links": args.n_links,
        "dimensionless_setup": dimensionless_setup(setup).to_dict(),
        "simulated_actuator": {
            "equivalent_action": "gain * commanded_action + bias",
            "gain": args.actuator_gain,
            "bias": args.actuator_bias,
        },
        "adaptation_contract": {
            "type": "bounded_action_direction_projected_rls",
            "calibration_seconds": args.calibration_seconds,
            "calibration_amplitude": args.calibration_amplitude,
            "estimate_frozen_after_calibration": True,
            "route_or_energy_parameters_changed": False,
            "maximum_per_step_correction": BoundedForceAdapter().correction_bound,
            "structural_residuals_require_replanning": True,
        },
        "episodes_per_mode": args.episodes,
        "seed_start": args.seed,
        "results": results,
        "summary_by_mode": {
            mode: {
                "successes": int(sum(row["success"] for row in rows)),
                "success_rate": float(np.mean([row["success"] for row in rows])),
                "mean_gain_absolute_error": float(
                    np.mean(
                        [row["adaptation"]["gain_absolute_error"] for row in rows]
                    )
                ),
                "mean_bias_absolute_error": float(
                    np.mean(
                        [row["adaptation"]["bias_absolute_error"] for row in rows]
                    )
                ),
                "max_required_rail_ratio": float(
                    max(row["rail_requirement"]["required_rail_ratio"] for row in rows)
                ),
            }
            for mode, rows in by_mode.items()
        },
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(output, args.out)
    print(
        f"baseline={output['summary_by_mode']['baseline']['success_rate']:.3f} "
        f"adaptive={output['summary_by_mode']['adaptive']['success_rate']:.3f}"
    )


if __name__ == "__main__":
    main()
