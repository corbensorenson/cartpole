#!/usr/bin/env python
"""Convert a serialized exact rollout prefix into an FDDP route segment."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

import mujoco
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
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.modal import StateScales, dimensionless_absolute_transform

try:
    from scripts.search_energy_shaping_feedback import (
        force_for_cart_acceleration,
        state_features,
    )
    from scripts.search_pfl_capture_controller import capture_blend, swing_acceleration
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from search_energy_shaping_feedback import (
        force_for_cart_acceleration,
        state_features,
    )
    from search_pfl_capture_controller import capture_blend, swing_acceleration
    from search_swingup_capture import lqr_action, lqr_gain


def find_trace(payload: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    for name, value in (
        ("final_eval.trace", payload.get("final_eval", {}).get("trace")),
        ("result.trajectory", payload.get("result", {}).get("trajectory")),
        ("trace_rollout.trace", payload.get("trace_rollout", {}).get("trace")),
        ("trace", payload.get("trace")),
    ):
        if isinstance(value, list) and value:
            return name, [dict(row) for row in value]
    raise ValueError("input artifact has no supported nonempty serialized trace")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="configs/swingup7_uniform.yaml")
    parser.add_argument("--n-links", type=int, required=True)
    parser.add_argument("--through-time", type=float, required=True)
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument(
        "--pfl-feedback",
        action="store_true",
        help="linearize the source PFL/LQR state-feedback law along the prefix",
    )
    parser.add_argument("--feedback-epsilon", type=float, default=1.0e-5)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.n_links < 1 or args.through_time <= 0.0 or args.feedback_epsilon <= 0.0:
        raise ValueError("link count and through-time must be positive")

    source_path = Path(args.input)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    trace_path, trace = find_trace(source)
    rows = [
        row
        for row in trace
        if float(row.get("time_seconds", np.inf)) <= args.through_time + 1e-12
    ]
    if not rows:
        raise ValueError("no trace rows occur at or before through-time")
    if not all("action" in row and "qpos" in row and "qvel" in row for row in rows):
        raise ValueError("every selected trace row must contain action, qpos, and qvel")

    cfg = copy.deepcopy(load_config(args.config))
    cfg["env"]["n_links"] = int(args.n_links)
    cfg["env"]["init_mode"] = "hanging"
    cfg["env"]["episode_seconds"] = max(
        float(cfg["env"]["episode_seconds"]), float(args.through_time + 1.0)
    )
    cfg["env"]["action_lqr_residual"] = {"enabled": False}
    cfg["env"]["action_lqr_switch"] = {"enabled": False}
    for key in (
        "init_angle_noise",
        "init_vel_noise",
        "init_cart_noise",
        "init_cart_vel_noise",
    ):
        cfg["env"][key] = 0.0
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0

    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    env.reset(seed=0)
    initial = data_state(env.data)
    nq = args.n_links + 1
    policy_dt = float(env.dt)
    spec = load_config(args.spec)
    distribution = spec["distribution"]
    transform = dimensionless_absolute_transform(
        env.n,
        StateScales(
            float(distribution["cart_position_abs_max"]),
            float(distribution["absolute_link_angle_abs_max"]),
            float(distribution["cart_velocity_abs_max"]),
            float(distribution["hinge_velocity_rms_max"]),
        ),
    )
    transition = MujocoTransition(env, coordinate_transform=transform)
    controls = np.asarray([row["action"] for row in rows], dtype=np.float64)
    serialized_physical = np.asarray(
        [np.r_[row["qpos"], row["qvel"]] for row in rows], dtype=np.float64
    )
    serialized_states = np.asarray(
        [transition.to_coordinates(state) for state in serialized_physical],
        dtype=np.float64,
    )
    nominal_states = np.vstack((transition.to_coordinates(initial), serialized_states))
    # Long chaotic swing-ups cannot be validated by replaying the entire open-
    # loop force trace: roundoff grows exponentially even when every recorded
    # transition is exact.  Validate each interval independently instead.
    one_step_errors = np.asarray(
        [
            np.max(
                np.abs(
                    transition(nominal_states[index], controls[index])
                    - nominal_states[index + 1]
                )
            )
            for index in range(controls.size)
        ],
        dtype=np.float64,
    )
    maximum_one_step_error = float(np.max(one_step_errors))
    if maximum_one_step_error > 1.0e-7:
        raise RuntimeError(
            f"serialized one-step dynamics differ by {maximum_one_step_error:.3e}"
        )
    feedback_gains = np.zeros((controls.size, transform.shape[0]), dtype=np.float64)
    maximum_action_error = 0.0
    if args.pfl_feedback:
        best = source.get("best")
        if not isinstance(best, dict) or not isinstance(best.get("vector"), list):
            raise ValueError("--pfl-feedback requires source best.vector parameters")
        parameters = np.asarray(best["vector"], dtype=np.float64)
        if parameters.shape != (13,):
            raise ValueError(
                "--pfl-feedback currently requires the 13-parameter PFL law"
            )
        capture_gain = lqr_gain(cfg, progress=1.0, fd_eps=1.0e-7, control_cost=1000.0)

        def policy_action(coordinate_state: np.ndarray, time_seconds: float) -> float:
            physical = transition.to_physical(coordinate_state)
            env.data.qpos[:] = physical[:nq]
            env.data.qvel[:] = physical[nq:]
            env.data.ctrl[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            features = state_features(env, time_seconds)
            acceleration = swing_acceleration(parameters, features, time_seconds)
            force = force_for_cart_acceleration(env, acceleration)
            swing = float(np.clip(force / env.force_limit, -1.0, 1.0))
            blend = capture_blend(parameters, features, env)
            capture = lqr_action(
                env,
                capture_gain,
                scale=float(parameters[8]),
                cart_target=0.0,
            )
            return float(np.clip((1.0 - blend) * swing + blend * capture, -1.0, 1.0))

        action_errors: list[float] = []
        for step, nominal in enumerate(nominal_states[:-1]):
            time_seconds = step * env.dt
            action_errors.append(
                abs(policy_action(nominal, time_seconds) - controls[step])
            )
            for column in range(nominal.size):
                delta = np.zeros_like(nominal)
                delta[column] = args.feedback_epsilon
                feedback_gains[step, column] = (
                    policy_action(nominal + delta, time_seconds)
                    - policy_action(nominal - delta, time_seconds)
                ) / (2.0 * args.feedback_epsilon)
        maximum_action_error = float(max(action_errors))
        # JSON round-tripping of a chaotic trajectory can move a reconstructed
        # nonlinear policy by a few 1e-3 near a saturated switching surface.
        # Keep the recorded action as the nominal control and reject only a
        # discrepancy large enough to indicate the wrong policy/configuration.
        if maximum_action_error > 1.0e-2:
            raise RuntimeError(
                f"reconstructed PFL actions differ from trace by {maximum_action_error:.3e}"
            )
    env.close()
    terminal = serialized_physical[-1]
    output = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "claim_status": "deterministic_trace_prefix_not_solution",
        "not_solution": True,
        "summary": "Exact serialized deterministic prefix materialized as an FDDP route segment.",
        "selected_state": {
            "qpos": initial[:nq].astype(float).tolist(),
            "qvel": initial[nq:].astype(float).tolist(),
            "state_index": 0,
        },
        "terminal_state": {
            "qpos": terminal[:nq].astype(float).tolist(),
            "qvel": terminal[nq:].astype(float).tolist(),
        },
        "controller": {
            "type": "materialized_deterministic_trace_prefix",
            "controls": controls.astype(float).tolist(),
            "feedback_gains": feedback_gains.astype(float).tolist(),
            "feedback_type": (
                "finite_difference_exact_pfl_policy" if args.pfl_feedback else "zero"
            ),
            "feedback_epsilon": float(args.feedback_epsilon),
            "horizon_steps": int(controls.size),
            "horizon_seconds": float(controls.size * policy_dt),
            "policy_dt": policy_dt,
            "coordinate_transform": transform.astype(float).tolist(),
            "source": file_metadata(source_path),
            "source_trace_path": trace_path,
        },
        "search": {
            "iterations": 0,
            "is_feasible": True,
            "cost": None,
            "nominal_coordinate_states": nominal_states.astype(float).tolist(),
            "maximum_serialized_one_step_error": maximum_one_step_error,
            "maximum_reconstructed_action_error": maximum_action_error,
        },
        "evidence": {
            "config": file_metadata(Path(args.config)),
            "controls_sha256": data_sha256(controls.astype(float).tolist()),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(output, args.out)
    print(
        f"wrote {args.out}: steps={controls.size} through={controls.size * policy_dt:.3f}s "
        f"one_step_error={maximum_one_step_error:.3e}"
    )


if __name__ == "__main__":
    main()
