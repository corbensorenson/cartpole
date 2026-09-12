#!/usr/bin/env python
"""Run the reset-free global-discovery gate on the exact MuJoCo plant.

This evaluator intentionally uses no saved handoff state, state list, or MLX
checkpoint. It is a small CPU-safe reference evaluator for the current
two-expert prototype: a saved feedback swing controller followed by a finite
difference LQR capture/stabilize controller. Failed runs are still valuable
because the complete physical trace is retained for the next search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import (
    data_sha256,
    file_metadata,
    git_metadata,
    runtime_metadata,
    text_sha256,
    utc_timestamp,
)
from gcartpole.env import NLinkCartPoleEnv

try:
    from scripts.probe_swingup_trajectory import (
        DEFAULT_KD,
        DEFAULT_KNOTS,
        DEFAULT_KP,
        DEFAULT_TRAJECTORY_SECONDS,
        trajectory_action,
    )
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from probe_swingup_trajectory import (
        DEFAULT_KD,
        DEFAULT_KNOTS,
        DEFAULT_KP,
        DEFAULT_TRAJECTORY_SECONDS,
        trajectory_action,
    )
    from search_swingup_capture import lqr_action, lqr_gain


def json_safe(value: Any) -> Any:
    """Convert NumPy scalars/arrays nested in MuJoCo metadata to JSON types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def load_swing_controller(path: str | None) -> dict[str, Any]:
    if path is None:
        return {
            "type": "cart_position_pd_fixed_knots",
            "trajectory_seconds": float(DEFAULT_TRAJECTORY_SECONDS),
            "kp": float(DEFAULT_KP),
            "kd": float(DEFAULT_KD),
            "knots": DEFAULT_KNOTS.astype(float).tolist(),
        }
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("best_by"), dict):
        score_record = payload["best_by"].get("score")
        if isinstance(score_record, dict) and isinstance(score_record.get("controller"), dict):
            return dict(score_record["controller"])
    if isinstance(payload, dict) and isinstance(payload.get("best"), dict):
        best = payload["best"]
        if isinstance(best.get("controller"), dict):
            return dict(best["controller"])
    if isinstance(payload, dict) and isinstance(payload.get("controller"), dict):
        return dict(payload["controller"])
    if isinstance(payload, dict) and "knots" in payload:
        return dict(payload)
    raise ValueError(f"Could not load a swing controller from {path}")


def row_from_env(
    env: NLinkCartPoleEnv,
    *,
    step: int,
    stage: str,
    action: float,
    reward: float,
    info: dict[str, Any],
) -> dict[str, Any]:
    relative_angles, absolute_angles = env._angles()
    return {
        "step": int(step),
        "time_seconds": float(step * env.dt),
        "stage": str(stage),
        "action": float(action),
        "reward": float(reward),
        "qpos": np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist(),
        "qvel": np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist(),
        "x": float(info["x"]),
        "cart_velocity": float(env.data.qvel[0]),
        "relative_angles": relative_angles.astype(float).tolist(),
        "absolute_angles": absolute_angles.astype(float).tolist(),
        "max_abs_angle": float(info["max_abs_angle"]),
        "mean_abs_angle": float(info["mean_abs_angle"]),
        "hinge_velocity_rms": float(info["hinge_velocity_rms"]),
        "is_upright": bool(info["is_upright"]),
        "upright_streak_seconds": float(info["upright_streak_seconds"]),
        "max_upright_streak_seconds": float(info["max_upright_streak_seconds"]),
        "time_to_first_upright": info["time_to_first_upright"],
        "time_to_capture": info["time_to_capture"],
        "termination_reason": info.get("termination_reason"),
    }


def evaluate_episode(
    cfg: dict[str, Any],
    *,
    seed: int,
    seconds: float,
    swing_controller: dict[str, Any],
    capture_gain: np.ndarray,
    capture_enter_angle: float,
    capture_min_time: float,
    stabilize_enter_angle: float,
    stabilize_enter_streak: float,
    stabilize_hinge_rms: float,
    min_capture_seconds: float,
    lqr_capture_scale: float,
    lqr_stabilize_scale: float,
    hold_seconds: float,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=seed)
    _, reset_info = env.reset(seed=seed)
    initial_qpos = np.asarray(env.data.qpos, dtype=np.float64).astype(float).tolist()
    initial_qvel = np.asarray(env.data.qvel, dtype=np.float64).astype(float).tolist()

    knots = np.asarray(swing_controller["knots"], dtype=np.float64)
    swing_seconds = float(swing_controller["trajectory_seconds"])
    swing_kp = float(swing_controller["kp"])
    swing_kd = float(swing_controller["kd"])
    steps = min(env.max_steps, int(seconds / env.dt))
    stage = "swing"
    stage_enter_time = 0.0
    stage_events: list[dict[str, Any]] = [{"time_seconds": 0.0, "stage": stage, "reason": "reset"}]
    trajectory: list[dict[str, Any]] = []
    total_return = 0.0
    max_cart_abs = abs(float(reset_info["x"]))
    action_abs_max = 0.0
    done_events: list[dict[str, Any]] = []
    final_info: dict[str, Any] = dict(reset_info)

    for step in range(steps):
        t = step * env.dt
        _, absolute_angles = env._angles()
        hinge_rms = float(np.sqrt(np.mean(env.data.qvel[1 : 1 + env.n] ** 2)))
        max_abs_angle = float(np.max(np.abs(absolute_angles)))
        if stage == "swing" and t >= capture_min_time and max_abs_angle <= capture_enter_angle:
            stage = "capture"
            stage_enter_time = t
            stage_events.append(
                {
                    "time_seconds": float(t),
                    "stage": stage,
                    "reason": "capture_enter_angle",
                    "max_abs_angle": max_abs_angle,
                    "hinge_velocity_rms": hinge_rms,
                    "x": float(env.data.qpos[0]),
                }
            )
        if (
            stage == "capture"
            and t - stage_enter_time >= min_capture_seconds
            and max_abs_angle <= stabilize_enter_angle
            and float(env._info()["upright_streak_seconds"]) >= stabilize_enter_streak
            and hinge_rms <= stabilize_hinge_rms
        ):
            stage = "stabilize"
            stage_enter_time = t
            stage_events.append(
                {
                    "time_seconds": float(t),
                    "stage": stage,
                    "reason": "stabilize_gate",
                    "max_abs_angle": max_abs_angle,
                    "hinge_velocity_rms": hinge_rms,
                    "x": float(env.data.qpos[0]),
                }
            )

        if stage == "swing":
            action = trajectory_action(env, t, knots, swing_seconds, swing_kp, swing_kd)
        elif stage == "capture":
            action = lqr_action(env, capture_gain, scale=lqr_capture_scale, cart_target=0.0)
        else:
            action = lqr_action(env, capture_gain, scale=lqr_stabilize_scale, cart_target=0.0)

        action_abs_max = max(action_abs_max, abs(float(action)))
        _, reward, terminated, truncated, info = env.step([action])
        total_return += float(reward)
        max_cart_abs = max(max_cart_abs, abs(float(info["x"])))
        trajectory.append(
            row_from_env(
                env,
                step=step + 1,
                stage=stage,
                action=action,
                reward=reward,
                info=info,
            )
        )
        final_info = dict(info)
        if terminated or truncated:
            done_events.append(
                {
                    "step": int(step + 1),
                    "time_seconds": float((step + 1) * env.dt),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "success": bool(info.get("success", False)),
                    "termination_reason": info.get("termination_reason"),
                }
            )
            break

    env.close()
    success = bool(final_info.get("success", False))
    max_streak = float(final_info.get("max_upright_streak_seconds", 0.0))
    if success:
        failure_category = "success"
    elif final_info.get("termination_reason") == "rail_violation":
        failure_category = "rail_violation"
    elif final_info.get("time_to_first_upright") is None:
        failure_category = "no_swingup"
    elif max_streak < hold_seconds:
        failure_category = "capture_loss"
    else:
        failure_category = "post_capture_fall"
    return {
        "seed": int(seed),
        "success": success,
        "failure_category": failure_category,
        "episode_return": float(total_return),
        "simulated_steps": int(len(trajectory)),
        "simulated_seconds": float(len(trajectory) * env.dt),
        "initial_state": {"qpos": initial_qpos, "qvel": initial_qvel},
        "stage_events": stage_events,
        "max_cart_excursion": float(max_cart_abs),
        "action_abs_max": float(action_abs_max),
        "time_to_first_upright": final_info.get("time_to_first_upright"),
        "time_to_capture": final_info.get("time_to_capture"),
        "max_upright_streak_seconds": max_streak,
        "final_upright_streak_seconds": float(final_info.get("upright_streak_seconds", 0.0)),
        "termination_reason": final_info.get("termination_reason"),
        "done_events": done_events,
        "final_info": final_info,
        "trajectory": trajectory,
    }


def benchmark_checks(cfg: dict[str, Any], *, expected_links: int, progress: float, zero_noise: bool) -> dict[str, Any]:
    env_cfg = cfg["env"]
    checks = {
        "n_links": int(env_cfg.get("n_links")) == expected_links,
        "total_length": np.isclose(float(env_cfg.get("total_length")), 3.0),
        "total_mass": np.isclose(float(env_cfg.get("total_mass")), 1.0),
        "cart_mass": np.isclose(float(env_cfg.get("cart_mass")), 1.0),
        "rail_limit": np.isclose(float(env_cfg.get("rail_limit")), 3.0),
        "force_limit": np.isclose(float(env_cfg.get("force_limit")), 80.0),
        "timestep": np.isclose(float(env_cfg.get("timestep")), 0.005),
        "frame_skip": int(env_cfg.get("frame_skip")) == 4,
        "cart_damping": np.isclose(float(env_cfg.get("cart_damping")), 0.02),
        "joint_armature": np.isclose(float(env_cfg.get("joint_armature")), 0.0005),
        "init_mode": str(env_cfg.get("init_mode")) == "hanging",
        "progress": np.isclose(float(progress), 1.0),
        "no_state_list": str(env_cfg.get("init_mode")) not in {"state_list", "fixed_state"},
        "nonzero_distribution_noise": not zero_noise,
    }
    return {"checks": checks, "passed": bool(all(checks.values()))}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate reset-free global discovery on the exact hanging-start MuJoCo plant")
    parser.add_argument("--config", default="configs/swingup6_uniform.yaml")
    parser.add_argument("--swing-controller-json", default=None)
    parser.add_argument("--seeds", default="0,1,2,3")
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--out", default="runs/goal_global_discovery/eval_global_discovery.json")
    parser.add_argument("--capture-enter-angle", type=float, default=0.16)
    parser.add_argument("--capture-min-time", type=float, default=5.70)
    parser.add_argument("--stabilize-enter-angle", type=float, default=0.15)
    parser.add_argument("--stabilize-enter-streak", type=float, default=0.02)
    parser.add_argument("--stabilize-hinge-rms", type=float, default=1.0)
    parser.add_argument("--min-capture-seconds", type=float, default=0.50)
    parser.add_argument("--hold-seconds", type=float, default=5.0)
    parser.add_argument("--minimum-heldout", type=int, default=3)
    parser.add_argument("--lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--lqr-capture-scale", type=float, default=1.0)
    parser.add_argument("--lqr-stabilize-scale", type=float, default=1.0)
    parser.add_argument("--zero-noise", action="store_true", help="Diagnostic only; zero-noise runs cannot pass the distribution gate")
    parser.add_argument("--fail-on-gate", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    if not seeds:
        raise ValueError("--seeds must contain at least one integer")
    if args.minimum_heldout < 1 or args.minimum_heldout > len(seeds):
        raise ValueError("--minimum-heldout must be in [1, number of seeds]")

    cfg = apply_overrides(load_config(args.config), args.override)
    source_init_mode = str(cfg["env"].get("init_mode", "upright"))
    # The six-link training config still uses a curriculum init label. For
    # this gate, resolve it to the explicit hanging distribution and record it.
    cfg["env"] = {**cfg["env"], "init_mode": "hanging"}
    if args.zero_noise:
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0

    swing_controller = load_swing_controller(args.swing_controller_json)
    controller_manifest = {
        "swing": swing_controller,
        "capture": {
            "type": "finite_difference_lqr",
            "control_cost": float(args.lqr_control_cost),
            "capture_scale": float(args.lqr_capture_scale),
            "stabilize_scale": float(args.lqr_stabilize_scale),
        },
        "switch": {
            "capture_enter_angle": float(args.capture_enter_angle),
            "capture_min_time": float(args.capture_min_time),
            "stabilize_enter_angle": float(args.stabilize_enter_angle),
            "stabilize_enter_streak": float(args.stabilize_enter_streak),
            "stabilize_hinge_rms": float(args.stabilize_hinge_rms),
            "min_capture_seconds": float(args.min_capture_seconds),
        },
    }
    controller_sha256 = data_sha256(controller_manifest)
    gain = lqr_gain(cfg, progress=args.progress, fd_eps=1e-7, control_cost=args.lqr_control_cost)
    episodes = [
        evaluate_episode(
            cfg,
            seed=seed,
            seconds=args.seconds,
            swing_controller=swing_controller,
            capture_gain=gain,
            capture_enter_angle=args.capture_enter_angle,
            capture_min_time=args.capture_min_time,
            stabilize_enter_angle=args.stabilize_enter_angle,
            stabilize_enter_streak=args.stabilize_enter_streak,
            stabilize_hinge_rms=args.stabilize_hinge_rms,
            min_capture_seconds=args.min_capture_seconds,
            lqr_capture_scale=args.lqr_capture_scale,
            lqr_stabilize_scale=args.lqr_stabilize_scale,
            hold_seconds=args.hold_seconds,
        )
        for seed in seeds
    ]
    hold_count = sum(float(row["max_upright_streak_seconds"]) >= args.hold_seconds for row in episodes)
    exact = benchmark_checks(cfg, expected_links=6, progress=args.progress, zero_noise=args.zero_noise)
    gate = {
        "exact_benchmark": exact,
        "minimum_heldout_starts": int(args.minimum_heldout),
        "heldout_starts_with_hold": int(hold_count),
        "any_five_second_hold": bool(hold_count > 0),
        "same_controller_three_start_hold": bool(hold_count >= args.minimum_heldout),
        "passed": bool(exact["passed"] and hold_count >= args.minimum_heldout),
    }
    xml_env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=seeds[0])
    generated_xml_sha256 = text_sha256(xml_env.xml)
    xml_env.close()
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": not gate["passed"],
        "summary": "Global-discovery gate: exact hanging-start episodes with complete reset-free physical traces.",
        "source_config": {
            "path": str(Path(args.config)),
            "source_init_mode": source_init_mode,
            "resolved_sha256": data_sha256(cfg),
            "metadata": file_metadata(args.config),
        },
        "config_sha256": data_sha256(cfg),
        "generated_xml_sha256": generated_xml_sha256,
        "controller": controller_manifest,
        "controller_sha256": controller_sha256,
        "swing_controller_source": None if args.swing_controller_json is None else file_metadata(args.swing_controller_json),
        "evaluation": {
            "seeds": seeds,
            "seconds": float(args.seconds),
            "progress": float(args.progress),
            "zero_noise": bool(args.zero_noise),
            "action_frequency_hz": 50.0,
            "hold_seconds": float(args.hold_seconds),
        },
        "gate": gate,
        "episodes": episodes,
        "trajectory_sha256": data_sha256(episodes),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    dump_json(json_safe(result), Path(args.out))
    print(
        f"global_discovery_passed={gate['passed']} "
        f"holds={hold_count}/{len(episodes)} "
        f"best_streak={max(float(row['max_upright_streak_seconds']) for row in episodes):.3f}s"
    )
    print(f"Wrote {args.out}")
    if args.fail_on_gate and not gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
