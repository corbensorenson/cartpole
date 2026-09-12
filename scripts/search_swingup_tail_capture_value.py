#!/usr/bin/env python
"""Search swing-up tails using measured downstream capture value.

The tail is rolled out in batched exact MuJoCo.  Each candidate's actual
terminal state is then handed to a feedback checkpoint without resetting to a
planned state.  This is a discovery tool: its outputs are teacher proposals,
not canonical swing-up evidence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from torch_runtime import prepare_runtime

prepare_runtime()

import mujoco
import numpy as np
from mujoco import rollout as mujoco_rollout

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import data_sha256, git_metadata, runtime_metadata, utc_timestamp
from search_swingup_tail_action_cem import (
    interpolation_matrix,
    load_controller,
    load_tail_center,
    physical_metrics,
    replay_to_tail,
    score_batch,
)

try:
    from scripts.search_swingup_capture import lqr_action, lqr_gain
except ModuleNotFoundError:
    from search_swingup_capture import lqr_action, lqr_gain


def load_capture_policy(
    cfg: dict[str, Any],
    checkpoint: str,
    hidden_sizes: list[int],
) -> tuple[Any, Any]:
    from gcartpole.ppo_torch import ActorCritic, load_model, sample_action

    probe = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    model = ActorCritic(
        obs_dim=int(probe.observation_space.shape[0]),
        act_dim=int(probe.action_space.shape[0]),
        hidden_sizes=hidden_sizes,
        action_std_init=float(cfg["ppo"].get("action_std_init", 0.02)),
    )
    probe.close()
    load_model(model, checkpoint)
    model.eval()
    return model, sample_action


def capture_cost_from_info(info: dict[str, Any]) -> float:
    return float(
        30.0 * (float(info["max_abs_angle"]) / 0.15) ** 2
        + 10.0 * (float(info["hinge_velocity_rms"]) / 0.75) ** 2
        + 3.0 * (abs(float(info["x"])) / 1.25) ** 2
        + 3.0 * (abs(float(info["cart_velocity"])) / 0.50) ** 2
    )


def reset_capture_state(
    env: NLinkCartPoleEnv,
    qpos: np.ndarray,
    qvel: np.ndarray,
) -> None:
    """Reset bookkeeping, then restore the exact physical handoff state.

    Capture-training configs may schedule ``init_qpos_scale`` and
    ``init_qvel_scale`` for curriculum resets.  Calling ``reset(options=...)``
    alone would therefore evaluate a different, artificially quiet state.
    Keep those scales for observation normalization, but put the plant back at
    the exact state produced by the swing/tail rollout.
    """
    env.reset(seed=0, options={"qpos": qpos, "qvel": qvel})
    env.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
    env.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
    mujoco.mj_forward(env.model, env.data)
    env._last_potential_energy = env._potential_energy()
    env.max_cart_excursion = abs(float(env.data.qpos[0]))
    env._update_upright_tracking()


def evaluate_capture_batch(
    env: NLinkCartPoleEnv,
    model: Any,
    sample_action: Any,
    qpos_batch: np.ndarray,
    qvel_batch: np.ndarray,
    capture_steps: int,
    *,
    capture_mode: str,
    capture_gain: np.ndarray | None,
    capture_lqr_scale: float,
) -> dict[str, np.ndarray]:
    best_cost = np.full(len(qpos_batch), np.inf, dtype=np.float64)
    terminal_cost = np.full(len(qpos_batch), np.inf, dtype=np.float64)
    mean_cost = np.full(len(qpos_batch), np.inf, dtype=np.float64)
    max_streak = np.zeros(len(qpos_batch), dtype=np.float64)
    final_angle = np.full(len(qpos_batch), np.inf, dtype=np.float64)
    final_hinge = np.full(len(qpos_batch), np.inf, dtype=np.float64)
    final_cart = np.full(len(qpos_batch), np.inf, dtype=np.float64)
    final_cart_velocity = np.full(len(qpos_batch), np.inf, dtype=np.float64)

    for index, (qpos, qvel) in enumerate(zip(qpos_batch, qvel_batch)):
        reset_capture_state(env, qpos, qvel)
        costs: list[float] = []
        final_info: dict[str, Any] = {}
        for _ in range(capture_steps):
            if capture_mode == "lqr":
                if capture_gain is None:
                    raise ValueError("LQR capture mode requires a capture gain")
                action = lqr_action(
                    env,
                    capture_gain,
                    scale=capture_lqr_scale,
                    cart_target=0.0,
                )
            else:
                obs = env._get_obs()
                action, _, _ = sample_action(
                    model,
                    obs[None, :],
                    deterministic=True,
                )
                action = float(action[0, 0])
            _, _, terminated, truncated, info = env.step([float(action)])
            info["cart_velocity"] = float(env.data.qvel[0])
            costs.append(capture_cost_from_info(info))
            final_info = info
            max_streak[index] = max(
                max_streak[index],
                float(info.get("max_upright_streak_seconds", 0.0)),
            )
            if terminated or truncated:
                break
        if costs:
            best_cost[index] = min(costs)
            terminal_cost[index] = costs[-1]
            mean_cost[index] = float(np.mean(costs[-min(10, len(costs)) :]))
            final_angle[index] = float(final_info["max_abs_angle"])
            final_hinge[index] = float(final_info["hinge_velocity_rms"])
            final_cart[index] = abs(float(final_info["x"]))
            final_cart_velocity[index] = abs(float(final_info["cart_velocity"]))

    capture_cost = (
        0.60 * best_cost
        + 0.25 * terminal_cost
        + 0.15 * mean_cost
        - 1000.0 * max_streak
    )
    return {
        "capture_cost": capture_cost,
        "capture_best_cost": best_cost,
        "capture_terminal_cost": terminal_cost,
        "capture_mean_cost": mean_cost,
        "capture_max_streak": max_streak,
        "capture_final_angle": final_angle,
        "capture_final_hinge": final_hinge,
        "capture_final_cart": final_cart,
        "capture_final_cart_velocity": final_cart_velocity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--swing-controller-json", required=True)
    parser.add_argument("--controller-key", default=None)
    parser.add_argument("--init-tail-json", default=None)
    parser.add_argument("--init-tail-key", default="best")
    parser.add_argument("--capture-mode", choices=("torch", "lqr"), default="torch")
    parser.add_argument("--capture-checkpoint", default=None)
    parser.add_argument("--capture-hidden-sizes", default="")
    parser.add_argument("--capture-lqr-scale", type=float, default=0.5)
    parser.add_argument("--capture-lqr-control-cost", type=float, default=1000.0)
    parser.add_argument("--tail-start-seconds", type=float, default=11.0)
    parser.add_argument("--tail-seconds", type=float, default=5.0)
    parser.add_argument("--capture-seconds", type=float, default=2.0)
    parser.add_argument("--knot-count", type=int, default=32)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--elites", type=int, default=12)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--action-sigma", type=float, default=0.12)
    parser.add_argument("--sigma-decay", type=float, default=0.92)
    parser.add_argument("--sigma-floor", type=float, default=0.005)
    parser.add_argument("--angle-weight", type=float, default=15.0)
    parser.add_argument("--hinge-weight", type=float, default=100.0)
    parser.add_argument("--max-hinge-weight", type=float, default=12.0)
    parser.add_argument("--absolute-velocity-weight", type=float, default=15.0)
    parser.add_argument("--cart-weight", type=float, default=1.5)
    parser.add_argument("--cart-velocity-weight", type=float, default=2.0)
    parser.add_argument("--rail-penalty-limit", type=float, default=12.5)
    parser.add_argument("--rail-penalty-weight", type=float, default=20000.0)
    parser.add_argument("--best-score-weight", type=float, default=0.55)
    parser.add_argument("--terminal-score-weight", type=float, default=0.25)
    parser.add_argument("--tail-average-weight", type=float, default=0.20)
    parser.add_argument("--physical-weight", type=float, default=0.20)
    parser.add_argument("--capture-weight", type=float, default=1.0)
    parser.add_argument("--handoff-angle-limit", type=float, default=0.15)
    parser.add_argument("--handoff-hinge-limit", type=float, default=0.75)
    parser.add_argument("--handoff-cart-limit", type=float, default=1.25)
    parser.add_argument("--handoff-cart-velocity-limit", type=float, default=0.50)
    parser.add_argument("--seed", type=int, default=20260928)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    if args.knot_count < 2 or args.population < 2 or not (1 <= args.elites <= args.population):
        raise ValueError("invalid CEM dimensions")
    if args.capture_mode == "torch" and not args.capture_checkpoint:
        raise ValueError("--capture-checkpoint is required for torch capture mode")
    if args.capture_lqr_scale < 0.0 or args.capture_lqr_control_cost <= 0.0:
        raise ValueError("capture LQR scale must be nonnegative and control cost positive")
    cfg = apply_overrides(load_config(args.config), args.override)
    cfg["env"] = {
        **cfg["env"],
        "init_mode": "hanging",
        "init_angle_noise": 0.0,
        "init_vel_noise": 0.0,
        "init_cart_noise": 0.0,
        "init_cart_vel_noise": 0.0,
    }
    for key in ("init_angle_noise", "init_vel_noise", "init_cart_noise", "init_cart_vel_noise"):
        cfg["env"][f"{key}_start"] = 0.0
        cfg["env"][f"{key}_end"] = 0.0
    capture_residual_cfg = cfg["env"].get("action_lqr_residual")
    capture_residual_enabled = bool(
        isinstance(capture_residual_cfg, dict)
        and capture_residual_cfg.get("enabled", False)
    )
    # The source swing and optimized tail are direct-force artifacts.  The
    # residual teacher is enabled only while evaluating the downstream
    # capture checkpoint from each actual tail endpoint.
    if capture_residual_enabled:
        capture_residual_cfg["enabled"] = False
    controller = load_controller(args.swing_controller_json, args.controller_key)
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    initial_state = replay_to_tail(env, controller, args.tail_start_seconds)
    horizon_steps = max(2, int(round(args.tail_seconds / env.dt)))
    capture_steps = max(1, int(round(args.capture_seconds / env.dt)))
    interpolation = interpolation_matrix(args.knot_count, horizon_steps)
    pool = [mujoco.MjData(env.model) for _ in range(min(32, max(1, args.population // 16)))]
    hidden_sizes = [int(value) for value in args.capture_hidden_sizes.split(",") if value.strip()]
    if capture_residual_enabled:
        capture_residual_cfg["enabled"] = True
    capture_model = None
    sample_action = None
    capture_gain = None
    if args.capture_mode == "torch":
        capture_model, sample_action = load_capture_policy(
            cfg, str(args.capture_checkpoint), hidden_sizes
        )
    else:
        capture_gain = lqr_gain(
            cfg,
            progress=1.0,
            fd_eps=1e-7,
            control_cost=args.capture_lqr_control_cost,
        )
    rng = np.random.default_rng(args.seed)
    center = load_tail_center(args.init_tail_json, args.knot_count, args.init_tail_key)
    sigma = np.full(args.knot_count, args.action_sigma, dtype=np.float64)
    best_record: dict[str, Any] | None = None
    best_feasible: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []
    started = time.time()

    for iteration in range(args.iterations):
        knots = np.clip(
            center[None, :] + rng.normal(0.0, sigma, size=(args.population, args.knot_count)),
            -1.0,
            1.0,
        )
        knots[0] = center
        physical_cost, metrics = score_batch(
            env,
            initial_state,
            knots,
            interpolation,
            pool,
            min_tail_steps=max(1, int(round(0.20 / env.dt))),
            angle_weight=args.angle_weight,
            hinge_weight=args.hinge_weight,
            max_hinge_weight=args.max_hinge_weight,
            absolute_velocity_weight=args.absolute_velocity_weight,
            cart_weight=args.cart_weight,
            cart_velocity_weight=args.cart_velocity_weight,
            rail_penalty_limit=args.rail_penalty_limit,
            rail_penalty_weight=args.rail_penalty_weight,
            best_score_weight=args.best_score_weight,
            terminal_score_weight=args.terminal_score_weight,
            tail_average_weight=args.tail_average_weight,
            robust_window_steps=0,
        )
        capture = evaluate_capture_batch(
            env,
            capture_model,
            sample_action,
            metrics["qpos"][:, -1, :],
            metrics["qvel"][:, -1, :],
            capture_steps,
            capture_mode=args.capture_mode,
            capture_gain=capture_gain,
            capture_lqr_scale=args.capture_lqr_scale,
        )
        cost = args.physical_weight * physical_cost + args.capture_weight * capture["capture_cost"]
        order = np.argsort(cost)
        top = int(order[0])
        elite_knots = knots[order[: args.elites]]
        center = np.mean(elite_knots, axis=0)
        sigma = np.maximum(np.std(elite_knots, axis=0) * args.sigma_decay, args.sigma_floor)
        endpoint = len(metrics["max_angle"][top]) - 1
        record = {
            "iteration": iteration + 1,
            "cost": float(cost[top]),
            "physical_cost": float(physical_cost[top]),
            "capture_cost": float(capture["capture_cost"][top]),
            "capture_max_streak": float(capture["capture_max_streak"][top]),
            "capture_best_cost": float(capture["capture_best_cost"][top]),
            "endpoint_angle": float(metrics["max_angle"][top, endpoint]),
            "endpoint_hinge_rms": float(metrics["hinge_rms"][top, endpoint]),
            "endpoint_max_hinge": float(metrics["max_hinge"][top, endpoint]),
            "endpoint_cart_abs": float(metrics["qpos"][top, endpoint, 0]),
            "endpoint_cart_velocity": float(metrics["qvel"][top, endpoint, 0]),
            "rail": float(metrics["rail"][top]),
        }
        history.append(record)
        candidate = {
            **record,
            "knots": knots[top].astype(float).tolist(),
            "best_state": {
                "qpos": metrics["qpos"][top, endpoint].astype(float).tolist(),
                "qvel": metrics["qvel"][top, endpoint].astype(float).tolist(),
                "absolute_angles": metrics["absolute_angles"][top, endpoint].astype(float).tolist(),
            },
        }
        if best_record is None or candidate["cost"] < best_record["cost"]:
            best_record = candidate
        point_feasible = bool(
            record["endpoint_angle"] <= args.handoff_angle_limit
            and record["endpoint_hinge_rms"] <= args.handoff_hinge_limit
            and abs(record["endpoint_cart_abs"]) <= args.handoff_cart_limit
            and abs(record["endpoint_cart_velocity"]) <= args.handoff_cart_velocity_limit
        )
        if point_feasible and (best_feasible is None or candidate["cost"] < best_feasible["cost"]):
            best_feasible = candidate
        print(
            f"iter={iteration + 1:03d} cost={record['cost']:.3f} "
            f"capture={record['capture_cost']:.3f} streak={record['capture_max_streak']:.3f}s "
            f"angle={record['endpoint_angle']:.3f} hinge={record['endpoint_hinge_rms']:.3f} "
            f"x={record['endpoint_cart_abs']:.3f} xd={record['endpoint_cart_velocity']:.3f} "
            f"rail={record['rail']:.3f}",
            flush=True,
        )

    assert best_record is not None
    result = {
        "schema_version": 1,
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Exact-MuJoCo tail CEM ranked by downstream feedback capture from each actual terminal state.",
        "source_controller": controller,
        "source_controller_json": str(args.swing_controller_json),
        "controller_key": args.controller_key,
        "init_tail_json": args.init_tail_json,
        "init_tail_key": args.init_tail_key,
        "capture_mode": args.capture_mode,
        "capture_checkpoint": (
            None if args.capture_checkpoint is None else str(args.capture_checkpoint)
        ),
        "capture_lqr_scale": float(args.capture_lqr_scale),
        "capture_lqr_control_cost": float(args.capture_lqr_control_cost),
        "tail_start_seconds": float(args.tail_start_seconds),
        "tail_horizon_steps": int(horizon_steps),
        "tail_horizon_seconds": float(horizon_steps * env.dt),
        "capture_horizon_steps": int(capture_steps),
        "capture_horizon_seconds": float(capture_steps * env.dt),
        "search": {
            "seed": int(args.seed),
            "knot_count": int(args.knot_count),
            "population": int(args.population),
            "elites": int(args.elites),
            "iterations": int(args.iterations),
            "action_sigma": float(args.action_sigma),
            "sigma_decay": float(args.sigma_decay),
            "sigma_floor": float(args.sigma_floor),
            "physical_weight": float(args.physical_weight),
            "capture_weight": float(args.capture_weight),
            "wall_time_seconds": float(time.time() - started),
        },
        "best": best_record,
        "best_feasible": best_feasible,
        "history": history,
        "initial_state": {
            "qpos": initial_state[1 : 1 + env.model.nq].astype(float).tolist(),
            "qvel": initial_state[1 + env.model.nq : 1 + env.model.nq + env.model.nv].astype(float).tolist(),
        },
        "config_sha256": data_sha256(cfg),
        "runtime": runtime_metadata(),
        "git": git_metadata(Path(__file__).resolve().parents[1]),
    }
    env.close()
    dump_json(result, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
