#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import osqp
from scipy import sparse
from scipy.linalg import solve_discrete_are

from gcartpole.capture_envelope import validate_capture_config, validate_capture_states
from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.evidence import file_metadata, git_metadata, runtime_metadata, utc_timestamp
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
try:
    from scripts.make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics
except ModuleNotFoundError:
    from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics


class LinearMPC:
    def __init__(
        self,
        a: np.ndarray,
        b: np.ndarray,
        q: np.ndarray,
        r: np.ndarray,
        terminal_cost: np.ndarray,
        *,
        horizon: int,
        rail_constraint: float,
        force_bound: float = 1.0,
        fallback_gain: np.ndarray | None = None,
        fallback_scale: float = 1.3,
    ) -> None:
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.nx = self.a.shape[0]
        self.nu = self.b.shape[1]
        self.horizon = int(horizon)
        self.fallback_gain = None if fallback_gain is None else np.asarray(fallback_gain, dtype=np.float64)
        self.fallback_scale = float(fallback_scale)
        if self.horizon < 1:
            raise ValueError("MPC horizon must be positive")

        # Condense the dynamics into the action variables. The six-link linearized
        # system is poorly scaled; retaining every state as a QP variable makes the
        # equality-constrained OSQP form fail even for feasible near-origin states.
        state_transition = np.zeros((self.horizon * self.nx, self.nx), dtype=np.float64)
        action_transition = np.zeros(
            (self.horizon * self.nx, self.horizon * self.nu), dtype=np.float64
        )
        a_power = np.eye(self.nx, dtype=np.float64)
        for step in range(self.horizon):
            a_power = self.a @ a_power
            row = slice(step * self.nx, (step + 1) * self.nx)
            state_transition[row] = a_power
            for action_step in range(step + 1):
                col = slice(action_step * self.nu, (action_step + 1) * self.nu)
                action_transition[row, col] = np.linalg.matrix_power(
                    self.a, step - action_step
                ) @ self.b

        state_cost = sparse.block_diag(
            [sparse.csc_matrix(q)] * (self.horizon - 1)
            + [sparse.csc_matrix(terminal_cost)],
            format="csc",
        )
        action_cost = sparse.kron(sparse.eye(self.horizon), sparse.csc_matrix(r), format="csc")
        hessian = action_transition.T @ state_cost @ action_transition + action_cost
        hessian = 0.5 * (hessian + hessian.T)
        self.linear_cost_map = 2.0 * np.asarray(action_transition.T @ state_cost @ state_transition)

        action_bounds = sparse.eye(self.horizon * self.nu, format="csc")
        cart_rows = np.arange(0, self.horizon * self.nx, self.nx)
        self.rail_state_map = state_transition[cart_rows]
        rail_action_map = sparse.csc_matrix(action_transition[cart_rows])
        constraint_matrix = sparse.vstack([action_bounds, rail_action_map], format="csc")
        action_count = self.horizon * self.nu
        self.action_lower = np.full(action_count, -float(force_bound))
        self.action_upper = np.full(action_count, float(force_bound))
        self.rail_constraint = float(rail_constraint)
        rail_lower = np.full(self.horizon, -self.rail_constraint)
        rail_upper = np.full(self.horizon, self.rail_constraint)
        lower = np.r_[self.action_lower, rail_lower]
        upper = np.r_[self.action_upper, rail_upper]
        self.problem = osqp.OSQP()
        self.problem.setup(
            P=sparse.csc_matrix(2.0 * hessian),
            q=np.zeros(action_count),
            A=constraint_matrix,
            l=lower,
            u=upper,
            verbose=False,
            polishing=False,
            warm_starting=True,
            eps_abs=1e-5,
            eps_rel=1e-5,
            max_iter=5000,
        )

    def action(self, state: np.ndarray) -> tuple[float, str, int]:
        state = np.asarray(state, dtype=np.float64)
        rail_offset = self.rail_state_map @ state
        lower = np.r_[self.action_lower, -self.rail_constraint - rail_offset]
        upper = np.r_[self.action_upper, self.rail_constraint - rail_offset]
        self.problem.update(q=self.linear_cost_map @ state, l=lower, u=upper)
        result = self.problem.solve(raise_error=False)
        status = str(result.info.status)
        if result.x is None or result.info.status_val not in {1, 2}:
            if self.fallback_gain is None:
                return 0.0, status, int(result.info.iter)
            fallback = -self.fallback_scale * float(self.fallback_gain @ state)
            return float(np.clip(fallback, -1.0, 1.0)), f"{status};lqr_fallback", int(result.info.iter)
        return float(np.clip(result.x[0], -1.0, 1.0)), status, int(result.info.iter)


def state_from_env(env: NLinkCartPoleEnv) -> np.ndarray:
    qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    qpos[1:] = wrap_angle(qpos[1:])
    return np.r_[qpos, np.asarray(env.data.qvel, dtype=np.float64)]


def evaluate_episode(
    cfg: dict[str, Any],
    controller: LinearMPC,
    *,
    state: dict[str, Any],
    state_index: int,
    seed: int,
    progress: float,
) -> dict[str, Any]:
    env = NLinkCartPoleEnv(cfg, progress=progress, seed=seed)
    _, _ = env.reset(seed=seed, options={"qpos": state["qpos"], "qvel": state["qvel"]})
    episode_return = 0.0
    solver_iterations = []
    solver_statuses: dict[str, int] = {}
    terminated = False
    truncated = False
    info: dict[str, Any] = {}
    try:
        while not (terminated or truncated):
            action, status, iterations = controller.action(state_from_env(env))
            solver_iterations.append(iterations)
            solver_statuses[status] = solver_statuses.get(status, 0) + 1
            _, reward, terminated, truncated, info = env.step([action])
            episode_return += float(reward)
    finally:
        env.close()
    return {
        "episode": state_index,
        "state_index": state_index,
        "state_id": state.get("state_id"),
        "seed": seed,
        "return": episode_return,
        "length": int(info.get("step", 0)),
        "success": bool(info.get("success", False)),
        "termination_reason": info.get("termination_reason"),
        "time_to_first_upright": info.get("time_to_first_upright"),
        "time_to_capture": info.get("time_to_capture"),
        "max_upright_streak_seconds": float(info.get("max_upright_streak_seconds", 0.0)),
        "final_upright_streak_seconds": float(info.get("upright_streak_seconds", 0.0)),
        "max_cart_excursion": float(info.get("max_cart_excursion", 0.0)),
        "solver_status_counts": solver_statuses,
        "solver_iterations_mean": float(np.mean(solver_iterations)),
        "solver_iterations_max": int(np.max(solver_iterations)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate constrained linear MPC on frozen capture states")
    parser.add_argument("--config", default="configs/swingup6_capture_envelope.yaml")
    parser.add_argument("--dataset", default="runs/p1_capture_envelope/validation.json")
    parser.add_argument("--spec", default="benchmarks/p1_capture_envelope.yaml")
    parser.add_argument("--progress", type=float, default=0.10)
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--rail-constraint", type=float, default=2.90)
    parser.add_argument("--control-cost", type=float, default=1000.0)
    parser.add_argument("--terminal-cost", choices=["riccati", "scaled-q"], default="riccati")
    parser.add_argument("--terminal-q-factor", type=float, default=100.0)
    parser.add_argument("--fallback-lqr-scale", type=float, default=1.30)
    parser.add_argument("--cart-position-cost", type=float, default=0.1)
    parser.add_argument("--cart-velocity-cost", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=61501)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    spec = load_config(args.spec)
    source = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    split = source.get("split") if isinstance(source, dict) else None
    errors = validate_capture_config(cfg, spec)
    if split in spec.get("splits", {}):
        errors.extend(validate_capture_states(source, spec, str(split)))
    else:
        errors.append(f"dataset split {split!r} is not declared by the capture specification")
    if errors:
        raise ValueError("Capture benchmark validation failed:\n- " + "\n- ".join(errors))
    cfg["env"]["action_lqr_residual"]["enabled"] = False
    states = source.get("states", source) if isinstance(source, dict) else source
    if not isinstance(states, list) or not states:
        raise ValueError("MPC dataset must contain capture states")
    count = min(len(states), int(args.limit))
    a, b = finite_difference_dynamics(cfg, 1.0, 1e-7)
    n = int(cfg["env"]["n_links"])
    q_weights = {
        "cart_position": float(args.cart_position_cost),
        "absolute_angle": 100.0,
        "cart_velocity": float(args.cart_velocity_cost),
        "absolute_angular_velocity": 1.0,
        "relative_angle": 1.0,
        "relative_angular_velocity": 0.01,
    }
    q = absolute_angle_cost(n, q_weights)
    r = np.asarray([[float(args.control_cost)]], dtype=np.float64)
    riccati_cost = solve_discrete_are(a, b, q, r)
    lqr_gain = np.linalg.solve(b.T @ riccati_cost @ b + r, b.T @ riccati_cost @ a).reshape(-1)
    if args.terminal_cost == "riccati":
        terminal_cost = riccati_cost
    else:
        terminal_cost = float(args.terminal_q_factor) * q
    cost_scale = float(np.max(np.diag(terminal_cost)))
    q_scaled = q / cost_scale
    r_scaled = r / cost_scale
    terminal_cost_scaled = terminal_cost / cost_scale
    controller = LinearMPC(
        a,
        b,
        q_scaled,
        r_scaled,
        terminal_cost_scaled,
        horizon=args.horizon,
        rail_constraint=args.rail_constraint,
        fallback_gain=lqr_gain,
        fallback_scale=args.fallback_lqr_scale,
    )
    episodes = [
        evaluate_episode(
            cfg,
            controller,
            state=states[index],
            state_index=index,
            seed=args.seed + index,
            progress=args.progress,
        )
        for index in range(count)
    ]
    successes = np.asarray([row["success"] for row in episodes], dtype=np.float64)
    holds = np.asarray([row["max_upright_streak_seconds"] for row in episodes], dtype=np.float64)
    payload = {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Constrained linear MPC capture diagnostic with a scaled terminal state cost.",
        "progress": float(args.progress),
        "episodes": count,
        "success_rate": float(np.mean(successes)),
        "max_upright_streak_mean": float(np.mean(holds)),
        "max_upright_streak_median": float(np.median(holds)),
        "episode_results": episodes,
        "mpc": {
            "horizon_steps": int(args.horizon),
            "horizon_seconds": float(args.horizon * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]),
            "rail_constraint": float(args.rail_constraint),
            "control_cost": float(args.control_cost),
            "terminal_cost": args.terminal_cost,
            "terminal_q_factor": float(args.terminal_q_factor),
            "cost_normalization": cost_scale,
            "fallback_lqr_scale": float(args.fallback_lqr_scale),
            "q_weights": q_weights,
        },
        "evidence": {
            "config_path": str(Path(args.config)),
            "overrides": list(args.override),
            "spec": file_metadata(args.spec),
            "dataset": file_metadata(args.dataset),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, args.out)
    print(
        f"episodes={count} success_rate={payload['success_rate']:.3f} "
        f"median_hold={payload['max_upright_streak_median']:.3f}s"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
