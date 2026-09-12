#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

from torch_runtime import prepare_runtime

prepare_runtime()

from gcartpole.config import apply_overrides, dump_json, load_config
from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.evidence import git_metadata, runtime_metadata, utc_timestamp
from make_lqr_checkpoint import absolute_angle_cost, finite_difference_dynamics


class LinearMPC:
    """Unconstrained condensed linear MPC with action and rail-safe clipping."""

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
        fallback_gain: np.ndarray,
        fallback_scale: float,
    ) -> None:
        del rail_constraint
        self.a = np.asarray(a, dtype=np.float64)
        self.b = np.asarray(b, dtype=np.float64)
        self.horizon = int(horizon)
        self.fallback_gain = np.asarray(fallback_gain, dtype=np.float64)
        self.fallback_scale = float(fallback_scale)
        nx = self.a.shape[0]
        nu = self.b.shape[1]
        state_transition = np.zeros((self.horizon * nx, nx), dtype=np.float64)
        action_transition = np.zeros((self.horizon * nx, self.horizon * nu), dtype=np.float64)
        a_power = np.eye(nx, dtype=np.float64)
        for step in range(self.horizon):
            a_power = self.a @ a_power
            row = slice(step * nx, (step + 1) * nx)
            state_transition[row] = a_power
            for action_step in range(step + 1):
                col = slice(action_step * nu, (action_step + 1) * nu)
                action_transition[row, col] = np.linalg.matrix_power(
                    self.a, step - action_step
                ) @ self.b
        q_blocks = [q] * (self.horizon - 1) + [terminal_cost]
        q_bar = np.zeros((self.horizon * nx, self.horizon * nx), dtype=np.float64)
        for step, block in enumerate(q_blocks):
            row = slice(step * nx, (step + 1) * nx)
            q_bar[row, row] = block
        r_bar = np.kron(np.eye(self.horizon), r)
        hessian = action_transition.T @ q_bar @ action_transition + r_bar
        self.hessian = 0.5 * (hessian + hessian.T) + 1e-9 * np.eye(hessian.shape[0])
        self.linear_cost_map = action_transition.T @ q_bar @ state_transition

    def action(self, state: np.ndarray) -> tuple[float, str, int]:
        state = np.asarray(state, dtype=np.float64)
        try:
            controls = -np.linalg.solve(self.hessian, self.linear_cost_map @ state)
            action = float(np.clip(controls[0], -1.0, 1.0))
            return action, "solved", 0
        except np.linalg.LinAlgError:
            fallback = -self.fallback_scale * float(self.fallback_gain @ state)
            return float(np.clip(fallback, -1.0, 1.0)), "linalg_fallback", 0


def state_from_env(env: NLinkCartPoleEnv) -> np.ndarray:
    qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    qpos[1:] = wrap_angle(qpos[1:])
    return np.r_[qpos, np.asarray(env.data.qvel, dtype=np.float64)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate receding linear MPC on upright-start n-link states")
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress", type=float, default=1.0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--rail-constraint", type=float, default=2.90)
    parser.add_argument("--control-cost", type=float, default=1000.0)
    parser.add_argument("--terminal-q-factor", type=float, default=100.0)
    parser.add_argument("--fallback-lqr-scale", type=float, default=1.30)
    parser.add_argument("--episode-seconds", type=float, default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = apply_overrides(load_config(args.config), args.override)
    if args.episode_seconds is not None:
        cfg["env"]["episode_seconds"] = float(args.episode_seconds)
    a, b = finite_difference_dynamics(cfg, args.progress, 1e-7)
    n = int(cfg["env"]["n_links"])
    q_weights = {
        "cart_position": 0.1,
        "absolute_angle": 100.0,
        "cart_velocity": 0.1,
        "absolute_angular_velocity": 1.0,
        "relative_angle": 1.0,
        "relative_angular_velocity": 0.01,
    }
    q = absolute_angle_cost(n, q_weights)
    r = np.asarray([[float(args.control_cost)]], dtype=np.float64)
    riccati = solve_discrete_are(a, b, q, r)
    cost_scale = float(np.max(np.diag(riccati)))
    q_scaled = q / cost_scale
    r_scaled = r / cost_scale
    terminal = float(args.terminal_q_factor) * q / cost_scale
    gain = np.linalg.solve(b.T @ riccati @ b + r, b.T @ riccati @ a).reshape(-1)
    controller = LinearMPC(
        a,
        b,
        q_scaled,
        r_scaled,
        terminal,
        horizon=args.horizon,
        rail_constraint=args.rail_constraint,
        fallback_gain=gain,
        fallback_scale=args.fallback_lqr_scale,
    )

    episodes: list[dict[str, object]] = []
    for episode in range(int(args.episodes)):
        env = NLinkCartPoleEnv(cfg, progress=args.progress, seed=args.seed + episode)
        env.reset(seed=args.seed + episode)
        done = False
        episode_return = 0.0
        solver_statuses: dict[str, int] = {}
        solver_iterations: list[int] = []
        info: dict[str, object] = {}
        while not done:
            action, status, iterations = controller.action(state_from_env(env))
            solver_statuses[status] = solver_statuses.get(status, 0) + 1
            solver_iterations.append(iterations)
            _, reward, terminated, truncated, info = env.step([action])
            episode_return += float(reward)
            done = bool(terminated or truncated)
        episodes.append(
            {
                "episode": episode,
                "seed": args.seed + episode,
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
        )
        env.close()

    payload = {
        "generated_at": utc_timestamp(),
        "not_solution": True,
        "summary": "Upright-start receding linear MPC maintenance diagnostic; not final swing-up evidence.",
        "config": str(Path(args.config)),
        "progress": float(args.progress),
        "episodes": int(args.episodes),
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "max_upright_streak_mean": float(np.mean([row["max_upright_streak_seconds"] for row in episodes])),
        "max_upright_streak_median": float(np.median([row["max_upright_streak_seconds"] for row in episodes])),
        "episode_results": episodes,
        "mpc": {
            "horizon_steps": int(args.horizon),
            "horizon_seconds": float(args.horizon * cfg["env"]["timestep"] * cfg["env"]["frame_skip"]),
            "rail_constraint": float(args.rail_constraint),
            "control_cost": float(args.control_cost),
            "terminal_q_factor": float(args.terminal_q_factor),
            "fallback_lqr_scale": float(args.fallback_lqr_scale),
            "cost_normalization": cost_scale,
            "q_weights": q_weights,
        },
        "evidence": {
            "overrides": list(args.override),
            "runtime": runtime_metadata(),
            "git": git_metadata(Path(__file__).resolve().parents[1]),
        },
    }
    dump_json(payload, Path(args.out))
    print(
        f"episodes={args.episodes} success_rate={payload['success_rate']:.3f} "
        f"median_hold={payload['max_upright_streak_median']:.3f}s"
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
