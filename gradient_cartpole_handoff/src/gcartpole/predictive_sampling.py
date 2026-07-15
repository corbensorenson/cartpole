from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
from mujoco import rollout as mujoco_rollout


Array = np.ndarray


@dataclass(frozen=True)
class PredictiveSamplingConfig:
    horizon_steps: int = 150
    replan_steps: int = 5
    knot_count: int = 24
    iterations: int = 6
    population: int = 1024
    elites: int = 64
    archive_size: int = 1
    action_sigma: float = 0.7
    sigma_decay: float = 0.8
    sigma_floor: float = 0.025
    handoff_lyapunov: float = 1800.0
    handoff_cart_abs: float = 1.5
    handoff_angle_ratio: float = 1.0
    handoff_cart_velocity_ratio: float = 1.0
    handoff_hinge_velocity_ratio: float = 1.0

    def __post_init__(self) -> None:
        if (
            min(
                self.horizon_steps,
                self.replan_steps,
                self.knot_count,
                self.iterations,
                self.population,
                self.elites,
                self.archive_size,
            )
            < 1
        ):
            raise ValueError("planner counts must be positive")
        if self.knot_count < 2:
            raise ValueError("knot_count must be at least two")
        if max(self.elites, self.archive_size) > self.population:
            raise ValueError("elites and archive_size cannot exceed population")
        if (
            min(
                self.action_sigma,
                self.sigma_decay,
                self.sigma_floor,
                self.handoff_lyapunov,
                self.handoff_cart_abs,
                self.handoff_angle_ratio,
                self.handoff_cart_velocity_ratio,
                self.handoff_hinge_velocity_ratio,
            )
            <= 0.0
        ):
            raise ValueError("planner scales and handoff bounds must be positive")


def interpolation_matrix(knot_count: int, step_count: int) -> Array:
    if knot_count < 2 or step_count < 2:
        raise ValueError("knot_count and step_count must be at least two")
    source = np.linspace(0.0, 1.0, knot_count, dtype=np.float64)
    target = np.linspace(0.0, 1.0, step_count, dtype=np.float64)
    matrix = np.zeros((step_count, knot_count), dtype=np.float64)
    right = np.searchsorted(source, target, side="right")
    right = np.clip(right, 1, knot_count - 1)
    left = right - 1
    fraction = (target - source[left]) / (source[right] - source[left])
    rows = np.arange(step_count)
    matrix[rows, left] = 1.0 - fraction
    matrix[rows, right] += fraction
    return matrix


def shift_action_knots(
    knots: Array,
    *,
    elapsed_steps: int,
    horizon_steps: int,
) -> Array:
    knots = np.asarray(knots, dtype=np.float64)
    if knots.ndim != 1 or knots.size < 2:
        raise ValueError(
            "knots must be a one-dimensional array with at least two entries"
        )
    if elapsed_steps < 0 or horizon_steps < 2:
        raise ValueError(
            "elapsed_steps must be nonnegative and horizon_steps at least two"
        )
    source = np.linspace(0.0, horizon_steps - 1, knots.size, dtype=np.float64)
    query = source + float(elapsed_steps)
    shifted = np.interp(query, source, knots, right=0.0)
    shifted[-1] = 0.0
    return shifted


class PredictiveSamplingPlanner:
    """Multithreaded direct-action CEM using exact MuJoCo rollouts."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        frame_skip: int,
        force_limit: float,
        coordinate_transform: Array,
        lyapunov: Array,
        config: PredictiveSamplingConfig,
        rail_limit: float = 3.0,
        threads: int = 0,
    ) -> None:
        self.model = model
        self.frame_skip = int(frame_skip)
        self.force_limit = float(force_limit)
        self.rail_limit = float(rail_limit)
        self.transform = np.asarray(coordinate_transform, dtype=np.float64)
        self.lyapunov = np.asarray(lyapunov, dtype=np.float64)
        self.config = config
        self.nq = int(model.nq)
        self.nv = int(model.nv)
        self.nx = self.nq + self.nv
        if self.frame_skip < 1 or min(self.force_limit, self.rail_limit) <= 0.0:
            raise ValueError("frame_skip, force_limit, and rail_limit must be positive")
        if int(model.nu) != 1:
            raise ValueError("predictive sampling currently requires one actuator")
        if self.transform.shape != (self.nx, self.nx):
            raise ValueError("coordinate transform does not match model state")
        if self.lyapunov.shape != (self.nx, self.nx):
            raise ValueError("Lyapunov matrix does not match model state")
        self._interpolation = interpolation_matrix(
            config.knot_count, config.horizon_steps
        )
        thread_count = max(1, int(threads))
        self._rollout_data = [mujoco.MjData(model) for _ in range(thread_count)]
        self._state_size = mujoco.mj_stateSize(
            model, mujoco.mjtState.mjSTATE_FULLPHYSICS.value
        )

    def full_physics_state(self, data: mujoco.MjData) -> Array:
        state = np.empty(self._state_size, dtype=np.float64)
        mujoco.mj_getState(
            self.model,
            data,
            state,
            mujoco.mjtState.mjSTATE_FULLPHYSICS.value,
        )
        return state

    def actions_from_knots(self, knots: Array) -> Array:
        knots = np.asarray(knots, dtype=np.float64)
        if knots.ndim == 1:
            if knots.shape != (self.config.knot_count,):
                raise ValueError("knot vector has the wrong length")
            actions = np.clip(self._interpolation @ knots, -1.0, 1.0)
            return actions.astype(np.float32).astype(np.float64)
        if knots.ndim != 2 or knots.shape[1] != self.config.knot_count:
            raise ValueError("knot matrix has the wrong shape")
        actions = np.clip(knots @ self._interpolation.T, -1.0, 1.0)
        return actions.astype(np.float32).astype(np.float64)

    def handoff_state(self, qpos: Array, qvel: Array) -> tuple[bool, float]:
        qpos = np.asarray(qpos, dtype=np.float64).copy()
        qvel = np.asarray(qvel, dtype=np.float64)
        if qpos.shape != (self.nq,) or qvel.shape != (self.nv,):
            raise ValueError("qpos or qvel does not match the planner model")
        qpos[1:] = (qpos[1:] + np.pi) % (2.0 * np.pi) - np.pi
        coordinates = self.transform @ np.r_[qpos, qvel]
        value = float(max(0.0, coordinates @ self.lyapunov @ coordinates))
        accepted = (
            value <= self.config.handoff_lyapunov
            and abs(float(qpos[0])) <= self.config.handoff_cart_abs
            and float(np.max(np.abs(coordinates[1 : self.nq])))
            <= self.config.handoff_angle_ratio
            and abs(float(coordinates[self.nq]))
            <= self.config.handoff_cart_velocity_ratio
            and float(np.sqrt(np.mean(coordinates[self.nq + 1 :] ** 2)))
            <= self.config.handoff_hinge_velocity_ratio
        )
        return bool(accepted), value

    def rollout(self, initial_state: Array, knots: Array) -> tuple[Array, Array]:
        policy_actions = self.actions_from_knots(knots)
        if policy_actions.ndim == 1:
            policy_actions = policy_actions[None, :]
        controls = np.repeat(policy_actions, self.frame_skip, axis=1)
        controls = (controls * self.force_limit)[:, :, None]
        states, _ = mujoco_rollout.rollout(
            self.model,
            self._rollout_data,
            np.asarray(initial_state, dtype=np.float64)[None, :],
            controls,
            persistent_pool=True,
        )
        return states[:, self.frame_skip - 1 :: self.frame_skip], policy_actions

    def _coordinate_states(self, rollout_states: Array) -> tuple[Array, Array]:
        qpos = rollout_states[..., 1 : 1 + self.nq].copy()
        qvel = rollout_states[..., 1 + self.nq : 1 + self.nq + self.nv]
        qpos[..., 1:] = (qpos[..., 1:] + np.pi) % (2.0 * np.pi) - np.pi
        physical = np.concatenate((qpos, qvel), axis=-1)
        coordinates = np.einsum("ij,btj->bti", self.transform, physical)
        return coordinates, qpos[..., 0]

    def score(
        self, rollout_states: Array, policy_actions: Array
    ) -> tuple[Array, dict[str, Array]]:
        coordinates, cart = self._coordinate_states(rollout_states)
        values = np.maximum(
            0.0,
            np.einsum("bti,ij,btj->bt", coordinates, self.lyapunov, coordinates),
        )
        rail_hit = np.any(np.abs(cart) > self.rail_limit, axis=1)
        max_angle_ratio = np.max(np.abs(coordinates[..., 1 : self.nq]), axis=-1)
        cart_velocity_ratio = np.abs(coordinates[..., self.nq])
        hinge_velocity_ratio = np.sqrt(
            np.mean(coordinates[..., self.nq + 1 :] ** 2, axis=-1)
        )
        within_cart = np.abs(cart) <= self.config.handoff_cart_abs
        handoff_mask = (
            (values <= self.config.handoff_lyapunov)
            & within_cart
            & (max_angle_ratio <= self.config.handoff_angle_ratio)
            & (cart_velocity_ratio <= self.config.handoff_cart_velocity_ratio)
            & (hinge_velocity_ratio <= self.config.handoff_hinge_velocity_ratio)
        )
        latched = np.any(handoff_mask, axis=1) & ~rail_hit
        first_handoff = np.where(
            np.any(handoff_mask, axis=1),
            np.argmax(handoff_mask, axis=1),
            self.config.horizon_steps,
        )
        minimum_value = np.min(values, axis=1)
        terminal_value = values[:, -1]
        rail_ratio = np.abs(cart) / self.rail_limit
        capture_objective = 3_000.0 * np.log1p(values) + 1_000.0 * (
            max_angle_ratio**2 + cart_velocity_ratio**2 + hinge_velocity_ratio**2
        )
        minimum_capture_objective = np.min(capture_objective, axis=1)
        terminal_capture_objective = capture_objective[:, -1]
        score = (
            100_000_000.0 * rail_hit
            + terminal_capture_objective
            + 50.0 * np.log1p(minimum_value)
            + 0.1 * np.sum(np.log1p(values), axis=1)
            + 500.0 * np.sum(rail_ratio**8, axis=1)
            + 2.0 * np.sum(policy_actions**2, axis=1)
            - 100_000_000.0 * latched
            + 10.0 * first_handoff
        )
        return score, {
            "minimum_lyapunov": minimum_value,
            "minimum_capture_objective": minimum_capture_objective,
            "terminal_lyapunov": terminal_value,
            "terminal_capture_objective": terminal_capture_objective,
            "rail_hit": rail_hit,
            "latched": latched,
            "first_handoff_step": first_handoff,
            "max_cart_abs": np.max(np.abs(cart), axis=1),
        }

    def search(
        self,
        data: mujoco.MjData,
        *,
        rng: np.random.Generator,
        initial_knots: Array | None = None,
    ) -> dict[str, Any]:
        cfg = self.config
        center = (
            np.zeros(cfg.knot_count, dtype=np.float64)
            if initial_knots is None
            else np.asarray(initial_knots, dtype=np.float64).copy()
        )
        if center.shape != (cfg.knot_count,):
            raise ValueError("initial knots have the wrong shape")
        center[-1] = 0.0
        sigma = np.full(cfg.knot_count, cfg.action_sigma, dtype=np.float64)
        sigma[-1] = cfg.sigma_floor
        initial_state = self.full_physics_state(data)
        best_knots = center.copy()
        best_score = np.inf
        best_metrics: dict[str, Any] = {}
        history: list[dict[str, Any]] = []
        archive: list[dict[str, Any]] = []

        for iteration in range(cfg.iterations):
            candidates = np.empty((cfg.population, cfg.knot_count), dtype=np.float64)
            candidates[0] = best_knots
            if cfg.population > 1:
                candidates[1] = 0.0
            if cfg.population > 2:
                candidates[2:] = center + rng.normal(
                    0.0, sigma, size=(cfg.population - 2, cfg.knot_count)
                )
            candidates = np.clip(candidates, -1.0, 1.0)
            candidates[:, -1] = 0.0
            rollout_states, policy_actions = self.rollout(initial_state, candidates)
            scores, metrics = self.score(rollout_states, policy_actions)
            order = np.argsort(scores)
            top = int(order[0])
            if float(scores[top]) < best_score:
                best_score = float(scores[top])
                best_knots = candidates[top].copy()
                best_metrics = {
                    key: np.asarray(value)[top].item() for key, value in metrics.items()
                }
            archive_candidates = min(cfg.population, cfg.archive_size + 2)
            for candidate_index in order[:archive_candidates]:
                candidate_knots = candidates[int(candidate_index)].copy()
                if any(
                    np.array_equal(candidate_knots, np.asarray(row["knots"]))
                    for row in archive
                ):
                    continue
                archive.append(
                    {
                        "score": float(scores[int(candidate_index)]),
                        "knots": candidate_knots,
                        "metrics": {
                            key: np.asarray(value)[int(candidate_index)].item()
                            for key, value in metrics.items()
                        },
                    }
                )
            archive.sort(key=lambda row: float(row["score"]))
            del archive[cfg.archive_size :]
            history.append(
                {
                    "iteration": iteration + 1,
                    "score": float(scores[top]),
                    **{
                        key: np.asarray(value)[top].item()
                        for key, value in metrics.items()
                    },
                }
            )
            elite = candidates[order[: cfg.elites]]
            center = elite.mean(axis=0)
            sigma = np.maximum(cfg.sigma_decay * elite.std(axis=0), cfg.sigma_floor)
            center[-1] = 0.0
            sigma[-1] = cfg.sigma_floor

        return {
            "knots": best_knots,
            "actions": self.actions_from_knots(best_knots),
            "score": best_score,
            "metrics": best_metrics,
            "history": history,
            "archive": [
                {
                    "score": float(row["score"]),
                    "knots": np.asarray(row["knots"], dtype=np.float64),
                    "metrics": row["metrics"],
                }
                for row in archive
            ],
            "candidate_rollout_count": cfg.iterations * cfg.population,
        }
