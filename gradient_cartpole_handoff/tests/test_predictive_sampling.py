from __future__ import annotations

import unittest

import numpy as np

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.predictive_sampling import (
    PredictiveSamplingConfig,
    PredictiveSamplingPlanner,
    interpolation_matrix,
    shift_action_knots,
)


class PredictiveSamplingTests(unittest.TestCase):
    def test_archive_size_cannot_exceed_population(self) -> None:
        with self.assertRaisesRegex(ValueError, "archive_size"):
            PredictiveSamplingConfig(population=2, elites=1, archive_size=3)

    def test_interpolation_matrix_preserves_endpoints_and_linear_knots(self) -> None:
        matrix = interpolation_matrix(3, 5)
        np.testing.assert_allclose(
            matrix @ np.asarray([-1.0, 0.0, 1.0]), np.linspace(-1.0, 1.0, 5)
        )

    def test_shift_action_knots_advances_schedule_and_zeros_tail(self) -> None:
        shifted = shift_action_knots(
            np.asarray([0.0, 0.5, 1.0, 0.0]),
            elapsed_steps=2,
            horizon_steps=7,
        )
        np.testing.assert_allclose(shifted, [0.5, 1.0, 0.0, 0.0])

    def test_batched_rollout_matches_environment_policy_steps(self) -> None:
        cfg = load_config("configs/swingup6_capture_envelope.yaml")
        cfg["env"]["action_lqr_residual"]["enabled"] = False
        cfg["env"]["init_mode"] = "fixed_state"
        cfg["env"]["init_qpos"] = [0.0] * 7
        cfg["env"]["init_qvel"] = [0.0] * 7
        cfg["env"]["init_cart_noise"] = 0.0
        cfg["env"]["init_cart_vel_noise"] = 0.0
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=4)
        env.reset(seed=4)
        size = 2 * (env.n + 1)
        planner = PredictiveSamplingPlanner(
            env.model,
            frame_skip=env.frame_skip,
            force_limit=env.force_limit,
            coordinate_transform=np.eye(size),
            lyapunov=np.eye(size),
            config=PredictiveSamplingConfig(
                horizon_steps=5,
                replan_steps=1,
                knot_count=5,
                iterations=1,
                population=2,
                elites=1,
            ),
            rail_limit=env.rail_limit,
            threads=2,
        )
        initial = planner.full_physics_state(env.data)
        actions = np.asarray([0.0, 0.2, -0.1, 0.3, 0.0])
        rollout_states, rollout_actions = planner.rollout(initial, actions)
        expected = []
        for action in actions:
            env.step([float(action)])
            expected.append(np.r_[env.data.qpos.copy(), env.data.qvel.copy()])
        env.close()
        actual = np.concatenate(
            (
                rollout_states[0, :, 1 : 1 + size // 2],
                rollout_states[0, :, 1 + size // 2 : 1 + size],
            ),
            axis=1,
        )
        np.testing.assert_allclose(rollout_actions[0], actions, atol=1e-12)
        np.testing.assert_allclose(actual, np.asarray(expected), atol=1e-11)

    def test_handoff_requires_cold_velocity_at_the_same_state(self) -> None:
        cfg = load_config("configs/swingup6_capture_envelope.yaml")
        cfg["env"]["action_lqr_residual"]["enabled"] = False
        cfg["env"]["init_mode"] = "fixed_state"
        cfg["env"]["init_qpos"] = [0.0] * 7
        cfg["env"]["init_qvel"] = [0.0] * 7
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=5)
        env.reset(seed=5)
        size = 2 * (env.n + 1)
        planner = PredictiveSamplingPlanner(
            env.model,
            frame_skip=env.frame_skip,
            force_limit=env.force_limit,
            coordinate_transform=np.eye(size),
            lyapunov=np.eye(size),
            config=PredictiveSamplingConfig(
                horizon_steps=5,
                replan_steps=1,
                knot_count=5,
                iterations=1,
                population=2,
                elites=1,
                handoff_lyapunov=10.0,
                handoff_cart_abs=1.0,
                handoff_angle_ratio=0.2,
                handoff_cart_velocity_ratio=0.5,
                handoff_hinge_velocity_ratio=0.75,
            ),
            rail_limit=env.rail_limit,
            threads=2,
        )
        accepted, _ = planner.handoff_state(np.zeros(7), np.zeros(7))
        rejected, _ = planner.handoff_state(np.zeros(7), np.r_[0.0, np.ones(6)])
        env.close()
        self.assertTrue(accepted)
        self.assertFalse(rejected)


if __name__ == "__main__":
    unittest.main()
