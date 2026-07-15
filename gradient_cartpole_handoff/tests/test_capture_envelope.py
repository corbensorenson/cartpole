from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.capture_envelope import generate_capture_states, validate_capture_config, validate_capture_states
from gcartpole.capture_funnel import (
    CaptureFunnelModel,
    deterministic_stratified_split,
    effective_state,
    fit_capture_funnel,
    normalized_capture_coordinates,
    polynomial_features,
)
from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import MujocoTransition, data_state, state_difference
from gcartpole.ppo_mlx import select_evaluation_state_indices
from scripts.mine_capture_failures import build_mining_mixture
from scripts.evaluate_linear_mpc_capture import LinearMPC
from scripts.search_linear_policy import development_seed
from scripts.search_capture_recovery import recovery_residual
from scripts.search_capture_hybrid_schedule import evaluate_hybrid_schedule, resample_target_controller
from scripts.evaluate_feedback_mpc_capture import feedback_action, shift_schedule
from scripts.search_capture_target_schedule import evaluate_target_schedule, scheduled_cart_target
from scripts.search_swingup_capture import lqr_action
from scripts.search_ilqr_capture import load_initial_controls


ROOT = Path(__file__).resolve().parents[1]


class CaptureEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_config(ROOT / "benchmarks/p1_capture_envelope.yaml")

    def small_spec(self) -> dict:
        spec = copy.deepcopy(self.spec)
        spec["splits"]["test"]["count"] = 64
        return spec

    def test_generation_is_deterministic_and_within_bounds(self) -> None:
        spec = self.small_spec()
        first = generate_capture_states(spec, "test")
        second = generate_capture_states(spec, "test")
        self.assertEqual(first, second)
        self.assertEqual(validate_capture_states(first, spec, "test"), [])
        self.assertEqual(len({state["state_id"] for state in first["states"]}), 64)

    def test_split_seeds_produce_disjoint_states(self) -> None:
        spec = self.small_spec()
        spec["splits"]["validation"]["count"] = 64
        test_ids = {state["state_id"].split("-", 2)[-1] for state in generate_capture_states(spec, "test")["states"]}
        validation_ids = {
            state["state_id"].split("-", 2)[-1]
            for state in generate_capture_states(spec, "validation")["states"]
        }
        self.assertTrue(test_ids.isdisjoint(validation_ids))

    def test_state_list_reset_can_select_exact_heldout_index(self) -> None:
        spec = self.small_spec()
        payload = generate_capture_states(spec, "test")
        cfg = load_config(ROOT / "configs/swingup6_capture_envelope.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            cfg["env"]["init_states_path"] = str(path)
            env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
            try:
                index = 17
                env.reset(options={"state_index": index})
                self.assertEqual(env.last_init_state_index, index)
                np.testing.assert_allclose(env.data.qpos, payload["states"][index]["qpos"])
                np.testing.assert_allclose(env.data.qvel, payload["states"][index]["qvel"])
                direct_index = 23
                env.reset(
                    options={
                        "qpos": payload["states"][direct_index]["qpos"],
                        "qvel": payload["states"][direct_index]["qvel"],
                    }
                )
                np.testing.assert_allclose(env.data.qpos, payload["states"][direct_index]["qpos"])
                np.testing.assert_allclose(env.data.qvel, payload["states"][direct_index]["qvel"])
                with self.assertRaises(IndexError):
                    env.reset(options={"state_index": len(payload["states"])})
            finally:
                env.close()

    def test_frozen_config_rejects_easier_plant_and_gate(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_envelope.yaml")
        self.assertEqual(validate_capture_config(cfg, self.spec), [])

        easier = copy.deepcopy(cfg)
        easier["env"]["rail_limit"] = 9.0
        easier["env"]["force_limit"] = 120.0
        easier["env"]["success_sustain_seconds"] = 1.0
        errors = validate_capture_config(easier, self.spec)
        self.assertTrue(any("rail_limit" in error for error in errors))
        self.assertTrue(any("force_limit" in error for error in errors))
        self.assertTrue(any("success_sustain_seconds" in error for error in errors))

    def test_curriculum_validation_indices_are_fixed_and_unique(self) -> None:
        spec = self.small_spec()
        payload = generate_capture_states(spec, "test")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "states.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            first = select_evaluation_state_indices(path, episodes=32, seed=61201)
            second = select_evaluation_state_indices(path, episodes=32, seed=61201)
            different = select_evaluation_state_indices(path, episodes=32, seed=61202)
        self.assertEqual(first, second)
        self.assertNotEqual(first, different)
        self.assertEqual(len(set(first)), 32)
        self.assertTrue(all(0 <= index < 64 for index in first))

    def test_hard_mining_mixture_preserves_anchors_and_repeats_failures(self) -> None:
        states = [{"state_id": f"state-{index}"} for index in range(5)]
        mixture = build_mining_mixture(states, [0, 1, 2, 3], [1, 3], hard_repeat=2)
        self.assertEqual(len(mixture), 8)
        self.assertEqual([row["state_id"] for row in mixture[:4]], [f"state-{index}" for index in range(4)])
        self.assertEqual(sum(bool(row["mining"]["hard_failure"]) for row in mixture), 6)
        self.assertEqual(len({row["state_id"] for row in mixture}), len(mixture))

    def test_condensed_linear_mpc_matches_unconstrained_lqr(self) -> None:
        a = np.asarray([[0.9]], dtype=np.float64)
        b = np.asarray([[1.0]], dtype=np.float64)
        q = np.asarray([[1.0]], dtype=np.float64)
        r = np.asarray([[0.1]], dtype=np.float64)
        terminal = solve_discrete_are(a, b, q, r)
        gain = np.linalg.solve(b.T @ terminal @ b + r, b.T @ terminal @ a)
        controller = LinearMPC(
            a,
            b,
            q,
            r,
            terminal,
            horizon=5,
            rail_constraint=100.0,
        )

        action, status, _ = controller.action(np.asarray([0.5], dtype=np.float64))

        self.assertEqual(status, "solved")
        expected = float((-gain @ np.asarray([0.5])).item())
        self.assertAlmostEqual(action, expected, places=4)

    def test_linear_search_development_cohort_is_fixed_by_default(self) -> None:
        self.assertEqual(development_seed(100, 0, 0), 100)
        self.assertEqual(development_seed(100, 99, 0), 100)
        self.assertEqual(development_seed(100, 4, 5), 100)
        self.assertEqual(development_seed(100, 5, 5), 1100)

    def test_capture_recovery_residual_fades_to_zero(self) -> None:
        knots = np.ones(5, dtype=np.float64)
        self.assertAlmostEqual(recovery_residual(0.0, knots, recovery_seconds=2.0, fade_fraction=0.25), 1.0)
        self.assertAlmostEqual(recovery_residual(1.75, knots, recovery_seconds=2.0, fade_fraction=0.25), 0.5)
        self.assertEqual(recovery_residual(2.0, knots, recovery_seconds=2.0, fade_fraction=0.25), 0.0)

    def test_capture_target_schedule_returns_to_zero(self) -> None:
        knots = np.asarray([0.0, 1.0, -1.0], dtype=np.float64)
        self.assertEqual(scheduled_cart_target(0.0, knots, 2.0), 0.0)
        self.assertAlmostEqual(scheduled_cart_target(0.5, knots, 2.0), 0.5)
        self.assertEqual(scheduled_cart_target(2.0, knots, 2.0), 0.0)
        self.assertEqual(scheduled_cart_target(3.0, knots, 2.0), 0.0)

    def test_capture_target_rollout_exposes_required_timing_metrics(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_envelope.yaml")
        cfg["env"]["init_mode"] = "upright"
        cfg["env"]["episode_seconds"] = 0.04
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
        metrics = evaluate_target_schedule(
            cfg,
            progress=1.0,
            seed=17,
            gain=np.zeros(14, dtype=np.float64),
            target_knots=np.zeros(2, dtype=np.float64),
            schedule_seconds=0.02,
            lqr_scale=1.0,
        )
        self.assertIn("time_to_first_upright", metrics)
        self.assertIn("time_to_capture", metrics)
        self.assertIn("capture_start_time", metrics)
        self.assertIn("final_upright_streak_seconds", metrics)
        self.assertGreater(metrics["final_upright_streak_seconds"], 0.0)

    def test_hybrid_rollout_records_lyapunov_and_faded_residual(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_envelope.yaml")
        cfg["env"]["init_mode"] = "upright"
        cfg["env"]["episode_seconds"] = 0.06
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
        metrics = evaluate_hybrid_schedule(
            cfg,
            progress=1.0,
            seed=19,
            gain=np.zeros(14, dtype=np.float64),
            transform=np.eye(14, dtype=np.float64),
            lyapunov=np.eye(14, dtype=np.float64),
            target_knots=np.zeros(2, dtype=np.float64),
            target_seconds=0.02,
            residual_knots=np.ones(2, dtype=np.float64),
            residual_seconds=0.02,
            fade_fraction=0.5,
            lqr_scale=1.0,
            record_trajectory=True,
        )
        self.assertEqual(metrics["trajectory"][-1]["recovery_residual"], 0.0)
        self.assertIn("dimensionless_lyapunov_value", metrics["trajectory"][0])
        self.assertGreaterEqual(metrics["initial_lyapunov"], metrics["minimum_lyapunov"])
        self.assertIn("final_upright_streak_seconds", metrics)

    def test_hybrid_resampling_extends_target_and_initializes_zero_residual(self) -> None:
        controller = resample_target_controller(
            {"target_knots": [0.0, 1.0, -1.0], "lqr_scale": 1.4},
            old_seconds=2.0,
            target_count=5,
            target_seconds=3.0,
            residual_count=4,
        )
        np.testing.assert_allclose(controller["target_knots"], [0.0, 0.75, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(controller["residual_knots"], np.zeros(4))
        self.assertEqual(controller["lqr_scale"], 1.4)

    def test_feedback_mpc_action_matches_live_lqr(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_envelope.yaml")
        cfg["env"]["init_mode"] = "upright"
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=23)
        gain = np.linspace(-0.2, 0.3, 14, dtype=np.float64)
        try:
            env.reset()
            env.data.qpos[1] = 2.0 * np.pi + 0.03
            env.data.qvel[:] = np.linspace(-0.1, 0.1, 7)
            expected = lqr_action(env, gain, scale=1.2, cart_target=0.4)
            actual = feedback_action(
                env.data.qpos,
                env.data.qvel,
                gain,
                n_links=6,
                scale=1.2,
                cart_target=0.4,
            )
        finally:
            env.close()
        self.assertAlmostEqual(actual, expected)

    def test_feedback_mpc_warm_start_shifts_and_ends_at_zero(self) -> None:
        shifted = shift_schedule(
            np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
            elapsed_seconds=0.5,
            horizon_seconds=2.0,
        )
        np.testing.assert_allclose(shifted, [0.5, 0.5, 0.0])

    def test_ilqr_state_difference_wraps_relative_angles(self) -> None:
        first = np.asarray([0.2, 2.0 * np.pi - 0.1, -2.0 * np.pi + 0.2, 0.4, -0.3])
        second = np.zeros(5, dtype=np.float64)
        np.testing.assert_allclose(
            state_difference(first, second, n_links=2),
            [0.2, -0.1, 0.2, 0.4, -0.3],
            atol=1e-12,
        )

    def test_ilqr_warm_start_preserves_time_and_pads_with_zero(self) -> None:
        payload = {
            "controller": {
                "controls": [0.0, 0.5, 1.0],
                "horizon_seconds": 0.06,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "controller.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            controls = load_initial_controls(str(path), horizon_steps=5, policy_dt=0.02)
        np.testing.assert_allclose(controls, [0.0, 0.5, 1.0, 0.0, 0.0])

    def test_ilqr_dimensionless_mujoco_transition_linearizes(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_envelope.yaml")
        cfg["env"]["init_mode"] = "upright"
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=31)
        try:
            env.reset()
            transform = np.diag(np.linspace(0.5, 1.5, 14, dtype=np.float64))
            transition = MujocoTransition(env, coordinate_transform=transform)
            initial = transition.to_coordinates(data_state(env.data))
            np.testing.assert_allclose(transition.to_physical(initial), data_state(env.data))
            following = transition(initial, 0.1)
            state_matrix, input_matrix = transition.linearize(
                initial,
                0.1,
                state_epsilon=1e-5,
                action_epsilon=1e-4,
            )
        finally:
            env.close()
        self.assertEqual(following.shape, (14,))
        self.assertEqual(state_matrix.shape, (14, 14))
        self.assertEqual(input_matrix.shape, (14, 1))
        self.assertTrue(np.all(np.isfinite(state_matrix)))
        self.assertTrue(np.all(np.isfinite(input_matrix)))

    def test_policy_control_penalty_prefers_zero_residual(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_sac_boundary.yaml")
        cfg["env"]["init_mode"] = "upright"
        env = NLinkCartPoleEnv(cfg, progress=0.0, seed=11)
        try:
            env.reset(options={"state_index": 0})
            env.last_policy_action_norm[0] = 0.0
            zero_reward = env._reward(0.0)
            env.last_policy_action_norm[0] = 1.0
            residual_reward = env._reward(0.0)
        finally:
            env.close()
        self.assertAlmostEqual(zero_reward - residual_reward, 2.0, places=6)

    def test_capture_features_preserve_curriculum_state_scale(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_sac_boundary.yaml")
        cfg["env"]["init_mode"] = "fixed_state"
        cfg["env"]["init_qpos"] = [1.25, 0.15, -0.15, 0.15, -0.15, 0.15, -0.15]
        cfg["env"]["init_qvel"] = [0.5, 0.75, -0.75, 0.75, -0.75, 0.75, -0.75]
        env = NLinkCartPoleEnv(cfg, progress=0.0625, seed=13)
        try:
            obs, _ = env.reset()
            capture = obs[-15:]
        finally:
            env.close()
        self.assertEqual(obs.shape[0], 59)
        self.assertAlmostEqual(float(capture[0]), 1.0, places=5)
        self.assertAlmostEqual(float(capture[1]), 1.0, places=5)
        np.testing.assert_allclose(capture[2:8], [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], atol=1e-5)
        np.testing.assert_allclose(capture[8:14], [1.0, -1.0, 1.0, -1.0, 1.0, -1.0], atol=1e-5)

    def test_effective_funnel_state_matches_component_curriculum(self) -> None:
        qpos = np.asarray([1.0, 0.1, -0.2], dtype=np.float64)
        qvel = np.asarray([0.5, 0.3, -0.4], dtype=np.float64)
        effective_qpos, effective_qvel = effective_state(
            qpos,
            qvel,
            progress=0.5,
            qpos_scale_power=2.0,
            qvel_scale_power=3.0,
        )
        np.testing.assert_allclose(effective_qpos, 0.25 * qpos)
        np.testing.assert_allclose(effective_qvel, 0.125 * qvel)

    def test_polynomial_funnel_fit_round_trips_and_separates(self) -> None:
        rng = np.random.default_rng(19)
        coordinates = rng.normal(size=(200, 4))
        labels = (coordinates[:, 0] + 0.7 * coordinates[:, 1] > 0.0).astype(np.int64)
        identifiers = [f"training-{index}" for index in range(labels.size)]
        development, holdout = deterministic_stratified_split(labels, identifiers, 0.2)
        model, diagnostics = fit_capture_funnel(
            coordinates,
            labels,
            development,
            l2=0.001,
            max_iterations=300,
        )
        model.coordinate_bounds = {
            "cart_position_bound": 1.0,
            "angle_bound": 1.0,
            "cart_velocity_bound": 1.0,
            "hinge_velocity_bound": 1.0,
        }
        model.coordinate_abs_limits = np.full(4, 10.0, dtype=np.float64)
        features = polynomial_features(coordinates)
        logits = ((features - model.feature_mean) / model.feature_scale) @ model.weights + model.bias
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -60, 60)))
        self.assertTrue(diagnostics["optimizer_success"])
        self.assertGreater(float(np.mean((probabilities[holdout] >= 0.5) == labels[holdout])), 0.9)
        restored = CaptureFunnelModel.from_dict(model.to_dict())
        qpos = np.asarray([coordinates[0, 0], coordinates[0, 1]], dtype=np.float64)
        qvel = np.asarray([coordinates[0, 2], coordinates[0, 3]], dtype=np.float64)
        expected_coordinates = normalized_capture_coordinates(qpos, qvel, **model.coordinate_bounds)
        self.assertEqual(expected_coordinates.shape, (4,))
        self.assertAlmostEqual(restored.predict_probability(qpos, qvel), model.predict_probability(qpos, qvel))
        self.assertEqual(restored.domain_distance(qpos, qvel), 0.0)
        self.assertEqual(restored.predict_probability(100.0 * qpos, qvel), 0.0)
        self.assertGreater(restored.domain_distance(100.0 * qpos, qvel), 0.0)


if __name__ == "__main__":
    unittest.main()
