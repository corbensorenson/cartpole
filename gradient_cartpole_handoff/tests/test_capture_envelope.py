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
from gcartpole.ilqr import MujocoTransition, data_state, scalar_box_policy, state_difference
from gcartpole.multiple_shooting import pack_decision, shooting_sparsity, unpack_decision
from gcartpole.ppo_mlx import select_evaluation_state_indices
from gcartpole.trajectory_policy import trajectory_conditioned_features
from scripts.build_capture_supervisor_dataset import aligned_trajectory_steps
from scripts.build_feedback_mpc_teachers import failed_state_indices
from scripts.distill_capture_supervisor import grouped_source_split, source_balancing_weights
from scripts.distill_trajectory_conditioned_capture import parse_hidden_sizes
from scripts.evaluate_feedback_mpc_capture import feedback_action, shift_schedule
from scripts.evaluate_linear_mpc_capture import LinearMPC
from scripts.evaluate_trajectory_conditioned_capture import selected_indices
from scripts.label_capture_dagger_target import selected_queries
from scripts.mine_capture_failures import build_mining_mixture
from scripts.search_capture_hybrid_schedule import (
    evaluate_hybrid_schedule,
    resample_target_controller,
)
from scripts.search_capture_recovery import recovery_residual
from scripts.search_capture_target_schedule import evaluate_target_schedule, scheduled_cart_target
from scripts.search_ilqr_capture import load_initial_controls
from scripts.search_linear_policy import development_seed
from scripts.search_swingup_capture import lqr_action


ROOT = Path(__file__).resolve().parents[1]


class CaptureEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_config(ROOT / "benchmarks/p1_capture_envelope.yaml")

    def small_spec(self) -> dict:
        spec = copy.deepcopy(self.spec)
        spec["splits"]["test"]["count"] = 64
        return spec

    def test_trajectory_conditioned_features_include_initial_state_and_phase(self) -> None:
        observations = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        initial = np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
        features = trajectory_conditioned_features(
            observations,
            initial,
            np.asarray([0, 25]),
            maximum_steps=100,
        )
        np.testing.assert_allclose(
            features,
            [[1.0, 2.0, 5.0, 6.0, 0.0], [3.0, 4.0, 7.0, 8.0, 0.25]],
        )
        np.testing.assert_allclose(
            trajectory_conditioned_features(
                observations,
                None,
                np.asarray([0, 25]),
                maximum_steps=100,
                include_initial_observation=False,
            ),
            [[1.0, 2.0, 0.0], [3.0, 4.0, 0.25]],
        )

    def test_trajectory_policy_metadata_selects_source_partitions(self) -> None:
        metadata = {"train_sources": [4, 1], "validation_sources": [3]}
        self.assertEqual(selected_indices(6, metadata, "train", None), [4, 1])
        self.assertEqual(selected_indices(6, metadata, "all", 2), [1, 3])
        self.assertEqual(parse_hidden_sizes("128, 64"), [128, 64])

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

    def test_feedback_teacher_failure_indices_are_sorted(self) -> None:
        payload = {
            "evaluation": {
                "episode_results": [
                    {"state_index": 9, "success": False},
                    {"state_index": 2, "success": True},
                    {"state_index": 4, "success": False},
                ]
            }
        }
        self.assertEqual(failed_state_indices(payload), [4, 9])

    def test_supervisor_alignment_pairs_actions_with_pre_action_states(self) -> None:
        legacy = [
            {"qpos": [1.0, 2.0], "qvel": [3.0, 4.0], "action": 0.25},
            {"qpos": [5.0, 6.0], "qvel": [7.0, 8.0], "action": -0.5},
        ]
        aligned = aligned_trajectory_steps(legacy, [0.0, 0.1], [0.2, 0.3])
        np.testing.assert_allclose(aligned[0]["qpos"], [0.0, 0.1])
        np.testing.assert_allclose(aligned[1]["qpos"], [1.0, 2.0])
        self.assertEqual([row["action"] for row in aligned], [0.25, -0.5])

        explicit = [
            {
                "pre_action_qpos": [9.0, 10.0],
                "pre_action_qvel": [11.0, 12.0],
                "qpos": [1.0, 2.0],
                "qvel": [3.0, 4.0],
                "action": 0.75,
            }
        ]
        aligned = aligned_trajectory_steps(explicit, [0.0, 0.1], [0.2, 0.3])
        np.testing.assert_allclose(aligned[0]["qpos"], [9.0, 10.0])
        np.testing.assert_allclose(aligned[0]["qvel"], [11.0, 12.0])

    def test_supervisor_split_is_source_grouped_and_weights_sources_equally(self) -> None:
        sources = np.asarray([1, 1, 1, 2, 3, 3, 4, 4, 4, 4])
        tiers = np.asarray(["lqr"] * 4 + ["target"] * 6)
        train, validation = grouped_source_split(
            sources,
            tiers,
            validation_fraction=0.5,
            seed=47,
        )
        self.assertTrue(set(sources[train]).isdisjoint(set(sources[validation])))
        weights = source_balancing_weights(sources)
        totals = [float(np.sum(weights[sources == source])) for source in np.unique(sources)]
        np.testing.assert_allclose(totals, np.full(len(totals), totals[0]))
        early = source_balancing_weights(
            sources,
            np.asarray([0, 5, 10, 0, 0, 5, 0, 5, 10, 15]),
            early_weight=20.0,
            early_decay_steps=5.0,
        )
        self.assertGreater(early[0], early[2])
        early_totals = [float(np.sum(early[sources == source])) for source in np.unique(sources)]
        np.testing.assert_allclose(early_totals, np.full(len(early_totals), early_totals[0]))

    def test_dagger_target_query_selection_is_step_bounded_and_deterministic(self) -> None:
        payload = {
            "queries": [
                {"source_index": 3, "step": 1, "dimensionless_lyapunov_value": 40.0},
                {"source_index": 2, "step": 0, "dimensionless_lyapunov_value": 10.0},
                {"source_index": 1, "step": 1, "dimensionless_lyapunov_value": 20.0},
                {"source_index": 2, "step": 1, "dimensionless_lyapunov_value": 30.0},
            ]
        }
        rows = selected_queries(payload, source_step=1, max_lyapunov=35.0, limit=2)
        self.assertEqual([row["source_index"] for row in rows], [1, 2])

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

    def test_lqr_policy_switch_uses_hysteresis_without_changing_state(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_envelope.yaml")
        cfg["env"]["init_mode"] = "upright"
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
        cfg["env"]["action_lqr_residual"]["enabled"] = False
        cfg["env"]["action_lqr_switch"] = {
            "enabled": True,
            "state_gain": np.zeros(14).tolist(),
            "scale": 1.0,
            "cart_target": 0.0,
            "enter_max_abs_angle": 0.05,
            "enter_hinge_velocity_rms": 0.10,
            "enter_cart_abs": 0.25,
            "enter_cart_velocity_abs": 0.10,
            "exit_max_abs_angle": 0.10,
            "exit_hinge_velocity_rms": 0.20,
            "exit_cart_abs": 0.50,
            "exit_cart_velocity_abs": 0.20,
        }
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=31)
        try:
            env.reset()
            before_qpos = env.data.qpos.copy()
            before_qvel = env.data.qvel.copy()
            self.assertEqual(env._applied_action_norm(0.7), 0.0)
            self.assertTrue(env.lqr_switch_active)
            self.assertEqual(env.last_controller_mode, "lqr")
            self.assertEqual(env.lqr_switch_entry_count, 1)
            self.assertEqual(env.lqr_switch_lqr_steps, 1)
            np.testing.assert_allclose(env.data.qpos, before_qpos)
            np.testing.assert_allclose(env.data.qvel, before_qvel)

            env.data.qpos[1] = 0.075
            self.assertEqual(env._applied_action_norm(0.7), 0.0)
            self.assertTrue(env.lqr_switch_active)

            env.data.qpos[1] = 0.20
            self.assertEqual(env._applied_action_norm(0.7), 0.7)
            self.assertFalse(env.lqr_switch_active)
            self.assertEqual(env.last_controller_mode, "policy")
            self.assertEqual(env.lqr_switch_exit_count, 1)
            self.assertEqual(env.lqr_switch_policy_steps, 1)

            env.data.qpos[1] = 0.075
            self.assertEqual(env._applied_action_norm(-0.4), -0.4)
            self.assertFalse(env.lqr_switch_active)
            self.assertEqual(env.lqr_switch_policy_steps, 2)
        finally:
            env.close()

    def test_lqr_switch_rejects_residual_blending(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_envelope.yaml")
        cfg["env"]["action_lqr_switch"] = {"enabled": True}
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=37)
        try:
            env.reset()
            with self.assertRaisesRegex(ValueError, "mutually exclusive"):
                env.step([0.0])
        finally:
            env.close()

    def test_lqr_policy_switch_can_require_dimensionless_lyapunov_funnel(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_envelope.yaml")
        cfg["env"]["init_mode"] = "upright"
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
        cfg["env"]["action_lqr_residual"]["enabled"] = False
        cfg["env"]["action_lqr_switch"] = {
            "enabled": True,
            "state_gain": np.zeros(14).tolist(),
            "scale": 1.0,
            "cart_target": 0.0,
            "enter_max_abs_angle": 0.20,
            "enter_hinge_velocity_rms": 0.20,
            "enter_cart_abs": 0.50,
            "enter_cart_velocity_abs": 0.20,
            "exit_max_abs_angle": 0.30,
            "exit_hinge_velocity_rms": 0.30,
            "exit_cart_abs": 0.75,
            "exit_cart_velocity_abs": 0.30,
            "enter_lyapunov_max": 0.001,
            "exit_lyapunov_max": 0.01,
            "state_transform": np.eye(14).tolist(),
            "lyapunov_matrix": np.eye(14).tolist(),
        }
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=41)
        try:
            env.reset()
            env.data.qpos[1] = 0.075
            self.assertEqual(env._applied_action_norm(0.6), 0.6)
            self.assertFalse(env.lqr_switch_active)

            env.data.qpos[1] = 0.02
            self.assertEqual(env._applied_action_norm(0.6), 0.0)
            self.assertTrue(env.lqr_switch_active)
            self.assertAlmostEqual(
                env._lqr_switch_lyapunov_value(cfg["env"]["action_lqr_switch"]),
                0.0004,
            )

            env.data.qpos[1] = 0.075
            self.assertEqual(env._applied_action_norm(0.6), 0.0)
            self.assertTrue(env.lqr_switch_active)

            env.data.qpos[1] = 0.11
            self.assertEqual(env._applied_action_norm(0.6), 0.6)
            self.assertFalse(env.lqr_switch_active)
        finally:
            env.close()

    def test_lqr_switch_supports_checkpoint_compatible_tanh_squashing(self) -> None:
        cfg = load_config(ROOT / "configs/swingup6_capture_envelope.yaml")
        cfg["env"]["init_mode"] = "upright"
        cfg["env"]["init_angle_noise"] = 0.0
        cfg["env"]["init_vel_noise"] = 0.0
        env = NLinkCartPoleEnv(cfg, progress=1.0, seed=43)
        controller = {
            "state_gain": [2.0] + [0.0] * 13,
            "scale": 1.0,
            "cart_target": 0.0,
            "action_squash": "tanh",
        }
        try:
            env.reset()
            env.data.qpos[0] = 0.5
            action, _ = env._lqr_action_bias(controller)
            self.assertAlmostEqual(action, -float(np.tanh(1.0)))
            controller["action_squash"] = "clip"
            action, _ = env._lqr_action_bias(controller)
            self.assertEqual(action, -1.0)
            controller["action_squash"] = "invalid"
            with self.assertRaisesRegex(ValueError, "action_squash"):
                env._lqr_action_bias(controller)
        finally:
            env.close()

    def test_ilqr_state_difference_wraps_relative_angles(self) -> None:
        first = np.asarray([0.2, 2.0 * np.pi - 0.1, -2.0 * np.pi + 0.2, 0.4, -0.3])
        second = np.zeros(5, dtype=np.float64)
        np.testing.assert_allclose(
            state_difference(first, second, n_links=2),
            [0.2, -0.1, 0.2, 0.4, -0.3],
            atol=1e-12,
        )

    def test_ilqr_scalar_box_policy_clamps_and_disables_feedback(self) -> None:
        feedforward, feedback, active = scalar_box_policy(
            qu=-4.0,
            quu=2.0,
            qux=np.asarray([1.0, -2.0]),
            control=0.75,
        )
        self.assertEqual(feedforward, 0.25)
        np.testing.assert_allclose(feedback, np.zeros(2))
        self.assertTrue(active)

        feedforward, feedback, active = scalar_box_policy(
            qu=0.4,
            quu=2.0,
            qux=np.asarray([1.0, -2.0]),
            control=0.0,
        )
        self.assertAlmostEqual(feedforward, -0.2)
        np.testing.assert_allclose(feedback, [-0.5, 1.0])
        self.assertFalse(active)

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

    def test_ilqr_warm_start_can_use_recorded_rollout_actions(self) -> None:
        payload = {
            "result": {
                "trajectory": [
                    {"action": -0.5},
                    {"action": 0.25},
                    {"action": 0.75},
                ]
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rollout.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            controls = load_initial_controls(str(path), horizon_steps=2, policy_dt=0.02)
        np.testing.assert_allclose(controls, [-0.5, 0.25])

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
            env.step([0.1])
            np.testing.assert_allclose(
                following,
                transition.to_coordinates(data_state(env.data)),
                rtol=0.0,
                atol=1e-12,
            )
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

    def test_multiple_shooting_decision_and_sparsity(self) -> None:
        nodes = np.arange(12, dtype=np.float64).reshape(3, 4)
        controls = np.linspace(-1.0, 1.0, 6, dtype=np.float64)
        decision = pack_decision(nodes, controls)
        unpacked_nodes, unpacked_controls = unpack_decision(
            decision,
            nx=4,
            horizon_steps=6,
            segment_steps=2,
        )
        np.testing.assert_allclose(unpacked_nodes, nodes)
        np.testing.assert_allclose(unpacked_controls, controls)

        pattern = shooting_sparsity(nx=4, horizon_steps=6, segment_steps=2)
        self.assertEqual(pattern.shape, (28, 18))
        self.assertTrue(np.all(pattern.getnnz(axis=1) > 0))

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
