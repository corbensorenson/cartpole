"""Physical invariants at the inherited-controller and planner boundaries."""

import numpy as np
import mujoco
import json
import sys
from pathlib import Path

import tempfile
import unittest
from unittest.mock import patch

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.modal import StateScales, dimensionless_absolute_transform, dimensionless_wrapped_state
from scripts.pad_fddp_controller import pad_coordinate_state, pad_physical_state, pad_feedback_gains
from scripts.force_proposal_to_fddp import control_sample_index
from scripts.search_capture_online_mpc import fixed_state_cfg, interpolation_matrix, rollout_profiles
from scripts import force_proposal_to_fddp, pad_fddp_controller
from scripts.evaluate_fddp_two_expert import load_controller, validate_release_identity


class TransferContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)

    def test_padding_preserves_physical_state_and_feedback_action(self):
        scales = StateScales(1.25, 0.15, 0.5, 0.75)
        source = {"qpos": [0.2, 0.1, -0.03, 0.04], "qvel": [0.3, -0.2, 0.5, -0.6]}
        a = dimensionless_absolute_transform(3, scales)
        b = dimensionless_absolute_transform(5, scales)
        old = dimensionless_wrapped_state(np.array(source["qpos"]), np.array(source["qvel"]), a)
        padded = pad_physical_state(source, 3, 5)
        new = dimensionless_wrapped_state(np.array(padded["qpos"]), np.array(padded["qvel"]), b)
        np.testing.assert_allclose(pad_coordinate_state(old, 3, 5), new)
        gain = np.arange(8, dtype=float)[None, :]
        lifted_gain = pad_feedback_gains(gain, 3, 5)
        np.testing.assert_allclose(lifted_gain @ new, gain @ old)
        # Newly added coordinates must not steal any predecessor feedback channel.
        new[[4, 5, 10, 11]] += 100.0
        np.testing.assert_allclose(lifted_gain @ new, gain @ old)


    def test_saved_route_sampling_preserves_every_action_and_starts_tail_on_time(self):
        indices = [control_sample_index(i, 0.02, 4.56, 228) for i in range(230)]
        assert indices == list(range(228)) + [None, None]
        assert [control_sample_index(i, 0.01, 0.04, 2) for i in range(5)] == [0, 0, 1, 1, None]


    def test_padding_cannot_inherit_a_source_success(self):
        tmp_path = Path(self.directory.name)
        source = tmp_path / "source.json"
        out = tmp_path / "padded.json"
        source.write_text(json.dumps({
            "controller": {"controls": [0.0], "feedback_gains": [[0.0] * 4], "horizon_seconds": 0.01},
            "search": {"is_feasible": True, "converged": True, "nominal_coordinate_states": [[0.0] * 4] * 2},
            "result": {"success": True},
        }))
        with patch.object(sys, "argv", ["pad", "--source", str(source), "--source-links", "1", "--target-links", "2", "--out", str(out)]):
            pad_fddp_controller.main()
        result = json.loads(out.read_text())
        assert "result" not in result
        assert result["search"]["is_feasible"] is False
        assert result["controller"]["horizon_seconds"] == 0.01


    def test_adapter_preserves_controls_and_appends_zero_tail(self):
        tmp_path = Path(self.directory.name)
        source = tmp_path / "source.json"
        replay = tmp_path / "replay.json"
        tail = tmp_path / "tail.json"
        actions = [0.001, -0.002, 0.003, -0.001]
        source.write_text(json.dumps({"controller": {"controls": actions, "horizon_seconds": 0.08}}))
        common = ["adapter", "--config", "configs/swingup7_uniform.yaml", "--record-key", "controller"]
        with patch.object(sys, "argv", common + ["--proposal", str(source), "--out", str(replay)]):
            force_proposal_to_fddp.main()
        with patch.object(sys, "argv", common + ["--proposal", str(replay), "--source-trajectory-feedback", "--tail-seconds", "0.04", "--out", str(tail)]):
            force_proposal_to_fddp.main()
        first = json.loads(replay.read_text())
        second = json.loads(tail.read_text())
        np.testing.assert_array_equal(first["controller"]["controls"], np.array(actions, dtype=np.float32).astype(float))
        np.testing.assert_array_equal(second["controller"]["controls"], first["controller"]["controls"] + [0.0, 0.0])
        np.testing.assert_allclose(second["search"]["nominal_coordinate_states"][:5], first["search"]["nominal_coordinate_states"], atol=1e-12)
        assert np.shape(second["controller"]["feedback_gains"]) == (6, 16)


    def test_release_label_requires_matching_controller_and_execution_settings(self):
        path = Path("runs/swingup7_uniform/seven_link_swingup_manifest.json")
        manifest = json.loads(path.read_text())
        cfg = load_config("configs/swingup7_uniform.yaml")
        controller = load_controller(Path(manifest["controller"]["path"]), 7, load_config("benchmarks/p1_capture_envelope.yaml"))
        settings = dict(tracking_gain_scale=2.0, prelude_seconds=10.0, settle_mode="hanging_lqr", settle_scale=1.0, settle_control_cost=1000.0, shift_cart_nominal=True, phase_adaptive=False)
        validate_release_identity(manifest, cfg, controller, settings)
        with self.assertRaisesRegex(ValueError, "prelude_seconds"):
            validate_release_identity(manifest, cfg, controller, {**settings, "prelude_seconds": 0.0})
        with self.assertRaisesRegex(ValueError, "config_sha256"):
            validate_release_identity(manifest, load_config("configs/swingup8_uniform.yaml"), controller, settings)


    def test_online_mpc_rollout_matches_exact_live_policy_steps(self):
        cfg = load_config("configs/swingup8_split_unlock_continuation.yaml")
        state = {"qpos": [0.0] + [0.001] * 8, "qvel": [0.0] * 9}
        # A prior curriculum must not silently scale or perturb a saved handoff.
        cfg["env"].update(init_qpos_scale_start=0.1, init_qpos_scale_end=0.2)
        cfg = fixed_state_cfg(cfg, state, 1.0)
        env = NLinkCartPoleEnv(cfg, progress=0.9, seed=0)
        try:
            env.reset()
            np.testing.assert_allclose(env.data.qpos, state["qpos"], atol=1e-14)
            size = mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            initial = np.empty(size)
            mujoco.mj_getState(env.model, env.data, initial, mujoco.mjtState.mjSTATE_FULLPHYSICS)
            actions = np.array([0.020000003, -0.010000007, 0.030000011, 0.0])
            states = rollout_profiles(env, [mujoco.MjData(env.model)], initial, actions[None, :], interpolation_matrix(4, 4))
            actual = []
            for action in actions:
                env.step([action])
                actual.append(np.r_[env.data.qpos.copy(), env.data.qvel.copy()])
            assert states.shape[1] == len(actions)
            np.testing.assert_allclose(states[0, :, 1:], actual, atol=1e-10)
            np.testing.assert_allclose(states[0, :, 0], np.arange(1, 5) * env.dt, atol=1e-14)
        finally:
            env.close()
