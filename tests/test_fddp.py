from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from gcartpole.capture_constraints import (
    exact_constraint_margins,
    normalized_constraint_score,
    terminal_bounds,
)
from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import QuadraticTrajectoryCost
from gcartpole.modal import StateScales
from scripts.continue_fddp_homotopy import artifact_passes, candidate_alpha
from scripts.search_capture_sequence import fixed_state_cfg, load_state


class _LinearTransition:
    nx = 4
    env = SimpleNamespace(n=1)

    def __call__(self, state: np.ndarray, action: float) -> np.ndarray:
        return np.asarray(state, dtype=np.float64) + action

    def linearize(
        self,
        state: np.ndarray,
        action: float,
        *,
        state_epsilon: float,
        action_epsilon: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        del state, action, state_epsilon, action_epsilon
        return np.eye(self.nx), np.ones((self.nx, 1))


@unittest.skipUnless(importlib.util.find_spec("crocoddyl"), "Crocoddyl is optional")
class FDDPTests(unittest.TestCase):
    def test_action_model_matches_transition_and_derivative_shapes(self) -> None:
        from gcartpole.fddp import MujocoActionModel, rollout_controls

        cost = QuadraticTrajectoryCost(
            stage_state=np.eye(4),
            terminal_state=2.0 * np.eye(4),
            control=0.5,
            rail_soft_limit=10.0,
            rail_limit=20.0,
            rail_weight=1.0,
            wrap_angles=False,
        )
        model = MujocoActionModel(_LinearTransition(), cost)
        data = model.createData()
        state = np.asarray([0.1, -0.2, 0.3, -0.4])
        control = np.asarray([0.25])

        model.calc(data, state, control)
        model.calcDiff(data, state, control)

        np.testing.assert_allclose(data.xnext, state + control[0])
        np.testing.assert_allclose(data.Fx, np.eye(4))
        np.testing.assert_allclose(data.Fu, np.ones(4))
        self.assertEqual(data.Lx.shape, (4,))
        self.assertEqual(data.Lu.shape, (1,))
        self.assertEqual(data.Lxx.shape, (4, 4))
        self.assertEqual(data.Lxu.shape, (4,))
        self.assertEqual(data.Luu.shape, (1, 1))

        states = rollout_controls(model.transition, state, np.asarray([0.25, -0.1]))
        np.testing.assert_allclose(states, [state, state + 0.25, state + 0.15])


class FDDPContinuationTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("crocoddyl"), "Crocoddyl is optional")
    def test_feedback_warm_start_tracks_the_inherited_route(self) -> None:
        from scripts.search_fddp_capture import rebuild_feedback_warm_start

        transition = _LinearTransition()
        controls = np.asarray([0.1, 0.2])
        nominal_states = np.zeros((3, 4), dtype=np.float64)
        feedback_gains = np.ones((2, 4), dtype=np.float64)
        states = rebuild_feedback_warm_start(
            transition,
            np.zeros(4, dtype=np.float64),
            controls,
            nominal_states,
            feedback_gains,
            feedback_scale=0.5,
        )
        np.testing.assert_allclose(
            states,
            np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.1, 0.1, 0.1, 0.1],
                    [0.5, 0.5, 0.5, 0.5],
                ]
            ),
        )

    @unittest.skipUnless(importlib.util.find_spec("mujoco"), "MuJoCo is optional")
    def test_fixed_state_cfg_clears_curriculum_noise_and_scales(self) -> None:
        root = Path(__file__).resolve().parents[1]
        cfg = load_config(root / "configs/swingup8_capture_protected.yaml")
        state = {
            "qpos": [0.0, 0.0, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0],
            "qvel": [0.0] * 9,
        }
        fixed = fixed_state_cfg(cfg, state, 5.0)
        env = NLinkCartPoleEnv(fixed, progress=1.0, seed=991)
        try:
            env.reset(seed=991)
            np.testing.assert_allclose(env.data.qpos, state["qpos"], atol=1e-12)
            np.testing.assert_allclose(env.data.qvel, state["qvel"], atol=1e-12)
        finally:
            env.close()

    def test_candidate_alpha_stops_at_target(self) -> None:
        self.assertEqual(candidate_alpha(0.5, 1.0, 0.1), 0.6)
        self.assertEqual(candidate_alpha(0.95, 1.0, 0.1), 1.0)

    def test_load_state_accepts_replay_selected_state(self) -> None:
        state = {"qpos": [0.0, 3.141592653589793], "qvel": [0.0, 0.0]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.json"
            path.write_text(json.dumps({"selected_state": state}), encoding="utf-8")
            loaded, index = load_state(str(path), "selected")
        self.assertEqual(index, -1)
        self.assertEqual(loaded, state)

    def test_load_state_accepts_saved_terminal_state(self) -> None:
        state = {"qpos": [0.0, 3.141592653589793], "qvel": [0.0, 0.0]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "controller.json"
            path.write_text(json.dumps({"terminal_state": state}), encoding="utf-8")
            loaded, index = load_state(str(path), "terminal")
        self.assertEqual(index, -1)
        self.assertEqual(loaded, state)

    def test_load_state_accepts_global_proposal_best_state(self) -> None:
        state = {"qpos": [0.0, 0.25], "qvel": [0.0, -0.4]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "proposal.json"
            path.write_text(json.dumps({"best": {"best_state": state}}), encoding="utf-8")
            loaded, index = load_state(str(path), "best")
        self.assertEqual(index, -1)
        self.assertEqual(loaded, state)

    def test_load_state_accepts_endpoint_refiner_state(self) -> None:
        state = {"qpos": [0.0, 0.1], "qvel": [0.0, -0.2]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "endpoint.json"
            path.write_text(json.dumps({"best": {"endpoint": state}}), encoding="utf-8")
            loaded, index = load_state(str(path), "endpoint")
        self.assertEqual(index, -1)
        self.assertEqual(loaded, state)

    def test_artifact_requires_feasibility_and_strict_replay_success(self) -> None:
        payload = {
            "search": {"is_feasible": True},
            "result": {"success": True, "max_upright_streak_seconds": 10.0},
        }
        self.assertTrue(artifact_passes(payload))
        payload["result"]["max_upright_streak_seconds"] = 9.99
        self.assertFalse(artifact_passes(payload))
        payload["result"]["max_upright_streak_seconds"] = 10.0
        payload["search"]["is_feasible"] = False
        self.assertFalse(artifact_passes(payload))

    def test_proxddp_terminal_bounds_and_exact_margins(self) -> None:
        scales = StateScales(1.25, 0.15, 0.5, 0.75)
        lower, upper = terminal_bounds(
            2,
            scales,
            cart_abs=1.25,
            angle_abs=0.15,
            cart_velocity_abs=0.5,
            hinge_velocity_abs=0.75,
        )
        np.testing.assert_allclose(lower, -np.ones(6))
        np.testing.assert_allclose(upper, np.ones(6))
        states = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.5, 0.2, -0.1, 0.3, 0.1, -0.2],
            ]
        )
        margins = exact_constraint_margins(
            states,
            np.eye(6),
            lower,
            upper,
            rail_limit=2.4,
            handoff_lyapunov=1800.0,
        )
        self.assertAlmostEqual(margins["rail"], 1.9)
        self.assertAlmostEqual(margins["terminal_box"], 0.5)
        self.assertAlmostEqual(margins["minimum_hard"], 0.5)
        self.assertEqual(
            normalized_constraint_score(
                margins, handoff_lyapunov=1800.0, rail_limit=2.4
            ),
            0.0,
        )

        margins["lyapunov"] = -180.0
        self.assertAlmostEqual(
            normalized_constraint_score(
                margins, handoff_lyapunov=1800.0, rail_limit=2.4
            ),
            0.1,
        )


if __name__ == "__main__":
    unittest.main()
