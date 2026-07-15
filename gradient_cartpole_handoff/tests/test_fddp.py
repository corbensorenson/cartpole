from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

from gcartpole.ilqr import QuadraticTrajectoryCost
from scripts.continue_fddp_homotopy import artifact_passes, candidate_alpha


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
    def test_candidate_alpha_stops_at_target(self) -> None:
        self.assertEqual(candidate_alpha(0.5, 1.0, 0.1), 0.6)
        self.assertEqual(candidate_alpha(0.95, 1.0, 0.1), 1.0)

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


if __name__ == "__main__":
    unittest.main()
