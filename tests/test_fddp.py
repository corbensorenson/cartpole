from __future__ import annotations

import importlib.util
import unittest
from types import SimpleNamespace

import numpy as np

from gcartpole.capture_constraints import (
    exact_constraint_margins,
    normalized_constraint_score,
    terminal_bounds,
)
from gcartpole.ilqr import QuadraticTrajectoryCost
from gcartpole.modal import StateScales
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
