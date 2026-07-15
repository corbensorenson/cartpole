from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gcartpole.ilqr import stitch_feedback_trajectories
from scripts.evaluate_ilqr_chain_basin import aggregate, parse_radii
from scripts.refine_ilqr_capture_chain import (
    capture_selection_key,
    controls_from_knots,
    warm_start_controls,
)


class ILQRTests(unittest.TestCase):
    def test_stitch_feedback_trajectories_preserves_one_boundary_state(self) -> None:
        first_controls = np.asarray([0.1, 0.2])
        first_states = np.asarray([[0.0, 0.0], [1.0, 2.0], [3.0, 4.0]])
        first_gains = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        second_controls = np.asarray([-0.3])
        second_states = np.asarray([[3.0, 4.0], [5.0, 6.0]])
        second_gains = np.asarray([[5.0, 6.0]])

        controls, states, gains = stitch_feedback_trajectories(
            first_controls,
            first_states,
            first_gains,
            second_controls,
            second_states,
            second_gains,
        )

        np.testing.assert_allclose(controls, [0.1, 0.2, -0.3])
        np.testing.assert_allclose(
            states, [[0.0, 0.0], [1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        )
        np.testing.assert_allclose(gains, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_stitch_feedback_trajectories_rejects_boundary_reset(self) -> None:
        with self.assertRaisesRegex(ValueError, "trajectory boundary differs"):
            stitch_feedback_trajectories(
                np.asarray([0.0]),
                np.asarray([[0.0], [1.0]]),
                np.asarray([[0.0]]),
                np.asarray([0.0]),
                np.asarray([[1.1], [2.0]]),
                np.asarray([[0.0]]),
            )

    def test_warm_start_controls_slices_trajectory_and_zero_pads(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "controller.json"
            path.write_text(
                '{"result":{"trajectory":[{"action":-0.2},{"action":0.3},{"action":1.2}]}}',
                encoding="utf-8",
            )
            np.testing.assert_allclose(
                warm_start_controls(path, offset_steps=1, horizon_steps=4),
                [0.3, 1.0, 0.0, 0.0],
            )

    def test_controls_from_knots_interpolates_and_clips(self) -> None:
        np.testing.assert_allclose(
            controls_from_knots(np.asarray([-2.0, 0.0, 2.0]), 5),
            [-1.0, -1.0, 0.0, 1.0, 1.0],
        )

    def test_capture_selection_uses_uninterrupted_result_before_path_cost(self) -> None:
        low_cost_failure = {
            "label": "low_cost",
            "cost": 1.0,
            "result": {
                "success": False,
                "latched": False,
                "max_upright_streak_seconds": 0.2,
                "minimum_lyapunov": 100.0,
                "max_cart_excursion": 1.0,
            },
        }
        captured = {
            "label": "captured",
            "cost": 1000.0,
            "result": {
                "success": False,
                "latched": True,
                "max_upright_streak_seconds": 0.1,
                "minimum_lyapunov": 500.0,
                "max_cart_excursion": 2.0,
            },
        }

        self.assertEqual(
            min([low_cost_failure, captured], key=capture_selection_key)["label"],
            "captured",
        )

    def test_chain_basin_helpers_validate_and_aggregate(self) -> None:
        self.assertEqual(parse_radii("0.01, 0.2"), [0.01, 0.2])
        with self.assertRaises(ValueError):
            parse_radii("0.0")
        rows = [
            {
                "result": {
                    "success": True,
                    "latched": True,
                    "minimum_lyapunov": 10.0,
                    "max_cart_excursion": 1.0,
                }
            },
            {
                "result": {
                    "success": False,
                    "latched": False,
                    "minimum_lyapunov": 30.0,
                    "max_cart_excursion": 2.0,
                }
            },
        ]
        self.assertEqual(
            aggregate(rows),
            {
                "count": 2,
                "success_count": 1,
                "success_rate": 0.5,
                "latch_count": 1,
                "latch_rate": 0.5,
                "median_minimum_lyapunov": 20.0,
                "maximum_cart_excursion": 2.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
