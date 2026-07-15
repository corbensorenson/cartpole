from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from gcartpole.ilqr import stitch_feedback_trajectories
from scripts.evaluate_ilqr_chain_basin import aggregate, parse_radii, select_feedback_gains
from scripts.evaluate_trajectory_funnel_library import selection_key
from scripts.evaluate_receding_ilqr_capture import shift_controls
from scripts.refine_ilqr_capture_chain import (
    capture_selection_key,
    controls_from_knots,
    warm_start_controls,
)
from scripts.recenter_trajectory_controller import realized_trajectory
from scripts.search_ilqr_capture import (
    handoff_bounds_satisfied,
    interpolate_initial_state,
)
from scripts.search_capture_pipeline import (
    best_extension_stage,
    best_stage,
    parse_state_indices,
    stage_paths,
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

    def test_receding_ilqr_shift_preserves_remaining_controls(self) -> None:
        np.testing.assert_allclose(
            shift_controls(np.asarray([0.1, 0.2, 0.3, 0.4]), 2),
            [0.3, 0.4],
        )
        with self.assertRaises(ValueError):
            shift_controls(np.asarray([0.1, 0.2]), 3)

    def test_initial_state_interpolation_preserves_endpoints(self) -> None:
        source = {
            "state_id": "source",
            "qpos": [0.0, 0.1, 0.2],
            "qvel": [0.3, 0.4, 0.5],
        }
        target = {
            "state_id": "target",
            "qpos": [1.0, -0.1, -0.2],
            "qvel": [-0.3, -0.4, -0.5],
        }

        start = interpolate_initial_state(source, target, 0.0)
        end = interpolate_initial_state(source, target, 1.0)

        np.testing.assert_allclose(start["qpos"], source["qpos"])
        np.testing.assert_allclose(start["qvel"], source["qvel"])
        np.testing.assert_allclose(end["qpos"], target["qpos"])
        np.testing.assert_allclose(end["qvel"], target["qvel"])

    def test_strict_handoff_bounds_reject_hot_upright_state(self) -> None:
        cold = np.asarray([0.0, 0.02, -0.01, 0.1, 0.2, -0.1])
        hot = cold.copy()
        hot[4] = 2.0

        self.assertTrue(
            handoff_bounds_satisfied(
                cold,
                2,
                angle_abs=0.15,
                cart_velocity_abs=0.5,
                hinge_velocity_rms=0.75,
            )
        )
        self.assertFalse(
            handoff_bounds_satisfied(
                hot,
                2,
                angle_abs=0.15,
                cart_velocity_abs=0.5,
                hinge_velocity_rms=0.75,
            )
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

    def test_chain_basin_selects_raw_solver_gains_with_scale(self) -> None:
        applied = np.zeros((2, 3))
        solver = np.arange(6, dtype=np.float64).reshape(2, 3)
        controller = {"solver_feedback_gains": solver.tolist()}
        np.testing.assert_allclose(
            select_feedback_gains(controller, applied, "solver", 0.25),
            0.25 * solver,
        )
        np.testing.assert_allclose(
            select_feedback_gains(controller, applied, "applied", 1.0), applied
        )

    def test_recentered_trajectory_aligns_pre_action_states_and_controls(self) -> None:
        initial = {"qpos": [0.0, 0.1], "qvel": [0.2, 0.3]}
        trajectory = [
            {"qpos": [1.0, 1.1], "qvel": [1.2, 1.3], "action": 0.4},
            {"qpos": [2.0, 2.1], "qvel": [2.2, 2.3], "action": -0.5},
        ]
        controls, states = realized_trajectory(
            initial, trajectory, np.eye(4), horizon_steps=2
        )
        np.testing.assert_allclose(controls, [0.4, -0.5])
        np.testing.assert_allclose(
            states,
            [
                [0.0, 0.1, 0.2, 0.3],
                [1.0, 1.1, 1.2, 1.3],
                [2.0, 2.1, 2.2, 2.3],
            ],
        )

    def test_capture_pipeline_parses_unique_indices_and_stable_paths(self) -> None:
        self.assertEqual(parse_state_indices("4, 1,9"), [4, 1, 9])
        with self.assertRaisesRegex(ValueError, "unique"):
            parse_state_indices("1,1")
        paths = stage_paths(Path("runs/pipeline"), 4)
        self.assertEqual(
            paths["approach"],
            Path("runs/pipeline/validation_4/approach_ilqr.json"),
        )
        self.assertEqual(
            paths["approach_fddp"],
            Path("runs/pipeline/validation_4/approach_fddp.json"),
        )

    def test_capture_pipeline_selects_best_authoritative_stage(self) -> None:
        def payload(
            *, success: bool, latched: bool, hold: float, value: float, cart: float
        ) -> dict[str, object]:
            return {
                "result": {
                    "success": success,
                    "latched": latched,
                    "max_upright_streak_seconds": hold,
                    "minimum_lyapunov": value,
                    "max_cart_excursion": cart,
                }
            }

        stages = {
            "predictive": payload(
                success=False, latched=False, hold=0.06, value=1100000.0, cart=3.12
            ),
            "approach": payload(
                success=False, latched=False, hold=0.04, value=37000.0, cart=3.02
            ),
            "approach_fddp": payload(
                success=False, latched=False, hold=0.14, value=1700.0, cart=3.01
            ),
            "tail_seed": payload(
                success=False, latched=False, hold=0.04, value=45000.0, cart=3.00
            ),
        }
        name, selected = best_stage(stages)
        self.assertEqual(name, "approach_fddp")
        self.assertIs(selected, stages["approach_fddp"])

        stages["approach"]["controller"] = {"controls": [0.0]}
        stages["approach"]["search"] = {"terminal_lyapunov": 37000.0}
        stages["approach_fddp"]["controller"] = {"controls": [0.0]}
        stages["approach_fddp"]["search"] = {"terminal_lyapunov": 1700.0}
        stages["tail_seed"]["controller"] = {"controls": [0.0]}
        stages["tail_seed"]["search"] = {"tail_terminal_lyapunov": 220000.0}
        name, selected = best_extension_stage(stages)
        self.assertEqual(name, "approach_fddp")
        self.assertIs(selected, stages["approach_fddp"])

        stages["tail_ddp"] = payload(
            success=True, latched=True, hold=10.0, value=1.0, cart=2.8
        )
        name, selected = best_stage(stages)
        self.assertEqual(name, "tail_ddp")
        self.assertIs(selected, stages["tail_ddp"])

    def test_funnel_library_selection_prefers_success_then_hold(self) -> None:
        failure = {
            "controller_index": 0,
            "result": {
                "success": False,
                "max_upright_streak_seconds": 12.0,
                "minimum_lyapunov": 0.0,
                "max_cart_excursion": 0.0,
            },
        }
        short_success = {
            "controller_index": 1,
            "result": {
                "success": True,
                "max_upright_streak_seconds": 9.0,
                "minimum_lyapunov": 10.0,
                "max_cart_excursion": 2.0,
            },
        }
        long_success = {
            "controller_index": 2,
            "result": {
                "success": True,
                "max_upright_streak_seconds": 10.0,
                "minimum_lyapunov": 20.0,
                "max_cart_excursion": 2.0,
            },
        }
        self.assertIs(
            min([failure, short_success, long_success], key=selection_key), long_success
        )


if __name__ == "__main__":
    unittest.main()
