from __future__ import annotations

import unittest
from pathlib import Path

from scripts.continue_scp_capture import STAGES, artifact_matches_stage, stage_command


class SCPCurriculumTests(unittest.TestCase):
    def test_recipe_has_unique_ordered_stages_and_practical_finish(self) -> None:
        names = [str(stage["name"]) for stage in STAGES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(names[0], "loose_energy")
        self.assertEqual(names[-1], "v20_tight_velocity")
        self.assertIn("v10_energy", names)
        self.assertIn("v10_focus", names)
        self.assertEqual(STAGES[-1]["handoff_angle_abs"], 0.075)
        self.assertEqual(STAGES[-1]["handoff_cart_velocity_abs"], 0.25)
        self.assertEqual(STAGES[-1]["handoff_hinge_velocity_abs"], 0.375)

    def test_stage_command_includes_source_window_and_optional_bounds(self) -> None:
        command = stage_command(
            ".venv/bin/python",
            state_index=4,
            seed=64014,
            source=Path("source.json"),
            output=Path("output.json"),
            stage=STAGES[-1],
        )
        self.assertEqual(command[0], ".venv/bin/python")
        self.assertIn("--initial-controller", command)
        self.assertIn("source.json", command)
        self.assertIn("--window-offsets", command)
        self.assertIn("--handoff-angle-abs", command)
        self.assertIn("--handoff-cart-velocity-abs", command)
        self.assertIn("--handoff-hinge-velocity-abs", command)

    def test_resume_match_rejects_changed_stage_settings(self) -> None:
        stage = STAGES[0]
        payload = {
            "state_index": 4,
            "seed": 64014,
            "controller": {
                "initial_controller": {"sha256": "source-hash"},
                "iterations": stage["iterations"],
                "handoff_lyapunov": stage["handoff_lyapunov"],
                "window_steps": 30,
                "window_offsets": [0, 5, 10, 15, 20],
                "trust_region": 0.003,
                "box_penalty": stage["box_penalty"],
                "cart_penalty": stage["cart_penalty"],
                "handoff_angle_abs": 0.15,
                "handoff_cart_velocity_abs": 0.5,
                "handoff_hinge_velocity_abs": 0.75,
            },
        }
        self.assertTrue(
            artifact_matches_stage(
                payload,
                state_index=4,
                seed=64014,
                source_sha256="source-hash",
                stage=stage,
            )
        )
        changed = dict(stage, handoff_lyapunov=1000.0)
        self.assertFalse(
            artifact_matches_stage(
                payload,
                state_index=4,
                seed=64014,
                source_sha256="source-hash",
                stage=changed,
            )
        )


if __name__ == "__main__":
    unittest.main()
