from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv

ROOT = Path(__file__).resolve().parents[1]


class ExplicitMorphologyProfileTests(unittest.TestCase):
    def test_explicit_damping_profile_preserves_total(self) -> None:
        cfg = load_config(ROOT / "configs/generalized_n3_split_locked.yaml")
        cfg["morphology"]["damping_start"] = [0.0075, 0.0074, 0.0001]
        cfg["morphology"]["damping_end"] = [0.005, 0.005, 0.005]
        start = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
        middle = NLinkCartPoleEnv(cfg, progress=0.5, seed=0)
        end = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
        try:
            np.testing.assert_allclose(start.morphology.damping, [0.0075, 0.0074, 0.0001])
            np.testing.assert_allclose(end.morphology.damping, [0.005, 0.005, 0.005])
            self.assertAlmostEqual(float(np.sum(middle.morphology.damping)), 0.015)
        finally:
            start.close()
            middle.close()
            end.close()

    def test_ghost_profile_preserves_totals_and_reaches_uniform_target(self) -> None:
        cfg = load_config(ROOT / "configs/swingup8_ghost_continuation.yaml")
        start = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
        middle = NLinkCartPoleEnv(cfg, progress=0.5, seed=0)
        end = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
        try:
            for env in (start, middle, end):
                self.assertAlmostEqual(float(np.sum(env.morphology.lengths)), 3.0)
                self.assertAlmostEqual(float(np.sum(env.morphology.masses)), 1.0)
            self.assertLess(float(start.morphology.lengths[-1]), 0.03)
            self.assertLess(float(start.morphology.masses[-1]), 0.001)
            self.assertAlmostEqual(float(start.morphology.joint_stiffness[-1]), 0.001)
            self.assertAlmostEqual(float(end.morphology.joint_stiffness[-1]), 0.0)
            self.assertTrue(np.allclose(end.morphology.lengths, 3.0 / 8.0))
            self.assertTrue(np.allclose(end.morphology.masses, 1.0 / 8.0))
            self.assertGreater(float(middle.morphology.lengths[-1]), float(start.morphology.lengths[-1]))
            self.assertGreater(float(middle.morphology.masses[-1]), float(start.morphology.masses[-1]))
        finally:
            start.close()
            middle.close()
            end.close()

    def test_split_link_profile_preserves_totals_and_anneals_lock(self) -> None:
        cfg = load_config(ROOT / "configs/swingup8_split_link_continuation.yaml")
        start = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
        end = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
        try:
            self.assertAlmostEqual(float(np.sum(start.morphology.lengths)), 3.0)
            self.assertAlmostEqual(float(np.sum(start.morphology.masses)), 1.0)
            self.assertAlmostEqual(float(start.morphology.joint_stiffness[-1]), 1.0)
            self.assertAlmostEqual(float(end.morphology.joint_stiffness[-1]), 0.0)
            self.assertTrue(np.allclose(start.morphology.joint_lock, 0.0))
            self.assertTrue(np.allclose(end.morphology.joint_lock, 0.0))
            self.assertTrue(np.allclose(end.morphology.lengths, 3.0 / 8.0))
            self.assertTrue(np.allclose(end.morphology.masses, 1.0 / 8.0))
        finally:
            start.close()
            end.close()

    def test_split_locked_profile_uses_a_temporary_equality_constraint(self) -> None:
        cfg = load_config(ROOT / "configs/swingup8_split_locked_continuation.yaml")
        start = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
        middle = NLinkCartPoleEnv(cfg, progress=0.5, seed=0)
        end = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
        try:
            self.assertAlmostEqual(float(start.morphology.joint_lock[-1]), 1.0)
            self.assertAlmostEqual(float(middle.morphology.joint_lock[-1]), 0.5)
            self.assertAlmostEqual(float(end.morphology.joint_lock[-1]), 0.0)
            self.assertEqual(int(start.model.neq), 1)
            self.assertEqual(int(middle.model.neq), 1)
            self.assertEqual(int(end.model.neq), 0)
            self.assertIn('joint1="hinge_8"', start.xml)
            np.testing.assert_allclose(start.model.eq_solimp[0, :2], [0.9999, 0.9999])
            np.testing.assert_allclose(middle.model.eq_solimp[0, :2], [0.5, 0.5])
            np.testing.assert_allclose(start.model.eq_solref[0], [0.01, 1.0])
        finally:
            start.close()
            middle.close()
            end.close()

        logarithmic_cfg = copy.deepcopy(cfg)
        logarithmic_cfg["env"]["joint_lock_impedance_schedule"] = "log_compliance"
        logarithmic_middle = NLinkCartPoleEnv(
            logarithmic_cfg, progress=0.5, seed=0
        )
        try:
            expected_compliance = np.sqrt(0.0001 * 0.9999)
            np.testing.assert_allclose(
                logarithmic_middle.model.eq_solimp[0, :2],
                [1.0 - expected_compliance, 1.0 - expected_compliance],
            )
        finally:
            logarithmic_middle.close()

    def test_uniform_locked_stage_reaches_uniform_geometry_before_unlock(self) -> None:
        cfg = load_config(ROOT / "configs/swingup8_uniform_locked_morphology.yaml")
        start = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
        end = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
        try:
            self.assertAlmostEqual(float(start.morphology.joint_lock[-1]), 1.0)
            self.assertAlmostEqual(float(end.morphology.joint_lock[-1]), 1.0)
            self.assertTrue(np.allclose(end.morphology.lengths, 3.0 / 8.0))
            self.assertTrue(np.allclose(end.morphology.masses, 1.0 / 8.0))
            self.assertEqual(int(start.model.neq), 1)
            self.assertEqual(int(end.model.neq), 1)
        finally:
            start.close()
            end.close()

    def test_uniform_unlock_stage_changes_only_the_joint_constraint(self) -> None:
        cfg = load_config(ROOT / "configs/swingup8_uniform_unlock.yaml")
        start = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
        end = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
        try:
            self.assertTrue(np.allclose(start.morphology.lengths, end.morphology.lengths))
            self.assertTrue(np.allclose(start.morphology.masses, end.morphology.masses))
            self.assertEqual(int(start.model.neq), 1)
            self.assertEqual(int(end.model.neq), 0)
        finally:
            start.close()
            end.close()


if __name__ == "__main__":
    unittest.main()
