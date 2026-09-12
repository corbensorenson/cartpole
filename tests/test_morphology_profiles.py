from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv


ROOT = Path(__file__).resolve().parents[1]


class ExplicitMorphologyProfileTests(unittest.TestCase):
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
            self.assertTrue(np.allclose(end.morphology.lengths, 3.0 / 8.0))
            self.assertTrue(np.allclose(end.morphology.masses, 1.0 / 8.0))
            self.assertGreater(float(middle.morphology.lengths[-1]), float(start.morphology.lengths[-1]))
            self.assertGreater(float(middle.morphology.masses[-1]), float(start.morphology.masses[-1]))
        finally:
            start.close()
            middle.close()
            end.close()


if __name__ == "__main__":
    unittest.main()
