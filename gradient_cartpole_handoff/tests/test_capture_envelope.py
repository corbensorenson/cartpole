from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.capture_envelope import generate_capture_states, validate_capture_config, validate_capture_states
from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ppo_mlx import select_evaluation_state_indices
from scripts.mine_capture_failures import build_mining_mixture
from scripts.evaluate_linear_mpc_capture import LinearMPC


ROOT = Path(__file__).resolve().parents[1]


class CaptureEnvelopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_config(ROOT / "benchmarks/p1_capture_envelope.yaml")

    def small_spec(self) -> dict:
        spec = copy.deepcopy(self.spec)
        spec["splits"]["test"]["count"] = 64
        return spec

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


if __name__ == "__main__":
    unittest.main()
