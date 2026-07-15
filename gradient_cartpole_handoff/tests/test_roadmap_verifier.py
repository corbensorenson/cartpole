from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gcartpole.config import load_config
from gcartpole.evidence import data_sha256, file_sha256
from gcartpole.roadmap import (
    benchmark_snapshot,
    validate_canonical_config,
    validate_evaluation,
    validate_runtime_benchmark,
    validate_solution_artifacts,
)


ROOT = Path(__file__).resolve().parents[1]


class CanonicalBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cfg = load_config(ROOT / "configs/swingup7_uniform.yaml")

    def test_canonical_config_and_snapshot(self) -> None:
        self.assertEqual(validate_canonical_config(self.cfg), [])
        snapshot = benchmark_snapshot(self.cfg)
        self.assertEqual(snapshot["observation_dim"], 51)
        self.assertEqual(snapshot["action_dim"], 1)
        self.assertEqual(snapshot["action_frequency_hz"], 50.0)
        self.assertEqual(snapshot["max_steps"], 1500)
        self.assertTrue(all(abs(value - 3.0 / 7.0) < 1e-12 for value in snapshot["lengths"]))
        self.assertTrue(all(abs(value - 1.0 / 7.0) < 1e-12 for value in snapshot["masses"]))
        self.assertTrue(all(value == 0.0 for value in snapshot["frictionloss"]))

    def test_rejects_training_wheels_and_wrong_plant(self) -> None:
        cfg = copy.deepcopy(self.cfg)
        cfg["env"]["rail_limit"] = 10.0
        cfg["env"]["init_mode"] = "hanging_curriculum"
        cfg["morphology"]["end"]["alpha_mass"] = 0.1
        errors = validate_canonical_config(cfg)
        self.assertTrue(any("rail_limit" in error for error in errors))
        self.assertTrue(any("init_mode" in error for error in errors))
        self.assertTrue(any("alpha_mass" in error for error in errors))

    def test_runtime_contract(self) -> None:
        errors, snapshot = validate_runtime_benchmark(self.cfg)
        self.assertEqual(errors, [])
        self.assertEqual(snapshot["initial_info"]["max_abs_angle"], np.pi)
        self.assertEqual(snapshot["initial_info"]["max_cart_excursion"], 0.0)

    def test_rejects_incomplete_evaluation(self) -> None:
        snapshot = benchmark_snapshot(self.cfg)
        payload = {
            "episodes": 20,
            "success_rate": 1.0,
            "episode_results": [{"seed": index, "success": True} for index in range(20)],
            "evidence": {
                "deterministic_policy": True,
                "config": {"resolved_sha256": data_sha256(self.cfg)},
                "generated_xml_sha256": snapshot["generated_xml_sha256"],
                "environment": {
                    "n_links": 7,
                    "init_mode": "hanging",
                    "force_limit": 80.0,
                    "rail_limit": 3.0,
                    "observation_dim": 51,
                    "action_dim": 1,
                    "action_frequency_hz": 50.0,
                },
            },
        }
        errors = validate_evaluation(
            payload,
            required_episodes=20,
            minimum_success_rate=0.80,
            config_sha256=snapshot["config_sha256"],
            xml_sha256=snapshot["generated_xml_sha256"],
        )
        self.assertTrue(any("missing" in error for error in errors))

    def test_rejects_below_gate_and_duplicate_seeds(self) -> None:
        snapshot = benchmark_snapshot(self.cfg)
        episode = {
            "seed": 1,
            "return": 0.0,
            "termination_reason": "time_limit",
            "time_to_first_upright": None,
            "time_to_capture": None,
            "max_upright_streak_seconds": 0.0,
            "final_upright_streak_seconds": 0.0,
            "max_cart_excursion": 0.0,
            "success": False,
        }
        payload = {
            "episodes": 20,
            "success_rate": 0.79,
            "episode_results": [copy.deepcopy(episode) for _ in range(20)],
            "evidence": {
                "deterministic_policy": True,
                "config": {"resolved_sha256": snapshot["config_sha256"]},
                "generated_xml_sha256": snapshot["generated_xml_sha256"],
                "environment": {
                    "n_links": 7,
                    "init_mode": "hanging",
                    "force_limit": 80.0,
                    "rail_limit": 3.0,
                    "observation_dim": 51,
                    "action_dim": 1,
                    "action_frequency_hz": 50.0,
                },
            },
        }
        errors = validate_evaluation(
            payload,
            required_episodes=20,
            minimum_success_rate=0.80,
            config_sha256=snapshot["config_sha256"],
            xml_sha256=snapshot["generated_xml_sha256"],
        )
        self.assertTrue(any("below" in error for error in errors))
        self.assertTrue(any("not unique" in error for error in errors))

    def test_rejects_missing_weights_and_reset_video(self) -> None:
        snapshot = benchmark_snapshot(self.cfg)
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            run_dir = repo_root / "runs/swingup7_uniform"
            run_dir.mkdir(parents=True)
            manifest = {
                "architecture": "single_policy",
                "training": {"wall_clock_seconds": 1.0, "environment_steps": 1},
                "checkpoints": [
                    {"role": "policy", "path": "runs/swingup7_uniform/missing.safetensors", "sha256": "0" * 64}
                ],
            }
            (run_dir / "policy_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            def evaluation(count: int) -> dict:
                episodes = []
                for index in range(count):
                    episodes.append(
                        {
                            "seed": 1000 + index,
                            "return": 1.0,
                            "termination_reason": "time_limit",
                            "time_to_first_upright": 1.0,
                            "time_to_capture": 6.0,
                            "max_upright_streak_seconds": 24.0,
                            "final_upright_streak_seconds": 24.0,
                            "max_cart_excursion": 2.0,
                            "success": True,
                        }
                    )
                return {
                    "episodes": count,
                    "success_rate": 1.0,
                    "episode_results": episodes,
                    "evidence": {
                        "deterministic_policy": True,
                        "config": {"resolved_sha256": snapshot["config_sha256"]},
                        "generated_xml_sha256": snapshot["generated_xml_sha256"],
                        "environment": {
                            "n_links": 7,
                            "init_mode": "hanging",
                            "force_limit": 80.0,
                            "rail_limit": 3.0,
                            "observation_dim": 51,
                            "action_dim": 1,
                            "action_frequency_hz": 50.0,
                        },
                    },
                }

            for count in (20, 100):
                (run_dir / f"eval_swingup7_{count}.json").write_text(
                    json.dumps(evaluation(count)), encoding="utf-8"
                )
            video_path = run_dir / "seven_link_swingup_success.mp4"
            video_path.write_bytes(b"not-a-real-video")
            video_meta = {
                "video": {"sha256": file_sha256(video_path)},
                "generated_xml_sha256": snapshot["generated_xml_sha256"],
                "render": {
                    "reset_count": 1,
                    "completed_requested_steps": True,
                    "simulated_seconds": 30.0,
                    "seed": 1000,
                    "done_events": [{"terminated": False, "success": True}],
                },
            }
            (run_dir / "seven_link_swingup_success.video.json").write_text(
                json.dumps(video_meta), encoding="utf-8"
            )
            errors = validate_solution_artifacts(self.cfg, run_dir, repo_root)
            self.assertTrue(any("missing checkpoint" in error for error in errors))
            self.assertTrue(any("reset_count" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
