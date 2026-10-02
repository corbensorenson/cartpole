import json
from pathlib import Path

import numpy as np

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.route_feedback import TimeVaryingFeedbackRoute


def test_time_varying_route_advances_phase_and_applies_feedback(tmp_path):
    path = tmp_path / "route.json"
    path.write_text(
        json.dumps(
            {
                "controller": {
                    "controls": [0.25, -0.5],
                    "feedback_gains": np.zeros((2, 6)).tolist(),
                    "periodic_coordinate_errors": True,
                },
                "search": {
                    "nominal_coordinate_states": np.zeros((3, 6)).tolist(),
                },
            }
        ),
        encoding="utf-8",
    )
    route = TimeVaryingFeedbackRoute(path, 2, phase_window=1)

    first, first_index = route.action(np.zeros(3), np.zeros(3))
    second, second_index = route.action(np.zeros(3), np.zeros(3))

    assert first == 0.25
    assert second == -0.5
    assert first_index == 0
    assert second_index == 1
    assert route.phase_fraction == 1.0


def test_env_route_residual_and_handoff_distance_are_observable():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/swingup11_uniform.yaml")
    cfg["env"]["init_angle_noise"] = 0.0
    cfg["env"]["init_vel_noise"] = 0.0
    cfg["env"]["init_cart_vel_noise"] = 0.0
    cfg["env"]["obs_include_route_features"] = True
    cfg["env"]["action_route_residual"] = {
        "enabled": True,
        "path": str(root / "runs/generalized_solver/n11_p39_feedback_a1rm1.json"),
        "tracking_gain_scale": 0.5,
        "phase_window": 12,
        "residual_scale": 0.5,
    }
    cfg["env"]["reward"]["handoff_bank_path"] = str(
        root / "runs/generalized_solver/n11_p39_low_momentum_handoff_states.json"
    )
    cfg["env"]["reward"]["handoff_distance_progress"] = 1.0
    env = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    try:
        obs, info = env.reset(seed=0)
        assert obs.shape == (81,)
        assert np.isfinite(info["handoff_bank_distance"])
        _, _, terminated, truncated, info = env.step([0.0])
        assert not terminated
        assert not truncated
        assert info["controller_mode"] == "route_residual"
        assert info["route_index"] == 0
    finally:
        env.close()


def test_env_can_fade_route_into_an_angle_curriculum():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/swingup11_uniform.yaml")
    cfg["env"]["init_mode"] = "hanging_curriculum"
    cfg["env"]["init_angle_noise"] = 0.0
    cfg["env"]["init_vel_noise"] = 0.0
    cfg["env"]["init_cart_vel_noise"] = 0.0
    cfg["env"]["obs_include_route_features"] = True
    cfg["env"]["action_route_residual"] = {
        "enabled": True,
        "path": str(root / "runs/generalized_solver/n11_p39_feedback_a1rm1.json"),
        "tracking_gain_scale": 0.5,
        "phase_window": 12,
        "route_action_scale_start": 0.0,
        "route_action_scale_end": 1.0,
        "residual_scale_start": 1.0,
        "residual_scale_end": 0.5,
    }
    env = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
    try:
        env.reset(seed=0)
        _, _, _, _, early_info = env.step([0.0])
        assert early_info["action_bias_norm"] == 0.0
        env.set_progress(1.0)
        env.reset(seed=0)
        _, _, _, _, final_info = env.step([0.0])
        assert abs(final_info["action_bias_norm"]) > 0.1
        assert final_info["residual_scale"] == 0.5
    finally:
        env.close()


def test_env_uses_lqr_warm_start_before_route_expert():
    root = Path(__file__).resolve().parents[1]
    cfg = load_config(root / "configs/swingup11_uniform.yaml")
    cfg["env"]["init_mode"] = "hanging_curriculum"
    cfg["env"]["init_angle_noise"] = 0.0
    cfg["env"]["init_vel_noise"] = 0.0
    cfg["env"]["init_cart_vel_noise"] = 0.0
    cfg["env"]["action_lqr_residual"] = {
        "enabled": True,
        "state_gain": [0.0] * 24,
        "residual_scale": 0.05,
        "residual_action_limit": 0.005,
    }
    cfg["env"]["action_route_residual"] = {
        "enabled": True,
        "path": str(root / "runs/generalized_solver/n11_p39_feedback_a1rm1.json"),
        "tracking_gain_scale": 0.5,
        "phase_window": 12,
        "lqr_until_progress": 0.25,
    }
    env = NLinkCartPoleEnv(cfg, progress=0.0, seed=0)
    try:
        env.reset(seed=0)
        _, _, _, _, early_info = env.step([0.0])
        assert early_info["controller_mode"] == "lqr_residual"
        assert early_info["route_index"] is None
        env.set_progress(1.0)
        env.reset(seed=0)
        _, _, _, _, final_info = env.step([0.0])
        assert final_info["controller_mode"] == "route_residual"
        assert final_info["route_index"] == 0
    finally:
        env.close()
