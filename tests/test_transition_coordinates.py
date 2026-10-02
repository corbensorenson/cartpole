from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gcartpole.ilqr import MujocoTransition


def test_action_secant_uses_actual_quantized_action_separation():
    class QuantizedLinearMap(MujocoTransition):
        def __init__(self):
            self.nx = 2
            self.continuous_angles = True

        def __call__(self, state, action):
            return np.asarray(state) + float(np.float32(action))

    transition = QuantizedLinearMap()
    for action in (.7, 1.):
        _, b = transition.linearize(np.zeros(2), action, state_epsilon=1e-5, action_epsilon=1e-4)
        np.testing.assert_allclose(b, np.ones((2, 1)), atol=1e-12)
    import pytest
    with pytest.raises(ValueError, match="vanished"):
        transition.linearize(np.zeros(2), .7, state_epsilon=1e-5, action_epsilon=1e-12)


def test_physical_shooting_matches_uninterrupted_twelve_link_steps_exactly():
    from gcartpole.config import load_config
    from gcartpole.env import NLinkCartPoleEnv
    from gcartpole.ilqr import data_state
    from scripts.search_capture_sequence import fixed_state_cfg

    cfg = load_config("configs/swingup11_uniform.yaml")
    cfg["env"]["n_links"] = 12
    state = dict(qpos=[0., -np.pi] + [0.] * 11, qvel=[0.] * 13)
    cfg = fixed_state_cfg(cfg, state, 30.)
    env = NLinkCartPoleEnv(cfg, progress=1., seed=4)
    env.reset(seed=4)
    transition = MujocoTransition(env, continuous_angles=True)
    current = data_state(env.data).copy()
    try:
        for step in range(80):
            action = .05 * np.sin(step / 7.)
            current = transition(current, action)
            env.step([action])
            np.testing.assert_array_equal(current, data_state(env.data))
    finally:
        env.close()


def test_transformed_state_difference_wraps_angles_before_scaling() -> None:
    transition = MujocoTransition.__new__(MujocoTransition)
    transition.env = SimpleNamespace(n=1)
    transition.continuous_angles = False
    transition.coordinate_transform = np.diag([1.0, 2.0, 1.0, 1.0])
    transition.inverse_transform = np.linalg.inv(transition.coordinate_transform)

    physical_first = np.array([0.0, np.pi - 0.01, 0.0, 0.0])
    physical_second = np.array([0.0, -np.pi + 0.01, 0.0, 0.0])
    first = transition.coordinate_transform @ physical_first
    second = transition.coordinate_transform @ physical_second

    difference = transition.difference(first, second)

    np.testing.assert_allclose(difference, [0.0, -0.04, 0.0, 0.0], atol=1e-12)


def test_continuous_angles_preserve_full_rotations_and_euclidean_gaps():
    transition = MujocoTransition.__new__(MujocoTransition)
    transition.env = SimpleNamespace(n=1)
    transition.continuous_angles = True
    transition.coordinate_transform = np.diag([1.0, 2.0, 1.0, 1.0])
    transition.inverse_transform = np.linalg.inv(transition.coordinate_transform)
    first = np.array([0.0, 3 * np.pi + 0.01, 0.0, 0.0])
    second = np.array([0.0, 3 * np.pi - 0.01, 0.0, 0.0])
    np.testing.assert_allclose(transition.to_physical(transition.to_coordinates(first)), first)
    np.testing.assert_allclose(
        transition.difference(transition.to_coordinates(first), transition.to_coordinates(second)),
        [0.0, 0.04, 0.0, 0.0], atol=1e-12,
    )
