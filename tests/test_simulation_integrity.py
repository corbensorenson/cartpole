from pathlib import Path

import mujoco
import numpy as np
import pytest

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.simulation import SimulationError


@pytest.fixture
def env():
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs/swingup11_uniform.yaml")
    result = NLinkCartPoleEnv(cfg, progress=1.0, seed=0)
    result.reset(seed=0)
    yield result
    result.close()


def test_automatic_zero_reset_never_counts_as_upright(env, monkeypatch):
    def reset_on_step(model, data):
        mujoco.mj_resetData(model, data)
        data.time = float(model.opt.timestep)

    monkeypatch.setattr(mujoco, "mj_step", reset_on_step)
    _, _, terminated, _, info = env.step([0.0])
    assert terminated
    assert info["termination_reason"] == "simulation_invalid"
    assert not info["success"]
    assert not info["is_upright"]
    assert info["max_upright_streak_seconds"] == 0.0


def test_warning_rejects_optimizer_transition(env, monkeypatch):
    def warn_on_step(model, data):
        data.time += float(model.opt.timestep)
        data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number += 1

    transition = MujocoTransition(env)
    monkeypatch.setattr(mujoco, "mj_step", warn_on_step)
    with pytest.raises(SimulationError, match="warning"):
        transition(transition.to_coordinates(data_state(env.data)), 0.0)


def test_valid_environment_and_optimizer_steps_agree(env):
    transition = MujocoTransition(env)
    state = transition.to_coordinates(data_state(env.data))
    expected = transition(state, 0.125)
    _, _, terminated, _, info = env.step([0.125])
    assert not terminated
    assert info["simulation_error"] is None
    np.testing.assert_allclose(
        transition.to_coordinates(data_state(env.data)), expected, atol=1e-12, rtol=0
    )


def test_continuous_dynamics_derivative_matches_euclidean_fd_at_pi(env):
    env.data.qpos[:] = 0
    env.data.qpos[1] = np.pi
    env.data.qvel[:] = 0
    transform = np.eye(2 * (env.n + 1))
    transform[1, 1] = 2.0
    transition = MujocoTransition(env, transform, continuous_angles=True)
    state = transition.to_coordinates(data_state(env.data))
    a, _ = transition.linearize(state, 0.0, state_epsilon=1e-5, action_epsilon=1e-4)
    dx = np.zeros(state.size)
    dx[1] = 1e-5
    euclidean_fd = (transition(state + dx, 0.0) - transition(state - dx, 0.0)) / 2e-5
    np.testing.assert_allclose(a[:, 1], euclidean_fd, atol=1e-10)
    assert abs(a[1, 1]) < 2.0
