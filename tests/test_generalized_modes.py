from __future__ import annotations

import copy

import mujoco
import numpy as np

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.generalized_modes import (
    chain_normal_modes,
    modal_energy_acceleration_ratio,
)


def uniform_env(n_links: int) -> NLinkCartPoleEnv:
    cfg = copy.deepcopy(load_config("configs/swingup7_uniform.yaml"))
    cfg["env"]["n_links"] = int(n_links)
    cfg["env"]["init_angle_noise"] = 0.0
    cfg["env"]["init_vel_noise"] = 0.0
    cfg["env"]["init_cart_vel_noise"] = 0.0
    return NLinkCartPoleEnv(cfg, progress=1.0, seed=0)


def test_hanging_modes_are_mass_orthonormal_and_satisfy_eigenproblem():
    env = uniform_env(4)
    env.reset(seed=0)
    qpos = np.asarray(env.data.qpos).copy()
    qvel = np.asarray(env.data.qvel).copy()
    modes = chain_normal_modes(env, equilibrium="hanging")

    np.testing.assert_allclose(
        modes.relative_shapes.T @ modes.joint_mass_matrix @ modes.relative_shapes,
        np.eye(4),
        atol=1e-9,
    )
    np.testing.assert_allclose(
        modes.stiffness_matrix @ modes.relative_shapes,
        modes.joint_mass_matrix
        @ modes.relative_shapes
        @ np.diag(modes.squared_frequencies),
        atol=1e-8,
    )
    assert np.all(modes.squared_frequencies > 0.0)
    assert np.all(np.diff(modes.angular_frequencies) > 0.0)
    assert np.isclose(np.linalg.norm(modes.normalized_coupling), 1.0)
    np.testing.assert_allclose(env.data.qpos, qpos)
    np.testing.assert_allclose(env.data.qvel, qvel)
    env.close()


def test_modal_projection_reconstructs_small_relative_state():
    env = uniform_env(3)
    env.reset(seed=0)
    modes = chain_normal_modes(env, equilibrium="hanging")
    displacement = np.asarray([0.03, -0.02, 0.01])
    rates = np.asarray([-0.1, 0.2, -0.05])
    positions, velocities = modes.coordinates(
        modes.relative_equilibrium + displacement,
        rates,
    )
    np.testing.assert_allclose(
        modes.relative_shapes @ positions, displacement, atol=1e-12
    )
    np.testing.assert_allclose(modes.relative_shapes @ velocities, rates, atol=1e-12)
    env.close()


def test_modal_energy_law_has_constant_parameter_count_and_damps_internal_power():
    env = uniform_env(5)
    env.reset(seed=0)
    modes = chain_normal_modes(env, equilibrium="hanging")
    # Construct a pure internal-mode velocity.  With no collective pumping, the
    # command must oppose its input-power direction regardless of link count.
    modal_velocity = np.zeros(5)
    modal_velocity[2] = 1.0
    rates = modes.relative_shapes @ modal_velocity
    ratio, diagnostics = modal_energy_acceleration_ratio(
        modes,
        modes.relative_equilibrium,
        rates,
        total_energy_error=-1.0,
        energy_scale=float(env._energy_gap),
        collective_gain=0.0,
        internal_damping_gain=2.0,
    )
    power_direction = float(diagnostics["internal_power_direction"])
    assert ratio * power_direction <= 0.0
    assert len(diagnostics["modal_phase"]) == 5
    env.close()


def test_mode_analysis_restores_mujoco_time_and_state():
    env = uniform_env(2)
    env.reset(seed=0)
    env.step([0.1])
    before = (
        float(env.data.time),
        np.asarray(env.data.qpos).copy(),
        np.asarray(env.data.qvel).copy(),
        np.asarray(env.data.ctrl).copy(),
    )
    chain_normal_modes(env)
    after = (
        float(env.data.time),
        np.asarray(env.data.qpos).copy(),
        np.asarray(env.data.qvel).copy(),
        np.asarray(env.data.ctrl).copy(),
    )
    assert np.isclose(before[0], after[0])
    for first, second in zip(before[1:], after[1:]):
        np.testing.assert_allclose(first, second)
    # Restoring through mj_forward must leave a valid exact state.
    mujoco.mj_forward(env.model, env.data)
    env.close()
