from pathlib import Path

import mujoco
import numpy as np

from gcartpole.config import load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.simulation import advance_checked
from scripts.audit_control_frequency import cadence_config
from scripts.audit_action_precision import precision_step


def test_control_cadence_variants_keep_physics_and_held_force_identical():
    base = load_config(Path(__file__).resolve().parents[1]/"configs/swingup11_uniform.yaml")
    states = []
    for frame_skip in (4, 2, 1):
        cfg = cadence_config(base, 11, frame_skip)
        env = NLinkCartPoleEnv(cfg, progress=1., seed=0)
        assert env.model.opt.timestep == .005
        assert env.model.opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        assert env.dt == .005 * frame_skip
        env.data.qpos[:] = 0
        env.data.qpos[1] = np.pi
        env.data.qvel[:] = 0
        env.data.ctrl[0] = 12.
        mujoco.mj_forward(env.model, env.data)
        for _ in range(4//frame_skip):
            advance_checked(env.model, env.data, frame_skip)
        states.append(np.r_[env.data.qpos.copy(), env.data.qvel.copy()])
        env.close()
    np.testing.assert_array_equal(states[0], states[1])
    np.testing.assert_array_equal(states[0], states[2])
    assert base["env"]["frame_skip"] == 4


def test_diagnostic_float32_step_matches_canonical_states_and_success_tracking():
    base = load_config(Path(__file__).resolve().parents[1]/"configs/swingup11_uniform.yaml")
    cfg = cadence_config(base, 2, 2)
    first = NLinkCartPoleEnv(cfg, progress=1., seed=17)
    second = NLinkCartPoleEnv(cfg, progress=1., seed=17)
    first.reset(seed=17); second.reset(seed=17)
    for action in np.random.default_rng(14).uniform(-.1, .1, 50):
        _, _, terminated, truncated, _ = first.step([action])
        diagnostic = precision_step(second, [action], float32=True)
        assert diagnostic == (terminated, truncated)
        np.testing.assert_array_equal(first.data.qpos, second.data.qpos)
        np.testing.assert_array_equal(first.data.qvel, second.data.qvel)
        assert first.max_upright_streak_steps == second.max_upright_streak_steps
        assert first._success() == second._success()
    first.close(); second.close()
