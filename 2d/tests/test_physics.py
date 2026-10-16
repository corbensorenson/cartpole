import mujoco as mj
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from spatialpole import Config, SpatialCartpole
from spatialpole.control import balance_design, BalancePolicy, WorldBalancePolicy
from spatialpole.control import linearize, linearize_many, cost_derivatives, state_cost


@pytest.mark.parametrize("n", [1, 2, 5])
def test_unrestricted_physics(n):
    e = SpatialCartpole(Config(n_links=n))
    assert e.model.nq == 2+4*n and e.model.nv == 2+3*n
    assert e.model.nu == 2 and e.model.neq == 0
    assert list(e.model.jnt_type[2:]) == [mj.mjtJoint.mjJNT_BALL]*n
    assert not e.model.jnt_limited.any()
    assert np.count_nonzero(e.model.actuator_trnid[:,0] >= 2) == 0
    state, _ = e.reset(seed=1, options={"angle_noise":0})
    assert np.max(e.tilts()) == 0
    e.reset(seed=1, options={"mode":"hanging", "angle_noise":0})
    np.testing.assert_allclose(e.tilts(), np.pi)


def test_independent_xy_and_twist():
    e = SpatialCartpole()
    x = e.upright_state()
    dx = e.transition(x, [10.,0.])
    dy = e.transition(x, [0.,10.])
    assert dx[0] > 0 and dy[1] > 0
    np.testing.assert_allclose(dx[0], dy[1], rtol=1e-9)
    assert abs(dx[1])+abs(dy[0]) < 1e-12
    d = np.zeros(2*e.model.nv)
    d[4] = 1.2
    e.set_state(e.integrate(x,d))
    np.testing.assert_allclose(e.tilts(), 0., atol=1e-8)


def test_force_budget_and_replay():
    e = SpatialCartpole()
    x, _ = e.reset(seed=7)
    u = e.deliver([80.,80.])
    assert u.dtype == np.float32 and np.linalg.norm(u.astype(float)) <= 80
    expected = e.transition(x,u)
    e.transition(e.upright_state(),[-50.,20.])
    np.testing.assert_array_equal(e.transition(x,u), expected)
    with pytest.raises(ValueError):
        e.step_force([np.nan,0])


def test_balance_both_tilt_axes():
    e = SpatialCartpole(Config(episode_seconds=6.))
    design = balance_design(e)
    policy = BalancePolicy(e, design)
    x, _ = e.reset(seed=33)
    for t in range(e.config.horizon):
        x, _, term, trunc, info = e.step_force(policy(x))
        assert not term
    assert trunc and info["success"]
    assert np.max(e.tilts()) < 0.001


def test_gym_interface():
    e = SpatialCartpole()
    check_env(e, skip_render_check=True)


@pytest.mark.parametrize("n", [1,2])
def test_batched_derivatives_match_independent_stencil(n):
    e = SpatialCartpole(Config(n_links=n))
    x, _ = e.reset(seed=44, options={"angle_noise":0.3})
    force = np.array([5.,-8.])
    slow = linearize(e,x,force)
    fast = linearize_many(e,np.array([x]),np.array([force]))[0]
    for a,b in zip(slow,fast):
        np.testing.assert_allclose(a,b,atol=1e-9,rtol=1e-7)


def test_spatial_cost_gradient_at_large_rotation():
    e = SpatialCartpole(Config(n_links=2))
    ref = e.upright_state()
    rng = np.random.default_rng(654)
    x = e.integrate(ref,rng.normal(size=2*e.model.nv))
    W = state_cost(e)
    gradient,_ = cost_derivatives(e,x,ref,W)
    actual = []
    for j in range(2*e.model.nv):
        delta = np.zeros(2*e.model.nv)
        delta[j] = 1e-6
        hi = e.difference(e.integrate(x,delta),ref)
        lo = e.difference(e.integrate(x,-delta),ref)
        actual.append((hi@W@hi-lo@W@lo)/4e-6)
    np.testing.assert_allclose(gradient,actual,atol=1e-6,rtol=1e-6)


def test_world_feedback_is_invariant_to_passive_axial_twist():
    e = SpatialCartpole()
    design = balance_design(e)
    policy = WorldBalancePolicy(e,design)
    delta = np.zeros(2*e.model.nv)
    delta[2:4] = [0.03,-0.04]
    x = e.integrate(e.upright_state(),delta)
    spin = np.zeros_like(delta)
    spin[4] = 1.5
    twisted = e.integrate(x,spin)
    np.testing.assert_allclose(policy(x),policy(twisted),atol=1e-10)
