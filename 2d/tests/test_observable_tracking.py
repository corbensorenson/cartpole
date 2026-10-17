import mujoco as mj
import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.physical import direction_tracking_factors
from spatialpole.conditioning import axial_gauge_state


def native_observables(env,state):
    env.set_state(state)
    directions = env.data.xmat[env.link_ids].reshape(-1,3,3)[:,:,2].copy()
    rates = []
    for body,direction in zip(env.link_ids,directions):
        velocity = np.zeros(6)
        mj.mj_objectVelocity(env.model,env.data,mj.mjtObj.mjOBJ_BODY,int(body),velocity,0)
        rates.append(np.cross(velocity[:3],direction))
    return np.r_[state[:2],directions.ravel(),state[env.model.nq:env.model.nq+2],np.array(rates).ravel()]


@pytest.mark.parametrize('n',[1,3,13])
def test_tracking_factor_matches_independent_native_observable_direction(n):
    env = SpatialCartpole(Config(n_links=n))
    rng = np.random.default_rng(918)
    nx = 2*env.model.nv
    state = env.integrate(env.upright_state(),rng.normal(size=nx)*.2)
    weights = rng.uniform(.2,4.,nx)
    live = env.state().copy()
    factor = direction_tracking_factors(env,state[None,:],weights)[0]
    np.testing.assert_array_equal(env.state(),live)
    direction = rng.normal(size=nx)
    h = 1e-6
    expected = np.sqrt(weights)*(native_observables(env,env.integrate(state,h*direction))-
                                 native_observables(env,env.integrate(state,-h*direction)))/(2*h)
    np.testing.assert_allclose(factor@direction,expected,rtol=2e-7,atol=2e-8)
    assert env.native_steps==0
    env.close()


def test_tracking_cost_annihilates_axial_frame_gauge_tangent():
    env = SpatialCartpole(Config(n_links=3))
    rng = np.random.default_rng(973)
    state = env.integrate(env.upright_state(),rng.normal(size=2*env.model.nv)*.3)
    factor = direction_tracking_factors(env,state[None,:],np.ones(2*env.model.nv))[0]
    h = 1e-6
    angles = rng.normal(size=3)
    tangent = (env.difference(axial_gauge_state(env,state,h*angles),state)-
               env.difference(axial_gauge_state(env,state,-h*angles),state))/(2*h)
    np.testing.assert_allclose(factor@tangent,0.,atol=2e-8)
    env.close()


def test_axial_spin_remains_physical_but_has_no_instantaneous_rod_rate_cost():
    env = SpatialCartpole(Config(n_links=3))
    rng = np.random.default_rng(974)
    state = env.integrate(env.upright_state(),rng.normal(size=2*env.model.nv)*.1)
    factor = direction_tracking_factors(env,state[None,:],np.ones(2*env.model.nv))[0]
    delta = np.zeros(2*env.model.nv)
    delta[env.model.nv+4+3*2] = 1.
    np.testing.assert_allclose(factor@delta,0.,atol=2e-9)
    spinning = state.copy()
    spinning[env.model.nq+4+3*2] += .5
    moved = env.transition(spinning,[0.,0.])
    assert abs(env.difference(moved,spinning)[4+3*2])>.001
    env.close()
