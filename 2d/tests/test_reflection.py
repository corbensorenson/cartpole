import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.symmetry import reflect_state,clean_planar_reference
from spatialpole.transfer import lift_planar_state
from spatialpole.affine_value import ParkedAffineValuePolicy


@pytest.mark.parametrize('n',[1,3,13])
def test_native_full_spatial_dynamics_obey_plane_reflection(n):
    env = SpatialCartpole(Config(n_links=n))
    rng = np.random.default_rng(825)
    state = env.integrate(env.upright_state(),rng.normal(size=2*env.model.nv)*.2)
    mirrored = reflect_state(env,state)
    for t in range(10):
        force = np.array([9*np.sin(t+.3),-7*np.cos(t*.4)])
        state = env.transition(state,force)
        mirrored = env.transition(mirrored,force*[1,-1])
        np.testing.assert_allclose(env.difference(mirrored,reflect_state(env,state)),0.,atol=3e-11)
    env.close()


@pytest.mark.parametrize('n',[1,3,13])
def test_reference_cleanup_is_small_and_never_changes_live_state(n):
    env = SpatialCartpole(Config(n_links=n))
    q,v = np.r_[.1,-np.pi,np.full(n-1,.03)],np.full(n+1,.02)
    exact = lift_planar_state(env,q,v)
    noisy = env.integrate(exact,np.full(2*env.model.nv,1e-13))
    xs,us = np.array([noisy,noisy]),np.array([[2.,1e-12]])
    original_x,original_u,live = xs.copy(),us.copy(),env.state().copy()
    cleaned,forces,record = clean_planar_reference(env,xs,us)
    np.testing.assert_array_equal(xs,original_x)
    np.testing.assert_array_equal(us,original_u)
    np.testing.assert_array_equal(env.state(),live)
    np.testing.assert_array_equal(cleaned,np.array([reflect_state(env,x) for x in cleaned]))
    assert forces[0,1]==0 and record['max_normalized_state_change']<1e-10
    assert env.native_steps==0
    env.close()


def test_reference_cleanup_rejects_real_spatial_motion():
    env = SpatialCartpole(Config(n_links=3))
    state = env.upright_state()
    state[1] = .01
    with pytest.raises(ValueError,match='tolerance'):
        clean_planar_reference(env,np.array([state,state]),np.zeros((1,2)))
    env.close()


@pytest.mark.parametrize('n',[1,3,13])
def test_native_dynamics_are_invariant_under_xy_cart_origin_translation(n):
    env = SpatialCartpole(Config(n_links=n))
    rng = np.random.default_rng(826)
    state = env.integrate(env.upright_state(),rng.normal(size=2*env.model.nv)*.1)
    shifted = state.copy()
    translation = np.array([.4,-.2])
    shifted[:2] += translation
    for t in range(10):
        force = np.array([4*np.sin(t+.2),-3*np.cos(t*.3)])
        state = env.transition(state,force)
        shifted = env.transition(shifted,force)
        expected = state.copy()
        expected[:2] += translation
        np.testing.assert_allclose(env.difference(shifted,expected),0.,atol=3e-10)
    env.close()


@pytest.mark.parametrize('align,offset',[(True,[0.,0.]),(False,[.003,-.004])])
def test_reference_origin_alignment_leaves_observed_cart_and_payload_untouched(align,offset):
    env = SpatialCartpole(Config(n_links=1))
    ref,nx = env.upright_state(),2*env.model.nv
    gains = np.zeros((2,2,nx))
    gains[:,0,0],gains[:,1,1] = 1.,1.
    original = np.repeat(ref[None,:],3,axis=0)
    payload = dict(states=original,actions=np.zeros((2,2)),gains=gains,feed=np.zeros((2,2)),
        roots=np.repeat(np.eye(nx)[None,:,:],3,axis=0),residuals=np.zeros((3,nx)),tracking_R=np.eye(2),
        reference=ref,K=np.zeros((2,nx)),P=np.eye(nx),Q=np.eye(nx),R=np.eye(2),
        park_steps=0,swing_steps=2,parking_reference=ref,parking_K=np.zeros((2,nx)),
        align_gauge=False,online_gauge=False,native_search=False,align_cart_origin=align,
        cart_reference_offset=np.array(offset))
    observed = ref.copy()
    observed[:2] = [.01,-.02]
    env.set_state(observed)
    policy = ParkedAffineValuePolicy(env,payload)
    force = policy(observed,0)
    np.testing.assert_array_equal(env.state(),observed)
    np.testing.assert_array_equal(original,np.repeat(ref[None,:],3,axis=0))
    expected_origin = observed[:2] if align else np.array(offset)
    np.testing.assert_array_equal(policy.route.states[:,:2],np.repeat(expected_origin[None,:],3,axis=0))
    np.testing.assert_allclose(force,expected_origin-observed[:2],atol=1e-15)
    assert env.native_steps==0
    policy.close()
    env.close()
