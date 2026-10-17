import numpy as np
import pytest
from spatialpole.transfer import regrid_planar_material_states,decode_planar_native_states,lift_planar_state
from spatialpole import Config,SpatialCartpole


@pytest.mark.parametrize('n',[1,3,13])
def test_native_planar_decoder_recovers_winding_rates_and_quaternion_sign_flips(n):
    env = SpatialCartpole(Config(n_links=n))
    time = np.linspace(0,1,57)
    q,v = np.zeros((len(time),n+1)),np.zeros((len(time),n+1))
    q[:,0],v[:,0] = .2*time,.2
    q[:,1],v[:,1] = np.pi+4*np.pi*time,4*np.pi
    for j in range(2,n+1):
        q[:,j],v[:,j] = .1*j*np.sin(2*np.pi*time),.2*np.pi*j*np.cos(2*np.pi*time)
    states = np.array([lift_planar_state(env,a,b) for a,b in zip(q,v)])
    for j in range(n):
        states[::3,2+4*j:6+4*j] *= -1
    original,live = states.copy(),env.state().copy()
    decoded = decode_planar_native_states(env,states)
    np.testing.assert_allclose(decoded,np.c_[q,v],atol=8e-15)
    for row,state in zip(decoded,states):
        encoded = lift_planar_state(env,row[:n+1],row[n+1:])
        np.testing.assert_allclose(env.difference(encoded,state),0.,atol=2e-14)
    np.testing.assert_array_equal(states,original)
    np.testing.assert_array_equal(env.state(),live)
    assert env.native_steps==0
    env.close()


def test_native_planar_decoder_rejects_spatial_reference():
    env = SpatialCartpole()
    state = env.upright_state()
    state[1] = .001
    with pytest.raises(ValueError,match='reflection-fixed'):
        decode_planar_native_states(env,np.array([state,state]))
    env.close()


def test_native_planar_decoder_rejects_aliased_rates():
    env = SpatialCartpole()
    state = env.upright_state()
    state[env.model.nq+3] = 200.
    with pytest.raises(ValueError,match='sampling'):
        decode_planar_native_states(env,np.array([state,state]))
    env.close()


@pytest.mark.parametrize('source_count,target_count',[(14,13),(3,2),(1,3)])
def test_material_transfer_preserves_cart_and_rigid_unwrapped_motion(source_count,target_count):
    p = np.zeros((3,2*(source_count+1)))
    p[:,0],p[:,source_count+1] = [.1,.2,.3],[.4,.5,.6]
    p[:,1],p[:,source_count+2] = [-np.pi,-2*np.pi,-3*np.pi],[.2,.3,.4]
    original = p.copy()
    out = regrid_planar_material_states(p,np.ones(source_count),np.ones(target_count))
    np.testing.assert_array_equal(p,original)
    np.testing.assert_array_equal(out[:,0],p[:,0])
    np.testing.assert_array_equal(out[:,target_count+1],p[:,source_count+1])
    np.testing.assert_allclose(np.cumsum(out[:,1:target_count+1],axis=1),np.repeat(p[:,1:2],target_count,axis=1),atol=2e-15)
    np.testing.assert_allclose(np.cumsum(out[:,target_count+2:],axis=1),np.repeat(p[:,source_count+2:source_count+3],target_count,axis=1),atol=2e-15)


def test_material_transfer_matches_analytic_shape_and_rate_field():
    ns,nt = 14,13
    centers = (np.arange(ns)+.5)/ns
    target = (np.arange(nt)+.5)/nt
    angles,rates = -np.pi+.4*centers,.2+.3*centers
    physical = np.r_[.7,np.diff(np.r_[0.,angles]),.1,np.diff(np.r_[0.,rates])][None,:]
    out = regrid_planar_material_states(physical,np.ones(ns),np.ones(nt))[0]
    np.testing.assert_allclose(np.cumsum(out[1:nt+1]),-np.pi+.4*target,atol=2e-15)
    np.testing.assert_allclose(np.cumsum(out[nt+2:]),.2+.3*target,atol=2e-15)


def test_material_transfer_rejects_invalid_geometry():
    with pytest.raises(ValueError):
        regrid_planar_material_states(np.zeros((2,6)),[1.,0.],[1.])
