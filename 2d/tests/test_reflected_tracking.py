import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import axial_aligned_affine_force
from spatialpole.symmetry import reflect_tracking_suffix,reflect_state


@pytest.mark.parametrize('n',[1,3,14])
def test_reflected_suffix_preserves_prefix_and_transports_force_value_native_flow(n):
    env=SpatialCartpole(Config(n_links=n));rng=np.random.default_rng(8800+n)
    nx=2*env.model.nv;ny=nx-n;H=2
    states=np.array([env.integrate(env.upright_state(),rng.normal(size=nx)*.2) for _ in range(H+1)])
    payload=dict(states=states,actions=rng.normal(size=(H,2)),feed=rng.normal(size=(H,2)),
        gains=rng.normal(size=(H,2,nx))*.2,roots=rng.normal(size=(H+1,ny,nx)),
        residuals=rng.normal(size=(H+1,ny)))
    mirrored=reflect_tracking_suffix(env,payload,start=1)
    for key in payload:
        assert mirrored[key][:1].tobytes()==payload[key][:1].tobytes()
    state=env.integrate(states[1],rng.normal(size=nx)*.03)
    reflected=reflect_state(env,state);force_parity=np.array([1.,-1.])
    raw=axial_aligned_affine_force(env,state,states[1],payload['actions'][1],payload['feed'][1],payload['gains'][1])
    mirror_raw=axial_aligned_affine_force(env,reflected,mirrored['states'][1],
        mirrored['actions'][1],mirrored['feed'][1],mirrored['gains'][1])
    np.testing.assert_allclose(mirror_raw,force_parity*raw,atol=2e-14,rtol=2e-14)
    original_value=payload['roots'][1]@env.difference(state,states[1])+payload['residuals'][1]
    mirrored_value=mirrored['roots'][1]@env.difference(reflected,mirrored['states'][1])+mirrored['residuals'][1]
    np.testing.assert_allclose(original_value,mirrored_value,atol=2e-14,rtol=2e-14)
    actual=env.transition(state,env.deliver(raw))
    mirror_actual=env.transition(reflected,env.deliver(mirror_raw))
    np.testing.assert_allclose(mirror_actual,reflect_state(env,actual),atol=3e-13,rtol=3e-13)
    assert env.native_steps==8
    env.close()
