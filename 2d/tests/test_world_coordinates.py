import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.world_coordinates import (native_geometry,world_coordinate_error,
    world_coordinate_jacobian,physical_tracking_factor,ParkedWorldCoordinatePolicy)
from spatialpole.quotient import axial_quotient_frame
from spatialpole.conditioning import axial_gauge_state


@pytest.mark.parametrize('n',[1,3,14])
def test_physical_cost_matches_native_feature_secants_and_reflection(n):
    env=SpatialCartpole(Config(n_links=n));rng=np.random.default_rng(1100+n)
    nx,nv=2*env.model.nv,env.model.nv
    displacement=np.zeros(nx)
    # Reflection-fixed moving state, with cumulative tilts and angular rates.
    displacement[0]=.4
    displacement[3:nv:3]=rng.normal(size=n)*.2
    displacement[nv]=.3
    displacement[nv+3:2*nv:3]=rng.normal(size=n)*.4
    reference=env.integrate(env.upright_state(),displacement)
    R,w=native_geometry(env,reference)
    L=physical_tracking_factor(env,reference)
    weights=np.r_[[4.,4.],np.full(2*n,40.),[.05,.05],np.tile([.05,.05,.005],n)]
    direction=rng.normal(size=nx);epsilon=1e-6
    plus=world_coordinate_error(env,env.integrate(reference,epsilon*direction),reference,R,w)
    minus=world_coordinate_error(env,env.integrate(reference,-epsilon*direction),reference,R,w)
    secant=(plus-minus)/(2*epsilon)
    np.testing.assert_allclose(np.linalg.norm(L@direction)**2,np.dot(weights*secant,secant),rtol=2e-9,atol=2e-9)
    frame=axial_quotient_frame(env,reference,planar=True)
    cost=L@frame['E'];G=cost.T@cost;signs=frame['signs']
    np.testing.assert_allclose(G*signs[:,None]*signs[None,:],G,atol=2e-12)
    assert np.linalg.matrix_rank(L[:,nv:])==nv
    assert env.native_steps==0
    env.close()


@pytest.mark.parametrize('n',[1,3,14])
def test_native_physical_feature_jacobian_gauge_kernel_and_all_velocities(n):
    env=SpatialCartpole(Config(n_links=n));rng=np.random.default_rng(402+n)
    nx,nv=2*env.model.nv,env.model.nv
    reference=env.integrate(env.upright_state(),rng.normal(size=nx)*.3)
    R,w=native_geometry(env,reference)
    np.testing.assert_array_equal(world_coordinate_error(env,reference,reference,R,w),np.zeros(4+5*n))
    before=env.state()
    J=world_coordinate_jacobian(env,reference)
    np.testing.assert_array_equal(env.state(),before)
    f=axial_quotient_frame(env,reference)
    np.testing.assert_allclose(J@f['D'],0.,atol=3e-14)
    assert np.linalg.matrix_rank(J)==nx-n
    assert np.linalg.matrix_rank(J[:,nv:])==nv
    T=J@f['E']
    np.testing.assert_allclose(np.linalg.solve(T,J),f['F'],atol=4e-14,rtol=4e-14)
    eps=1e-6
    independent=np.empty_like(J)
    for k in range(nx):
        direction=np.eye(nx)[k]*eps
        minus=world_coordinate_error(env,env.integrate(reference,-direction),reference,R,w)
        plus=world_coordinate_error(env,env.integrate(reference,direction),reference,R,w)
        independent[:,k]=(plus-minus)/(2*eps)
    np.testing.assert_allclose(J,independent,atol=5e-9,rtol=5e-9)
    state=env.integrate(reference,rng.normal(size=nx)*.15)
    actual=world_coordinate_error(env,state,reference,R,w)
    changed=axial_gauge_state(env,state,rng.uniform(-2.,2.,n))
    np.testing.assert_allclose(world_coordinate_error(env,changed,reference,R,w),actual,atol=4e-14,rtol=4e-14)
    assert env.native_steps==0
    env.close()


def test_physical_spin_and_cart_velocity_are_observed_and_antipode_is_not_zero():
    env=SpatialCartpole(Config(n_links=3));reference=env.upright_state();R,w=native_geometry(env,reference)
    state=reference.copy();state[env.model.nq:env.model.nq+2]=[.2,-.3]
    state[env.model.nq+4]=.4
    error=world_coordinate_error(env,state,reference,R,w)
    np.testing.assert_allclose(error[8:10],[.2,-.3])
    np.testing.assert_allclose(error[10:].reshape(3,3),np.tile([0.,0.,.4],(3,1)),atol=1e-15)
    state=reference.copy();state[2:6]=[0.,0.,1.,0.]
    error=world_coordinate_error(env,state,reference,R,w)
    np.testing.assert_array_equal(error[2:4],[np.pi,0.])
    assert np.linalg.norm(error[2:8])>=np.pi
    env.close()


def test_time_varying_physical_transport_preserves_affine_models_forces_values_and_costs():
    env=SpatialCartpole(Config(n_links=3));rng=np.random.default_rng(9001);nx=2*env.model.nv;ny=nx-3
    states=[env.integrate(env.upright_state(),rng.normal(size=nx)*.2) for _ in range(2)]
    frames=[axial_quotient_frame(env,s) for s in states]
    J=[world_coordinate_jacobian(env,s) for s in states]
    T=[j@f['E'] for j,f in zip(J,frames)]
    inverse=np.linalg.inv(T[0])
    A,B,d=rng.normal(size=(ny,ny)),rng.normal(size=(ny,2)),rng.normal(size=ny)
    K,S,L=rng.normal(size=(2,ny)),rng.normal(size=(12,ny)),rng.normal(size=(17,ny))
    residual,feed=rng.normal(size=12),rng.normal(size=2)
    y,u=rng.normal(size=ny),rng.normal(size=2);z=T[0]@y
    predicted=(T[1]@A@inverse)@z+(T[1]@B)@u+T[1]@d
    np.testing.assert_allclose(predicted,T[1]@(A@y+B@u+d),atol=3e-13,rtol=3e-13)
    np.testing.assert_allclose(feed-(K@inverse)@z,feed-K@y,atol=4e-14)
    np.testing.assert_allclose((S@inverse)@z+residual,S@y+residual,atol=4e-14)
    np.testing.assert_allclose((L@inverse)@z,L@y,atol=4e-14)
    np.testing.assert_allclose(K@inverse@J[0],K@frames[0]['F'],atol=4e-14)
    env.close()


def test_physical_feedback_owns_geometry_and_preserves_actual_state_clock_counters():
    env=SpatialCartpole(Config(n_links=1));reference=env.upright_state()
    scratch=SpatialCartpole(env.config);R,w=native_geometry(scratch,reference);scratch.close()
    payload=dict(world_coordinate_feedback=True,park_steps=1499,states=np.array([reference,reference]),
        actions=np.array([[1.,-2.]]),feed=np.array([[.1,.2]]),swing_steps=1,
        parking_reference=reference,parking_K=np.zeros((2,10)),world_gains=np.zeros((1,2,9)),
        world_reference_rotations=np.array([R,R]),world_reference_omega=np.array([w,w]))
    payload['world_gains'][0,0,0]=2.
    policy=ParkedWorldCoordinatePolicy(env,payload)
    payload['world_gains'][0,0,0]=999.  # external aliases cannot change frozen feedback
    state=reference.copy();state[0]=.1
    env.set_state(state);env.data.time=22.25;env.steps=11;env.streak=3;env.best_streak=5;env.min_plane_margin=1.2;env.warning_seen=True
    force=policy(state,1499)
    np.testing.assert_allclose(force,[.9,-1.8],atol=2e-16)
    np.testing.assert_array_equal(env.state(),state)
    assert (env.data.time,env.steps,env.native_steps,env.streak,env.best_streak,env.min_plane_margin,env.warning_seen)==(22.25,11,0,3,5,1.2,True)
    assert policy.scratch.native_steps==0 and policy.metadata['coordinate_forward_queries']==2
    policy.close();env.close()
