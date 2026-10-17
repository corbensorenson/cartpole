import mujoco as mj
import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.input_derivatives import input_jacobians


def test_input_map_matches_independent_linear_mechanics_held_input_rk4_polynomial():
    e=SpatialCartpole(Config(n_links=1));state=e.upright_state();e.set_state(state)
    nv=e.model.nv;nx=2*nv
    M=np.empty((nv,nv));mj.mj_fullM(e.model,M,e.data.qM)
    mass=e.model.body_mass[e.link_ids[0]];lever=e.model.body_ipos[e.link_ids[0],2]
    gravity=np.diag([0.,0.,mass*9.81*lever,mass*9.81*lever,0.])
    A=np.zeros((nx,nx));A[:nv,nv:]=np.eye(nv)
    A[nv:,:nv]=np.linalg.solve(M,gravity)
    A[nv:,nv:]=-np.linalg.solve(M,np.diag(e.model.dof_damping))
    B=np.zeros((nx,2));B[nv:]=np.linalg.solve(M,np.eye(nv)[:,:2])
    generator=np.zeros((nx+2,nx+2));generator[:nx,:nx]=A;generator[:nx,nx:]=B
    z=generator*e.config.timestep
    polynomial=np.eye(nx+2)+z+z@z/2+z@z@z/6+z@z@z@z/24
    oracle=np.linalg.matrix_power(polynomial,e.config.frame_skip)[:nx,nx:]
    observed,predicted=input_jacobians(e,state[None,:],np.zeros((1,2)),eps=1/32)
    np.testing.assert_allclose(observed[0],oracle,atol=3e-12,rtol=2e-10)
    np.testing.assert_array_equal(predicted[0],state)
    assert e.native_steps==36
    e.close()


@pytest.mark.parametrize('n',[1,3,14])
def test_native_literal_input_stencil_both_axes_owned_data_and_exact_counts(n):
    e=SpatialCartpole(Config(n_links=n));rng=np.random.default_rng(391+n)
    state=e.integrate(e.upright_state(),rng.normal(size=2*e.model.nv)*.2)
    force=np.array([9.,-13.]);eps=1/32
    e.set_state(state);e.data.time=12.75;e.steps=7
    observed,predicted=input_jacobians(e,state[None,:],force[None,:],eps)
    np.testing.assert_array_equal(e.state(),state)
    assert e.data.time==12.75 and e.steps==7 and e.native_steps==36
    oracle=SpatialCartpole(e.config)
    base=oracle.transition(state,force,quantize=False)
    np.testing.assert_array_equal(predicted[0],base)
    for axis in range(2):
        values=[]
        for k in (1.,-1.,2.,-2.):
            u=force.copy();u[axis]+=k*eps
            values.append(oracle.difference(oracle.transition(state,u,quantize=False),base))
        expected=(8*values[0]-8*values[1]-values[2]+values[3])/(12*eps)
        np.testing.assert_allclose(observed[0,:,axis],expected,atol=2e-13,rtol=2e-13)
    oracle.close();e.close()


@pytest.mark.parametrize('sign',[-1.,1.])
def test_input_component_cap_uses_literal_inward_probes(sign):
    e=SpatialCartpole(Config(n_links=2));state=e.upright_state();force=np.array([sign*80.,0.]);eps=1/32
    B,base=input_jacobians(e,state[None,:],force[None,:],eps)
    oracle=SpatialCartpole(e.config);values=[]
    for k in (1.,2.,3.,4.):
        u=force.copy();u[0]-=sign*k*eps
        assert abs(u[0])<=80
        values.append(oracle.difference(oracle.transition(state,u,quantize=False),base[0]))
    expected=-sign*(48*values[0]-36*values[1]+16*values[2]-3*values[3])/(12*eps)
    np.testing.assert_allclose(B[0,:,0],expected,atol=3e-13,rtol=3e-13)
    oracle.close();e.close()
