"""Physical checks independent of the mostly planar swing references."""
import mujoco as mj
import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.control import balance_design,WorldBalancePolicy


@pytest.mark.parametrize('n',[1,3,7,8,9,10,11,12])
def test_persistent_native_multiaxis_motion_matches_environment(n):
    env = SpatialCartpole(Config(n_links=n))
    initial,_ = env.reset(seed=0,options={'mode':'hanging','angle_noise':0.})
    data = mj.MjData(env.model)
    data.qpos[:],data.qvel[:] = initial[:env.model.nq],initial[env.model.nq:]
    mj.mj_forward(env.model,data)
    state = initial.copy()
    t = np.arange(100)*env.config.dt
    forces = np.c_[1.2*np.sin(2*np.pi*t/1.4),2.*np.cos(2*np.pi*t/2.3)].astype(np.float32)
    cart,rods = [],[]
    for force in forces:
        data.ctrl[:] = force
        for _ in range(env.config.frame_skip):
            mj.mj_step(env.model,data)
        mj.mj_forward(env.model,data)
        expected = np.r_[data.qpos,data.qvel]
        state = env.transition(state,force)
        np.testing.assert_array_equal(state,expected)
        cart.append(data.qpos[:2].copy())
        rods.append(data.xmat[env.link_ids].reshape(-1,3,3)[:,:2,2].copy())
    for horizontal in (np.array(cart),np.array(rods).reshape(-1,2)):
        singular = np.linalg.svd(horizontal-horizontal.mean(axis=0),compute_uv=False)
        assert singular[1]/singular[0]>.02
    env.close()


@pytest.mark.parametrize('n',[1,3,7,8,9,10,11,12])
def test_force_response_has_no_preferred_horizontal_axis(n):
    env = SpatialCartpole(Config(n_links=n))
    initial,_ = env.reset(seed=0,options={'mode':'hanging','angle_noise':0.})
    outputs = []
    for force in ([10.,0.],[0.,10.]):
        state = initial.copy()
        for _ in range(25):
            state = env.transition(state,force)
        outputs.append((state.copy(),env.data.xmat[env.link_ids].reshape(-1,3,3)[:,:,2].copy()))
    (x,dx),(y,dy) = outputs
    np.testing.assert_allclose(x[0],y[1],atol=1e-12)
    np.testing.assert_allclose(dy,np.c_[-dx[:,1],dx[:,0],dx[:,2]],atol=1e-11)
    assert x[0]>.1 and y[1]>.1 and np.max(np.abs(dy[:,1]))>.02
    env.close()


@pytest.mark.parametrize('n',[1,3,7,8,9,10,11,12])
def test_every_ball_joint_retains_both_transverse_directions_and_spin(n):
    env = SpatialCartpole(Config(n_links=n))
    initial,_ = env.reset(seed=0,options={'mode':'hanging','angle_noise':0.})
    for joint in range(n):
        for axis in (0,1):
            increment = np.zeros(2*env.model.nv)
            increment[2+3*joint+axis] = .12
            env.transition(env.integrate(initial,increment),[0.,0.])
            direction = env.data.xmat[env.link_ids[joint]].reshape(3,3)[:,2]
            assert np.linalg.norm(direction[:2])>.05
    spinning = initial.copy()
    spinning[env.model.nq+4+3*(n-1)] = .5
    moved = env.transition(spinning,[0.,0.])
    assert abs(env.difference(moved,spinning)[4+3*(n-1)])>.001
    env.close()


@pytest.mark.parametrize('n',[3,11,12])
def test_axis_separated_upright_design_rotates_with_physical_world(n):
    env = SpatialCartpole(Config(n_links=n))
    design = balance_design(env,axis_separated=True,value_backend='finite_qr' if n==12 else 'dare')
    assert design['closed_loop_radius']<1.
    rng = np.random.default_rng(418)
    state = env.integrate(env.upright_state(),rng.normal(size=2*env.model.nv)*1e-6)
    rotated = state.copy()
    rotated[:2] = [-state[1],state[0]]
    rotated[env.model.nq:env.model.nq+2] = [-state[env.model.nq+1],state[env.model.nq]]
    q = np.array([np.sqrt(.5),0.,0.,np.sqrt(.5)])
    mj.mju_mulQuat(rotated[2:6],q,state[2:6])
    # The root's local angular velocity stays the same under world rotation;
    # every relative joint state, including free axial spin, is also retained.
    policy = WorldBalancePolicy(env,design)
    force,turned = policy(state),policy(rotated)
    np.testing.assert_allclose(turned,[-force[1],force[0]],atol=1e-6)
    ix = np.array([0]+[3+3*j for j in range(n)])
    ix = np.r_[ix,ix+env.model.nv]
    if 'value_factor' in design:
        assert np.linalg.svd(design['value_factor'][:,ix],compute_uv=False)[-1]>0.
        np.testing.assert_allclose(design['P'],design['value_factor'].T@design['value_factor'],rtol=2e-13,atol=1e-10)
    else:
        assert np.linalg.eigvalsh(design['P'][np.ix_(ix,ix)])[0]>0.
    env.close()


def test_finite_qr_and_dare_agree_on_well_conditioned_native_model():
    env = SpatialCartpole(Config(n_links=3))
    dare = balance_design(env,axis_separated=True)
    qr = balance_design(env,axis_separated=True,value_backend='finite_qr')
    np.testing.assert_allclose(qr['K'],dare['K'],rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(qr['P'],dare['P'],rtol=1e-7,atol=1e-7)
    env.close()
