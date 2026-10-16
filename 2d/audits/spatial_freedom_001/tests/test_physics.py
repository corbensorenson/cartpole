import mujoco as mj
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from spatialpole import Config, SpatialCartpole
from spatialpole.control import balance_design, BalancePolicy, WorldBalancePolicy
from spatialpole.control import linearize, linearize_many, cost_derivatives, state_cost,bounded_action_step
from spatialpole.physical import world_errors, world_cost_derivatives,direction_residuals
from spatialpole.transfer import lift_planar_state
from spatialpole.conditioning import axial_gauge_state,axial_gauge_route


@pytest.mark.parametrize("n", [1, 2, 5, 7])
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


def test_physical_terminal_error_distinguishes_hanging_from_upright():
    e = SpatialCartpole()
    policy = WorldBalancePolicy(e,{})
    x,_ = e.reset(seed=1,options={"mode":"hanging","angle_noise":0.})
    assert np.linalg.norm(policy.error(x)[2:4]) == pytest.approx(np.pi)
    assert np.linalg.norm(policy.error(e.upright_state())[2:4]) == 0


def test_cart_armature_is_explicit_and_zero_in_the_benchmark():
    e = SpatialCartpole(Config(n_links=4))
    np.testing.assert_array_equal(e.model.dof_armature[:2],0.)
    altered = SpatialCartpole(Config(n_links=4,cart_armature=0.0005))
    np.testing.assert_array_equal(altered.model.dof_armature[:2],0.0005)
    with pytest.raises(ValueError):
        Config(cart_armature=np.nan)


@pytest.mark.parametrize("n",[1,3,6])
def test_batched_physical_kinematics_match_native(n):
    e = SpatialCartpole(Config(n_links=n))
    rng = np.random.default_rng(764)
    states = np.array([e.integrate(e.upright_state(),rng.normal(size=2*e.model.nv)) for _ in range(8)])
    policy = WorldBalancePolicy(e,{})
    expected = np.array([policy.error(x) for x in states])
    np.testing.assert_allclose(world_errors(e,states),expected,atol=1e-10,rtol=1e-10)


def test_world_cost_derivatives_against_native_directional_difference():
    e = SpatialCartpole(Config(n_links=3))
    rng = np.random.default_rng(870)
    x = e.integrate(e.upright_state(),rng.normal(size=2*e.model.nv)*0.5)
    W = state_cost(e)
    grads,hess = world_cost_derivatives(e,np.array([x]),W)
    d = rng.normal(size=2*e.model.nv)
    native = WorldBalancePolicy(e,{})
    hi,lo = native.error(e.integrate(x,d*1e-6)),native.error(e.integrate(x,-d*1e-6))
    actual = (hi@W@hi-lo@W@lo)/4e-6
    np.testing.assert_allclose(grads[0]@d,actual,atol=1e-5,rtol=1e-6)
    assert np.min(np.linalg.eigvalsh(hess[0])) > -1e-8


def test_native_derivative_worker_parity():
    a,b = SpatialCartpole(Config(n_links=3)),SpatialCartpole(Config(n_links=3))
    b.derivative_workers = 4
    rng = np.random.default_rng(133)
    states = np.array([a.integrate(a.upright_state(),rng.normal(size=2*a.model.nv)*0.3) for _ in range(6)])
    actions = rng.uniform(-15,15,(6,2))
    one,four = linearize_many(a,states,actions),linearize_many(b,states,actions)
    for pair1,pair4 in zip(one,four):
        for x,y in zip(pair1,pair4):
            np.testing.assert_array_equal(x,y)
    assert a.native_steps == b.native_steps
    a.close()
    b.close()


def test_circular_action_qp_kkt_and_feedback_derivative():
    u = np.array([0.9,0.1])
    g = np.array([3.,-2.])
    H = np.array([[3.,0.4],[0.4,1.5]])
    cross = np.array([[0.3,-0.8],[0.4,1.1]])
    k,K,active = bounded_action_step(u,g,H,cross,1.)
    v,gradient = u+k,g+H@k
    multiplier = -gradient@v/(v@v)
    assert active and multiplier > 0
    assert np.linalg.norm(v) <= 1+1e-12
    np.testing.assert_allclose(gradient+multiplier*v,0.,atol=1e-10)
    for j in range(2):
        hi = bounded_action_step(u,g+cross[:,j]*1e-5,H,cross,1.)[0]
        lo = bounded_action_step(u,g-cross[:,j]*1e-5,H,cross,1.)[0]
        np.testing.assert_allclose((hi-lo)/2e-5,-K[:,j],atol=1e-7)


def test_force_derivative_uses_interior_secant_at_axis_bound():
    e = SpatialCartpole()
    x = e.upright_state()
    A,B = linearize(e,x,np.array([80.,0.]))
    _,near = linearize(e,x,np.array([79.999,0.]))
    np.testing.assert_allclose(B,near,atol=1e-7,rtol=1e-5)
    fast = linearize_many(e,np.array([x]),np.array([[80.,0.]]))[0]
    np.testing.assert_allclose(B,fast[1],atol=1e-9)
    e.close()


def test_direction_running_cost_is_continuous_across_joint_pi():
    e = SpatialCartpole(Config(n_links=2))
    states = np.array([lift_planar_state(e,[0,np.pi+d,1.],np.zeros(3)) for d in (-1e-7,1e-7)])
    residuals = direction_residuals(e,states)
    assert abs(np.sum(residuals[0]**2)-np.sum(residuals[1]**2)) < 1e-5
    for x,r in zip(states,residuals):
        e.set_state(x)
        directions = e.data.xmat[e.link_ids].reshape(-1,3,3)[:,:,2]
        np.testing.assert_allclose(r[2:e.model.nv].reshape(-1,3),directions-[0,0,1],atol=1e-12)


@pytest.mark.parametrize('n',[1,3,6])
def test_axial_gauge_preserves_native_spatial_physics_and_feedback(n):
    e = SpatialCartpole(Config(n_links=n))
    rng = np.random.default_rng(852)
    x = e.integrate(e.upright_state(),rng.normal(size=2*e.model.nv))
    ref = e.integrate(e.upright_state(),rng.normal(size=2*e.model.nv)*0.3)
    angles = rng.uniform(-2.,2.,n)
    gx,gr = axial_gauge_state(e,x,angles),axial_gauge_state(e,ref,angles)
    expected = axial_gauge_state(e,e.transition(x,[13.,-20.],quantize=False),angles)
    actual = e.transition(gx,[13.,-20.],quantize=False)
    np.testing.assert_allclose(e.difference(actual,expected),0.,atol=1e-10)
    K = rng.normal(size=(1,2,2*e.model.nv))
    _,newK = axial_gauge_route(e,np.array([ref]),K,angles)
    np.testing.assert_allclose(K[0]@e.difference(x,ref),newK[0]@e.difference(gx,gr),atol=1e-10)
    e.close()
