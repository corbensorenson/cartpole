import numpy as np
import pytest
from spatialpole import Config, SpatialCartpole
from spatialpole.affine_value import affine_tracking_values, AffineValuePolicy,axial_aligned_affine_force
from spatialpole.conditioning import axial_gauge_state,axial_gauge_route


def test_affine_qr_matches_independent_condensed_problem():
    rng = np.random.default_rng(972)
    H,nx,nu = 5,4,2
    models = [(rng.normal(size=(nx,nx))*.3,rng.normal(size=(nx,nu))) for _ in range(H)]
    defects = rng.normal(size=(H,nx))*.4
    factors = [np.diag(rng.uniform(.2,2,nx)) for _ in range(H)]
    terminal = rng.normal(size=(nx+2,nx))
    terminal_r = rng.normal(size=nx+2)
    R = np.diag([.3,.7])
    feed,K,roots,residuals = affine_tracking_values(models,defects,factors,R,terminal,terminal_r)
    initial = rng.normal(size=nx)
    # Independent dense condensation of all physical errors, retaining all
    # defect and terminal affine terms. Solve the full open-loop action vector.
    influence,bias = np.zeros((nx,H*nu)),initial.copy()
    M,r = [],[]
    for t,(A,B) in enumerate(models):
        M.append(factors[t]@influence)
        r.append(factors[t]@bias)
        influence,bias = A@influence,A@bias+defects[t]
        influence[:,t*nu:(t+1)*nu] += B
    M.append(terminal@influence)
    r.append(terminal@bias+terminal_r)
    M.append(np.kron(np.eye(H),np.linalg.cholesky(R).T))
    r.append(np.zeros(H*nu))
    optimum = np.linalg.lstsq(np.vstack(M),-np.concatenate(r),rcond=None)[0].reshape(H,nu)
    state,actions = initial.copy(),[]
    for t,(A,B) in enumerate(models):
        action = feed[t]-K[t]@state
        actions.append(action)
        state = A@state+B@action+defects[t]
    np.testing.assert_allclose(actions,optimum,rtol=2e-12,atol=2e-12)
    # Value differences are independent of the discarded scalar constant.
    assert np.linalg.norm(feed)>0.1
    assert np.linalg.norm(residuals[0])>0.1


@pytest.mark.parametrize('sign',[-1.,1.])
def test_search_retains_affine_residual_and_owns_native_state(sign):
    env = SpatialCartpole(Config(n_links=1))
    state,_ = env.reset(seed=51,options={'mode':'upright','angle_noise':0.})
    nx = 2*env.model.nv
    root = np.zeros((1,nx))
    root[0,0] = 1.
    balance = dict(reference=state,K=np.zeros((2,nx)))
    policy = AffineValuePolicy(env,np.array([state,state]),np.zeros((1,2)),
        np.zeros((1,2)),np.zeros((1,2,nx)),[root,root],
        [np.zeros(1),np.array([sign*.002])],np.eye(2)*1e-9,balance,search=True)
    before = env.state().copy()
    time,native = env.data.time,env.native_steps
    action = policy(state,0)
    np.testing.assert_array_equal(env.state(),before)
    assert env.data.time==time and env.native_steps==native
    assert action.dtype==np.float32 and np.linalg.norm(action.astype(float))<=80
    assert sign*action[0]<0
    candidates = policy.trace[0]['candidates']
    assert 1<=len(candidates)<=16
    assert len({c['action'].tobytes() for c in candidates})==len(candidates)
    assert policy.metadata['predictor_native_steps']==4*len(candidates)
    for c in candidates:
        successor = env.transition(state,c['action'])
        np.testing.assert_array_equal(successor,c['successor'])
    chosen = next(c for c in candidates if np.array_equal(c['action'],action))
    assert chosen['cost']<=candidates[0]['cost']
    policy.close()
    env.close()


def test_axial_gauge_transports_full_affine_value_and_force_law():
    env = SpatialCartpole(Config(n_links=3))
    rng = np.random.default_rng(995)
    nx = 2*env.model.nv
    state = env.integrate(env.upright_state(),rng.normal(size=nx)*.2)
    reference = env.integrate(env.upright_state(),rng.normal(size=nx)*.1)
    angles = rng.uniform(-2,2,3)
    transformed_state = axial_gauge_state(env,state,angles)
    K,S = rng.normal(size=(1,2,nx)),rng.normal(size=(1,nx,nx))
    refs,newK = axial_gauge_route(env,np.array([reference]),K,angles)
    _,newS = axial_gauge_route(env,np.array([reference]),S,angles)
    feed,residual = rng.normal(size=2),rng.normal(size=nx)
    old_error,new_error = env.difference(state,reference),env.difference(transformed_state,refs[0])
    np.testing.assert_allclose(feed-K[0]@old_error,feed-newK[0]@new_error,atol=2e-13)
    np.testing.assert_allclose(S[0]@old_error+residual,newS[0]@new_error+residual,atol=2e-13)
    env.close()


@pytest.mark.parametrize('n',[1,3,11,12])
def test_online_axial_force_is_invariant_to_observed_passive_frames(n):
    env = SpatialCartpole(Config(n_links=n))
    rng = np.random.default_rng(999)
    nx = 2*env.model.nv
    reference = env.integrate(env.upright_state(),rng.normal(size=nx)*.1)
    state = env.integrate(reference,rng.normal(size=nx)*.15)
    gain,action,feed = rng.normal(size=(2,nx)),rng.normal(size=2),rng.normal(size=2)
    original = axial_aligned_affine_force(env,state,reference,action,feed,gain)
    native_before = env.native_steps
    transformed = axial_gauge_state(env,state,rng.uniform(-2.,2.,n))
    result = axial_aligned_affine_force(env,transformed,reference,action,feed,gain)
    np.testing.assert_allclose(result,original,atol=5e-13,rtol=5e-13)
    np.testing.assert_array_equal(env.state(),transformed)
    assert env.native_steps==native_before
    env.close()
