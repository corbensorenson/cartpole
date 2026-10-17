import numpy as np
import pytest
from spatialpole import Config, SpatialCartpole
from spatialpole.aligned_value import (AlignedNativeValuePolicy, aligned_next_value,
    relative_value_cost, aligned_value_policy_from_payload)


def test_relative_affine_cost_matches_independent_absolute_costs():
    rng = np.random.default_rng(1491)
    S,r = rng.normal(size=(13,10)),rng.normal(size=13)
    e0,e = rng.normal(size=(2,10))
    d0,d = rng.normal(size=(2,2))
    R = np.array([[2.,.3],[.3,1.]])
    absolute = lambda x,u: np.linalg.norm(S@x+r)**2+u@R@u
    np.testing.assert_allclose(relative_value_cost(S,r,e0,e,R,d0,d),absolute(e,d)-absolute(e0,d0),rtol=2e-14,atol=2e-14)


def make_policy(env,state,sign=1.,force=None):
    nx = 2*env.model.nv
    root = np.zeros((1,nx))
    root[0,0] = 1.
    return AlignedNativeValuePolicy(env,np.array([state,state]),np.zeros((1,2)),
        np.array([np.zeros(2) if force is None else force]),np.zeros((1,2,nx)),[root,root],
        [np.zeros(1),np.array([sign*.002])],np.eye(2)*1e-9,
        dict(reference=state,K=np.zeros((2,nx))))


@pytest.mark.parametrize('sign',[-1.,1.])
def test_aligned_search_nonzero_residual_native_trace_and_live_isolation(sign):
    env = SpatialCartpole(Config(n_links=1))
    state,_ = env.reset(seed=51,options={'mode':'upright','angle_noise':0.})
    env.data.time = 12.34
    env.steps,env.streak,env.best_streak = 23,7,9
    env.min_plane_margin,env.warning_seen = 1.23,True
    before = env.state()
    policy = make_policy(env,state,sign)
    action = policy(state,0)
    np.testing.assert_array_equal(env.state(),before)
    assert (env.data.time,env.native_steps,env.steps,env.streak,env.best_streak,env.min_plane_margin,env.warning_seen)==(12.34,0,23,7,9,1.23,True)
    assert sign*action[0]<0
    row = policy.trace[0]
    records = row['candidates']
    assert 1<=len(records)<=16 and len({c['action'].tobytes() for c in records})==len(records)
    assert policy.metadata['predictor_native_steps']==4*len(records)
    assert records[row['selected_index']]['cost']<=0
    oracle = SpatialCartpole(env.config)
    ref,root,angles = aligned_next_value(oracle,policy.states[1],policy.roots[1],records[0]['successor'])
    np.testing.assert_array_equal(angles,row['anchor_angles'])
    error0 = oracle.difference(records[0]['successor'],ref)
    baseline_value = root@error0+policy.residuals[1]
    baseline_delta = records[0]['action'].astype(float)-policy.actions[0]
    for c in records:
        assert c['action'].dtype==np.float32 and np.linalg.norm(c['action'].astype(float))<=80
        np.testing.assert_array_equal(oracle.transition(state,c['action']),c['successor'])
        value = root@oracle.difference(c['successor'],ref)+policy.residuals[1]
        delta = c['action'].astype(float)-policy.actions[0]
        independent = value@value-baseline_value@baseline_value+delta@policy.R@delta-baseline_delta@policy.R@baseline_delta
        np.testing.assert_allclose(c['cost'],independent,atol=2e-21,rtol=1e-7)
    tape = policy.forecast_tapes
    np.testing.assert_array_equal(tape['native_search_offsets'],[0,len(records)])
    np.testing.assert_array_equal(tape['native_search_selected'],[action])
    assert len(tape['native_search_successor'])==len(records)
    oracle.close()
    policy.close()
    env.close()


def test_candidate_disk_cap_and_invalid_baseline_fallback():
    env = SpatialCartpole(Config(n_links=1))
    state = env.upright_state()
    policy = make_policy(env,state,force=np.array([1e5,-2e5]))
    policy(state,0)
    records = policy.trace[0]['candidates']
    assert len(records)<=16
    assert all(np.linalg.norm(c['action'].astype(float))<=80 for c in records)
    policy.close()
    state[0] = 3.1
    policy = make_policy(env,state)
    action = policy(state,0)
    assert all(not c['valid'] for c in policy.trace[0]['candidates'])
    assert policy.trace[0]['baseline_fallback']
    np.testing.assert_array_equal(action,policy.trace[0]['candidates'][0]['action'])
    policy.close()
    env.close()


def test_rejects_unmatched_averaged_value_roots():
    env = SpatialCartpole(Config(n_links=1))
    with pytest.raises(ValueError,match='matching'):
        aligned_value_policy_from_payload(env,dict(aligned_native_value=True,native_search=True,
            online_gauge=True,affine_value=True,align_gauge=True,reflection_averaged_feedback=True))
    env.close()
