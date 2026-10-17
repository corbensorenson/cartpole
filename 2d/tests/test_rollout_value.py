import math
import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import axial_aligned_affine_force
from spatialpole.aligned_value import aligned_next_value
from spatialpole.rollout_value import NativeRolloutValuePolicy


def policy_for(env,state,sign=1.,general=False):
    nx,H = 2*env.model.nv,7
    rng = np.random.default_rng(701)
    gain = rng.normal(size=(H,2,nx))*.2 if general else np.zeros((H,2,nx))
    feed = rng.normal(size=(H,2))*.4 if general else np.zeros((H,2))
    root = np.diag(rng.uniform(.1,2,nx)) if general else np.eye(1,nx)
    residual = rng.normal(size=len(root))*.1 if general else np.array([sign*.002])
    L = np.diag(rng.uniform(.1,1,nx)) if general else np.zeros((nx,nx))
    ref = env.upright_state() if general else state
    return NativeRolloutValuePolicy(env,np.repeat(ref[None,:],H+1,axis=0),np.zeros((H,2)),
        feed,gain,np.repeat(root[None,:,:],H+1,axis=0),np.repeat(residual[None,:],H+1,axis=0),
        np.eye(2)*(.03 if general else 1e-10),dict(reference=ref,K=np.zeros((2,nx))),
        stage_factors=np.repeat(L[None,:,:],H+1,axis=0),horizon=4)


@pytest.mark.parametrize('n',[1,3])
def test_rollout_cost_all_future_feedback_native_paths_and_live_isolation(n):
    env = SpatialCartpole(Config(n_links=n))
    nx = 2*env.model.nv
    state = env.integrate(env.upright_state(),np.random.default_rng(27).normal(size=nx)*.02)
    env.set_state(state)
    env.data.time=17.25
    env.steps,env.streak,env.best_streak,env.min_plane_margin,env.warning_seen=11,3,5,1.2,True
    policy = policy_for(env,state,general=True)
    selected = policy(state,0)
    np.testing.assert_array_equal(env.state(),state)
    assert (env.data.time,env.steps,env.native_steps,env.streak,env.best_streak,env.min_plane_margin,env.warning_seen)==(17.25,11,0,3,5,1.2,True)
    records = policy.trace[0]['candidates']
    assert 1<=len(records)<=16 and len({r['actions'][0].tobytes() for r in records})==len(records)
    assert all(r['used_steps']==4 for r in records)
    assert policy.metadata['predictor_native_steps']==16*len(records)==sum(r['native_steps'] for r in records)
    oracle = SpatialCartpole(env.config)
    anchors=[]
    for j in range(1,5):
        factor = policy.roots[j] if j==4 else policy.stage_factors[j]
        ref,root,angles = aligned_next_value(oracle,policy.states[j],factor,records[0]['states'][j])
        residual = policy.residuals[j] if j==4 else np.zeros(len(root))
        anchors.append((ref,root,residual))
        np.testing.assert_array_equal(angles,policy.trace[0]['anchor_angles'][j-1])
    absolute=[]
    for record in records:
        costs=[]
        for j in range(4):
            if j:
                expected = axial_aligned_affine_force(oracle,record['states'][j],policy.states[j],
                    policy.actions[j],policy.feed[j],policy.gains[j])
                np.testing.assert_array_equal(expected,record['raw'][j])
                np.testing.assert_array_equal(oracle.deliver(expected),record['actions'][j])
            assert np.linalg.norm(record['actions'][j].astype(float))<=80
            oracle.min_plane_margin,oracle.warning_seen=env.config.plane_limit,False
            successor = oracle.transition(record['states'][j],record['actions'][j])
            np.testing.assert_array_equal(successor,record['states'][j+1])
            assert oracle.min_plane_margin==record['plane_margins'][j]
            assert oracle.warning_seen==record['physics_warnings'][j]
            ref,root,residual = anchors[j]
            value = root@oracle.difference(successor,ref)+residual
            du = record['actions'][j].astype(float)-policy.actions[j]
            costs.extend([float(value@value),float(du@policy.R@du)])
        absolute.append(math.fsum(costs))
    for cost,record in zip(absolute,records):
        np.testing.assert_allclose(record['cost'],cost-absolute[0],atol=3e-15,rtol=3e-10)
        np.testing.assert_allclose(record['cost'],math.fsum(record['state_cost_differences'].tolist()+record['input_cost_differences'].tolist()),atol=0,rtol=0)
    chosen = policy.trace[0]['selected_index']
    assert records[chosen]['cost']<=0
    np.testing.assert_array_equal(selected,records[chosen]['actions'][0])
    np.testing.assert_array_equal(oracle.transition(state,selected),records[chosen]['states'][1])
    assert len(policy.trace)==1  # future feedback did not invoke recursive search
    tape=policy.forecast_tapes
    np.testing.assert_array_equal(tape['native_rollout_offsets'],[0,len(records)])
    assert tape['native_rollout_states'].shape==(len(records),5,len(state))
    oracle.close();policy.close();env.close()


@pytest.mark.parametrize('sign',[-1.,1.])
def test_four_command_terminal_affine_residual_changes_first_force(sign):
    env=SpatialCartpole(Config(n_links=1));state=env.upright_state()
    p=policy_for(env,state,sign)
    action=p(state,0)
    assert sign*action[0]<0
    p.close();env.close()


def test_exit_at_later_forecast_step_rejected_and_partial_queries_counted():
    env=SpatialCartpole(Config(n_links=1));state=env.upright_state()
    state[0],state[env.model.nq]=2.98,.4
    p=policy_for(env,state)
    action=p(state,0)
    row=p.trace[0];records=row['candidates']
    assert row['baseline_fallback'] and row['selected_index']==-1
    assert all(not r['valid'] for r in records)
    assert records[0]['used_steps']==4
    assert any(0<r['used_steps']<4 and np.any(r['plane_margins']<0) for r in records[1:])
    assert p.metadata['predictor_native_steps']==sum(r['native_steps'] for r in records)
    assert all(r['native_steps']==4*r['used_steps'] for r in records)
    np.testing.assert_array_equal(action,records[0]['actions'][0])
    p.close();env.close()


def test_injected_intermediate_warning_invalidates_finite_baseline(monkeypatch):
    env=SpatialCartpole(Config(n_links=1));state=env.upright_state();p=policy_for(env,state)
    original=p.predictor.transition
    count=0
    def marked_transition(x,u):
        nonlocal count
        result=original(x,u);count+=1
        if count==2:
            p.predictor.warning_seen=True
        return result
    monkeypatch.setattr(p.predictor,'transition',marked_transition)
    p(state,0)
    row=p.trace[0];records=row['candidates']
    assert records[0]['physics_warnings'][1] and not records[0]['valid']
    assert row['selected_index']>0 and records[row['selected_index']]['valid']
    assert not np.any(records[row['selected_index']]['physics_warnings'])
    p.close();env.close()


def test_all_future_force_deliveries_respect_joint_disk_and_dedup():
    env=SpatialCartpole(Config(n_links=1));state=env.upright_state();p=policy_for(env,state)
    p.feed[:]=[1e5,-2e5]
    p(state,0)
    records=p.trace[0]['candidates']
    assert 1<=len(records)<=16 and len({r['actions'][0].tobytes() for r in records})==len(records)
    maximum=0.
    for r in records:
        forces=r['actions'][:r['used_steps']]
        maximum=max(maximum,float(np.max(np.linalg.norm(forces.astype(float),axis=1))))
    assert 79.999<maximum<=80.
    assert p.metadata['predictor_native_steps']<=16*4*4
    p.close();env.close()
