import json
from dataclasses import asdict
import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.controller_bank import ControllerBankPolicy


def payload_for(env,forces):
    H=env.config.horizon-1;nx=2*env.model.nv;ref=env.upright_state()
    states=np.repeat(ref[None,:],H+1,axis=0)
    count=len(forces)
    tables=dict(states=np.repeat(states[None,:,:],count,axis=0),
        actions=np.array([np.tile([force,0.],(H,1)) for force in forces]),
        feed=np.zeros((count,H,2)),gains=np.zeros((count,H,2,nx)),
        roots=np.zeros((count,H+1,nx,nx)),residuals=np.zeros((count,H+1,nx)))
    base={k:v[0].copy() for k,v in tables.items()}
    return dict(base,**{'bank_'+k:v for k,v in tables.items()},controller_bank=True,
        config=json.dumps(asdict(env.config)),park_steps=1,swing_steps=1,
        affine_value=True,native_search=False,align_gauge=True,online_gauge=True,
        parking_reference=ref,parking_K=np.zeros((2,nx)),reference=ref,K=np.zeros((2,nx)),
        P=np.eye(nx),Q=np.eye(nx),R=np.eye(2),tracking_R=np.eye(2))


def advance(env,policy,state,step):
    raw=policy(state,step);action=env.deliver(raw)
    state,_,terminated,truncated,info=env.step_force(action)
    policy.observe_transition(state,step,action)
    return state,info,terminated or truncated


@pytest.mark.parametrize('forces,selected,predictor_steps',[
    ([10.,0.,-10.],1,16),([10.,11.,12.,13.,0.],4,28)])
def test_first_failed_then_passing_complete_forecast_and_all_successors(forces,selected,predictor_steps):
    env=SpatialCartpole(Config(n_links=1,episode_seconds=.08,sustain_seconds=.04,plane_limit=.001))
    p=payload_for(env,forces);policy=ControllerBankPolicy(env,p)
    state,_=env.reset(options=dict(mode='upright',angle_noise=0.,velocity_noise=0.))
    # Use exact stationary state rather than reset's independently noisy velocity.
    state=env.upright_state();env.set_state(state)
    for step in range(env.config.horizon):
        state,info,_=advance(env,policy,state,step)
    assert info['success'] and policy.selected==selected and policy.metadata['launch_admitted']
    assert len(policy.metadata['launch_forecasts'])==selected+1
    assert policy.metadata['predictor_native_steps']==predictor_steps
    assert policy.metadata['verified_selected_transitions']==3
    assert policy.metadata['final_selected_transition_verified']
    assert env.native_steps==16
    policy.close();env.close()


def test_forecast_owns_data_clocks_counters_and_profile_arrays():
    env=SpatialCartpole(Config(n_links=1,episode_seconds=.08,sustain_seconds=.04))
    p=payload_for(env,[0.,0.,0.]);policy=ControllerBankPolicy(env,p)
    p['bank_actions'][:]=100.;p['actions'][:]=100.
    state=env.upright_state();env.set_state(state);env.data.time=2.75;env.steps=1
    env.streak=1;env.best_streak=1;env.min_plane_margin=2.9
    raw=policy(state,1)
    np.testing.assert_array_equal(raw,np.zeros(2));np.testing.assert_array_equal(env.state(),state)
    assert (env.data.time,env.steps,env.native_steps,env.streak,env.best_streak,env.min_plane_margin)==(2.75,1,0,1,1,2.9)
    assert policy.metadata['predictor_native_steps']==12
    with pytest.raises(ValueError):
        policy.tables['actions'][0,0,0]=1.
    with pytest.raises(RuntimeError,match='observer'):
        policy(state,2)
    policy.close();env.close()


def test_final_transition_mismatch_is_detected_without_another_policy_call():
    env=SpatialCartpole(Config(n_links=1,episode_seconds=.04,sustain_seconds=.02))
    policy=ControllerBankPolicy(env,payload_for(env,[0.,0.,0.]))
    state=env.upright_state();env.set_state(state);env.steps=1
    raw=policy(state,1);action=env.deliver(raw)
    actual,*_=env.step_force(action)
    actual[0]=np.nextafter(actual[0],np.inf)
    with pytest.raises(RuntimeError,match='byte mismatch'):
        policy.observe_transition(actual,1,action)
    assert not policy.metadata.get('final_selected_transition_verified',False)
    policy.close();env.close()


def test_exhausted_bank_is_explicit_and_original_failure_reproduces():
    env=SpatialCartpole(Config(n_links=1,episode_seconds=.08,sustain_seconds=.04,plane_limit=.001))
    policy=ControllerBankPolicy(env,payload_for(env,[10.,11.,12.]))
    state=env.upright_state();env.set_state(state)
    state,_,_=advance(env,policy,state,0)
    state,info,done=advance(env,policy,state,1)
    assert done and not info['success'] and not policy.metadata['launch_admitted']
    assert policy.selected==0 and len(policy.metadata['launch_forecasts'])==3
    assert policy.metadata['predictor_native_steps']==12
    policy.close();env.close()


@pytest.mark.parametrize('flag',['native_search','online_parking','align_cart_origin','world_coordinate_feedback'])
def test_incompatible_policy_families_are_rejected(flag):
    env=SpatialCartpole(Config(n_links=1,episode_seconds=.08,sustain_seconds=.04))
    p=payload_for(env,[0.,0.,0.]);p[flag]=True
    with pytest.raises(ValueError):
        ControllerBankPolicy(env,p)
    env.close()
