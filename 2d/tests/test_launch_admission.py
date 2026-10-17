import numpy as np
import pytest
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import affine_policy_from_payload


def payload(env,profiles):
    state,nx,H = env.upright_state(),2*env.model.nv,env.config.horizon
    gains = np.zeros((H,2,nx))
    gains[:,0,0] = gains[:,1,1] = 10000.
    return dict(states=np.repeat(state[None,:],H+1,axis=0),actions=np.zeros((H,2)),
        gains=gains,feed=np.zeros((H,2)),roots=np.repeat(np.eye(nx)[None,:,:],H+1,axis=0),
        residuals=np.zeros((H+1,nx)),tracking_R=np.eye(2),reference=state,
        K=np.zeros((2,nx)),P=np.eye(nx),Q=np.eye(nx),R=np.eye(2),
        park_steps=1,swing_steps=2,parking_reference=state,parking_K=np.zeros((2,nx)),
        align_gauge=False,online_gauge=False,native_search=False,
        launch_reference_profiles=np.array(profiles),affine_value=True)


@pytest.mark.parametrize('profiles,selected,admitted',[
    ([[1.,0.],[0.,0.]],1,True),
    ([[0.,0.],[0.,0.]],0,True),
    ([[1.,0.],[-1.,0.]],0,False)])
def test_launch_forecasts_are_isolated_causal_and_match_selected_actual_suffix(profiles,selected,admitted):
    env = SpatialCartpole(Config(n_links=1,plane_limit=.005,episode_seconds=.12,sustain_seconds=.06))
    state = env.upright_state()
    env.set_state(state)
    policy = affine_policy_from_payload(env,payload(env,profiles))
    state,*_ = env.step_force(policy(state,0))
    before = (env.state(),env.native_steps,env.steps,env.streak,env.best_streak,
              env.min_plane_margin,env.warning_seen,env.data.time)
    first = policy(state,1)
    np.testing.assert_array_equal(env.state(),before[0])
    assert (env.native_steps,env.steps,env.streak,env.best_streak,
            env.min_plane_margin,env.warning_seen,env.data.time)==before[1:]
    assert policy.metadata['selected_launch_profile']==selected
    assert policy.metadata['launch_admitted']==admitted
    assert policy.metadata['predictor_native_steps']==sum(r['native_steps'] for r in policy.metadata['launch_forecasts'])
    assert 0<policy.metadata['predictor_native_steps']<=2*5*4
    states,actions = [state],[]
    for step in range(1,env.config.horizon):
        force = first if step==1 else policy(state,step)
        action = env.deliver(force)
        state,_,terminated,truncated,info = env.step_force(action)
        states.append(state)
        actions.append(action)
        if terminated or truncated:
            break
    np.testing.assert_array_equal(states,policy.forecast_tapes[f'forecast_{selected}_states'])
    np.testing.assert_array_equal(actions,policy.forecast_tapes[f'forecast_{selected}_actions'])
    policy.close()
    env.close()


def test_admission_rejects_invalid_profile_table():
    env = SpatialCartpole()
    with pytest.raises(ValueError,match='two finite'):
        affine_policy_from_payload(env,payload(env,[[0.,np.nan],[0.,0.]]))
    env.close()
