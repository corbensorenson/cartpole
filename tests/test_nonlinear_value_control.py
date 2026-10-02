import numpy as np
import pytest
from gcartpole.nonlinear_value_control import select_nonlinear_value_action
from gcartpole.simulation import SimulationError


def test_nonlinear_value_action_improves_baseline_and_uses_delivered_actions():
    actions=[]
    def transition(state,action):
        actions.append(action);return state-action-.2*action**2
    state=np.array([1.]);before=state.copy()
    action,diag=select_nonlinear_value_action(transition,state,np.eye(1),.5)
    expected=(-1+np.sqrt(1.8))/.4
    assert abs(action-expected)<1e-6
    assert diag['predicted_value']<1e-12
    assert diag['improves_baseline']
    assert diag['predictor_evaluations']==len(actions)
    assert all(x==float(np.float32(x)) and -1<=x<=1 for x in actions)
    np.testing.assert_array_equal(state,before)


def test_baseline_is_retained_when_global_scalar_search_is_not_better():
    action,diag=select_nonlinear_value_action(lambda x,u:np.array([u-.123456]),
                                            np.zeros(1),np.eye(1),.123456,max_iterations=1)
    assert action==float(np.float32(.123456))
    assert diag['predicted_value']==diag['baseline_predicted_value']


def test_invalid_and_out_of_rail_previews_are_rejected():
    def transition(x,u):
        if u<-.2:raise SimulationError('invalid trial')
        return np.array([2*u])
    action,diag=select_nonlinear_value_action(transition,np.zeros(1),np.eye(1),.8,rail_limit=.5)
    assert abs(action)<1e-6
    assert diag['invalid_previews']>0
    assert np.isinf(diag['baseline_predicted_value'])
    with pytest.raises(SimulationError,match='no finite'):
        select_nonlinear_value_action(lambda x,u:np.array([np.nan]),np.zeros(1),np.eye(1),0.)


def test_factored_mp_preview_preserves_weak_error_and_action_penalty():
    # Strong residual components cancel exactly; the weak component and
    # declared effort cost still determine the native delivered action.
    factor=np.array([[1e20,1e20],[0.,1.]])
    def forecast(state,action):return np.array([action,-action])
    target=np.array([.3,-.3])
    action,diag=select_nonlinear_value_action(forecast,np.zeros(2),factor,.3,
        error_function=lambda x:x-target,control_cost=1.,reference_action=0.,
        decimal_digits=80,candidate_actions=[0.])
    assert action==float(np.float32(action))
    assert abs(action-.15)<1e-6
    np.testing.assert_allclose(diag['predicted_value'],.045,rtol=1e-12)
    assert diag['improves_baseline']


def test_native_preview_does_not_modify_live_mujoco_state():
    from gcartpole.config import load_config
    from gcartpole.env import NLinkCartPoleEnv
    from gcartpole.ilqr import MujocoTransition,data_state
    cfg=load_config('configs/swingup11_uniform.yaml');cfg['env']['n_links']=1
    env=NLinkCartPoleEnv(cfg,progress=1.,seed=12);env.reset(seed=12)
    before=data_state(env.data).copy();time_before=env.data.time
    predictor=MujocoTransition(env,continuous_angles=True)
    try:
        select_nonlinear_value_action(predictor,before,np.eye(4),0.,grid_points=3,max_iterations=2,
            error_function=lambda x:x-before,decimal_digits=80)
        np.testing.assert_array_equal(data_state(env.data),before)
        assert env.data.time==time_before
    finally:env.close()
