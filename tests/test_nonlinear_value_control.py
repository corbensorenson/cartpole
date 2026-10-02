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
