import copy
import numpy as np
import pytest
from scripts.search_inverse_spline import inherited_hanging_state


def source(qpos):
    return dict(selected_state=dict(qpos=qpos, qvel=[0.]*len(qpos)),
                search=dict(spline_control_points=[qpos]*8))


@pytest.mark.parametrize('qpos', [[0., np.pi, 0., 0.], [0., -np.pi, 0., 0.],
                                [0., 3*np.pi, 2*np.pi, -2*np.pi]])
def test_saved_stationary_hanging_winding_is_preserved_exactly(qpos):
    data = source(qpos)
    observed = inherited_hanging_state(data, 3)
    np.testing.assert_array_equal(observed, qpos)
    observed[0] = 1.
    assert data['selected_state']['qpos'][0] == 0.


@pytest.mark.parametrize('change', ['cart', 'angle', 'velocity', 'node', 'dimensions', 'nan'])
def test_equivalent_winding_does_not_admit_a_different_physical_start(change):
    data = copy.deepcopy(source([0., -np.pi, 0., 0.]))
    if change == 'cart': data['selected_state']['qpos'][0] = .01
    elif change == 'angle': data['selected_state']['qpos'][1] += .01
    elif change == 'velocity': data['selected_state']['qvel'][1] = .01
    elif change == 'node': data['search']['spline_control_points'][0] = [0., np.pi, 0., 0.]
    elif change == 'dimensions': data['selected_state']['qpos'].pop()
    else: data['selected_state']['qpos'][1] = float('nan')
    with pytest.raises(ValueError, match='stationary canonical hanging'):
        inherited_hanging_state(data, 3)
