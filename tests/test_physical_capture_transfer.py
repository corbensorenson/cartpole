import numpy as np
import pytest
from scripts.transfer_physical_capture_route import transfer_physical_nodes, retime_physical_curve


def test_adjacent_transfer_preserves_cart_and_interpolates_physical_absolute_rates():
    source = np.array([[.3, .1, .4, .7, .2, .6], [-.2, -.1, -.4, -.7, -.2, -.6]])
    mapped = transfer_physical_nodes(source, 3)
    np.testing.assert_array_equal(mapped[:, 0], source[:, 0])
    np.testing.assert_array_equal(mapped[:, 4], source[:, 3])
    np.testing.assert_allclose(np.cumsum(mapped[:, 1:4],axis=1), [[.1,.3,.5],[-.1,-.3,-.5]])
    np.testing.assert_allclose(np.cumsum(mapped[:, 5:8],axis=1), [[.2,.5,.8],[-.2,-.5,-.8]])


def test_same_count_transfer_preserves_the_complete_physical_curve():
    source = np.random.default_rng(23).normal(size=(4,8))
    np.testing.assert_allclose(transfer_physical_nodes(source,3), source, atol=5e-16)


def test_default_retiming_preserves_saved_states_and_delivered_controls_exactly():
    rng = np.random.default_rng(17)
    states, actions = rng.normal(size=(5, 6)), rng.normal(size=4).astype(np.float32).astype(float)
    for duration in [None, .08]:
        result, controls = retime_physical_curve(states, actions, .02, duration)
        np.testing.assert_array_equal(result, states)
        np.testing.assert_array_equal(controls, actions)


def test_time_dilation_preserves_winding_and_scales_analytic_velocity():
    t = np.arange(5)*.02
    position = np.column_stack([t**3, 2*np.pi+t**2])
    velocity = np.column_stack([3*t**2, 2*t])
    states = np.column_stack([position, velocity])
    actions = np.linspace(-.6, .6, 4)
    retimed, controls = retime_physical_curve(states, actions, .02, .16)
    q = np.arange(9)*.01
    np.testing.assert_allclose(retimed[:, :2], np.column_stack([q**3, 2*np.pi+q**2]), atol=1e-14)
    np.testing.assert_allclose(retimed[:, 2:], .5*np.column_stack([3*q**2, 2*q]), atol=1e-12)
    assert len(controls) == 8 and np.max(np.abs(controls)) <= .6
    np.testing.assert_array_equal(retimed[[0,-1], :2], states[[0,-1], :2])


@pytest.mark.parametrize('duration', [0, -.1, .031, np.nan])
def test_retiming_rejects_invalid_or_unticked_duration(duration):
    with pytest.raises(ValueError, match='tick aligned'):
        retime_physical_curve(np.zeros((5,4)), np.zeros(4), .02, duration)
