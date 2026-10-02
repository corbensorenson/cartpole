import numpy as np
from scripts.transfer_physical_capture_route import transfer_physical_nodes


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
