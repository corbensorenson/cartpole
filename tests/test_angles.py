import numpy as np

from gcartpole.angles import wrap_angle


def test_wrapping_preserves_small_representable_angles_exactly():
    angles = np.array([-1e-30, -1e-18, -1e-13, 0.0, 1e-13, 1e-18, 1e-30])
    np.testing.assert_array_equal(wrap_angle(angles), angles)


def test_wrapping_keeps_the_existing_pi_boundary_and_periodic_convention():
    angles = np.array([-5 * np.pi, -np.pi, np.pi, 3 * np.pi, 0.25 + 4 * np.pi])
    np.testing.assert_allclose(wrap_angle(angles), [-np.pi, -np.pi, -np.pi, -np.pi, 0.25], atol=1e-14)
