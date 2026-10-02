import numpy as np

from scripts.generate_reverse_descent_seeds import reverse_descent


def test_reversal_keeps_actions_paired_with_reversed_intervals_and_negates_velocities():
    states = np.array([[0., 0., 1., 2.], [1., np.pi, 3., 4.],
                       [2., 3 * np.pi, 5., 6.]])
    hanging = np.array([0., np.pi, 0., 0.])
    reverse, controls, mismatch = reverse_descent(states, [0.2, -0.4], hanging)
    np.testing.assert_array_equal(controls, [-0.4, 0.2])
    np.testing.assert_array_equal(reverse[:, 2:], -states[::-1, 2:])
    np.testing.assert_array_equal(reverse[:, 0], [0., -1., -2.])
    np.testing.assert_allclose(reverse[0, 1], np.pi)
    np.testing.assert_array_equal(mismatch, [0., 0., -5., -6.])
    # Preserve the mismatch for evidence; the actual reverse state is not
    # silently replaced by a hanging equilibrium.
    np.testing.assert_array_equal(reverse[0, 2:], [-5., -6.])


def test_balanced_force_waveform_removes_impulse_and_first_moment():
    from scripts.screen_reverse_descent_seeds import balanced_waveform
    original = np.random.default_rng(10).normal(size=40)
    corrected = balanced_waveform(original)
    assert abs(np.sum(corrected)) < 1e-13
    assert abs(np.linspace(-1, 1, 40) @ corrected) < 1e-13
