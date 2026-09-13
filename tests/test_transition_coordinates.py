from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gcartpole.ilqr import MujocoTransition


def test_transformed_state_difference_wraps_angles_before_scaling() -> None:
    transition = MujocoTransition.__new__(MujocoTransition)
    transition.env = SimpleNamespace(n=1)
    transition.coordinate_transform = np.diag([1.0, 2.0, 1.0, 1.0])
    transition.inverse_transform = np.linalg.inv(transition.coordinate_transform)

    physical_first = np.array([0.0, np.pi - 0.01, 0.0, 0.0])
    physical_second = np.array([0.0, -np.pi + 0.01, 0.0, 0.0])
    first = transition.coordinate_transform @ physical_first
    second = transition.coordinate_transform @ physical_second

    difference = transition.difference(first, second)

    np.testing.assert_allclose(difference, [0.0, -0.04, 0.0, 0.0], atol=1e-12)
