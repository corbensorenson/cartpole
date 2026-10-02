from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from gcartpole.ilqr import MujocoTransition


def test_transformed_state_difference_wraps_angles_before_scaling() -> None:
    transition = MujocoTransition.__new__(MujocoTransition)
    transition.env = SimpleNamespace(n=1)
    transition.continuous_angles = False
    transition.coordinate_transform = np.diag([1.0, 2.0, 1.0, 1.0])
    transition.inverse_transform = np.linalg.inv(transition.coordinate_transform)

    physical_first = np.array([0.0, np.pi - 0.01, 0.0, 0.0])
    physical_second = np.array([0.0, -np.pi + 0.01, 0.0, 0.0])
    first = transition.coordinate_transform @ physical_first
    second = transition.coordinate_transform @ physical_second

    difference = transition.difference(first, second)

    np.testing.assert_allclose(difference, [0.0, -0.04, 0.0, 0.0], atol=1e-12)


def test_continuous_angles_preserve_full_rotations_and_euclidean_gaps():
    transition = MujocoTransition.__new__(MujocoTransition)
    transition.env = SimpleNamespace(n=1)
    transition.continuous_angles = True
    transition.coordinate_transform = np.diag([1.0, 2.0, 1.0, 1.0])
    transition.inverse_transform = np.linalg.inv(transition.coordinate_transform)
    first = np.array([0.0, 3 * np.pi + 0.01, 0.0, 0.0])
    second = np.array([0.0, 3 * np.pi - 0.01, 0.0, 0.0])
    np.testing.assert_allclose(transition.to_physical(transition.to_coordinates(first)), first)
    np.testing.assert_allclose(
        transition.difference(transition.to_coordinates(first), transition.to_coordinates(second)),
        [0.0, 0.04, 0.0, 0.0], atol=1e-12,
    )
