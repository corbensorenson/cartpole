from __future__ import annotations

import unittest

import numpy as np

from gcartpole.capture_funnel import normalized_capture_coordinates
from gcartpole.modal import (
    StateScales,
    conjugate_mode_groups,
    dimensionless_absolute_transform,
    grouped_modal_amplitudes,
    modal_decomposition,
    real_schur_decomposition,
    scale_feedback_by_schur_group,
    transform_dynamics,
    transform_feedback_gain,
)


class ModalTests(unittest.TestCase):
    def test_dimensionless_transform_matches_capture_coordinates(self) -> None:
        scales = StateScales(1.25, 0.15, 0.50, 0.75)
        qpos = np.asarray([0.5, 0.1, -0.04], dtype=np.float64)
        qvel = np.asarray([-0.2, 0.3, -0.45], dtype=np.float64)
        transform = dimensionless_absolute_transform(2, scales)
        actual = transform @ np.r_[qpos, qvel]
        expected = normalized_capture_coordinates(
            qpos,
            qvel,
            cart_position_bound=scales.cart_position,
            angle_bound=scales.absolute_angle,
            cart_velocity_bound=scales.cart_velocity,
            hinge_velocity_bound=scales.hinge_velocity,
        )
        np.testing.assert_allclose(actual, expected)

    def test_dynamics_and_feedback_transform_preserve_next_state(self) -> None:
        state_matrix = np.asarray(
            [
                [1.0, 0.1, 0.0, 0.0],
                [0.0, 1.0, 0.1, 0.0],
                [0.0, 0.0, 0.9, 0.2],
                [0.0, 0.0, -0.1, 0.8],
            ],
            dtype=np.float64,
        )
        input_matrix = np.asarray([[0.0], [0.1], [0.2], [0.3]], dtype=np.float64)
        transform = dimensionless_absolute_transform(1, StateScales(2.0, 0.2, 1.0, 0.5))
        transformed_a, transformed_b = transform_dynamics(state_matrix, input_matrix, transform)
        state = np.asarray([0.4, -0.1, 0.2, 0.3], dtype=np.float64)
        action = 0.25
        np.testing.assert_allclose(
            transform @ (state_matrix @ state + input_matrix[:, 0] * action),
            transformed_a @ (transform @ state) + transformed_b[:, 0] * action,
        )

        gain = np.asarray([0.2, -0.3, 0.4, -0.5], dtype=np.float64)
        transformed_gain = transform_feedback_gain(gain, transform)
        self.assertAlmostEqual(float(gain @ state), float(transformed_gain @ (transform @ state)))

    def test_modal_coordinates_reconstruct_complex_pair(self) -> None:
        state_matrix = np.asarray(
            [
                [0.9, -0.2, 0.0],
                [0.2, 0.9, 0.0],
                [0.0, 0.0, 1.1],
            ],
            dtype=np.float64,
        )
        input_matrix = np.asarray([[1.0], [0.5], [0.25]], dtype=np.float64)
        decomposition = modal_decomposition(state_matrix, input_matrix)
        state = np.asarray([0.2, -0.4, 0.7], dtype=np.float64)
        coordinates = decomposition.coordinates(state)
        np.testing.assert_allclose(decomposition.reconstruct(coordinates), state, atol=1e-12)
        groups = conjugate_mode_groups(decomposition.eigenvalues)
        amplitudes = grouped_modal_amplitudes(coordinates, groups)
        self.assertEqual(len(groups), 2)
        self.assertEqual(amplitudes.shape, (2,))
        self.assertTrue(np.all(amplitudes >= 0.0))

        schur_decomposition = real_schur_decomposition(state_matrix, input_matrix)
        schur_coordinates = schur_decomposition.coordinates(state)
        np.testing.assert_allclose(schur_decomposition.reconstruct(schur_coordinates), state, atol=1e-12)
        np.testing.assert_allclose(
            np.sum(schur_decomposition.grouped_amplitudes(state) ** 2),
            np.linalg.norm(state) ** 2,
            atol=1e-12,
        )
        self.assertEqual(sorted(len(group) for group in schur_decomposition.groups), [1, 2])
        gain = np.asarray([0.2, -0.1, 0.3], dtype=np.float64)
        np.testing.assert_allclose(
            scale_feedback_by_schur_group(gain, schur_decomposition, {}),
            gain,
            atol=1e-12,
        )
        scaled = scale_feedback_by_schur_group(gain, schur_decomposition, {0: 2.0})
        original_schur_gain = gain @ schur_decomposition.orthogonal_basis
        scaled_schur_gain = scaled @ schur_decomposition.orthogonal_basis
        np.testing.assert_allclose(
            scaled_schur_gain[list(schur_decomposition.groups[0])],
            2.0 * original_schur_gain[list(schur_decomposition.groups[0])],
            atol=1e-12,
        )

    def test_state_scales_reject_nonpositive_values(self) -> None:
        with self.assertRaises(ValueError):
            StateScales(1.0, 0.0, 1.0, 1.0)


if __name__ == "__main__":
    unittest.main()
