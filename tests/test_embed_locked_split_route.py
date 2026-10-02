import numpy as np

from scripts.embed_locked_split_route import (
    embed_coordinate_states,
    embed_feedback_gains,
    embed_physical_state,
)


def test_coordinate_split_duplicates_absolute_angle_and_zeroes_inserted_rate():
    source = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    target = embed_coordinate_states(source, split_link=2)

    np.testing.assert_allclose(target, [[1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 6.0, 0.0]])


def test_feedback_split_preserves_both_source_coordinate_blocks():
    source = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    target = embed_feedback_gains(source, split_link=2)

    np.testing.assert_allclose(target, [[1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 6.0, 0.0]])


def test_physical_split_starts_with_zero_inserted_relative_joint():
    qpos = np.array([0.2, 0.1, 0.2, 0.3])
    qvel = np.array([-0.4, 0.5, 0.6, 0.7])

    target_qpos, target_qvel = embed_physical_state(qpos, qvel, split_link=2)

    np.testing.assert_allclose(target_qpos, [0.2, 0.1, 0.2, 0.0, 0.3])
    np.testing.assert_allclose(target_qvel, [-0.4, 0.5, 0.6, 0.0, 0.7])
