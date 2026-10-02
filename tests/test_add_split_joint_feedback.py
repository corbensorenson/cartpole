import numpy as np

from scripts.add_split_joint_feedback import add_feedback


def test_split_feedback_can_be_limited_to_a_phase_window():
    payload = {
        "controller": {
            "controls": [0.0] * 4,
            "feedback_gains": np.zeros((4, 8)).tolist(),
        }
    }
    updated = add_feedback(
        payload,
        split_link=1,
        angle_gain=2.0,
        rate_gain=-3.0,
        start_step=1,
        end_step=3,
    )
    gains = np.asarray(updated["controller"]["feedback_gains"])
    np.testing.assert_allclose(gains[[0, 3]], 0.0)
    np.testing.assert_allclose(gains[1, [1, 2, 6]], [-2.0, 2.0, -3.0])
    assert updated["split_joint_feedback"]["end_step"] == 3
