import copy

import numpy as np
import pytest

from scripts.render_parked_evaluation_trace import validate_trace
from scripts.render_fddp_parked_route_video import draw_frame


def trace_fixture():
    rows = [dict(step=i+1, time_seconds=(i+1)*.02, qpos=[0., 0.], qvel=[0., 0.],
                 action=0., x=0., phase="capture_lqr", max_abs_angle=0., is_upright=True)
            for i in range(4)]
    return dict(n_links=1, environment=dict(action_frequency_hz=50., episode_seconds=.08, rail_limit=3.),
                episode_results=[dict(success=True, termination_reason="time_limit", final_info={},
                                      max_upright_streak_seconds=.08, trajectory=rows)])


def test_video_refuses_missing_steps_and_false_pose_success():
    payload = trace_fixture()
    assert len(validate_trace(payload, 0, required_hold=.04)[1]) == 4
    skipped = copy.deepcopy(payload); skipped["episode_results"][0]["trajectory"].pop(1)
    with pytest.raises(ValueError, match="complete"):
        validate_trace(skipped, 0, required_hold=.04)
    false_pose = copy.deepcopy(payload); false_pose["episode_results"][0]["trajectory"][0]["qpos"][1] = np.pi
    with pytest.raises(ValueError, match="physical pose"):
        validate_trace(false_pose, 0, required_hold=.04)


@pytest.mark.parametrize("angle", [0., np.pi/2, np.pi])
def test_twenty_link_camera_keeps_extreme_full_chain_poses_inside_frame(angle):
    q = np.zeros(21); q[0] = 3.; q[1] = angle
    frame = draw_frame(width=1280, height=720, qpos=q, action=0., time_seconds=0.,
                       phase="capture_lqr", max_abs_angle=angle, hinge_velocity_rms=0.,
                       upright_streak=0., max_upright_streak=0., cart_trace=[],
                       rail_limit=3., n_links=20, total_length=3.)
    tip = np.all(frame == np.array([251, 191, 36]), axis=2)
    ys, xs = np.where(tip)
    assert len(xs) > 0
    assert np.min(xs) > 64 and np.max(xs) < 1216
    assert np.min(ys) > 116 and np.max(ys) < 614
