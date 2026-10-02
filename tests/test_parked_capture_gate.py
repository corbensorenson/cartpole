from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_fddp_parked_route as evaluator


def controller(defer=False):
    return dict(
        controls=np.full(4, 0.2), nominal_states=np.zeros((5, 4)),
        feedback_gains=np.zeros((4, 4)), transform=np.eye(4),
        lqr_scale=1., continuous_angles=False,
        defer_handoff_until_horizon=defer,
        capture_gate=dict(lyapunov=.1, cart_abs=.1, angle_abs=.15,
                          cart_velocity_abs=.5, hinge_velocity_rms=.75),
    )


def test_capture_gate_uses_translated_equilibrium_and_velocity_limits():
    policy = controller()
    assert evaluator.capture_gate_satisfied(np.array([-.5, 0., 0., 0.]), 1, policy, np.eye(4), -.5)
    assert not evaluator.capture_gate_satisfied(np.array([-.5, 0., .51, 0.]), 1, policy, np.eye(4), -.5)
    assert not evaluator.capture_gate_satisfied(np.array([-.5, .16, 0., 0.]), 1, policy, np.eye(4), -.5)
    assert not evaluator.capture_gate_satisfied(np.array([np.nan, 0., 0., 0.]), 1, policy, np.eye(4), -.5)


def test_physical_capture_metric_reconstructs_the_transported_factor(monkeypatch):
    from gcartpole.evidence import data_sha256
    policy = controller()
    mapping = np.diag([.8, 6.7, 2., 1.3])
    design_factor = np.diag([1., 2., 3., 4.])
    physical_factor = design_factor @ mapping
    policy.update(physical_shooting=True, normalized_cost_mapping=mapping.tolist(),
                  lqr_decimal_digits=80,
                  lyapunov_metadata=dict(source="lqr_high_precision_discrete_lyapunov",
                      factor_sha256=data_sha256(physical_factor.tolist()),
                      matrix_sha256=data_sha256((physical_factor.T @ physical_factor).tolist())))
    monkeypatch.setattr(evaluator, "finite_difference_dynamics", lambda *args: (np.eye(4), np.ones((4, 1))))
    def reconstruct(a, b, gain, transform, **kwargs):
        np.testing.assert_array_equal(transform, mapping)
        return design_factor.T @ design_factor, design_factor, {}
    monkeypatch.setattr(evaluator, "high_precision_lyapunov_factor", reconstruct)
    metric = evaluator.capture_metric({}, policy, np.zeros(4))
    np.testing.assert_array_equal(metric, physical_factor.T @ physical_factor)
    np.testing.assert_array_equal(policy["checked_capture_value_factor"], physical_factor)


@pytest.mark.parametrize("defer,expected", [(False, ["swing_route_feedback"] + ["capture_lqr"] * 4),
                                           (True, ["swing_route_feedback"] * 4 + ["capture_lqr"])])
def test_parked_controller_latches_gate_without_reset_and_preserves_legacy_clock(monkeypatch, defer, expected):
    instances = []

    class FakeEnv:
        n, dt, max_steps = 1, .02, 5

        def __init__(self, *args, **kwargs):
            self.data = SimpleNamespace(qpos=np.array([-.5, 1.]), qvel=np.zeros(2), qacc_warmstart=np.zeros(2))
            self.steps, self.resets = 0, 0
            instances.append(self)

        def reset(self, **kwargs):
            self.resets += 1
            return None, {}

        def step(self, action):
            self.steps += 1
            # The gate opens once, then closes. Capture must remain latched.
            self.data.qpos[1] = 0. if self.steps == 1 else 1.
            info = dict(x=-.5, max_abs_angle=abs(self.data.qpos[1]),
                        hinge_velocity_rms=0., absolute_angular_velocity_rms=0.,
                        is_upright=self.steps == 1, termination_reason="time_limit")
            return None, 0., False, self.steps == self.max_steps, info

        def close(self):
            pass

    monkeypatch.setattr(evaluator, "NLinkCartPoleEnv", FakeEnv)
    monkeypatch.setattr(evaluator, "lqr_action", lambda *args, **kwargs: .7)
    result = evaluator.run_episode(
        {}, controller(defer), np.zeros(4), np.zeros(4), seed=1,
        park_seconds=0., cart_target=-.5, tracking_gain_scale=1.,
        phase_adaptive=False, phase_window=0, include_trace=True,
        lyapunov=None if defer else np.eye(4),
    )
    assert [row["phase"] for row in result["trajectory"]] == expected
    assert result["first_handoff_step"] == (4 if defer else 1)
    assert result["handoff_reason"] == ("route_end" if defer else "saved_state_gate")
    assert instances[0].resets == 1
    assert instances[0].steps == 5


def test_five_second_hold_followed_by_rail_failure_is_not_full_episode_success(monkeypatch):
    class HoldThenFail:
        n, dt, max_steps = 1, 1., 10

        def __init__(self, *args, **kwargs):
            self.data = SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2), qacc_warmstart=np.zeros(2))
            self.steps = 0

        def reset(self, **kwargs):
            return None, {}

        def step(self, action):
            self.steps += 1
            self.data.qpos[0] = 3.1 if self.steps == 7 else 0.
            info = dict(x=float(self.data.qpos[0]), max_abs_angle=0.,
                        hinge_velocity_rms=0., absolute_angular_velocity_rms=0.,
                        is_upright=True, success=self.steps >= 5,
                        max_upright_streak_seconds=float(self.steps),
                        termination_reason="rail_limit" if self.steps == 7 else None)
            return None, 0., self.steps == 7, False, info

        def close(self):
            pass

    monkeypatch.setattr(evaluator, "NLinkCartPoleEnv", HoldThenFail)
    monkeypatch.setattr(evaluator, "lqr_action", lambda *args, **kwargs: 0.)
    result = evaluator.run_episode(
        {"env": {"rail_limit": 3.}}, controller(True), np.zeros(4), np.zeros(4), seed=1,
        park_seconds=0., cart_target=0., tracking_gain_scale=1.,
        phase_adaptive=False, phase_window=0, include_trace=False,
    )
    assert result["success"]
    assert result["max_upright_streak_seconds"] == 7.
    assert result["length"] == 7
    assert not result["full_episode_success"]
