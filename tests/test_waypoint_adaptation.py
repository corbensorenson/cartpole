from __future__ import annotations

import numpy as np

from gcartpole import waypoint_adaptation
from gcartpole.waypoint_adaptation import adapt_route_to_waypoints


class DoubleIntegrator:
    def __call__(self, state: np.ndarray, action: float) -> np.ndarray:
        position, velocity = state
        return np.array([position + velocity + 0.5 * action, velocity + action])

    @staticmethod
    def difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
        return np.asarray(first) - np.asarray(second)


def test_waypoint_adaptation_retargets_each_exact_segment() -> None:
    transition = DoubleIntegrator()
    reference_controls = np.array([0.2, -0.2, 0.1, -0.1])
    reference_states = [np.zeros(2)]
    for action in reference_controls:
        reference_states.append(transition(reference_states[-1], float(action)))
    shifted_controls = 0.8 * reference_controls
    result = adapt_route_to_waypoints(
        transition,
        np.zeros(2),
        shifted_controls,
        np.asarray(reference_states),
        segment_steps=2,
        max_evaluations=30,
        endpoint_weight=1000.0,
        control_regularization=1.0e-8,
        rail_soft_limit=10.0,
        rail_weight=0.0,
        endpoint_tolerance=1.0e-5,
    )
    assert result.success
    np.testing.assert_allclose(result.states[[2, 4]], np.asarray(reference_states)[[2, 4]], atol=1e-5)


def test_waypoint_adaptation_falls_back_when_dense_svd_fails(monkeypatch) -> None:
    real_least_squares = waypoint_adaptation.least_squares
    calls: list[dict[str, object]] = []

    def flaky_least_squares(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise np.linalg.LinAlgError("synthetic dense SVD failure")
        return real_least_squares(*args, **kwargs)

    monkeypatch.setattr(waypoint_adaptation, "least_squares", flaky_least_squares)
    transition = DoubleIntegrator()
    controls = np.array([0.2, -0.2])
    states = [np.zeros(2)]
    for action in controls:
        states.append(transition(states[-1], float(action)))

    result = adapt_route_to_waypoints(
        transition,
        np.zeros(2),
        controls,
        np.asarray(states),
        segment_steps=2,
        max_evaluations=30,
        rail_soft_limit=10.0,
        endpoint_tolerance=1.0e-5,
    )

    assert result.success
    assert calls[0]["x_scale"] == "jac"
    assert calls[1]["x_scale"] == 1.0
    assert calls[1]["tr_solver"] == "lsmr"
    assert result.segments[0]["optimizer_fallback"] == "dense_svd_to_lsmr"
