from __future__ import annotations

import pytest

from scripts.run_similarity_continuation import (
    feedback_gain_candidates,
    materialized_route,
    next_step,
    scale_at_progress,
    segment_feedback_candidates,
    successful_replay,
)


def test_similarity_scale_paths_hit_exact_endpoints() -> None:
    for path in ("linear", "log"):
        assert scale_at_progress(2.0, 0.0, path=path) == pytest.approx(1.0)
        assert scale_at_progress(2.0, 1.0, path=path) == pytest.approx(2.0)
        assert scale_at_progress(0.5, 1.0, path=path) == pytest.approx(0.5)


def test_linear_and_log_similarity_paths_are_deterministic() -> None:
    assert scale_at_progress(2.0, 0.5, path="linear") == pytest.approx(1.5)
    assert scale_at_progress(2.0, 0.5, path="log") == pytest.approx(2.0**0.5)


def test_continuation_step_grows_after_acceptance_and_halves_after_rejection() -> None:
    assert next_step(0.1, accepted=True, growth=1.5, minimum=0.01, maximum=0.2) == pytest.approx(0.15)
    assert next_step(0.1, accepted=False, growth=1.5, minimum=0.01, maximum=0.2) == pytest.approx(0.05)
    assert next_step(0.01, accepted=False, growth=1.5, minimum=0.01, maximum=0.2) == pytest.approx(0.01)


def test_feedback_gain_screen_is_bounded_stable_and_excludes_identity() -> None:
    assert feedback_gain_candidates(None) == [0.75, 1.25, 0.5, 1.5, 2.0]
    assert feedback_gain_candidates([1.0, 0.75, 0.75, 1.25]) == [0.75, 1.25]
    with pytest.raises(ValueError, match="finite and positive"):
        feedback_gain_candidates([0.0])


def test_segment_feedback_screen_parses_unique_positive_pairs() -> None:
    assert segment_feedback_candidates(None)[0] == (1.0, 1.25)
    assert segment_feedback_candidates(["1,1", "1,1.25", "1,1.25"]) == [
        (1.0, 1.25)
    ]
    with pytest.raises(ValueError, match="SWING,TAIL"):
        segment_feedback_candidates(["1"])
    with pytest.raises(ValueError, match="finite and positive"):
        segment_feedback_candidates(["1,0"])


def test_route_and_replay_classifiers_require_positive_evidence(tmp_path) -> None:
    replay = tmp_path / "replay.json"
    replay.write_text('{"result":{"success":true,"latched":true}}')
    assert successful_replay(replay)
    replay.write_text('{"result":{"success":false,"latched":true}}')
    assert not successful_replay(replay)

    route = tmp_path / "route.json"
    route.write_text(
        '{"controller":{"materialized_swing_prefix_steps":2,'
        '"materialized_lqr_tail_steps":3}}'
    )
    assert materialized_route(route)
    route.write_text('{"controller":{"materialized_lqr_tail_steps":0}}')
    assert not materialized_route(route)
