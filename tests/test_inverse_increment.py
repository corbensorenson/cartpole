import copy

import pytest

from scripts.synthesize_inverse_increment import exact_candidate_passed, validate_increment


def passing_fixture():
    cfg = dict(env=dict(n_links=4, episode_seconds=30., timestep=.005, frame_skip=4,
                        success_sustain_seconds=5., rail_limit=3.))
    payload = dict(selected_state=dict(qpos=[0.]*4),
                   result=dict(success=True, trajectory_integrity=True, termination_reason="time_limit",
                               length=1500, max_upright_streak_seconds=24., max_cart_excursion=2.8),
                   controller=dict(final_trajectory_diagnostics=dict(is_feasible=True,
                                                                     maximum_dynamics_defect=0., initial_state_gap=0.)))
    source_cfg = copy.deepcopy(cfg); source_cfg["env"]["n_links"] = 3
    return cfg, payload, dict(search=dict(spline_control_points=[[0.]*4]), effective_config=source_cfg)


def test_increment_requires_feasible_full_replay_and_adjacent_matching_sources():
    cfg, payload, spline = passing_fixture()
    assert exact_candidate_passed(payload, cfg)
    validate_increment(spline, payload, cfg)
    for change in (dict(length=400), dict(max_upright_streak_seconds=4.9),
                   dict(max_cart_excursion=3.01), dict(trajectory_integrity=False)):
        bad = copy.deepcopy(payload); bad["result"].update(change)
        assert not exact_candidate_passed(bad, cfg)
    bad = copy.deepcopy(payload)
    bad["controller"]["final_trajectory_diagnostics"]["maximum_dynamics_defect"] = 1e-4
    assert not exact_candidate_passed(bad, cfg)
    with pytest.raises(ValueError, match="successful"):
        validate_increment(spline, bad, cfg)
    cfg["env"]["n_links"] = 5
    with pytest.raises(ValueError, match="increment"):
        validate_increment(spline, payload, cfg)
