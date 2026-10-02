import copy

import pytest

from scripts.verify_frontier_release import check_episode


@pytest.mark.parametrize('failure', [
    {'termination_reason': 'rail_violation'}, {'length': 1499},
    {'max_cart_excursion': 3.01}, {'max_upright_streak_seconds': 4.98},
    {'full_episode_success': False}, {'max_cart_excursion': float('nan')},
    {'final_info': {'simulation_error': 'warning reset'}},
])
def test_frontier_verifier_rejects_invalid_or_partial_episode_even_when_success_was_latched(failure):
    cfg = {'env': {'episode_seconds': 30., 'timestep': .005, 'frame_skip': 4,
                   'rail_limit': 3., 'success_sustain_seconds': 5.}}
    good = dict(success=True, full_episode_success=True, length=1500,
                termination_reason='time_limit', max_cart_excursion=2.1,
                max_upright_streak_seconds=7.5, final_info={'simulation_error': None})
    assert check_episode(good, cfg)
    bad = copy.deepcopy(good)
    bad.update(failure)
    assert not check_episode(bad, cfg)
