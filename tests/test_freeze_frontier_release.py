import copy

import pytest

from scripts.freeze_frontier_release import validate_development_cohorts


def evidence():
    parameters = dict(park_seconds=16., cart_target=-.05, tracking_gain_scale=1.)
    cohorts = []
    for size, seed in ((20, 1000), (100, 1100)):
        cohorts.append((size, dict(episodes=size, full_episode_successes=size, zero_noise=False,
            controller_sha256='controller', n_links=13, config=dict(sha256='config'),
            capture_gain_sha256='capture', settle_gain_sha256='settle', generated_xml_sha256='xml',
            seed_start=seed, **parameters)))
    kwargs = dict(n_links=13, controller_sha256='controller', config_sha256='config',
        parameters=parameters, capture_gain_sha256='capture', settle_gain_sha256='settle',
        generated_xml_sha256='xml', reserved_seeds={2000})
    return cohorts, kwargs


def test_matching_development_policy_can_be_frozen():
    cohorts, kwargs = evidence()
    validate_development_cohorts(cohorts, **kwargs)


@pytest.mark.parametrize('key,value', [('park_seconds', 17.), ('tracking_gain_scale', .75),
    ('capture_gain_sha256', 'other'), ('settle_gain_sha256', 'other'), ('n_links', 12),
    ('zero_noise', True), ('full_episode_successes', 19), ('generated_xml_sha256', 'other')])
def test_successful_cohort_with_different_policy_is_rejected(key, value):
    cohorts, kwargs = evidence()
    cohorts[0][1][key] = value
    with pytest.raises(ValueError):
        validate_development_cohorts(cohorts, **kwargs)


@pytest.mark.parametrize('seed', [1010, 2000])
def test_overlapping_or_reserved_development_seeds_are_rejected(seed):
    cohorts, kwargs = evidence()
    cohorts = copy.deepcopy(cohorts)
    cohorts[1][1]['seed_start'] = seed
    with pytest.raises(ValueError, match='disjoint'):
        validate_development_cohorts(cohorts, **kwargs)
