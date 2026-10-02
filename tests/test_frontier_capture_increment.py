"""Guards for reusing a development repair and separating physical from virtual evidence."""
import json
import pytest
from scripts.solve_frontier_capture_increment import (
    FIRST_PARAMETERS, physical_replay_passed, validate_adopted_repair,
    validate_development_seeds,
)
from gcartpole.evidence import file_metadata


def test_development_seed_guard_checks_whole_cohorts():
    validate_development_seeds(14,20264100)
    # The synthesis seed is outside reserved ranges, but its twenty-noisy cohort isn't.
    with pytest.raises(ValueError,match='reserved'):
        validate_development_seeds(14,213900)
    # A one-episode offset intersects the reserved hundred despite a safe first seed.
    with pytest.raises(ValueError,match='reserved'):
        validate_development_seeds(14,213901)


def test_physical_success_does_not_require_virtual_feasibility():
    cfg={'env':dict(episode_seconds=30,timestep=.005,frame_skip=4,rail_limit=3,
                    success_sustain_seconds=5)}
    result=dict(success=True,trajectory_integrity=True,termination_reason='time_limit',
                length=1500,max_cart_excursion=2.,max_upright_streak_seconds=7.,
                final_info={'simulation_error':None})
    assert physical_replay_passed(result,cfg)
    for patch in ({'length':1499},{'max_cart_excursion':3.01},{'trajectory_integrity':False},
                  {'max_upright_streak_seconds':float('nan')},{'final_info':{'simulation_error':'bad state'}}):
        assert not physical_replay_passed(result|patch,cfg)


def test_adopted_repair_requires_same_source_recipe_and_completed_execution(tmp_path):
    cfg=tmp_path/'config.yaml';cfg.write_text('target')
    source=tmp_path/'source.json';source.write_text('accepted controller')
    init=tmp_path/'initializer.json'
    init.write_text(json.dumps(dict(source=file_metadata(source),config=file_metadata(cfg),
                                  controller={'virtual_tail_seconds':5.})))
    repair=tmp_path/'result.json'
    data={'shared_capture_recipe':dict(parameters=FIRST_PARAMETERS|dict(seed=20264100,config=str(cfg)),
                                     initialization=file_metadata(init))}
    repair.write_text(json.dumps(data))
    execution=tmp_path/'execution.json'
    execution.write_text(json.dumps({'status':'finished','returncode':0}))
    validate_adopted_repair(repair,cfg,source,20264100)
    data['shared_capture_recipe']['parameters']['iterations']=19
    repair.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='recipe'):
        validate_adopted_repair(repair,cfg,source,20264100)
    data['shared_capture_recipe']['parameters']['iterations']=20
    repair.write_text(json.dumps(data))
    execution.write_text(json.dumps({'status':'running','returncode':0}))
    with pytest.raises(ValueError,match='finish'):
        validate_adopted_repair(repair,cfg,source,20264100)
    execution.write_text(json.dumps({'status':'finished','returncode':0}))
    source.write_text('different controller')
    with pytest.raises(ValueError,match='predecessor'):
        validate_adopted_repair(repair,cfg,source,20264100)
