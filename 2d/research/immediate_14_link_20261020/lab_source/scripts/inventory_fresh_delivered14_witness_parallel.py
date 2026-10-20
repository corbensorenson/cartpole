"""Canonical witness/reproduction receipt with sequential development coverage."""
import argparse
import json
from pathlib import Path
from eleven_synthesis_protocol import read,sha
from delivered_fresh_context14_parallel import FRESH


def main():
    p=argparse.ArgumentParser()
    for name in ('parent','run','audit','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();assert not a.out.exists();prior=read(a.parent)
    assert prior['passed'] and prior['plan_found'] and not prior['qualified'] and not prior['task_success']
    s,g=read(a.run/'summary.json'),read(a.audit/'summary.json')
    for folder,result in ((a.run,s),(a.audit,g)):
        assert (folder/'completed.json').exists() and result['passed'] and result['task_success'] and not result['qualified']
        assert result['controller_identity']==prior['controller_identity'] and result['result']['seed']==prior['seed']
        assert result['native_steps']==12000 and result['new_predictor_native_steps']==0 and result['all_forecast_arrays_byte_exact']
        r=result['result'];assert r['steps']==1500 and r['success'] and r['best_hold_seconds']>=5. and r['physical_velocities']==44
        assert r['angle_noise']==.02 and r['park_seconds']==0. and r['plane_margin']>=0. and not r['physics_warning'] and r['max_force_norm']<=80.
    assert g['source_summary_sha256']==sha(a.run/'summary.json') and g['source_actual_sha256']==sha(a.run/'actual.npz')
    assert g['canonical_reset_actual_controls_states_scores_cache_and_cold_independently_byte_exact']
    for key,value in s.items():assert g[key]==value
    assert read(a.run/'frozen_manifest.json')['inventory_sha256']==sha(a.parent)
    native=s['native_steps']+g['native_steps'];covered=prior['fresh_case_ordinal_before']+1
    assert 1<=covered<=32 and FRESH[covered-1]==prior['seed']
    report=dict(passed=True,links=14,qualified=False,task_success=True,seed=prior['seed'],controller_identity=prior['controller_identity'],
        parent_inventory=str(a.parent),parent_inventory_sha256=sha(a.parent),run=str(a.run),audit=str(a.audit),
        summary_sha256=sha(a.run/'summary.json'),audit_summary_sha256=sha(a.audit/'summary.json'),
        actual_sha256=sha(a.run/'actual.npz'),manifest_sha256=sha(a.run/'frozen_manifest.json'),
        actor_native_steps=s['native_steps'],audit_native_steps=g['native_steps'],new_native_steps=native,
        total_new_target_preparation_native_steps=prior['total_new_target_preparation_native_steps']+native,
        total_new_target_preparation_native_steps_bounds=[x+native for x in prior['total_new_target_preparation_native_steps_bounds']],
        result=s['result'],predictor_native_steps=0,inherited_predictor_query_intervals=s['inherited_predictor_query_intervals'],
        original_reset_episodes=1,independent_reproduction_episodes=1,new_noisy_samples=0,reserved_cases_consumed=0,
        fresh_case_ordinal=covered,whole_fresh_cohort_qualified=False,cohort_manifest_sha256=prior['cohort_manifest_sha256'],source_registration_fresh_cases_consumed=1,
        studies={str(folder):{file.relative_to(folder).as_posix():dict(sha256=sha(file),bytes=file.stat().st_size)
            for file in sorted(folder.rglob('*')) if file.is_file()} for folder in (a.run,a.audit)},
        scope='one canonical fresh original-reset success and independent reproduction; all32 cohort proof still required')
    with a.out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print({k:v for k,v in report.items() if k!='studies'},flush=True)


if __name__=='__main__':main()
