"""Seal registered-case preparation and independent proof without qualification."""
import argparse
import json
from pathlib import Path
from delivered_fresh_context14_parallel import registered
from eleven_synthesis_protocol import read,sha


def main():
    p=argparse.ArgumentParser()
    for name in ('run','audit','out'):p.add_argument('--'+name,required=True,type=Path)
    a=p.parse_args();assert not a.out.exists();m=read(a.run/'frozen_manifest.json');registration=Path(m['registration'])
    bank,runtime,cm,rm,expected,prior=registered(registration)
    s,g=read(a.run/'summary.json'),read(a.audit/'summary.json')
    for folder,result in ((a.run,s),(a.audit,g)):
        assert (folder/'completed.json').exists() and result['passed'] and not result['qualified'] and not result['task_success']
        assert result['seed']==rm['seed'] and result['controller_identity']==cm['controller_identity']
    assert g['source_summary_sha256']==sha(a.run/'summary.json') and g['source_manifest_sha256']==sha(a.run/'frozen_manifest.json')
    assert g['actor_native_steps']==s['native_steps']<=9072324 and g['native_steps']<=384324 and g['plan_found']==s['plan_found']
    assert sha(registration/'frozen_manifest.json')==m['registration_manifest_sha256']==g['registration_manifest_sha256']
    native=s['native_steps']+g['native_steps']
    report=dict(passed=True,links=14,qualified=False,task_success=False,seed=rm['seed'],controller_identity=cm['controller_identity'],
        parent_inventory=rm['previous_witness_inventory'],parent_inventory_sha256=rm['previous_witness_inventory_sha256'],
        run=str(a.run),audit=str(a.audit),registration=str(registration),summary_sha256=sha(a.run/'summary.json'),
        audit_summary_sha256=sha(a.audit/'summary.json'),manifest_sha256=sha(a.run/'frozen_manifest.json'),plan_found=s['plan_found'],
        actor_native_steps=s['native_steps'],audit_native_steps=g['native_steps'],new_native_steps=native,
        total_new_target_preparation_native_steps=prior['total_new_target_preparation_native_steps']+native,
        total_new_target_preparation_native_steps_bounds=[x+native for x in prior['total_new_target_preparation_native_steps_bounds']],
        fresh_case_ordinal_before=rm['fresh_case_ordinal_before'],cohort_manifest_sha256=rm['cohort_manifest_sha256'],source_registration_fresh_cases_consumed=1,
        original_reset_episodes=0,new_noisy_samples=0,reserved_cases_consumed=0,
        studies={str(folder):{file.relative_to(folder).as_posix():dict(sha256=sha(file),bytes=file.stat().st_size)
            for file in sorted(folder.rglob('*')) if file.is_file()} for folder in (a.run,a.audit,registration)},
        scope='registered fresh original observation and unchanged preparation/audit; canonical witness required before next fresh case')
    with a.out.open('x',encoding='utf-8') as f:json.dump(report,f,indent=2);f.write('\n')
    print({k:v for k,v in report.items() if k!='studies'},flush=True)


if __name__=='__main__':main()
