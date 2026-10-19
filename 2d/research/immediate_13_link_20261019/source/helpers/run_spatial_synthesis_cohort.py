"""Extend a completed pilot without reruns, then optionally run fresh qualification."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from eleven_synthesis_protocol import read, sha
from spatial_synthesis_cohort_protocol import complete, manifest_check, PILOT, ADDITIONAL, reserved


def worker(out):
    from spatialpole.experiment import write_json
    m,_=manifest_check(out);fresh=m['cohort']=='qualification32'
    write_json(out/'worker.json',dict(pid=os.getpid(),started_utc=time.time()))
    rows=[];consumed=[]
    if not fresh:
        for old in read(Path(m['parent_run'])/'summary.json')['results']:
            rows.append(dict(old,inherited=True,new_actor_native_steps=0,new_audit_native_steps=0,
                inherited_from=m['parent_run'],origin_inventories=str(out/'inventories')))
    env=dict(os.environ);env['PYTHONPATH']=str(out/'frozen_package')
    def execute(script,args,seed):
        command=[sys.executable,'-u',str(out/'helpers'/script),*map(str,args)]
        child=subprocess.Popen(command,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        write_json(out/'status.json',dict(state='running',cohort=m['cohort'],owner_pid=os.getpid(),
            seed=seed,child_pid=child.pid,command=command,completed_noisy_cases=len(rows),updated_utc=time.time()))
        if child.wait()!=0:raise RuntimeError(f'Child failed; inspect retained case/audit: {script}, seed {seed}')
    for seed in m['new_seeds']:
        case,audit=out/f'seed{seed}',out/f'audit{seed}'
        assert not case.exists() and not audit.exists(), 'Partial evidence must be inspected; never repeat its computation'
        if fresh:
            consumed.append(seed)
            write_json(out/'consumed_cases.json',dict(seeds=consumed,controller_identity=m['controller_identity']))
        arguments=['--source-inventory',out/'inventories/source.json','--model-inventory',out/'inventories/model.json',
            '--port-inventory',out/'inventories/port.json','--runtime',m['runtime'],'--seed',seed,'--out',case]
        if fresh:arguments.append('--qualification')
        execute(m['evaluation_driver'],arguments,seed)
        result=read(case/'summary.json');cm=read(case/'frozen_manifest.json')
        assert cm['source_sha256']==m['source_sha256'] and cm['bundle_sha256']==m['source_bundle_sha256']
        assert cm['solver_receipt_sha256']==m['solver_receipt_sha256']
        for name in m['algorithm_names']:assert cm['helper_sha256'][name]==m['helper_sha256'][name]
        assert bool(cm['qualification_attempt'])==fresh
        write_json(out/'current_case.json',dict(seed=seed,run=str(case),audit=str(audit),
            completed_actor_native_steps=result['inclusive_native_steps'],audit_pending=result['result']['success']))
        audit_steps=0
        if result['result']['success']:
            execute(m['audit_driver'],['--run',case,'--out',audit],seed)
            proof=read(audit/'summary.json');assert proof['passed'] and proof['actual_cold_byte_exact']
            audit_steps=proof['native_steps']
        rows.append(dict(seed=seed,run=str(case),audit=str(audit) if audit_steps else None,
            success=result['result']['success'],inherited=False,
            new_actor_native_steps=result['inclusive_native_steps'],new_audit_native_steps=audit_steps,
            source_case_summary_sha256=sha(case/'summary.json'),
            source_audit_summary_sha256=sha(audit/'summary.json') if audit_steps else None))
        finished=len(rows)==len(m['seeds']);failed=not rows[-1]['success']
        summary=dict(cohort=m['cohort'],results=rows,noisy_passes=sum(r['success'] for r in rows),
            passes=sum(r['success'] for r in rows)+int(not fresh),cohort_finished=finished,stopped_on_failure=failed,
            inclusive_native_steps=sum(r['new_actor_native_steps']+r['new_audit_native_steps'] for r in rows),
            new_actor_native_steps=sum(r['new_actor_native_steps'] for r in rows),
            new_case_audit_native_steps=sum(r['new_audit_native_steps'] for r in rows),
            controller_identity=m['controller_identity'],qualified=False,qualification_attempt=fresh,
            reserved_cases_consumed=len(consumed),known_nominal_equivalence=not fresh,
            inherited_noisy_cases=0 if fresh else 12,
            scope='fixed algorithm; inherited pilot and separately proven nominal cost zero new physics; new cases/audits charged once')
        write_json(out/'summary.json',summary)
        print(f"{m['cohort']} seed={seed} success={rows[-1]['success']} passes={summary['passes']}",flush=True)
        if failed:break
    write_json(out/'completed.json',dict(status='completed',all_passed=finished and not failed))
    write_json(out/'status.json',dict(state='cases_complete',cohort=m['cohort'],passes=summary['passes'],owner_pid=os.getpid()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent-run');p.add_argument('--out',required=True)
    p.add_argument('--cohort',choices=('development45','qualification32'));p.add_argument('--advance',action='store_true')
    p.add_argument('--worker',action='store_true');a=p.parse_args();out=Path(a.out).resolve()
    from spatialpole.experiment import freeze_inputs,write_json
    if a.worker:
        try:worker(out)
        except BaseException as exception:
            write_json(out/'failure.json',dict(type=type(exception).__name__,error=str(exception),
                accounting_complete=False,current_case_receipt='current_case.json',child_counters_retained=True))
            raise
        return
    parent=Path(a.parent_run).resolve()
    # Authentic parent completion and complete independent case proofs precede any output.
    parent_proof=complete(parent);pm=read(parent/'frozen_manifest.json')
    assert parent_proof['cohort']==('pilot12' if a.cohort=='development45' else 'development45')
    assert parent_proof['passes']==(13 if a.cohort=='development45' else 45)
    fresh=a.cohort=='qualification32';new_seeds=reserved(pm['links']) if fresh else ADDITIONAL
    expected=new_seeds if fresh else PILOT+ADDITIONAL
    reservation=out.parent/f'reserved_{pm["links"]}_immediate_{new_seeds[0]}.json'
    if fresh:assert not reservation.exists(), 'Reserved states already assigned; inspect, never reuse'
    assert not out.exists()
    freeze_inputs(out,a)
    assert read(out/'inputs.json')['source_sha256']==pm['source_sha256']
    assert read(out/'inputs.json')['runtime']==read(parent/'inputs.json')['runtime']
    (out/'helpers').mkdir();shutil.copytree(parent/'frozen_package',out/'frozen_package')
    shutil.copytree(parent/'inventories',out/'inventories')
    runner='evaluate_twelve_observed_synthesis.py' if pm['links']==12 else 'evaluate_spatial_observed_case.py'
    auditor='audit_twelve_synthesis_case.py' if pm['links']==12 else 'audit_spatial_synthesis_case.py'
    additions={Path(__file__).name,'spatial_synthesis_cohort_protocol.py',runner}
    helpers=set(pm['helper_sha256'])|additions
    for name in helpers:
        source=Path(__file__).with_name(name) if name in additions else parent/'helpers'/name
        shutil.copyfile(source,out/'helpers'/name)
    from spatial_synthesis_cohort_protocol import ALGORITHM
    for name in ALGORITHM:assert sha(out/'helpers'/name)==pm['helper_sha256'][name]
    manifest=dict(pm,cohort=a.cohort,seeds=expected,new_seeds=new_seeds,
        parent_run=str(parent),parent_summary_sha256=sha(parent/'summary.json'),
        helper_sha256={name:sha(out/'helpers'/name) for name in sorted(helpers)},
        algorithm_names=list(ALGORITHM),evaluation_driver=runner,audit_driver=auditor,
        inherited_noisy_cases=0 if fresh else 12,qualified=False,reserved_cases_consumed=0)
    write_json(out/'parent_integrity_audit.json',parent_proof)
    write_json(out/'frozen_manifest.json',manifest)
    manifest_check(out)
    if fresh:
        with reservation.open('x') as stream:
            import json
            json.dump(dict(run=str(out),seeds=expected,controller_identity=manifest['controller_identity'],
                frozen_manifest_sha256=sha(out/'frozen_manifest.json')),stream,indent=2)
    env=dict(os.environ);env['PYTHONPATH']=str(out/'frozen_package')
    command=[sys.executable,'-u',str(out/'helpers'/Path(__file__).name),'--worker','--out',str(out)]
    write_json(out/'worker_command.json',dict(command=command,pythonpath=env['PYTHONPATH']))
    result=subprocess.run(command,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:raise SystemExit(result.returncode)
    if not read(out/'completed.json')['all_passed']:return
    proof=complete(out);gate=out.with_name(out.name+'_audit')
    freeze_inputs(gate,a)
    shutil.copyfile(out/'helpers/spatial_synthesis_cohort_protocol.py',gate/'protocol.py')
    write_json(gate/'summary.json',proof);write_json(gate/'completed.json',dict(status='completed'))
    write_json(out/'status.json',dict(state='cohort_audited',cohort=a.cohort,passes=proof['passes'],audit=str(gate)))
    print(f"GATE PASSED {a.cohort}: {proof['passes']}",flush=True)
    if a.advance and not fresh:
        target=out.parent/f'observed_synthesis_qualification32_n{pm["links"]}_001'
        assert not target.exists()
        command=[sys.executable,'-u',str(out/'helpers'/Path(__file__).name),'--parent-run',str(out),
            '--cohort','qualification32','--out',str(target)]
        result=subprocess.run(command,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode:raise SystemExit(result.returncode)


if __name__=='__main__':main()
