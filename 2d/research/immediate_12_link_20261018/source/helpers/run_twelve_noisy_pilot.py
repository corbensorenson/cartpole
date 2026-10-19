"""Frozen twelve-start development pilot, inheriting the already audited first case."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from eleven_synthesis_protocol import read, sha

SEEDS = list(range(20261116,20261128))
ALGORITHM = ('spatial_observed_synthesis.py','spatial_synthesis_geometry.py',
             'eleven_value_sequence.py','donor_conic_step.py',
             'verify_eleven_sequence_refinement.py','eleven_sequence_qp.py','trajectory_expert_support.py')


def worker(out):
    from spatialpole.experiment import write_json
    manifest = read(out/'frozen_manifest.json')
    for name,digest in manifest['helper_sha256'].items():
        assert sha(out/'helpers'/name) == digest
    for name,digest in manifest['source_sha256'].items():
        assert sha(out/'frozen_package/spatialpole'/name) == digest
    for name,digest in manifest['inventory_sha256'].items():
        assert sha(out/'inventories'/name) == digest
    first = read(out/'inventories/first.json')
    source, audit = Path(first['run']), Path(first['audit'])
    assert sha(source/'summary.json') == first['summary_sha256']
    assert sha(audit/'summary.json') == first['audit_summary_sha256']
    prior = read(source/'frozen_manifest.json')
    assert prior['bundle_sha256'] == manifest['source_bundle_sha256']
    assert sha(Path(manifest['runtime'])/'receipt.json') == manifest['solver_receipt_sha256']
    for name in ALGORITHM[:4]:
        assert prior['helper_sha256'][name] == manifest['helper_sha256'][name]
    # The old successful case never entered the lazy constrained-solve branch.
    # Its unused dependency packaging fix requires no repeated model/episode.
    import numpy as np
    for certificate in (source/'planning').rglob('certificate.npz'):
        with np.load(certificate) as z:
            assert str(z['kind']) == 'zero_residual'
    env = dict(os.environ); env['PYTHONPATH'] = str(out/'frozen_package')
    rows = []
    def execute(script,args):
        command=[sys.executable,'-u',str(out/'helpers'/script),*map(str,args)]
        child=subprocess.Popen(command,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
        write_json(out/'status.json',dict(state='running',seed=seed,child_pid=child.pid,command=command,
            completed_cases=len(rows),updated_utc=time.time()))
        if child.wait() != 0:
            raise RuntimeError(f'Child failed; retain partial evidence: {script}, seed {seed}')
    for seed in SEEDS:
        if seed == 20261116:
            case, proof = source, audit
            own_actor=own_audit=0
            inherited=True
        else:
            case,proof=out/f'seed{seed}',out/f'audit{seed}'
            assert not case.exists() and not proof.exists(), 'Partial evidence must be inspected, never restarted'
            execute('evaluate_twelve_observed_synthesis.py',[
                '--source-inventory',out/'inventories/source.json','--model-inventory',out/'inventories/model.json',
                '--port-inventory',out/'inventories/port.json','--runtime',manifest['runtime'],
                '--seed',seed,'--out',case])
            result=read(case/'summary.json')
            current=read(case/'frozen_manifest.json')
            assert current['bundle_sha256'] == manifest['source_bundle_sha256']
            assert current['source_sha256'] == manifest['source_sha256']
            assert current['solver_receipt_sha256'] == manifest['solver_receipt_sha256']
            for name in ALGORITHM:
                assert current['helper_sha256'][name] == manifest['helper_sha256'][name]
            own_actor=result['inclusive_native_steps']; own_audit=0; inherited=False
            if result['result']['success']:
                execute('audit_twelve_synthesis_case.py',['--run',case,'--out',proof])
                own_audit=read(proof/'summary.json')['native_steps']
        result=read(case/'summary.json')
        if result['result']['success']:
            check=read(proof/'summary.json')
            assert (proof/'completed.json').exists() and check['passed'] and check['actual_cold_byte_exact']
            assert check['case_summary_sha256'] == sha(case/'summary.json')
            assert check['actual_sha256'] == sha(case/'actual.npz') and check['physical_velocities'] == 38
        rows.append(dict(seed=seed,run=str(case),audit=str(proof) if own_audit or inherited else None,
            success=result['result']['success'],inherited=inherited,
            new_actor_native_steps=own_actor,new_audit_native_steps=own_audit,
            source_case_summary_sha256=sha(case/'summary.json'),
            source_audit_summary_sha256=sha(proof/'summary.json') if own_audit or inherited else None))
        summary=dict(results=rows,passes=sum(row['success'] for row in rows),
            cohort_finished=len(rows)==12,stopped_on_failure=not rows[-1]['success'],
            inclusive_native_steps=sum(r['new_actor_native_steps']+r['new_audit_native_steps'] for r in rows),
            controller_identity=manifest['controller_identity'],qualified=False,reserved_cases_consumed=0,
            known_nominal_evidence_sha256=manifest['inventory_sha256']['source.json'],
            scope='twelve original-noise development starts, plus separately proven saved nominal; first noisy case inherited at zero new cost')
        write_json(out/'summary.json',summary)
        print(json.dumps(dict(seed=seed,passes=summary['passes'],cases=len(rows))),flush=True)
        if not rows[-1]['success']:
            break
    write_json(out/'status.json',dict(state='completed',passes=summary['passes'],completed_cases=len(rows)))
    write_json(out/'completed.json',dict(status='completed',all_passed=summary['passes']==12))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-inventory');p.add_argument('--model-inventory')
    p.add_argument('--port-inventory');p.add_argument('--first-inventory');p.add_argument('--runtime')
    p.add_argument('--out',required=True);p.add_argument('--worker',action='store_true')
    a=p.parse_args();out=Path(a.out).resolve()
    if a.worker:
        from spatialpole.experiment import write_json
        try:
            worker(out)
        except BaseException as exception:
            write_json(out/'failure.json',dict(type=type(exception).__name__,error=str(exception),
                accounting_complete=False,child_receipts_retained=True))
            raise
        return
    from spatialpole.experiment import freeze_inputs,write_json
    from evaluate_twelve_observed_synthesis import HELPERS
    first,source=read(a.first_inventory),read(a.source_inventory)
    assert first['passed'] and first['summary']['result']['success']
    assert first['source_sha256']==source['source_sha256']
    freeze_inputs(out,a)
    assert read(out/'inputs.json')['source_sha256']==first['source_sha256']
    (out/'helpers').mkdir();(out/'inventories').mkdir()
    helpers=set(HELPERS)|{'run_twelve_noisy_pilot.py','audit_twelve_synthesis_case.py','audit_spatial_synthesis_models.py'}
    for name in helpers:
        shutil.copyfile(Path(__file__).with_name(name),out/'helpers'/name)
    shutil.copytree(out/'source',out/'frozen_package/spatialpole')
    for name,path in (('source.json',a.source_inventory),('model.json',a.model_inventory),
                      ('port.json',a.port_inventory),('first.json',a.first_inventory)):
        shutil.copyfile(path,out/'inventories'/name)
    manifest=dict(helper_sha256={name:sha(out/'helpers'/name) for name in sorted(helpers)},
        source_sha256=first['source_sha256'],inventory_sha256={p.name:sha(p) for p in (out/'inventories').iterdir()},
        source_bundle_sha256=read(Path(first['run'])/'frozen_manifest.json')['bundle_sha256'],
        runtime=str(Path(a.runtime).resolve()),solver_receipt_sha256=sha(Path(a.runtime)/'receipt.json'),
        seeds=SEEDS,links=12,qualified=False,reserved_cases_consumed=0)
    identity=dict(algorithm_sha256={name:manifest['helper_sha256'][name] for name in ALGORITHM},
        source_sha256=manifest['source_sha256'],bundle_sha256=manifest['source_bundle_sha256'],
        solver_receipt_sha256=manifest['solver_receipt_sha256'],max_planning_native=7620324,max_case_native=7632324)
    manifest['controller_identity']=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    write_json(out/'frozen_manifest.json',manifest)
    env=dict(os.environ);env['PYTHONPATH']=str(out/'frozen_package')
    command=[sys.executable,'-u',str(out/'helpers'/Path(__file__).name),'--worker','--out',str(out)]
    write_json(out/'worker_command.json',dict(command=command,pythonpath=env['PYTHONPATH']))
    result=subprocess.run(command,env=env,creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__=='__main__':
    main()
