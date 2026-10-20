"""Reject partial development evidence before any fresh initialization exists."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


ROOT=Path(__file__).resolve().parents[1]
EXPECTED=[20261016]+list(range(20261116,20261128))+list(range(20261416,20261448))
FRESH=list(range(20263816,20263848))


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


class IncompleteDevelopment(RuntimeError):pass


def require_all45(proof,candidate_folder):
    """Pure evidence admission; this function never reserves or samples a case."""
    proof,candidate_folder=Path(proof),Path(candidate_folder)
    assert (proof/'completed.json').exists()
    s=read(proof/'summary.json');assert s['passed'] is True and s['links']==14 and s['qualified'] is False
    if not (s['all45_development_passed'] is True and s['fresh_gate_satisfied'] is True and s['verified_development_states']==45):
        raise IncompleteDevelopment(f"Only{s['verified_development_states']}/45 development states; all45={s['all45_development_passed']},fresh_gate={s['fresh_gate_satisfied']}")
    assert [row['seed'] for row in s['results']]==EXPECTED and len(s['results'])==45
    assert s['native_steps']==s['new_controller_callbacks']==s['new_solver_calls']==0
    assert s['fresh_qualification_activated'] is False and s['reserved_cases_consumed']==s['new_noisy_samples']==0
    m=read(proof/'frozen_manifest.json');assert sha(proof/'frozen_manifest.json')==s['source_manifest_sha256']
    assert m['require_all45'] is True and m['through_seed']==20261447 and m['max_native_steps']==0
    for name,value in m['helper_sha256'].items():
        assert Path(name).name==name and sha(proof/name)==value==sha(Path(__file__).with_name(name))
    # The potentially expensive candidate import and packet checks occur only
    # after exact development coverage has passed. No fresh label is accessed.
    sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
    from delivered_candidate14 import candidate
    bank,runtime,cm=candidate(candidate_folder)
    assert cm['controller_identity']==m['controller_identity']==s['controller_identity']
    assert sha(candidate_folder/'frozen_manifest.json')==m['candidate_manifest_sha256']
    assert cm['source_sha256']==m['source_sha256'] and cm['registered_development_seeds']==EXPECTED
    assert cm['fresh_qualification_seeds']==FRESH and cm['fresh_qualification_activated'] is False
    assert len(bank)==15 and s['exact_source_constructor_proofs_inherited']==15
    bindings={row['seed']:row for row in s['bindings']};assert sorted(bindings)==list(range(20261418,20261448))
    for row in s['results']:
        assert row['commands']==1500 and row['seconds']==30. and row['best_hold_seconds']>=5.
        assert row['minimum_native_substep_margin']>=0 and row['peak_delivered_force']<=80.
        if row['seed']<20261418:
            assert row['proof_kind']=='unchanged_exact_source_constructor'
            assert sha(Path(row['source_inventory']))==row['source_inventory_sha256']
        else:
            path=Path(row['witness_inventory']);w=read(path);b=bindings[row['seed']]
            assert sha(path)==row['witness_inventory_sha256']==b['witness_inventory_sha256']
            prep=path.with_name(path.name.replace('_witness_inventory_','_inventory_'))
            assert sha(prep)==b['preparation_inventory_sha256']==w['parent_inventory_sha256']
            assert w['seed']==row['seed'] and w['passed'] and w['task_success'] and not w['qualified']
            assert w['controller_identity']==cm['controller_identity'] and w['actual_sha256']==row['actual_sha256']
    return dict(passed=True,controller_identity=cm['controller_identity'],development_states=45,
        development_proof_summary_sha256=sha(proof/'summary.json'),development_proof_manifest_sha256=sha(proof/'frozen_manifest.json'),
        candidate_manifest_sha256=sha(candidate_folder/'frozen_manifest.json'),fresh_seeds=FRESH,
        fresh_registration_or_sampling_performed=False,qualified=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--proof',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--out',required=True);p.add_argument('--expect-incomplete',action='store_true');a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    shutil.copyfile(Path(__file__),out/'driver.py')
    def write(name,value):(out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    write('inputs.json',dict(args=vars(a),command=sys.argv,driver_sha256=sha(out/'driver.py'),
        proof_summary_sha256=sha(Path(a.proof)/'summary.json'),max_native_steps=0,max_wall_seconds=900))
    try:
        try:
            admitted=require_all45(a.proof,a.candidate);rejected=False;reason=None
        except IncompleteDevelopment as exc:
            if not a.expect_incomplete:raise
            admitted=None;rejected=True;reason=str(exc)
        if a.expect_incomplete:assert rejected,'Expected partial proof was admitted'
        assert time.monotonic()-started<900.
        result=dict(passed=True,qualified=False,task_success=False,expected_partial_proof_rejected=rejected,reason=reason,admission=admitted,
            native_steps=0,new_solver_calls=0,new_controller_callbacks=0,fresh_labels_opened=0,fresh_cases_reserved=0,
            candidate_numerical_loader_not_called_on_partial_proof=rejected,
            scope='fresh-boundary evidence gate only; no reset, reservation, qualification run or accepted count')
        write('summary.json',result);write('completed.json',dict(status='completed'));print(result,flush=True)
    except BaseException as exc:
        write('failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=0,accounting_complete=True));raise


if __name__=='__main__':main()
