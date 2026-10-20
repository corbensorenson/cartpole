"""Join all32 original-task proofs and inclusive ledgers without physical reruns."""
import argparse
import hashlib
from pathlib import Path
import shutil
import time
from delivered_fresh_context14_parallel import FRESH,CASE_NATIVE_CAP,cohort,previous,read,sha,serialized_args
from delivered_fresh_gate14 import require_all45
from spatialpole.experiment import freeze_inputs,write_json
from spatialpole.native_mpc import exact_bytes
from audit_spatial_synthesis_models import load
from audit_delivered_development14 import score,studies


def main():
    p = argparse.ArgumentParser()
    for name in ('cohort','out'):
        p.add_argument('--'+name,required=True,type=Path)
    a = p.parse_args()
    a.cohort,a.out = a.cohort.resolve(),a.out.resolve()
    bank,runtime,cm,m,development = cohort(a.cohort)
    admission = require_all45(m['development_proof'],m['candidate'])
    assert admission['controller_identity'] == cm['controller_identity']
    assert read(a.cohort/'consumed_cases.json')['seeds'] == FRESH
    assert sorted(int(path.stem) for path in (a.cohort/'case_claims').glob('*.json')) == FRESH
    assert sorted(int(path.name) for path in (a.cohort/'cases').iterdir()) == FRESH
    for seed in FRESH:
        receipt = a.cohort/'cases'/str(seed)/'witness_inventory.json'
        assert receipt.exists() and read(receipt)['passed'] and read(receipt)['task_success']
    freeze_inputs(a.out,serialized_args(a))
    shutil.copyfile(a.cohort/'frozen_manifest.json',a.out/'cohort-manifest.json')
    started = time.monotonic()
    rows = []
    ledger = development['total_new_target_preparation_native_steps']
    bounds = development['total_new_target_preparation_native_steps_bounds']
    initial_ledger = ledger
    prior_inventory = None
    incremental_native = 0

    def checkpoint():
        assert time.monotonic()-started < 2400

    try:
        for index,seed in enumerate(FRESH):
            ledger = development['total_new_target_preparation_native_steps']
            bounds = development['total_new_target_preparation_native_steps_bounds']
            case = a.cohort/'cases'/str(seed)
            prep_path,witness_path = case/'preparation_inventory.json',case/'witness_inventory.json'
            prep = read(prep_path)
            witness = previous(witness_path,cm,seed+1)
            assert prep['passed'] and prep['plan_found'] and not prep['task_success'] and not prep['qualified']
            assert prep['seed'] == witness['seed'] == seed
            assert prep['controller_identity'] == witness['controller_identity'] == cm['controller_identity']
            assert prep['cohort_manifest_sha256'] == witness['cohort_manifest_sha256'] == sha(a.cohort/'frozen_manifest.json')
            assert prep['fresh_case_ordinal_before'] == index and witness['fresh_case_ordinal'] == index+1
            assert prep['source_registration_fresh_cases_consumed'] == witness['source_registration_fresh_cases_consumed'] == 1
            assert prep['parent_inventory'] == (str(prior_inventory) if prior_inventory else None)
            assert prep['parent_inventory_sha256'] == (sha(prior_inventory) if prior_inventory else None)
            assert witness['parent_inventory_sha256'] == sha(prep_path)
            assert prep['actor_native_steps'] <= 9072324 and prep['audit_native_steps'] <= 384324
            assert prep['new_native_steps'] == prep['actor_native_steps']+prep['audit_native_steps']
            assert witness['actor_native_steps'] == witness['audit_native_steps'] == 12000
            assert witness['new_native_steps'] == 24000
            assert prep['total_new_target_preparation_native_steps'] == ledger+prep['new_native_steps']
            assert prep['total_new_target_preparation_native_steps_bounds'] == [x+prep['new_native_steps'] for x in bounds]
            assert witness['total_new_target_preparation_native_steps'] == prep['total_new_target_preparation_native_steps']+24000
            assert witness['total_new_target_preparation_native_steps_bounds'] == [x+24000 for x in prep['total_new_target_preparation_native_steps_bounds']]
            studies(prep,checkpoint)
            studies(witness,checkpoint)
            pr,pa,wr,wa = map(Path,(prep['run'],prep['audit'],witness['run'],witness['audit']))
            ps,proof,ws,replica = [read(folder/'summary.json') for folder in (pr,pa,wr,wa)]
            assert proof['passed'] and proof['plan_found'] and proof['source_summary_sha256'] == sha(pr/'summary.json')
            assert proof['source_manifest_sha256'] == sha(pr/'frozen_manifest.json')
            assert ps['native_steps'] == proof['actor_native_steps'] == prep['actor_native_steps']
            assert proof['native_steps'] == prep['audit_native_steps']
            assert sha(pr/'summary.json') == prep['summary_sha256'] and sha(pa/'summary.json') == prep['audit_summary_sha256']
            manifest = read(pr/'frozen_manifest.json')
            assert manifest['qualification_attempt'] and manifest['noise'] == .02 and manifest['conditioning_steps'] == 0
            registration = Path(prep['registration'])
            rm = read(registration/'frozen_manifest.json')
            assert rm['seed'] == seed and rm['fresh_index'] == index and rm['new_noisy_samples'] == 1
            assert sha(registration/'registered_initial.npz') == rm['registered_initial_sha256']
            assert sha(registration/'frozen_manifest.json') == manifest['registration_manifest_sha256'] == proof['registration_manifest_sha256']
            assert sha(a.cohort/'case_claims'/f'{seed}.json') == rm['case_claim_sha256']
            assert rm['previous_closed_native_steps'] == ledger and rm['previous_closed_native_bounds'] == bounds
            tape = load(wr/'actual.npz')
            assert hashlib.sha256(tape['states'][0].tobytes()).hexdigest() == rm['observed_initial_bytes_sha256']
            exact_bytes(tape['states'][0],load(registration/'registered_initial.npz')['state'])
            exact_bytes(tape['states'][0],load(pr/'observed_state.npz')['state'])
            exact_bytes(tape['states'][0],load(pr/'planning/observed_state.npz')['state'])
            assert ws['task_success'] and replica['task_success'] and ws['result'] == witness['result']
            assert read(wr/'frozen_manifest.json')['qualification_attempt']
            assert read(wr/'frozen_manifest.json')['cohort_manifest_sha256'] == sha(a.cohort/'frozen_manifest.json')
            assert read(wr/'frozen_manifest.json')['inventory_sha256'] == sha(prep_path)
            row = score(tape,ws['result'],seed)
            row.update(witness_inventory=str(witness_path),witness_inventory_sha256=sha(witness_path),
                preparation_inventory_sha256=sha(prep_path),actual_sha256=witness['actual_sha256'],
                canonical_and_reproduction_arrays_byte_exact=True)
            rows.append(row)
            ledger,bounds = witness['total_new_target_preparation_native_steps'],witness['total_new_target_preparation_native_steps_bounds']
            incremental_native += prep['new_native_steps'] + witness['new_native_steps']
            prior_inventory = None
            checkpoint()
            write_json(a.out/'progress.json',dict(verified_fresh_cases=len(rows),native_steps=0))
        assert len(rows) == 32 and [row['seed'] for row in rows] == FRESH
        ledger = initial_ledger + incremental_native
        bounds = [x + incremental_native for x in development['total_new_target_preparation_native_steps_bounds']]
        assert incremental_native <= 32*CASE_NATIVE_CAP == m['total_native_cap']
        result = dict(passed=True,qualified=True,task_success=True,links=14,controller_identity=cm['controller_identity'],
            cohort='qualification32',noisy_passes=32,passes=32,case_seeds=FRESH,results=rows,
            source_cohort_manifest_sha256=sha(a.cohort/'frozen_manifest.json'),
            all45_development_gate_bound=True,source_reserved_cases_consumed=32,native_steps=0,new_native_steps=0,
            new_solver_calls=0,new_controller_callbacks=0,new_noisy_samples=0,reserved_cases_consumed=0,
            referenced_fresh_native_steps=ledger-initial_ledger,total_new_target_preparation_native_steps=ledger,
            total_new_target_preparation_native_steps_bounds=bounds,original_reset_and_independent_reproduction_pairs=32,
            minimum_hold_seconds=min(row['best_hold_seconds'] for row in rows),
            minimum_plane_margin=min(row['minimum_native_substep_margin'] for row in rows),
            maximum_force_norm=max(row['peak_delivered_force'] for row in rows),
            scope='complete frozen original-noise fresh32 physical/model/audit/reproduction and accounting proof; runtime fresh resynthesis, video and publication remain')
        write_json(a.out/'summary.json',result)
        write_json(a.out/'completed.json',dict(status='completed'))
        print({key:value for key,value in result.items() if key!='results'},flush=True)
    except BaseException as exc:
        write_json(a.out/'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=0,accounting_complete=True,checked_cases=len(rows)))
        raise


if __name__ == '__main__':
    main()
