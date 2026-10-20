"""Consume precisely the next fresh label before observing its original reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
from delivered_fresh_context14_parallel import (FRESH, REGISTRATION, cohort, prefix_prior, read, sha, serialized_args)


def main():
    p = argparse.ArgumentParser()
    for name in ('cohort', 'out'):
        p.add_argument('--' + name, required=True, type=Path)
    p.add_argument('--previous-witness-inventory', type=Path)
    p.add_argument('--seed', required=True, type=int)
    a = p.parse_args()
    a.cohort,a.out = a.cohort.resolve(),a.out.resolve()
    bank, runtime, cm, m, development = cohort(a.cohort)
    assert a.out == a.cohort/'cases'/str(a.seed)/'registration'
    assert read(a.cohort/'execution_activation.json')['cohort_manifest_sha256'] == sha(a.cohort/'frozen_manifest.json')
    index = FRESH.index(a.seed)
    consumed = read(a.cohort / 'consumed_cases.json')['seeds']
    assert consumed == FRESH[:index]
    prior = prefix_prior(a.cohort, cm, m, development, index, a.previous_witness_inventory)
    assert not a.out.exists()
    from spatialpole import Config, SpatialCartpole
    from spatialpole.experiment import freeze_inputs, write_json
    import numpy as np
    freeze_inputs(a.out, serialized_args(a))
    assert read(a.out / 'inputs.json')['source_sha256'] == cm['source_sha256']
    claim = a.cohort / 'case_claims' / f'{a.seed}.json'
    with claim.open('x', encoding='utf-8') as stream:
        json.dump(dict(seed=a.seed,fresh_index=index,registration=str(a.out.resolve()),pid=os.getpid(),
            cohort_manifest_sha256=sha(a.cohort / 'frozen_manifest.json')),stream,indent=2)
    pending = a.cohort / f'consumed_cases_{a.seed}.pending'
    with pending.open('x', encoding='utf-8') as stream:
        json.dump(dict(seeds=FRESH[:index + 1],controller_identity=cm['controller_identity']),stream,indent=2)
    os.replace(pending, a.cohort / 'consumed_cases.json')
    env = SpatialCartpole(Config(n_links=14))
    try:
        state, _ = env.reset(seed=a.seed,options=dict(mode='hanging',angle_noise=.02))
        assert env.native_steps == 0
        np.savez_compressed(a.out / 'registered_initial.npz',state=state)
        manifest = dict(registration_protocol=REGISTRATION,cohort=str(a.cohort.resolve()),
            cohort_manifest_sha256=sha(a.cohort / 'frozen_manifest.json'),fresh_index=index,
            seed=a.seed,config=cm['config'],noise=.02,conditioning_steps=0,new_noisy_samples=1,
            candidate=m['candidate'],candidate_manifest_sha256=m['candidate_manifest_sha256'],controller_identity=cm['controller_identity'],
            case_claim_sha256=sha(claim),registered_initial_sha256=sha(a.out / 'registered_initial.npz'),
            observed_initial_bytes_sha256=hashlib.sha256(state.tobytes()).hexdigest(),
            previous_witness_inventory=None,
            previous_witness_inventory_sha256=None,
            previous_closed_native_steps=prior['total_new_target_preparation_native_steps'],
            previous_closed_native_bounds=prior['total_new_target_preparation_native_steps_bounds'],
            fresh_case_ordinal_before=index,completed_development_proofs_before=45,
            fresh_qualification_activated=True,qualified=False)
        write_json(a.out / 'frozen_manifest.json',manifest)
        write_json(a.out / 'summary.json',dict(passed=True,qualified=False,task_success=False,native_steps=0,
            seed=a.seed,controller_identity=cm['controller_identity'],new_noisy_samples=1,reserved_cases_consumed=1,
            fresh_observation_stored_before_prepare=True))
        write_json(a.out / 'completed.json',dict(status='completed'))
    except BaseException as exc:
        write_json(a.out / 'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=env.native_steps,
            fresh_seed_claimed=True,label_must_not_be_redrawn=True,accounting_complete=True))
        raise
    finally:
        env.close()


if __name__ == '__main__':
    main()
