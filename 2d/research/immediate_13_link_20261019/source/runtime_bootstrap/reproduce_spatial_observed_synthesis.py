"""Relocated, hash-bound reproduction of an already recorded spatial case."""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import sys

ALGORITHM=('spatial_observed_synthesis.py','spatial_synthesis_geometry.py',
    'eleven_value_sequence.py','donor_conic_step.py','verify_eleven_sequence_refinement.py',
    'eleven_sequence_qp.py','trajectory_expert_support.py')
REPRODUCTION_PROTOCOL=2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def contained(folder,name):
    path=(folder/name).resolve()
    assert path.is_relative_to(folder.resolve())
    return path


def compare_planning_files(expected_case,generated):
    """Compare complete data, or exact recorded digests of regenerable models."""
    import numpy as np
    expected_case,generated=Path(expected_case),Path(generated)
    original=expected_case/'planning'
    before={p.relative_to(original).as_posix() for p in original.rglob('*.npz')}
    after={p.relative_to(generated).as_posix() for p in generated.rglob('*.npz')}
    omitted={}
    receipts=[expected_case/name for name in ('omitted_planning_models.json','omitted_planning_caches.json')
        if (expected_case/name).exists()]
    assert len(receipts)<=1, 'Ambiguous planning cache receipts'
    if receipts:
        receipt=receipts[0]
        proof=read(receipt)
        legacy=receipt.name=='omitted_planning_models.json'
        assert (proof['schema'],proof['reason'])==((1,'regenerable_native_model_cache') if legacy
            else (2,'regenerable_planning_caches'))
        assert proof['case_summary_sha256']==sha(expected_case/'summary.json')
        assert proof['case_manifest_sha256']==sha(expected_case/'frozen_manifest.json')
        omitted=proof['files']
        for name,row in omitted.items():
            pattern=(r'iteration[0-9]+/native_models\.npz' if legacy
                else r'iteration[0-9]+/(native_models|coordinates)\.npz')
            assert re.fullmatch(pattern,name), 'Only native model or value-coordinate caches may be omitted'
            assert re.fullmatch('[0-9a-f]{64}',row['sha256'])
            assert type(row['bytes']) is int and row['bytes']>0
        assert not before.intersection(omitted), 'Provided files cannot also be declared omitted'
    assert before.union(omitted)==after and after, 'Planning files differ from recorded inventory'
    parity=[]
    for name in sorted(after):
        new=contained(generated,name)
        if name in omitted:
            row=omitted[name]
            assert new.stat().st_size==row['bytes'] and sha(new)==row['sha256'], 'Regenerated model differs from recorded file digest'
            same=True;comparison='recorded_model_digest'
        else:
            old=contained(original,name)
            with np.load(old) as x,np.load(new) as y:
                assert set(x.files)==set(y.files)
                for key in x.files:
                    xx,yy=x[key],y[key]
                    assert xx.dtype==yy.dtype and xx.shape==yy.shape and xx.tobytes()==yy.tobytes()
            same=sha(old)==sha(new);comparison='provided_arrays'
        parity.append(dict(path=name,arrays_byte_exact=True,file_byte_exact=same,comparison=comparison))
    return parity


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',required=True)
    p.add_argument('--runtime-root',required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--out',required=True);p.add_argument('--expected-case')
    p.add_argument('--query-parity-only',action='store_true');a=p.parse_args()
    study,runtime,out=map(lambda x:Path(x).resolve(),(a.study,a.runtime_root,a.out))
    assert not out.exists()
    assert (a.query_parity_only and a.seed==20261016 and not a.expected_case) or (
        not a.query_parity_only and a.seed!=20261016 and a.expected_case), (
        'Use nominal query parity, or explicitly reproduce an existing noisy case; no new samples')
    m=read(study/'frozen_manifest.json');links=m['links'];assert links>=12
    for section,folder in (('helper_sha256','helpers'),('source_sha256','frozen_package/spatialpole'),
                           ('source_bundle_sha256','reference')):
        for name,digest in m[section].items():
            assert sha(contained(study/folder,name))==digest
    assert sha(runtime/'receipt.json')==m['solver_receipt_sha256']
    for name,digest in read(runtime/'receipt.json')['files_sha256'].items():
        assert sha(contained(runtime,name))==digest
    sys.path.insert(0,str(study/'frozen_package'));sys.path.insert(0,str(study/'helpers'))
    import numpy as np
    import spatialpole
    import spatial_observed_synthesis as synthesis
    from spatialpole import Config, SpatialCartpole
    from spatialpole.experiment import freeze_inputs,write_json
    from spatialpole.native_mpc import exact_bytes
    from donor_conic_step import solver_module
    assert Path(spatialpole.__file__).resolve().parent==study/'frozen_package/spatialpole'
    assert Path(synthesis.__file__).resolve()==study/'helpers/spatial_observed_synthesis.py'
    config=Config(n_links=links);budget=synthesis.planning_budget(config)
    assert m.get('max_planning_native',budget)==budget
    assert m.get('max_case_native',budget+12000)==budget+12000
    payload=dict(algorithm_sha256={name:m['helper_sha256'][name] for name in ALGORITHM},
        source_sha256=m['source_sha256'],bundle_sha256=m['source_bundle_sha256'],
        solver_receipt_sha256=m['solver_receipt_sha256'],max_planning_native=budget,
        max_case_native=budget+12000)
    digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
    identities={digest(dict(links=links,**payload))}
    if links==12:identities.add(digest(payload))
    assert m['controller_identity'] in identities
    solver=solver_module(runtime)
    assert Path(solver.__file__).resolve().is_relative_to(runtime/'site')
    driver='evaluate_twelve_observed_synthesis' if links==12 else 'evaluate_spatial_observed_case'
    evaluator=importlib.import_module(driver)
    assert Path(evaluator.__file__).resolve()==study/'helpers'/(driver+'.py')
    expected_case=Path(a.expected_case).resolve() if a.expected_case else None
    if expected_case:
        assert (expected_case/'completed.json').exists()
        old,cm=read(expected_case/'summary.json'),read(expected_case/'frozen_manifest.json')
        assert old['result']['success'] and old['result']['seed']==a.seed
        assert old['result']['angle_noise']==.02 and cm['config']==config.__dict__
        assert cm['source_sha256']==m['source_sha256'] and cm['bundle_sha256']==m['source_bundle_sha256']
        assert cm['solver_receipt_sha256']==m['solver_receipt_sha256']
        for name in ALGORITHM:assert cm['helper_sha256'][name]==m['helper_sha256'][name]
    freeze_inputs(out,a)
    assert read(out/'inputs.json')['runtime']==read(study/'inputs.json')['runtime']
    assert read(out/'inputs.json')['source_sha256']==m['source_sha256']
    shutil.copytree(study/'helpers',out/'frozen_helpers')
    def load(path):
        with np.load(path) as z:return {name:z[name] for name in z.files}
    bundle=dict(v=load(study/'reference/input_nominal.npz'),
        m=load(study/'reference/native_models.npz'),c=load(study/'reference/coordinates.npz'),
        expected=load(study/'reference/actual.npz'),w=np.zeros((1500,2)))
    owner=SpatialCartpole(config);planning_cost=0;completed_actual_native=0
    try:
        initial,_=owner.reset(seed=a.seed,options=dict(mode='hanging',
            angle_noise=0. if a.query_parity_only else .02))
        if a.query_parity_only:exact_bytes(initial,bundle['v']['states'][0])
        plan=synthesis.synthesize(config,initial,bundle,runtime,out/'planning')
        planning_cost=read(out/'planning/summary.json')['inclusive_native_steps']
        assert plan is not None and plan.native_steps==planning_cost
        if a.query_parity_only:
            assert plan.reused_source and planning_cost==0
            owner.set_state(initial);policy=synthesis.PlannedPolicy(owner,plan)
            try:
                for t,state in enumerate(plan.expected['states'][:-1]):
                    owner.set_state(state);raw=policy(state,t)
                    exact_bytes(raw,plan.expected['raw_actions'][t])
                    exact_bytes(owner.deliver(raw),plan.expected['actions'][t])
                    policy.observe_transition(plan.expected['states'][t+1],t,plan.expected['actions'][t])
            finally:policy.close()
            result=dict(query_callbacks=1500,inclusive_native_steps=0,
                actual_benchmark_episodes=0,arrays_byte_exact=True,
                scope='relocated frozen imports, source callbacks and solver root; zero physics')
        else:
            assert planning_cost==old['planning_native_steps']
            row=evaluator.actual_episode(config,plan,a.seed,out)
            completed_actual_native=row['native_steps']+row['proof_native_steps']
            assert row['success'] and row['steps']==1500 and row['replay_exact']
            actual,prior=load(out/'actual.npz'),load(expected_case/'actual.npz')
            assert set(actual)==set(prior)
            for name in actual:exact_bytes(actual[name],prior[name])
            parity=compare_planning_files(expected_case,out/'planning')
            result=dict(result=row,inclusive_native_steps=planning_cost+row['native_steps']+row['proof_native_steps'],
                planning_native_steps=planning_cost,actual_benchmark_episodes=1,arrays_byte_exact=True,
                planning_parity=parity,expected_case=dict(case_summary_sha256=sha(expected_case/'summary.json'),
                    actual_tape_sha256=sha(expected_case/'actual.npz'),all_arrays_byte_exact=True,
                    actual_file_byte_exact=sha(expected_case/'actual.npz')==sha(out/'actual.npz')),
                scope='complete recorded-case resynthesis, original-reset feedback and cold replay; no new sample')
        assert owner.native_steps==0
        write_json(out/'summary.json',dict(passed=True,**result,seed=a.seed,links=links,
            query_parity_only=a.query_parity_only,controller_identity=m['controller_identity'],
            study_manifest_sha256=sha(study/'frozen_manifest.json'),
            solver_receipt_sha256=sha(runtime/'receipt.json'),
            imported_package=str(Path(spatialpole.__file__).resolve()),
            imported_synthesis=str(Path(synthesis.__file__).resolve()),
            imported_solver=str(Path(solver.__file__).resolve()),
            qualified=False,fresh_noisy_samples=0,reserved_cases_consumed=0))
        write_json(out/'completed.json',dict(status='completed'))
        print(json.dumps(dict(passed=True,links=links,query_only=a.query_parity_only,
            native_steps=result['inclusive_native_steps'])),flush=True)
    except BaseException as exception:
        write_json(out/'failure.json',dict(type=type(exception).__name__,error=str(exception),
            completed_planning_native_steps=planning_cost,geometry_native_steps=owner.native_steps,
            completed_actual_native_steps=completed_actual_native,
            owned_partial_counters=['planning/failure.json','actual_failure.json'],accounting_complete=False))
        raise
    finally:owner.close()


if __name__=='__main__':main()
