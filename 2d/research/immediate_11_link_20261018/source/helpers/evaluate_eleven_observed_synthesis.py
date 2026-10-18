"""Run the standard development pilot from a fully frozen state-only planner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
import spatialpole
from spatialpole import Config, SpatialCartpole
from spatialpole.experiment import episode, freeze_inputs, write_json
from eleven_synthesis_protocol import controller_identity, seeds, validate_parent, HARNESSES

HELPERS = ('evaluate_eleven_observed_synthesis.py', 'eleven_observed_synthesis.py',
    'eleven_initial_continuation.py', 'eleven_secant_corrector.py',
    'eleven_value_sequence.py', 'prepare_eleven_observed_target.py',
    'prepare_eleven_newton_seed.py', 'probe_eleven_native_chart.py',
    'eleven_compression_features.py', 'eleven_load_support.py',
    'donor_conic_step.py', 'verify_eleven_sequence_refinement.py',
    'eleven_sequence_qp.py', 'trajectory_expert_support.py', 'eleven_synthesis_protocol.py')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def worker(out):
    from eleven_initial_continuation import load
    from eleven_observed_synthesis import synthesize, PlannedPolicy, MAX_MODELS, MAX_NATIVE
    assert Path(spatialpole.__file__).resolve().parent == (out/'frozen_package/spatialpole').resolve()
    manifest = read(out/'frozen_manifest.json')
    cohort = manifest.get('cohort', 'pilot13')
    assert manifest['seeds'] == seeds(cohort)
    assert manifest['max_models'] == MAX_MODELS and manifest['max_planning_native'] == MAX_NATIVE
    for name, digest in manifest['helper_sha256'].items():
        assert sha(out/'helpers'/name) == digest
    for name, digest in manifest['source_sha256'].items():
        assert sha(out/'frozen_package/spatialpole'/name) == digest
    for name, digest in manifest['bundle_sha256'].items():
        assert sha(out/'reference'/name) == digest
    runtime = Path(manifest['runtime'])
    assert sha(runtime/'receipt.json') == manifest['solver_receipt_sha256']
    from donor_conic_step import solver_module
    solver = solver_module(runtime)
    write_json(out/'solver_preflight.json', dict(passed=True, native_steps=0,
        clarabel_version=solver.__version__, solver_receipt_sha256=manifest['solver_receipt_sha256']))
    bundle = dict(v=load(out/'reference/virtual_reference.npz'),
        m=load(out/'reference/native_models.npz'), c=load(out/'reference/coordinates.npz'),
        w=load(out/'reference/model_plan.npz')['w'], expected=load(out/'reference/accepted_anchor.npz'))
    if not (out/'source_guard.json').exists():
        source_plan = synthesize(bundle['expected']['states'][0], bundle, manifest['runtime'], out/'source_guard')
        guard = SpatialCartpole(Config(n_links=11))
        try:
            guard.set_state(bundle['expected']['states'][0]); policy = PlannedPolicy(guard, source_plan)
            try:
                for t, state in enumerate(bundle['expected']['states'][:-1]):
                    guard.set_state(state); policy(state, t)
                    policy.observe_transition(bundle['expected']['states'][t+1], t, bundle['expected']['actions'][t])
            finally:
                policy.close()
            assert guard.native_steps == source_plan.native_steps == 0
        finally:
            guard.close()
        write_json(out/'source_guard.json', dict(passed=True, callbacks=1500, native_steps=0))
    rows = []; total = 0; stopped = False
    write_json(out/'worker.json', dict(pid=os.getpid(), started_utc=time.time()))
    for seed in manifest['seeds']:
        folder = out/f'seed{seed}'; case_summary = folder/'summary.json'
        if (folder/'completed.json').exists():
            row = read(case_summary)
            assert row['frozen_manifest_sha256'] == sha(out/'frozen_manifest.json')
        else:
            assert not folder.exists(), 'Partial case retained: inspect it before resuming; do not repeat its physics.'
            folder.mkdir()
            inherited = manifest.get('inherited_cases', {}).get(str(seed))
            if inherited:
                source = Path(inherited['source_case'])
                assert sha(source/'summary.json') == inherited['case_summary_sha256']
                assert sha(source/'actual.npz') == inherited['actual_sha256']
                assert sha(Path(inherited['case_audit'])/'summary.json') == inherited['case_audit_sha256']
                original = read(source/'summary.json'); assert original['result']['success']
                row = dict(original, inherited_from=str(source),
                    source_case_summary_sha256=inherited['case_summary_sha256'],
                    source_audit_summary_sha256=inherited['case_audit_sha256'],
                    source_inclusive_native_steps=original['inclusive_native_steps'],
                    source_planning_native_steps=original['planning_native_steps'],
                    planning_native_steps=0, inclusive_native_steps=0, new_actual_benchmark_episodes=0,
                    frozen_manifest_sha256=sha(out/'frozen_manifest.json'))
            else:
                noise = 0. if seed == 20261016 else .02
                env = SpatialCartpole(Config(n_links=11))
                try:
                    observed, _ = env.reset(seed=seed, options=dict(mode='hanging', angle_noise=noise))
                    assert env.native_steps == 0
                finally:
                    env.close()
                start = time.perf_counter()
                plan = synthesize(observed, bundle, manifest['runtime'], folder/'planning')
                planning_seconds = time.perf_counter()-start
                planning = read(folder/'planning/summary.json')
                if plan is None:
                    result = dict(seed=seed, angle_noise=noise, initialization='hanging', success=False,
                        planning_failed=True, native_steps=0, proof_native_steps=0,
                        actual_benchmark_episodes=0)
                else:
                    result = episode(Config(n_links=11), lambda e:PlannedPolicy(e, plan),
                        seed, 'hanging', noise, folder/'actual.npz')
                    result['actual_benchmark_episodes'] = 1
                    assert result['replay_exact'] and result['steps'] == 1500 and result['success']
                count = planning['inclusive_native_steps']+result['native_steps']+result['proof_native_steps']
                assert count <= MAX_NATIVE+12000
                row = dict(result=result, planning_wall_seconds=planning_seconds,
                    planning_native_steps=planning['inclusive_native_steps'], inclusive_native_steps=count,
                    frozen_manifest_sha256=sha(out/'frozen_manifest.json'), qualified=False,
                    new_actual_benchmark_episodes=result['actual_benchmark_episodes'],
                    reserved_cases_consumed=int(cohort == 'qualification32'))
            write_json(case_summary, row); write_json(folder/'completed.json', dict(status='completed'))
        rows.append(row); total += row['inclusive_native_steps']
        stopped = not row['result']['success']
        finished = len(rows) == len(manifest['seeds'])
        summary = dict(cohort=cohort, results=rows, passes=sum(r['result']['success'] for r in rows),
            expected_episodes=len(manifest['seeds']), cohort_finished=finished, stopped_on_failure=stopped,
            inclusive_native_steps=total, frozen_manifest_sha256=sha(out/'frozen_manifest.json'),
            actual_benchmark_episodes=sum(r.get('new_actual_benchmark_episodes', r['result']['actual_benchmark_episodes']) for r in rows),
            inherited_cases=sum('inherited_from' in r for r in rows), controller_identity=controller_identity(manifest),
            qualification_attempt=cohort == 'qualification32',
            qualification_gate_passed=cohort == 'qualification32' and finished and not stopped,
            reserved_start=20263516 if cohort == 'qualification32' else None,
            qualified=False, reserved_cases_consumed=len(rows) if cohort == 'qualification32' else 0,
            scope='fixed observed-state synthesis, original task; inherited development evidence costs zero new physics; qualification is fresh')
        write_json(out/'summary.json', summary)
        print(json.dumps(dict(seed=seed, success=row['result']['success'],
            passes=summary['passes'], cases=len(rows), native_steps=total)), flush=True)
        if stopped:
            break
    write_json(out/'completed.json', dict(status='completed', all_passed=finished and not stopped))


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', required=True)
    p.add_argument('--reference'); p.add_argument('--audit'); p.add_argument('--runtime')
    p.add_argument('--cohort', choices=['pilot13', 'development45', 'qualification32'], default='pilot13')
    p.add_argument('--parent-run'); p.add_argument('--parent-audit')
    p.add_argument('--worker', action='store_true'); a = p.parse_args(); out = Path(a.out).resolve()
    if a.worker:
        worker(out); return
    parent = parent_manifest = None; inherited = {}
    if a.cohort == 'pilot13':
        assert a.reference and a.audit and a.runtime
        reference = Path(a.reference); audit = read(Path(a.audit)/'summary.json')
        assert audit['passed'] and any(r['full_native_success'] and
            r['source_summary_sha256'] == sha(reference/'summary.json') for r in audit['results'])
        candidate = None
    else:
        assert a.parent_run and a.parent_audit
        parent = Path(a.parent_run).resolve()
        _, parent_proof, parent_manifest = validate_parent(parent, a.parent_audit, a.cohort)
        candidate = Path(parent_manifest.get('candidate_run', str(parent)))
        reference = candidate/'reference'
        if a.cohort == 'development45':
            for seed in seeds('pilot13'):
                proof_ref = parent_proof['case_audits'][str(seed)]
                proof_path = Path(proof_ref['path']); proof = read(proof_path/'summary.json')
                source = parent/f'seed{seed}'
                assert sha(proof_path/'summary.json') == proof_ref['summary_sha256']
                assert proof['actual_sha256'] == sha(source/'actual.npz')
                inherited[str(seed)] = dict(source_case=str(source),
                    case_summary_sha256=sha(source/'summary.json'), actual_sha256=sha(source/'actual.npz'),
                    case_audit=str(proof_path), case_audit_sha256=sha(proof_path/'summary.json'))
    runtime = Path(a.runtime or parent_manifest['runtime']).resolve()
    if not (runtime/'receipt.json').exists() and runtime.name == 'site':
        runtime = runtime.parent
    assert (runtime/'receipt.json').exists() and (runtime/'site').is_dir()
    freeze_inputs(out, a); (out/'helpers').mkdir(); (out/'reference').mkdir()
    for name in HELPERS:
        source = Path(__file__).parent/name if candidate is None or name in HARNESSES else candidate/'helpers'/name
        shutil.copyfile(source, out/'helpers'/name)
    package_source = out/'source' if candidate is None else candidate/'frozen_package/spatialpole'
    shutil.copytree(package_source, out/'frozen_package/spatialpole')
    for name in ('virtual_reference.npz', 'native_models.npz', 'coordinates.npz', 'model_plan.npz'):
        shutil.copyfile(reference/name, out/'reference'/name)
    shutil.copyfile((reference.parent if candidate is None else reference)/'accepted_anchor.npz', out/'reference/accepted_anchor.npz')
    parent_inventory = read(Path(__file__).resolve().parents[1]/'output/eleven_full_noise_witness_inventory_001.json')
    sources = {name:sha(out/'source'/name) for name in parent_inventory['source_sha256']}
    assert sources == parent_inventory['source_sha256']
    receipt = runtime/'receipt.json'
    assert sha(receipt) == parent_inventory['solver_receipt_sha256']
    manifest = dict(
        helper_sha256={name:sha(out/'helpers'/name) for name in HELPERS}, source_sha256=sources,
        bundle_sha256={p.name:sha(p) for p in sorted((out/'reference').iterdir())},
        source_audit_sha256=sha(Path(a.audit if parent is None else a.parent_audit)/'summary.json'),
        runtime=str(runtime), solver_receipt_sha256=sha(receipt),
        max_models=4, max_planning_native=7044324, max_case_native=7056324,
        cohort=a.cohort, seeds=seeds(a.cohort), inherited_cases=inherited,
        candidate_run=str(out if candidate is None else candidate), qualified=False,
        scope='frozen state-only planner; seed labels are confined to original reset and reporting')
    manifest['controller_identity'] = controller_identity(manifest)
    if parent is not None:
        assert manifest['controller_identity'] == controller_identity(parent_manifest)
        assert read(out/'inputs.json')['runtime'] == read(parent/'inputs.json')['runtime']
        shutil.copyfile(parent/'summary.json', out/'parent_summary.json')
        shutil.copyfile(Path(a.parent_audit)/'summary.json', out/'parent_integrity_audit.json')
    write_json(out/'frozen_manifest.json', manifest)
    if a.cohort == 'qualification32':
        reservation = out.parent/'reserved_11_immediate_20263516.json'
        with reservation.open('x') as stream:
            json.dump(dict(run=str(out), seeds=seeds(a.cohort), controller_identity=manifest['controller_identity'],
                frozen_manifest_sha256=sha(out/'frozen_manifest.json')), stream, indent=2)
    env = dict(os.environ); env['PYTHONPATH'] = str(out/'frozen_package')
    command = [sys.executable, '-u', str(out/'helpers'/Path(__file__).name), '--worker', '--out', str(out)]
    write_json(out/'worker_command.json', dict(command=command, pythonpath=env['PYTHONPATH']))
    result = subprocess.run(command, env=env)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
