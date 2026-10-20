"""One post-qualification resynthesis from extracted code and a saved observation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


NATIVE_CAP = 9072324 + 84000
WALL_CAP = 16200


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def worker(args):
    job_path = args.job.resolve()
    job_root,job = job_path.parent,read(job_path)
    payload = args.payload.resolve()
    out = args.worker_out.resolve() if args.worker_out else job_root/'execution'
    assert not out.exists() and not out.is_relative_to(job_root/'expected-planning')
    assert job['first_seed'] == 20263816 and job['max_native_steps'] == NATIVE_CAP
    assert job['max_wall_seconds'] == WALL_CAP and sha(__file__) == job['worker_sha256']
    for name,digest in job['payload_bootstrap_sha256'].items():
        path = (payload/name).resolve()
        assert not Path(name).is_absolute() and '..' not in Path(name).parts and path.is_relative_to(payload)
        assert sha(path) == digest
    for name in job['expected_planning_npz']:
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
    qualification = read(job_root/'source-qualification-summary.json')
    assert sha(job_root/'source-qualification-summary.json') == job['qualification_summary_sha256']
    assert qualification['qualified'] and qualification['noisy_passes'] == 32 and qualification['links'] == 14
    assert qualification['case_seeds'][0] == job['first_seed']
    assert qualification['controller_identity'] == job['controller_identity']
    out.mkdir(parents=True)
    write(out/'worker.json',dict(pid=os.getpid()))
    started = time.monotonic()
    layout,relocation,provenance = [read(payload/name) for name in ('layout.json','relocation.json','archive-provenance.json')]
    forbidden = {str(Path(p).resolve()).casefold() for p in layout['original_path_to_blob']}
    forbidden.update(str(Path(row['source']).resolve()).casefold() for row in layout['bootstrap'].values())
    directories = [Path(p).resolve() for p in relocation['original_scientific_code_directories']]
    directories += [Path(provenance['original_development_view']).resolve(),Path(job['original_cohort']).resolve()]
    expected_directory = (job_root/'expected-planning').resolve()
    other_expected = {(job_root/name).resolve() for name in ('expected-actual.npz','source-prepare-summary.json')}
    comparison_allowed = False

    def deny_original(event,values):
        if event != 'open' or not isinstance(values[0],(str,bytes,os.PathLike)):
            return
        path = Path(os.fsdecode(values[0])).resolve()
        if str(path).casefold() in forbidden or any(path.is_relative_to(p) for p in directories):
            raise RuntimeError('Original scientific/cohort path requested: '+str(path))
        if path.is_relative_to(expected_directory) or path in other_expected:
            flags = values[2] if len(values)>2 and isinstance(values[2],int) else 0
            mode = values[1] if len(values)>1 and isinstance(values[1],str) else ''
            writing = any(c in mode for c in 'wax+') or bool(flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if not comparison_allowed or writing:
                raise RuntimeError('Expected result accessed during preparation or opened for writing')

    sys.addaudithook(deny_original)
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(payload/'scripts'))
    sys.path.insert(0,str(payload/'src'))
    from materialize_delivered_candidate14 import materialize
    from delivered_candidate14 import candidate
    from delivered_planner14 import prepare,controller
    from spatialpole import Config,SpatialCartpole
    from spatialpole.native_mpc import exact_bytes
    from audit_spatial_synthesis_models import load
    import importlib.metadata
    import numpy as np
    import mujoco
    original_inputs = read(payload/'original-candidate/inputs.json')['runtime']
    environment = {name:os.environ[name] for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS') if name in os.environ}
    assert environment == original_inputs['thread_environment'] == {}
    assert sys.version == original_inputs['python'] and np.__version__ == original_inputs['numpy'] and mujoco.__version__ == original_inputs['mujoco']
    assert {name:importlib.metadata.version(name) for name in original_inputs['packages']} == original_inputs['packages']
    for name in ('spatialpole','delivered_candidate14','delivered_planner14','general_delivered_mean14','successful_profiles14'):
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(payload)

    def no_reset(*unused,**ignored):
        raise RuntimeError('Resynthesis reuses its recorded observation; reset is forbidden')

    SpatialCartpole.reset = no_reset
    materialize(payload,out/'runtime-view',require_hardlinks=True)
    bank,runtime,cm = candidate(out/'runtime-view')
    assert cm['controller_identity'] == job['controller_identity']
    assert sha(job_root/'observed-state.npz') == job['observed_file_sha256']
    observed = load(job_root/'observed-state.npz')['state']
    assert observed.shape == (102,) and observed.dtype == np.float64 and np.isfinite(observed).all()
    assert hashlib.sha256(observed.tobytes()).hexdigest() == job['observed_bytes_sha256']
    env = cold = policy = None
    preparation_native = 0
    preparation_returned = False
    states,raws,actions,tilts = [],[],[],[]
    try:
        planning_begun = time.monotonic()
        report = prepare(observed,bank,runtime,out/'planning')
        planning_wall_seconds = time.monotonic()-planning_begun
        preparation_returned = True
        preparation_native = report['native_steps']
        assert report['plan_found'] and preparation_native <= 9072324
        comparison_allowed = True
        expected_report = read(job_root/'source-prepare-summary.json')
        for key in ('mode','stage','selected_iteration','selected_profile_index','native_steps'):
            assert report[key] == expected_report[key]
        actual_files = {p.relative_to(out/'planning').as_posix() for p in (out/'planning').rglob('*.npz')}
        assert actual_files == set(job['expected_planning_npz'])
        planning_bindings = {}
        for name,expected_hash in job['expected_planning_npz'].items():
            reference = job_root/'expected-planning'/name
            assert sha(reference) == expected_hash
            actual,expected = load(out/'planning'/name),load(reference)
            assert actual.keys() == expected.keys()
            for key in actual:
                exact_bytes(actual[key],expected[key])
            planning_bindings[name] = dict(expected_sha256=expected_hash,regenerated_sha256=sha(out/'planning'/name),all_arrays_byte_exact=True)
        write(out/'planning-array-parity.json',planning_bindings)
        assert sha(job_root/'expected-actual.npz') == job['expected_actual_sha256']
        expected = load(job_root/'expected-actual.npz')
        exact_bytes(observed,expected['states'][0])
        env,cold = SpatialCartpole(Config(n_links=14)),SpatialCartpole(Config(n_links=14))
        env.set_state(observed)
        exact_bytes(env.state(),observed)
        policy = controller(env.config,report,out/'planning',bank)
        states.append(observed.copy())
        execution_started = time.monotonic()

        def protected():
            return (env.state().tobytes(),env.data.time,env.steps,env.streak,env.best_streak,
                env.min_plane_margin,env.warning_seen,env.native_steps,env.data.ctrl.tobytes())

        for phase in range(1500):
            before = protected()
            raw = policy(states[-1],phase)
            assert protected() == before
            exact_bytes(raw,expected['raw_actions'][phase])
            force = env.deliver(raw)
            exact_bytes(force,expected['actions'][phase])
            state,_,done,truncated,info = env.step_force(force)
            exact_bytes(state,expected['states'][phase+1])
            exact_bytes(env.tilts(),expected['tilts'][phase])
            states.append(state.copy());raws.append(raw.copy());actions.append(force);tilts.append(env.tilts())
            assert phase == 1499 or not (done or truncated)
            if phase%100 == 0:
                write(out/'progress.json',dict(stage='native execution',commands=phase+1,actual_native_steps=env.native_steps))
            assert time.monotonic()-execution_started < 900 and time.monotonic()-started < WALL_CAP
        replay = observed.copy()
        for phase,force in enumerate(actions):
            replay = cold.transition(replay,force)
            exact_bytes(replay,states[phase+1]);exact_bytes(cold.tilts(),tilts[phase])
        predictor = getattr(getattr(policy,'predictor',None),'native_steps',0)
        assert predictor <= 72000 and env.native_steps == cold.native_steps == 6000
        assert env.min_plane_margin == cold.min_plane_margin and env.warning_seen == cold.warning_seen
        assert time.monotonic()-execution_started < 900
        assert info['success'] and info['best_hold_seconds'] >= 5. and not info['physics_warning']
        assert info['plane_margin'] >= 0 and float(np.linalg.norm(np.asarray(actions).astype(float),axis=1).max()) <= 80.
        np.savez_compressed(out/'actual.npz',states=np.asarray(states),raw_actions=np.asarray(raws),
            actions=np.asarray(actions,np.float32),tilts=np.asarray(tilts),config=np.array(json.dumps(env.config.__dict__)))
        if report['mode'] == 'native_control':
            policy.save(out/'native_queries.npz')
            actual_trace = load(out/'native_queries.npz')
            expected_trace = load(out/'planning'/f"iteration{report['selected_iteration']}"/'native_control/native_queries.npz')
            assert actual_trace.keys() == expected_trace.keys()
            for key in actual_trace:
                exact_bytes(actual_trace[key],expected_trace[key])
        total = preparation_native+env.native_steps+cold.native_steps+predictor
        assert total <= NATIVE_CAP and time.monotonic()-started < WALL_CAP
        result = dict(passed=True,qualified=False,source_qualified=True,links=14,controller_identity=cm['controller_identity'],
            seed=job['first_seed'],mode=report['mode'],preparation_native_steps=preparation_native,actual_native_steps=6000,
            planning_wall_seconds=planning_wall_seconds,total_worker_wall_seconds=time.monotonic()-started,
            cold_native_steps=6000,predictor_native_steps=predictor,new_native_steps=total,
            all_regenerated_planning_npz_arrays_byte_exact=True,source_numerical_audit_inherited_after_literal_packet_parity=True,
            full_native_episode_and_independent_cold_byte_exact=True,all_raw_delivered_controls_states_tilts_byte_exact=True,
            original_and_staged_scientific_paths_blocked=True,expected_results_blocked_until_prepare_finished=True,
            callbacks_use_real_native_predictions_without_cache=True,new_noisy_samples=0,reserved_cases_consumed=0,
            best_hold_seconds=info['best_hold_seconds'],source_qualification_summary_sha256=job['qualification_summary_sha256'],
            source_actual_sha256=job['expected_actual_sha256'],actual_sha256=sha(out/'actual.npz'),
            planning_array_parity_sha256=sha(out/'planning-array-parity.json'),park_seconds=0.,physical_velocities=44,
            scope='one required extracted-runtime first-fresh resynthesis; no new qualification sample, hardware or added-gust claim')
        write(out/'summary.json',result);write(out/'completed.json',dict(status='completed'))
        print(result,flush=True)
    except BaseException as exc:
        failure = out/'planning/failure.json'
        if failure.exists():
            preparation_native = read(failure).get('native_steps',preparation_native)
        known = preparation_native+getattr(env,'native_steps',0)+getattr(cold,'native_steps',0)+getattr(getattr(policy,'predictor',None),'native_steps',0)
        write(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),known_native_steps=known,native_upper_bound=NATIVE_CAP,
            accounting_complete=preparation_returned or (failure.exists() and read(failure).get('accounting_complete',False)),
            source_job_sha256=sha(job_path),new_noisy_samples=0,reserved_cases_consumed=0))
        if states:
            np.savez_compressed(out/'partial_actual_failure.npz',states=np.asarray(states),raw_actions=np.asarray(raws),actions=np.asarray(actions),tilts=np.asarray(tilts))
        raise
    finally:
        if policy is not None:policy.close()
        if env is not None:env.close()
        if cold is not None:cold.close()


def parent(args):
    from qualified_delivered_source14_parallel import first_qualified
    admission = first_qualified(args.proof)
    source = args.archive_run.resolve()
    payload = source/'extracted-payload'
    summary = read(source/'summary.json')
    assert (source/'completed.json').exists() and summary['passed']
    assert summary['controller_identity'] == admission['controller_identity']
    assert summary['all_member_crc_and_sha256_checked'] and summary['extracted_identity_and_relative_materialization_passed']
    assert sha(source/'member-manifest.json') == summary['member_manifest_sha256']
    archive_members = read(source/'member-manifest.json')['members']
    out = args.out.resolve()
    assert not out.is_relative_to(Path(admission['cohort']))
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'driver.py')
    witness = admission['witness']
    preparation = read(witness['parent_inventory'])
    original = Path(preparation['run'])
    records = preparation['studies'][str(original)]
    expected = {}

    def install(src,dst,digest):
        assert sha(src) == digest and not dst.exists()
        dst.parent.mkdir(parents=True,exist_ok=True)
        try:os.link(src,dst)
        except OSError:shutil.copyfile(src,dst)
        assert sha(dst) == digest

    for path in sorted((original/'planning').rglob('*.npz')):
        name = path.relative_to(original/'planning').as_posix()
        digest = records['planning/'+name]['sha256']
        install(path,out/'expected-planning'/name,digest)
        expected[name] = digest
    install(original/'observed_state.npz',out/'observed-state.npz',records['observed_state.npz']['sha256'])
    install(original/'summary.json',out/'source-prepare-summary.json',preparation['summary_sha256'])
    install(Path(witness['run'])/'actual.npz',out/'expected-actual.npz',witness['actual_sha256'])
    install(Path(admission['proof'])/'summary.json',out/'source-qualification-summary.json',admission['proof_summary_sha256'])
    import numpy as np
    with np.load(out/'observed-state.npz') as z:observed = z['state']
    job = dict(controller_identity=admission['controller_identity'],first_seed=admission['first_seed'],original_cohort=admission['cohort'],
        qualification_summary_sha256=admission['proof_summary_sha256'],observed_file_sha256=sha(out/'observed-state.npz'),
        observed_bytes_sha256=hashlib.sha256(observed.tobytes()).hexdigest(),expected_actual_sha256=witness['actual_sha256'],
        expected_planning_npz=expected,max_native_steps=NATIVE_CAP,max_wall_seconds=WALL_CAP,
        source_archive_member_manifest_sha256=summary['member_manifest_sha256'],worker_sha256=sha(out/'driver.py'),
        payload_bootstrap_sha256={name:row['sha256'] for name,row in archive_members.items() if not name.startswith('blobs/')},
        job_data_paths_relative=True,runtime_payload_supplied_explicitly=True,new_noisy_samples=0,reserved_cases_consumed=0)
    write(out/'job.json',job)
    command = [sys.executable,'-I','-B','-u',str(out/'driver.py'),'--job',str(out/'job.json'),'--payload',str(payload)]
    write(out/'command.json',dict(command=command,thread_environment_changes={}))
    with (out/'worker.log').open('x',encoding='utf-8') as log:
        options = dict(cwd=out,stdout=log,stderr=subprocess.STDOUT)
        if os.name == 'nt':options['creationflags'] = subprocess.CREATE_NO_WINDOW
        else:options['start_new_session'] = True
        process = subprocess.Popen(command,**options)
        write(out/'worker_handle.json',dict(pid=process.pid,command=command))
        try:
            code = process.wait(timeout=WALL_CAP)
            assert code == 0,'Extracted resynthesis failed; inspect retained worker.log and counter files'
        except BaseException as exc:
            if process.poll() is None:
                if os.name == 'nt':subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT,check=False)
                else:os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=30)
            write(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),unsealed_native_upper_bound=NATIVE_CAP,
                accounting_complete=False,new_noisy_samples=0,reserved_cases_consumed=0))
            raise
    result = read(out/'execution/summary.json')
    assert result['passed'] and result['controller_identity'] == admission['controller_identity']
    result.update(source_job_sha256=sha(out/'job.json'),source_worker_summary_sha256=sha(out/'execution/summary.json'))
    write(out/'summary.json',result);write(out/'completed.json',dict(status='completed'))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proof',type=Path);p.add_argument('--archive-run',type=Path);p.add_argument('--out',type=Path)
    p.add_argument('--job',type=Path);p.add_argument('--payload',type=Path);p.add_argument('--worker-out',type=Path)
    args = p.parse_args()
    if args.job:
        assert args.payload is not None
        worker(args)
    else:
        assert args.proof is not None and args.archive_run is not None and args.out is not None
        parent(args)


if __name__ == '__main__':
    main()
