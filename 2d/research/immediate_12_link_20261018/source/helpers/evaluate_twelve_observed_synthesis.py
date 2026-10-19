"""One frozen, original-noise twelve-link development case; no qualification claim."""
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
from spatialpole.experiment import freeze_inputs, write_json
from spatialpole.native_mpc import exact_bytes
from eleven_synthesis_protocol import read, sha

HELPERS = ('evaluate_twelve_observed_synthesis.py', 'spatial_observed_synthesis.py',
           'spatial_synthesis_geometry.py', 'eleven_value_sequence.py',
           'donor_conic_step.py', 'eleven_synthesis_protocol.py',
           'verify_eleven_sequence_refinement.py', 'eleven_sequence_qp.py',
           'trajectory_expert_support.py')


def load(path):
    with np.load(path) as z:
        return {name:z[name] for name in z.files}


def actual_episode(config, plan, seed, out):
    """Own both counters even if feedback parity or another assertion fails."""
    from spatial_observed_synthesis import PlannedPolicy
    env, cold = SpatialCartpole(config), SpatialCartpole(config)
    policy = None
    try:
        initial, _ = env.reset(seed=seed, options=dict(mode='hanging', angle_noise=.02))
        exact_bytes(initial, plan.expected['states'][0])
        policy = PlannedPolicy(env, plan)
        assert env.native_steps == cold.native_steps == 0
        states, actions, raws, tilts = [initial], [], [], []
        start = time.perf_counter()
        for t in range(config.horizon):
            before, clock = env.state().tobytes(), env.data.time
            force = policy(states[-1], t)
            assert env.state().tobytes() == before and env.data.time == clock
            delivered = env.deliver(force)
            state, _, done, truncated, info = env.step_force(delivered)
            states.append(state); actions.append(delivered); raws.append(force); tilts.append(env.tilts())
            policy.observe_transition(state, t, delivered)
            if done or truncated:
                break
        state = initial.copy()
        for t, force in enumerate(actions):
            state = cold.transition(state, force)
            exact_bytes(state, states[t+1])
        np.savez_compressed(out/'actual.npz', states=np.array(states),
            actions=np.array(actions, dtype=np.float32), raw_actions=np.array(raws), tilts=np.array(tilts),
            config=np.array(json.dumps(config.__dict__)), policy_metadata=np.array(json.dumps(policy.metadata)))
        result = dict(seed=seed, initialization='hanging', angle_noise=.02, **info,
            steps=len(actions), native_steps=env.native_steps, proof_native_steps=cold.native_steps,
            replay_exact=True, wall_seconds=time.perf_counter()-start,
            max_force_norm=float(np.max(np.linalg.norm(actions, axis=1))), policy_metadata=policy.metadata,
            actual_benchmark_episodes=1)
        return result
    except BaseException as exception:
        write_json(out/'actual_failure.json', dict(type=type(exception).__name__, error=str(exception),
            actual_native_steps=env.native_steps, cold_native_steps=cold.native_steps,
            inclusive_native_steps=env.native_steps+cold.native_steps, accounting_complete=False))
        raise
    finally:
        if policy is not None:
            policy.close()
        env.close(); cold.close()


def worker(out):
    from spatial_observed_synthesis import synthesize, planning_budget
    assert Path(spatialpole.__file__).resolve().parent == (out/'frozen_package/spatialpole').resolve()
    manifest = read(out/'frozen_manifest.json')
    for category, directory in (('helper_sha256','helpers'), ('source_sha256','frozen_package/spatialpole'),
                                ('bundle_sha256','reference')):
        for name, digest in manifest[category].items():
            assert sha(out/directory/name) == digest
    config = Config(n_links=12)
    assert manifest['config'] == config.__dict__ and manifest['max_planning_native'] == planning_budget(config)
    runtime = Path(manifest['runtime'])
    assert sha(runtime/'receipt.json') == manifest['solver_receipt_sha256']
    from donor_conic_step import solver_module
    # Check the lazy constrained-solve import before spending native model steps.
    from verify_eleven_sequence_refinement import check
    solver = solver_module(runtime)
    write_json(out/'solver_preflight.json', dict(passed=True, version=solver.__version__, native_steps=0))
    bundle = dict(v=load(out/'reference/input_nominal.npz'), m=load(out/'reference/native_models.npz'),
        c=load(out/'reference/coordinates.npz'), expected=load(out/'reference/actual.npz'), w=np.zeros((1500,2)))
    env = SpatialCartpole(config)
    try:
        observed, _ = env.reset(seed=manifest['seed'], options=dict(mode='hanging', angle_noise=.02))
        assert env.native_steps == 0
        assert not np.array_equal(observed, bundle['v']['states'][0])
    finally:
        env.close()
    write_json(out/'worker.json', dict(pid=os.getpid(), started_utc=time.time()))
    start = time.perf_counter()
    try:
        plan = synthesize(config, observed, bundle, runtime, out/'planning')
        planning = read(out/'planning/summary.json')
        elapsed = time.perf_counter()-start
        result = (actual_episode(config, plan, manifest['seed'], out) if plan is not None else
            dict(seed=manifest['seed'], initialization='hanging', angle_noise=.02, success=False,
                 planning_failed=True, native_steps=0, proof_native_steps=0, actual_benchmark_episodes=0))
        count = planning['inclusive_native_steps']+result['native_steps']+result['proof_native_steps']
        assert count <= planning_budget(config)+12000
        write_json(out/'summary.json', dict(result=result, planning_wall_seconds=elapsed,
            planning_native_steps=planning['inclusive_native_steps'], inclusive_native_steps=count,
            frozen_manifest_sha256=sha(out/'frozen_manifest.json'), source_nominal_reused=True,
            new_noisy_development_states=int(not manifest['qualification_attempt']),
            new_noisy_qualification_states=int(manifest['qualification_attempt']),
            actual_benchmark_episodes=result['actual_benchmark_episodes'],
            qualified=False, qualification_attempt=manifest['qualification_attempt'],
            reserved_cases_consumed=int(manifest['qualification_attempt']),
            scope='one original-noise state; frozen state-only planner; qualification requires the complete independent cohort gate'))
        write_json(out/'completed.json', dict(status='completed', task_success=result['success']))
        print((out/'summary.json').read_text(), flush=True)
    except BaseException as exception:
        write_json(out/'failure.json', dict(type=type(exception).__name__, error=str(exception),
            accounting_complete=False, counters_retained_in_owned_planning_and_actual_failure_receipts=True))
        raise


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source-inventory'); p.add_argument('--model-inventory')
    p.add_argument('--port-inventory'); p.add_argument('--runtime'); p.add_argument('--seed', type=int, default=20261116)
    p.add_argument('--out', required=True); p.add_argument('--worker', action='store_true')
    p.add_argument('--qualification', action='store_true')
    a = p.parse_args(); out = Path(a.out).resolve()
    if a.worker:
        worker(out); return
    source, model, port = map(read, (a.source_inventory, a.model_inventory, a.port_inventory))
    assert source['passed'] and source['summary']['nominal_source_law_admitted'] and model['passed'] and port['passed']
    reference, cache = Path(source['run']), Path(model['run'])
    assert sha(reference/'summary.json') == source['summary_sha256']
    assert sha(cache/'summary.json') == model['summary_sha256']
    # The prior saved source-query gate must bind this exact generic planner.
    port_summary = port['summary']
    for name in ('spatial_observed_synthesis.py', 'spatial_synthesis_geometry.py',
                 'eleven_value_sequence.py', 'donor_conic_step.py'):
        assert sha(Path(__file__).with_name(name)) == port_summary['helper_sha256'][name]
    runtime = Path(a.runtime).resolve()
    assert (runtime/'receipt.json').exists() and (runtime/'site').is_dir()
    payloads = {'input_nominal.npz':reference/'input_nominal.npz', 'coordinates.npz':reference/'coordinates.npz',
                'actual.npz':reference/'actual.npz', 'native_models.npz':cache/'native_models.npz'}
    for name, path in payloads.items():
        inventory = model if name == 'native_models.npz' else source
        assert sha(path) == inventory['files'][name]['sha256']
    freeze_inputs(out, a)
    assert read(out/'inputs.json')['source_sha256'] == source['source_sha256'] == model['source_sha256']
    (out/'helpers').mkdir(); (out/'reference').mkdir()
    for name in HELPERS:
        shutil.copyfile(Path(__file__).with_name(name), out/'helpers'/name)
    shutil.copytree(out/'source', out/'frozen_package/spatialpole')
    for name, path in payloads.items():
        shutil.copyfile(path, out/'reference'/name)
    manifest = dict(config=Config(n_links=12).__dict__, seed=a.seed, angle_noise=.02,
        helper_sha256={name:sha(out/'helpers'/name) for name in HELPERS},
        source_sha256=source['source_sha256'], bundle_sha256={name:sha(out/'reference'/name) for name in payloads},
        runtime=str(runtime), solver_receipt_sha256=sha(runtime/'receipt.json'),
        source_inventory_sha256=sha(a.source_inventory), model_inventory_sha256=sha(a.model_inventory),
        port_inventory_sha256=sha(a.port_inventory), max_planning_native=7620324,
        max_case_native=7632324, qualification_attempt=a.qualification,
        qualified=False, reserved_cases_consumed=0)
    algorithm = {key:manifest[key] for key in ('config','helper_sha256','source_sha256','bundle_sha256',
                                                'solver_receipt_sha256','max_planning_native','max_case_native')}
    algorithm['helper_sha256'] = {key:value for key,value in algorithm['helper_sha256'].items()
                                if key != Path(__file__).name}
    manifest['controller_identity'] = hashlib.sha256(json.dumps(algorithm, sort_keys=True).encode()).hexdigest()
    write_json(out/'frozen_manifest.json', manifest)
    env = dict(os.environ); env['PYTHONPATH'] = str(out/'frozen_package')
    command = [sys.executable, '-u', str(out/'helpers'/Path(__file__).name), '--worker', '--out', str(out)]
    write_json(out/'worker_command.json', dict(command=command, pythonpath=env['PYTHONPATH']))
    result = subprocess.run(command, env=env, creationflags=subprocess.CREATE_NO_WINDOW)
    if result.returncode:
        raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
