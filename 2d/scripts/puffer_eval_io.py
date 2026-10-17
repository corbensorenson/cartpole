"""State plans and independently replayed tapes for native PufferLib evaluation.

Import verifies recorded inputs/physics/gates. It does not prove neural inference
or training quality; those require a frozen trainer run and closed-loop rerun.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import struct

import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.env import model_xml
from spatialpole.experiment import freeze_inputs, write_json

REF = 'ea7f989371270d2948f96728bda8aec5e7634a4c'
ROOT = Path(__file__).resolve().parents[1]
HEADER = struct.Struct('<8sIIIIqdIQ')


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def observation(env, previous):
    state = env.state(); nq = env.model.nq
    return np.r_[state[:2]/3., state[2:nq], state[nq:nq+2]/4., state[nq+2:]/15.,
                 previous/np.float32(80), env.steps/1500.].astype(np.float32)


def create_plan(out, n, cases, scope, checkpoint=None, action_selection='unspecified'):
    """cases: seed, noise, mode (0 hanging / 1 upright / 2 diagnostic), state."""
    out = Path(out)
    assert 1 <= len(cases) <= 45 and len({x[0] for x in cases}) == len(cases)
    out.mkdir(parents=True, exist_ok=False)
    config = Config(n_links=n); env = SpatialCartpole(config)
    manifest = dict(schema=1, puffer_ref=REF, config=asdict(config), scope=scope,
                    action_selection=action_selection,
                    qualification=False, cases=[], checkpoint_sha256=None)
    try:
        with (out/'plan.bin').open('xb') as f:
            f.write(struct.pack('<8sIIII', b'SPPLAN01', 1, n, 4+7*n, len(cases)))
            for seed, noise, mode, state in cases:
                state = np.asarray(state, dtype='<f8')
                assert state.shape == (4+7*n,) and np.isfinite(state).all()
                assert mode in (0,1,2) and noise >= 0 and np.isfinite(noise)
                if scope == 'development':
                    assert mode == 0
                    canonical, _ = env.reset(seed=seed, options=dict(mode='hanging', angle_noise=noise))
                    exact(state, canonical)
                f.write(struct.pack('<qdI', seed, noise, mode)); f.write(state.tobytes())
                manifest['cases'].append(dict(seed=seed, angle_noise=noise, mode=mode,
                                               initial_state=state.tolist(), state_sha256=hashlib.sha256(state.tobytes()).hexdigest()))
        (out/'model.xml').write_text(model_xml(config), encoding='utf-8')
        manifest['model_sha256'] = hashlib.sha256((out/'model.xml').read_bytes()).hexdigest()
        manifest['plan_sha256'] = hashlib.sha256((out/'plan.bin').read_bytes()).hexdigest()
        if checkpoint:
            payload = Path(checkpoint).read_bytes(); assert payload
            (out/'frozen_policy.bin').write_bytes(payload)
            manifest['checkpoint_sha256'] = hashlib.sha256(payload).hexdigest()
        write_json(out/'plan.json', manifest)
    finally: env.close()
    return manifest


def record_dtype(n):
    return np.dtype([
        ('steps','<u4'), ('streak','<u4'), ('best','<u4'), ('flags','<u4'),
        ('native_steps','<u8'), ('margin','<f8'), ('reward','<f4'),
        ('observations','<f4',(7+7*n,)), ('normalized','<f4',(2,)),
        ('raw','<f8',(2,)), ('delivered','<f4',(2,)),
        ('state','<f8',(4+7*n,)), ('tilts','<f8',(n,))], align=False)


def import_traces(plan, traces, out):
    plan, traces, out = Path(plan), Path(traces), Path(out)
    manifest = json.loads((plan/'plan.json').read_text())
    config = Config(**manifest['config']); n = config.n_links
    assert config == Config(n_links=n) and manifest['qualification'] is False
    assert hashlib.sha256((plan/'plan.bin').read_bytes()).hexdigest() == manifest['plan_sha256']
    assert hashlib.sha256((plan/'model.xml').read_bytes()).hexdigest() == manifest['model_sha256']
    assert (plan/'model.xml').read_text(encoding='utf-8') == model_xml(config)
    with (plan/'plan.bin').open('rb') as f:
        assert f.read(24) == struct.pack('<8sIIII', b'SPPLAN01', 1, n, 4+7*n, len(manifest['cases']))
        for case in manifest['cases']:
            assert f.read(20) == struct.pack('<qdI', case['seed'], case['angle_noise'], case['mode'])
            assert f.read((4+7*n)*8) == np.asarray(case['initial_state'], dtype='<f8').tobytes()
        assert not f.read()
    args = argparse.Namespace(plan=str(plan), traces=str(traces), out=str(out))
    freeze_inputs(out, args)
    shutil.copytree(plan, out/'frozen_plan')
    shutil.copytree(ROOT/'native/puffer_spatial', out/'native_source')
    if manifest['checkpoint_sha256']:
        assert hashlib.sha256((plan/'frozen_policy.bin').read_bytes()).hexdigest() == manifest['checkpoint_sha256']
    filenames = sorted(traces.glob('*.sptrace'))
    assert [x.name for x in filenames] == [f'episode{i:04d}.sptrace' for i in range(len(manifest['cases']))]
    rows = []; proof_total = 0; actual_total = 0; previous_counter = None
    for path, case in zip(filenames, manifest['cases']):
        raw_bytes = path.read_bytes()
        magic, schema, links, size, obs_size, seed, noise, mode, start_counter = HEADER.unpack_from(raw_bytes)
        assert (magic,schema,links,size,obs_size) == (b'SPTRACE1',1,n,4+7*n,7+7*n)
        assert (seed,noise,mode) == (case['seed'],case['angle_noise'],case['mode'])
        offset = HEADER.size
        initial = np.frombuffer(raw_bytes, '<f8', size, offset).copy(); offset += size*8
        exact(initial, np.asarray(case['initial_state'], dtype='<f8'))
        if previous_counter is not None: assert start_counter == previous_counter
        dtype = record_dtype(n)
        assert (len(raw_bytes)-offset) % dtype.itemsize == 0
        records = np.frombuffer(raw_bytes, dtype, offset=offset)
        assert 1 <= len(records) <= 1501 and bool(records[-1]['flags'] & 2), 'incomplete episode'
        env = SpatialCartpole(config)
        try:
            if mode == 0:
                state, _ = env.reset(seed=seed, options=dict(mode='hanging', angle_noise=noise)); exact(initial,state)
            else:
                assert manifest['scope'] == 'diagnostic'
                env.set_state(initial)
            states = [initial]; actions = []; raws = []; normalized = []; tilts = []; observations = []
            previous = np.zeros(2,np.float32); rejected = None
            for i, record in enumerate(records):
                flags = int(record['flags']); advanced = bool(flags & 8)
                exact(record['observations'].copy(), observation(env, previous))
                if not advanced:
                    assert i == len(records)-1 and not np.isfinite(record['normalized']).all()
                    assert flags == 3 and record['steps'] == env.steps
                    exact(record['state'].copy(), env.state())
                    assert record['native_steps'] == start_counter+env.native_steps
                    rejected = record['normalized'].copy()
                    break
                assert np.isfinite(record['normalized']).all()
                raw = record['normalized'].astype(np.float64)*80.
                exact(raw, record['raw'].copy())
                exact(env.deliver(raw), record['delivered'].copy())
                state, _, terminated, truncated, info = env.step_force(raw)
                exact(state, record['state'].copy())
                assert np.max(abs(env.tilts()-record['tilts'])) < 5e-14
                assert (record['steps'],record['streak'],record['best']) == (env.steps,env.streak,env.best_streak)
                assert record['margin'] == env.min_plane_margin
                assert bool(flags & 1) == env.warning_seen
                assert bool(flags & 2) == (terminated or truncated)
                assert bool(flags & 4) == info['success']
                assert record['native_steps'] == start_counter+env.native_steps
                assert i == len(records)-1 or not (terminated or truncated)
                previous = record['delivered'].copy()
                states.append(state); actions.append(previous); raws.append(raw)
                normalized.append(record['normalized'].copy()); observations.append(record['observations'].copy())
                tilts.append(env.tilts())
            info = env.info()
            policy_failure = rejected is not None
            if policy_failure: info['success'] = False
            meta = dict(policy='recorded PufferLib 5 native evaluation', puffer_ref=REF,
                        park_steps=0, park_seconds=0., predictor_native_steps=0,
                        diagnostic=manifest['scope']=='diagnostic',
                        action_selection=manifest.get('action_selection','unspecified'),
                        checkpoint_sha256=manifest['checkpoint_sha256'], hardware_realtime_claim=False)
            if manifest['scope']=='diagnostic':meta['demonstration']='evaluation IO diagnostic'
            row = dict(seed=seed, initialization='hanging' if mode==0 else 'diagnostic', angle_noise=noise,
                       **info, policy_failure=policy_failure, steps=env.steps, native_steps=env.native_steps,
                       proof_native_steps=env.native_steps, replay_exact=True, park_seconds=0.,
                       max_force_norm=float(np.max(np.linalg.norm(actions,axis=1))) if actions else 0.,
                       max_xy=np.max(abs(np.array(states)[:,:2]),axis=0).tolist(),
                       trace_sha256=hashlib.sha256(raw_bytes).hexdigest())
            np.savez_compressed(out/f'seed{seed}.npz', states=states,
                actions=np.asarray(actions,np.float32).reshape(-1,2), raw_actions=np.asarray(raws).reshape(-1,2),
                tilts=np.asarray(tilts).reshape(-1,n), normalized_actions=np.asarray(normalized,np.float32).reshape(-1,2),
                observations=np.asarray(observations,np.float32).reshape(-1,7+7*n),
                rejected_normalized_action=rejected if policy_failure else np.empty(0,np.float32),
                config=json.dumps(asdict(config)), policy_metadata=json.dumps(meta))
            shutil.copy2(path,out/path.name)
            previous_counter = int(records[-1]['native_steps'])
            actual_total += previous_counter-start_counter
            proof_total += env.native_steps
            assert proof_total <= len(manifest['cases'])*6000
            rows.append(row)
        finally: env.close()
    accounting_verified=False
    if (traces/'accounting.json').is_file():
        accounting=json.loads((traces/'accounting.json').read_text())
        assert accounting['native_steps']==actual_total and accounting['control_transitions']*4==actual_total
        assert accounting['closed_environments']==1 and accounting['links']==n
        shutil.copy2(traces/'accounting.json',out/'native_accounting.json')
        accounting_verified=True
    report = dict(results=rows, passes=sum(r['success'] for r in rows), cases=len(rows),
        actual_native_steps=actual_total, additional_native_steps=proof_total,
        inclusive_native_steps=actual_total+proof_total, qualification=False,
        scope=manifest['scope'], checkpoint_sha256=manifest['checkpoint_sha256'],
        every_state_and_delivered_force_byte_exact=True, every_preobservation_byte_exact=True,
        every_world_tilt_and_gate_reconstructed=True, frozen_closed_loop_reexecution=False,
        native_shutdown_accounting_verified=accounting_verified,
        limitation='cold replay and interface evidence; does not prove neural inference, training quality or a new link certificate')
    write_json(out/'summary.json',report); write_json(out/'completed.json',dict(status='completed'))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command',required=True)
    make = sub.add_parser('plan'); make.add_argument('--links',type=int,required=True)
    make.add_argument('--out',required=True); make.add_argument('--checkpoint')
    make.add_argument('--cohort',choices=['pilot13','development45'],default='pilot13')
    read = sub.add_parser('import'); read.add_argument('--plan',required=True)
    read.add_argument('--traces',required=True); read.add_argument('--out',required=True)
    a = p.parse_args()
    if a.command == 'plan':
        seeds = [(20261016,0.)]+[(s,.02) for s in range(20261116,20261128)]
        if a.cohort == 'development45': seeds += [(s,.02) for s in range(20261416,20261448)]
        env = SpatialCartpole(Config(n_links=a.links)); cases=[]
        try:
            for seed,noise in seeds:
                state,_=env.reset(seed=seed,options=dict(mode='hanging',angle_noise=noise)); cases.append((seed,noise,0,state))
        finally: env.close()
        result=create_plan(a.out,a.links,cases,'development',a.checkpoint)
        print(json.dumps(dict(cases=len(cases),checkpoint_sha256=result['checkpoint_sha256'],qualification=False)))
    else:
        result=import_traces(a.plan,a.traces,a.out)
        print(json.dumps({k:result[k] for k in ('cases','passes','inclusive_native_steps','qualification','scope')}))
