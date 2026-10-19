"""Independent twelve-link original-noise development proof; no qualification claim."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.aligned_value import aligned_next_value
from spatialpole.direct_shooting import native_successors
from spatialpole.experiment import freeze_inputs, write_json
from spatialpole.native_mpc import exact_bytes
from audit_spatial_synthesis_models import audit as audit_model, certificate, models, load
from spatial_synthesis_geometry import defects, error, Dimensions
from eleven_synthesis_protocol import sha


def read(path):
    return json.loads(Path(path).read_text())


def raw_force(geometry, state, t, v, F, c, w):
    if np.array_equal(state, v['states'][t]):
        e = np.zeros(F.shape[1])
    else:
        ref, frame, _ = aligned_next_value(geometry, v['states'][t], F[t], state)
        e = frame @ geometry.difference(state, ref)
    return v['actions'][t].astype(float)+c['feed'][t]-c['K'][t]@(c['S'][t]@e)+c['Cinv'][t]@w[t]


def warm_start(planning, reference, initial, first_model, geometry, native):
    v, m, c = [load(reference/name) for name in
        ('input_nominal.npz', 'native_models.npz', 'coordinates.npz')]
    folder = planning/'warm_start'; row = read(folder/'summary.json')
    p = load(folder/'model_plan.npz'); proposal = load(folder/'virtual_proposal.npz')
    intervals = load(folder/'interval_check.npz')
    e0 = error(geometry, v, m['F'], initial, 0)
    exact_bytes(e0, p['initial_error'])
    u, e, w = certificate(folder/'certificate.npz', c, e0, v['states'], v['actions'].astype(float))
    np.testing.assert_allclose(u, p['controls'], atol=2e-12, rtol=1e-13)
    exact_bytes(e, p['errors']); exact_bytes(w, p['w'])
    models(m, e, p['controls']-v['actions'])
    delta = np.einsum('tij,tj->ti', m['E'], e)
    states = np.array([geometry.integrate(x, d) for x, d in zip(v['states'], delta)])
    states[0] = initial.copy(); exact_bytes(states, proposal['states'])
    actions = np.array([geometry.deliver(force) for force in p['controls']])
    exact_bytes(actions, proposal['actions'])
    exact_bytes(first_model['reference_states'], states)
    exact_bytes(first_model['reference_actions'], actions)
    exact_bytes(first_model['predicted'], intervals['predicted'])
    cost, maximum, residual = defects(geometry, intervals['predicted'], states[1:])
    exact_bytes(residual, intervals['residual'])
    baseline = m['predicted'].copy()
    baseline[0] = native_successors(native, initial[None], v['actions'][:1])[0]
    base, bmax, _ = defects(geometry, baseline, v['states'][1:])
    assert row['predictor_admitted'] and cost <= .8*base
    assert cost == row['physical_defect_cost'] and maximum == row['physical_defect_max']
    assert base == row['baseline_physical_defect_cost'] and bmax == row['baseline_physical_defect_max']
    assert row['native_steps'] == 6004
    return dict(passed=True, physical_defect_cost=cost, baseline_physical_defect_cost=base)


def check_update(folder, next_folder, initial, geometry):
    v, m, p = [load(folder/name) for name in ('virtual_reference.npz', 'native_models.npz', 'model_plan.npz')]
    updates = read(folder/'updates.json')['results']
    selected = [(i, row) for i, row in enumerate(updates) if row['accepted']]
    assert len(selected) == 1 and selected[0][0] == len(updates)-1
    # Match alpha rather than trial index: geometry-rejected scales have no file.
    row = selected[0][1]; data = None
    baseline, _, _ = defects(geometry, m['predicted'], v['states'][1:])
    delta = np.einsum('tij,tj->ti', m['E'], p['errors'])
    states = np.array([geometry.integrate(x, row['alpha']*d) for x, d in zip(v['states'], delta)])
    states[0] = initial.copy()
    actions = np.array([geometry.deliver(z) for z in v['actions'].astype(float)+
        row['alpha']*(p['controls']-v['actions'].astype(float))])
    for path in sorted(folder.glob('update[0-9].npz')):
        candidate = load(path)
        if np.array_equal(candidate['states'], states) and np.array_equal(candidate['actions'], actions):
            data = candidate; break
    assert data is not None
    next_model = load(next_folder/'native_models.npz')
    exact_bytes(states, next_model['reference_states']); exact_bytes(actions, next_model['reference_actions'])
    exact_bytes(data['predicted'], next_model['predicted'])
    cost, maximum, residual = defects(geometry, data['predicted'], states[1:])
    exact_bytes(residual, data['residual'])
    assert cost == row['physical_defect'] and maximum == row['maximum'] and cost <= .8*baseline
    assert np.max(abs(states[:, :2])) <= 3.
    for state in states[-251:]:
        geometry.set_state(state); assert np.max(geometry.tilts()) < .15
    return dict(passed=True, alpha=row['alpha'], physical_defect=cost, baseline=baseline)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--run', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args(); run, out = Path(a.run), Path(a.out)
    manifest = read(run/'frozen_manifest.json'); case = run
    seed = manifest['seed']
    assert (case/'completed.json').exists()
    row = read(case/'summary.json'); assert row['result']['success']
    assert 'inherited_from' not in row, 'Inherited cases use their hash-bound parent case audit.'
    assert row['frozen_manifest_sha256'] == sha(run/'frozen_manifest.json')
    assert row['result']['seed'] == seed
    assert manifest['config'] == asdict(Config(n_links=12))
    freeze_inputs(out, a)
    assert read(out/'inputs.json')['source_sha256'] == manifest['source_sha256']
    for name, digest in manifest['helper_sha256'].items():
        assert sha(run/'helpers'/name) == digest
    for name, digest in manifest['bundle_sha256'].items():
        assert sha(run/'reference'/name) == digest
    for name in ('audit_twelve_synthesis_case.py', 'audit_spatial_synthesis_models.py',
                 'spatial_synthesis_geometry.py', 'eleven_value_sequence.py',
                 'donor_conic_step.py', 'eleven_synthesis_protocol.py'):
        shutil.copyfile(Path(__file__).parent/name, out/name)
    assert sha(Path(__file__).with_name('spatial_synthesis_geometry.py')) == manifest['helper_sha256']['spatial_synthesis_geometry.py']
    tape = load(case/'actual.npz'); planning = case/'planning'; receipt = read(planning/'summary.json')
    geometry = SpatialCartpole(Config(n_links=12)); native = SpatialCartpole(geometry.config)
    native.derivative_workers = 4; cold = SpatialCartpole(geometry.config); results = []; updates = []
    try:
        assert json.loads(str(tape['config'])) == asdict(geometry.config)
        noise = .02
        initial, _ = geometry.reset(seed=seed, options=dict(mode='hanging', angle_noise=noise))
        exact_bytes(initial, tape['states'][0]); exact_bytes(initial, load(planning/'observed_state.npz')['state'])
        warm = None; derived = 0
        assert not receipt['reused_source']
        if not receipt['reused_source']:
            stages = receipt['results']; assert 1 <= len(stages) <= 4
            for i, stage in enumerate(stages):
                assert stage['iteration'] == i and not stage.get('model_rejected', False)
                folder = planning/f'iteration{i}'
                exact_bytes(load(folder/'virtual_reference.npz')['states'][0], initial)
                results.append(audit_model(folder, geometry, native, out, i))
                derived += Dimensions.from_config(geometry.config).model_native_steps+80+8*results[-1]['callbacks']
                derived += 6000*len(list(folder.glob('update[0-9].npz')))
                if i:
                    updates.append(check_update(planning/f'iteration{i-1}', folder, initial, geometry))
            first = load(planning/'iteration0/native_models.npz')
            warm = warm_start(planning, run/'reference', initial, first, geometry, native); derived += 6004
            folder = planning/f'iteration{len(stages)-1}'
            v, m, c, plan, expected = [load(folder/name) for name in
                ('virtual_reference.npz', 'native_models.npz', 'coordinates.npz', 'model_plan.npz', 'native_forecast.npz')]
            w = plan['w']; assert results[-1]['full_native_success']
        assert derived == receipt['inclusive_native_steps'] == row['planning_native_steps']
        assert row['inclusive_native_steps'] == derived+12000
        for name in ('states', 'actions', 'raw_actions', 'tilts'):
            exact_bytes(tape[name], expected[name])
        assert tape['states'].shape == (1501, 88) and tape['actions'].dtype == np.float32
        metadata = json.loads(str(tape['policy_metadata']))
        assert metadata == row['result']['policy_metadata'] and metadata['predictor_native_steps'] == derived
        assert metadata['park_seconds'] == 0. and metadata['physical_velocities'] == 38
        assert not metadata['hardware_realtime_claim']
        state = initial.copy(); streak = best = 0
        for t, action in enumerate(tape['actions']):
            raw = raw_force(geometry, tape['states'][t], t, v, m['F'], c, w)
            exact_bytes(raw, tape['raw_actions'][t]); exact_bytes(geometry.deliver(raw), action)
            assert np.linalg.norm(action.astype(float)) <= 80.
            state = cold.transition(state, action); exact_bytes(state, tape['states'][t+1])
            angles = cold.tilts(); exact_bytes(angles, tape['tilts'][t])
            streak = streak+1 if angles.max() < .15 else 0; best = max(best, streak)
        assert row['result']['angle_noise'] == noise and row['result']['initialization'] == 'hanging'
        assert best*.02 == row['result']['best_hold_seconds'] >= 5.
        assert cold.min_plane_margin == row['result']['plane_margin'] >= 0.
        assert not cold.warning_seen and not row['result']['physics_warning']
        assert row['result']['replay_exact'] and row['result']['steps'] == 1500
        assert row['result']['native_steps'] == row['result']['proof_native_steps'] == cold.native_steps == 6000
        assert geometry.native_steps == 0 and native.native_steps == 6080*len(results)+(4 if warm else 0)
        write_json(out/'summary.json', dict(passed=True, seed=seed,
            native_steps=native.native_steps+cold.native_steps, model_audits=results, update_audits=updates,
            warm_start=warm, reused_source=receipt['reused_source'], actual_cold_byte_exact=True,
            best_hold_seconds=best*.02, physical_velocities=38, source_native_steps=row['inclusive_native_steps'],
            frozen_manifest_sha256=sha(run/'frozen_manifest.json'), case_summary_sha256=sha(case/'summary.json'),
            actual_sha256=sha(case/'actual.npz'), qualified=False, reserved_cases_consumed=0,
            scope='twelve-link development only; original reset, all full native model centers/maps/certificates/callbacks, warm/accepted update lineage, actual feedback and new full cold replay'))
        write_json(out/'completed.json', dict(status='completed'))
        print(json.dumps(dict(passed=True, seed=seed, native_steps=native.native_steps+cold.native_steps)), flush=True)
    except BaseException as exception:
        write_json(out/'failure.json', dict(type=type(exception).__name__, error=str(exception),
            native_steps=native.native_steps+cold.native_steps, completed_model_audits=results)); raise
    finally:
        geometry.close(); native.close(); cold.close()


if __name__ == '__main__':
    main()
