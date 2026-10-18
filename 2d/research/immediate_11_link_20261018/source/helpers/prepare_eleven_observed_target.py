"""State-only warm start of a new full-noise target from an audited 11-link plan."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.direct_shooting import native_successors
from spatialpole.experiment import freeze_inputs, write_json
from spatialpole.native_mpc import exact_bytes
from eleven_initial_continuation import load
from eleven_value_sequence import solve_value_sequence, mapped_error
from prepare_eleven_newton_seed import defects
from probe_eleven_native_chart import error
from trajectory_expert_support import sha


def prepare(geometry, native, v, m, c, observed_initial, runtime, out):
    """No seed/case label enters the proposal; all physical rates are retained."""
    initial_error = error(geometry, v, dict(F=m['F']), observed_initial, 0)
    controls, errors, w, proof = solve_value_sequence(c, initial_error, v['states'],
        v['actions'].astype(float), runtime, out/'certificate.npz', time_limit=120.)
    assert mapped_error(m, errors, controls-v['actions']) < 1e-6
    np.savez_compressed(out/'model_plan.npz', controls=controls, errors=errors, w=w,
                        initial_error=initial_error)
    baseline_predicted = m['predicted'].copy()
    baseline_predicted[0] = native_successors(native, observed_initial[None], v['actions'][:1])[0]
    baseline, bmax, _ = defects(geometry, baseline_predicted, v['states'][1:])
    delta = np.einsum('tij,tj->ti', m['E'], errors)
    states = np.array([geometry.integrate(x, d) for x, d in zip(v['states'], delta)])
    states[0] = observed_initial.copy()
    actions = np.array([geometry.deliver(u) for u in controls])
    last = []
    for state in states[-251:]:
        geometry.set_state(state); last.append(geometry.tilts())
    margin = float(3.-np.max(abs(states[:, :2]))); tilt = float(np.max(last))
    geometry_passed = bool(margin >= 0. and tilt < .15)
    score = maximum = None; admitted = False
    if geometry_passed:
        predicted = native_successors(native, states[:-1], actions)
        score, maximum, residual = defects(geometry, predicted, states[1:])
        admitted = bool(score <= .8*baseline)
        np.savez_compressed(out/'interval_check.npz', predicted=predicted, residual=residual)
    np.savez_compressed(out/'virtual_proposal.npz', states=states, actions=actions,
                        fraction=np.array(1.))
    if admitted:
        np.savez_compressed(out/'accepted_predictor.npz', states=states, actions=actions,
                            fraction=np.array(1.))
        exact_bytes(states[0], observed_initial)
    return dict(predictor_admitted=admitted, geometry_passed=geometry_passed,
        baseline_physical_defect_cost=baseline, baseline_physical_defect_max=bmax,
        physical_defect_cost=score, physical_defect_max=maximum,
        virtual_plane_margin=margin, last5_virtual_max_tilt=tilt, proof=proof,
        original_target_bytes_preserved=True, physical_velocities=35)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--reference', required=True)
    p.add_argument('--audit', required=True); p.add_argument('--observed', required=True)
    p.add_argument('--runtime', required=True); p.add_argument('--out', required=True)
    a = p.parse_args(); reference, out = Path(a.reference), Path(a.out)
    audit = json.loads((Path(a.audit)/'summary.json').read_text())
    assert audit['passed'] and any(row['full_native_success'] and
        row['source_summary_sha256'] == sha(reference/'summary.json') for row in audit['results'])
    freeze_inputs(out, a)
    # The solver is given this state vector, never the filename or seed label.
    observed = load(a.observed)
    assert json.loads(str(observed['config'])) == asdict(Config(n_links=11))
    np.savez_compressed(out/'observed_state.npz', state=observed['states'][0])
    for name in ('eleven_initial_continuation.py', 'eleven_value_sequence.py',
                 'prepare_eleven_newton_seed.py', 'probe_eleven_native_chart.py',
                 'eleven_compression_features.py', 'eleven_load_support.py',
                 'donor_conic_step.py', 'verify_eleven_sequence_refinement.py', 'eleven_sequence_qp.py'):
        shutil.copyfile(Path(__file__).parent/name, out/name)
    v, m, c = [load(reference/name) for name in
        ('virtual_reference.npz', 'native_models.npz', 'coordinates.npz')]
    native = SpatialCartpole(Config(n_links=11)); native.derivative_workers = 4
    geometry = SpatialCartpole(native.config)
    try:
        result = prepare(geometry, native, v, m, c, observed['states'][0], a.runtime, out)
        assert native.native_steps <= 6004 and geometry.native_steps == 0
        write_json(out/'summary.json', dict(completed=True, **result, native_steps=native.native_steps,
            source_sha256={name:sha(reference/name) for name in
                ('virtual_reference.npz', 'native_models.npz', 'coordinates.npz', 'summary.json')},
            observed_state_sha256=sha(out/'observed_state.npz'), qualified=False,
            reserved_cases_consumed=0, actual_benchmark_episodes=0,
            scope='state-only virtual warm start for another full-noise development case; not a task rollout'))
        write_json(out/'completed.json', dict(status='completed'))
        print(json.dumps(dict(**result, native_steps=native.native_steps)), flush=True)
    except BaseException as exception:
        write_json(out/'failure.json', dict(type=type(exception).__name__, error=str(exception),
                                          native_steps=native.native_steps)); raise
    finally:
        geometry.close(); native.close()


if __name__ == '__main__':
    main()
