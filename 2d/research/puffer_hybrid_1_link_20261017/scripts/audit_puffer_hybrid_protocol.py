"""Independently reconstruct hybrid inputs, inference, scores and native replays."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from spatialpole import Config, SpatialCartpole
from spatialpole.experiment import freeze_inputs, write_json
from puffer_mingru_cpu import MeanMinGRU, observation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out, run = Path(a.out), Path(a.run)
    freeze_inputs(out, a)
    assert (run/'completed.json').is_file()
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    summary = json.loads((run/'summary.json').read_text())
    cohort = summary['cohort']
    assert cohort in ('development45', 'qualification32')
    expected = ([20261016]+list(range(20261116, 20261128))+list(range(20261416, 20261448))
                if cohort == 'development45' else list(range(20330116, 20330148)))
    assert [r['seed'] for r in summary['results']] == expected
    assert summary['passes'] == len(expected) and summary['full_first_closed_loop_reexecution_byte_exact']
    assert json.loads((run/'inputs.json').read_text())['source_sha256'] == json.loads((out/'inputs.json').read_text())['source_sha256']
    for name, digest in summary['controller_files_sha256'].items():
        assert sha(run/name) == digest
    shutil.copy2(run/'summary.json', out/'input_summary.json')
    with np.load(run/'input_route.npz') as z:
        route = {k: z[k] for k in z.files}
    geometry, native = SpatialCartpole(Config(n_links=1)), SpatialCartpole(Config(n_links=1))
    source = Path(__file__).parent/'puffer_mingru_cpu.py'
    assert source.read_bytes() == (run/'puffer_mingru_cpu.py').read_bytes()
    rows = []
    try:
        for case in summary['results']:
            seed = case['seed']
            with np.load(run/f'seed{seed}.npz') as z:
                states, actions, raws, recorded = z['states'], z['actions'], z['raw_actions'], z['tilts']
                observations, normalized = z['nn_observations'], z['nn_normalized_actions']
                metadata = json.loads(str(z['policy_metadata']))
            assert states.shape == (1501, 11) and actions.shape == (1500, 2)
            assert observations.shape == (1100, 14) and normalized.shape == (1100, 2)
            assert case['success'] and case['replay_exact'] and case['native_steps'] == case['proof_native_steps'] == 6000
            assert metadata['hybrid_handoff_step'] == 400 and metadata['handoff_actual_native_steps'] == 1600
            assert metadata['handoff_actual_seconds'] == 8. and not metadata['physical_state_reset_at_handoff']
            assert np.array_equal(metadata['handoff_state'], states[400])
            assert np.array_equal(metadata['handoff_previous_force'], actions[399])
            initial, _ = geometry.reset(seed=seed, options=dict(mode='hanging', angle_noise=case['angle_noise']))
            assert np.array_equal(initial, states[0])
            tilts = []
            net = MeanMinGRU(run/'policy.bin')
            for t in range(1500):
                if t < 400:
                    raw = route['actions'][t]-route['gains'][t]@geometry.difference(states[t], route['states'][t])
                else:
                    obs = observation(states[t], actions[t-1], t-400, 1, True)
                    assert obs.tobytes() == observations[t-400].tobytes()
                    value = net(obs)
                    assert value.tobytes() == normalized[t-400].tobytes()
                    raw = value.astype(float)*80.
                assert raw.tobytes() == raws[t].tobytes()
                assert geometry.deliver(raw).tobytes() == actions[t].tobytes()
                geometry.set_state(states[t+1])
                tilts.append(geometry.tilts())
            tilts = np.array(tilts)
            assert np.array_equal(tilts, recorded)
            inside = np.max(tilts, axis=1) < .15
            edges = np.diff(np.r_[False, inside, False].astype(int))
            longest = (np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)).max(initial=0)*.02
            assert abs(longest-case['best_hold_seconds']) < 1e-12 and longest >= 5.
            assert case['plane_margin'] >= 0 and not case['physics_warning']
            assert np.max(abs(states[:, :2])) <= 3. and np.linalg.norm(actions.astype(float), axis=1).max() <= 80.
            replayed = cohort == 'qualification32' or seed == expected[0]
            if replayed:
                native.min_plane_margin, native.warning_seen = 3., False
                state = states[0]
                for t, force in enumerate(actions):
                    state = native.transition(state, force)
                    assert state.tobytes() == states[t+1].tobytes()
                assert native.min_plane_margin == case['plane_margin'] and not native.warning_seen
            rows.append(dict(seed=seed, hold_seconds=float(longest), native_replayed=replayed,
                             observations_and_inference_byte_exact=True, all_physical_rates_retained=True,
                             tape_sha256=sha(run/f'seed{seed}.npz')))
        with np.load(run/f'seed{expected[0]}.npz') as first, np.load(run/'repeat_first.npz') as repeat:
            assert all(first[k].tobytes() == repeat[k].tobytes() for k in
                       ('states', 'actions', 'raw_actions', 'tilts', 'nn_observations', 'nn_normalized_actions'))
        cost = 6000 if cohort == 'development45' else 192000
        assert native.native_steps == cost and geometry.native_steps == 0
        report = dict(passed=True, cohort=cohort, results=rows, native_steps=cost,
                      development_summary_sha256=sha(run/'summary.json') if cohort == 'development45' else None,
                      qualification_summary_sha256=sha(run/'summary.json') if cohort == 'qualification32' else None,
                      controller_files_sha256=summary['controller_files_sha256'],
                      actual_30s_clock_and_8s_handoff_verified=True,
                      pure_RL_swing_up=False, qualified_hybrid_swing_up=cohort == 'qualification32',
                      scope='independent actual hybrid controller and protocol audit')
        write_json(out/'summary.json', report)
        write_json(out/'completed.json', dict(status='completed'))
        print({k: v for k, v in report.items() if k not in ('results', 'controller_files_sha256')}, flush=True)
    except BaseException as error:
        write_json(out/'failure.json', dict(type=type(error).__name__, error=str(error), native_steps=native.native_steps))
        raise
    finally:
        geometry.close()
        native.close()


if __name__ == '__main__':
    main()
