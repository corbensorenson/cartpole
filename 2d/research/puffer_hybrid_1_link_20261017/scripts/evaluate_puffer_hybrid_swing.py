"""Canonical hanging swing with classical motion and learned, sign-aware capture."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

from spatialpole import Config, SpatialCartpole
from spatialpole.control import RoutePolicy
from spatialpole.experiment import episode, freeze_inputs, write_json
from puffer_mingru_cpu import MeanMinGRU, observation


class HybridCapture:
    def __init__(self, env, payload, checkpoint):
        self.live, self.geometry = env, SpatialCartpole(env.config)
        self.classical = RoutePolicy(self.geometry, payload['states'], payload['actions'], payload['gains'], payload)
        self.network = MeanMinGRU(checkpoint)
        self.previous = np.zeros(2, np.float32)
        self.nn_observations, self.nn_actions = [], []
        self.metadata = dict(policy='classical swing / learned CPU Puffer capture',
                             park_steps=0, park_seconds=0., route_seconds=8., hybrid_handoff_step=400,
                             canonical_quaternion_sign=True, physical_state_reset_at_handoff=False,
                             learned_clock='commands since activation; actual episode clock unchanged',
                             predictor_native_steps=0, hardware_realtime_claim=False,
                             checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest())

    def __call__(self, state, step):
        before = (self.live.state().tobytes(), self.live.steps, self.live.native_steps, self.live.data.time)
        if step < 400:
            force = self.classical(state, step)
        else:
            if step == 400:
                assert not self.nn_observations and not np.any(self.network.memory)
                self.geometry.set_state(state)
                self.metadata.update(handoff_state=state.tolist(), handoff_previous_force=self.previous.tolist(),
                                     handoff_actual_native_steps=self.live.native_steps,
                                     handoff_actual_seconds=self.live.steps*self.live.config.dt,
                                     handoff_world_tilt=float(self.geometry.tilts().max()))
            obs = observation(state, self.previous, step-400, 1, canonical_sign=True)
            action = self.network(obs)
            self.nn_observations.append(obs.copy())
            self.nn_actions.append(action.copy())
            force = action.astype(float)*80.
        after = (self.live.state().tobytes(), self.live.steps, self.live.native_steps, self.live.data.time)
        assert after == before
        return force

    def observe_transition(self, state, step, delivered):
        self.previous = delivered.copy()

    @property
    def forecast_tapes(self):
        return dict(nn_observations=np.array(self.nn_observations, np.float32).reshape(-1, 14),
                    nn_normalized_actions=np.array(self.nn_actions, np.float32).reshape(-1, 2))

    def close(self):
        assert self.geometry.native_steps == 0
        self.geometry.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--route', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--capture-proof', required=True)
    ap.add_argument('--legacy-run', required=True)
    ap.add_argument('--cohort', choices=['pilot13', 'development45', 'qualification32'], required=True)
    ap.add_argument('--parent-run')
    ap.add_argument('--protocol-audit')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    out = Path(a.out)
    freeze_inputs(out, a)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(out/'input_route.npz') == 'dc41fdb176bc244474e9e106b88e4ddec52e1052870dda05df89d8c0ed2cc316'
    shutil.copy2(a.checkpoint, out/'policy.bin')
    assert sha(out/'policy.bin') == 'b8134743906935239d5ae8366951a1f9ab411f3fc33a09a724413aefad06632a'
    for name in ('puffer_mingru_cpu.py', 'puffer_training_support.py'):
        shutil.copy2(Path(__file__).parent/name, out/name)
    capture = Path(a.capture_proof)
    assert (capture/'completed.json').is_file()
    local = json.loads((capture/'summary.json').read_text())
    assert all(r['success'] for r in local['results'] if r['canonical_sign'] and r['phase'] == 400)
    assert len([r for r in local['results'] if r['canonical_sign'] and r['phase'] == 400]) == 3
    assert (capture/'puffer_mingru_cpu.py').read_bytes() == (out/'puffer_mingru_cpu.py').read_bytes()
    identity = {name: sha(out/name) for name in ('input_route.npz', 'policy.bin', 'driver.py',
                                                'puffer_mingru_cpu.py', 'puffer_training_support.py')}
    if a.cohort != 'pilot13':
        parent = Path(a.parent_run)
        assert (parent/'completed.json').is_file()
        s = json.loads((parent/'summary.json').read_text())
        expected, count = ('pilot13', 13) if a.cohort == 'development45' else ('development45', 45)
        assert s['cohort'] == expected and s['passes'] == len(s['results']) == count
        assert s['controller_files_sha256'] == identity
        assert json.loads((parent/'inputs.json').read_text())['source_sha256'] == json.loads((out/'inputs.json').read_text())['source_sha256']
        shutil.copy2(parent/'summary.json', out/'parent_summary.json')
        if a.cohort == 'qualification32':
            audit = json.loads(Path(a.protocol_audit).read_text())
            assert audit['passed'] and audit['development_summary_sha256'] == sha(parent/'summary.json')
            shutil.copy2(a.protocol_audit, out/'development_protocol_audit.json')
    cases = [(20261016, 0.)]+[(s, .02) for s in range(20261116, 20261128)]
    if a.cohort == 'development45':
        cases += [(s, .02) for s in range(20261416, 20261448)]
    if a.cohort == 'qualification32':
        cases = [(s, .02) for s in range(20330116, 20330148)]
    write_json(out/'frozen_controller.json', dict(files_sha256=identity, cases=cases, handoff_step=400,
               actual_horizon=1500, source_runtime_sha256=json.loads((out/'inputs.json').read_text())['source_sha256']))
    with np.load(out/'input_route.npz') as z:
        payload = {k: z[k] for k in z.files}
    assert Config(**json.loads(str(payload['config']))) == Config(n_links=1)
    legacy, rows, total = Path(a.legacy_run), [], 0
    cap = 12000*(len(cases)+1)
    for seed, noise in cases:
        assert total+12000 <= cap
        row = episode(Config(n_links=1), lambda env: HybridCapture(env, payload, out/'policy.bin'),
                      seed, 'hanging', noise, out/f'seed{seed}.npz')
        assert row['replay_exact']
        row['inclusive_native_steps'] = row['native_steps']+row['proof_native_steps']
        total += row['inclusive_native_steps']
        prior = legacy/'nominal.npz' if noise == 0 else legacy/f'validation_{seed-20261116}.npz'
        if prior.is_file() and seed in [20261016, *range(20261116, 20261128)]:
            with np.load(prior) as old, np.load(out/f'seed{seed}.npz') as new:
                assert np.array_equal(old['states'][:401], new['states'][:401])
                assert np.array_equal(old['actions'][:400], new['actions'][:400])
            row['legacy_400_control_prefix_byte_exact'] = True
        rows.append(row)
        write_json(out/'summary.json', dict(results=rows, passes=sum(r['success'] for r in rows),
                   cohort=a.cohort, inclusive_native_steps=total, controller_files_sha256=identity,
                   reserved_cases_consumed=len(rows) if a.cohort == 'qualification32' else 0,
                   qualified=False, scope='original hanging hybrid classical swing and learned capture'))
        print(json.dumps({k: row[k] for k in ('seed', 'success', 'steps', 'best_hold_seconds')}), flush=True)
    repeat = episode(Config(n_links=1), lambda env: HybridCapture(env, payload, out/'policy.bin'),
                     *cases[0][:1], 'hanging', cases[0][1], out/'repeat_first.npz')
    total += repeat['native_steps']+repeat['proof_native_steps']
    assert total <= cap and repeat['replay_exact']
    with np.load(out/f'seed{cases[0][0]}.npz') as original, np.load(out/'repeat_first.npz') as repeated:
        assert all(original[k].tobytes() == repeated[k].tobytes() for k in
                   ('states', 'actions', 'raw_actions', 'tilts', 'nn_observations', 'nn_normalized_actions'))
    final = json.loads((out/'summary.json').read_text())
    final.update(inclusive_native_steps=total, full_first_closed_loop_reexecution_byte_exact=True,
                 reexecution_native_steps=repeat['native_steps']+repeat['proof_native_steps'])
    write_json(out/'summary.json', final)
    write_json(out/'completed.json', dict(status='completed'))


if __name__ == '__main__':
    main()
