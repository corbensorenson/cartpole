"""Native plant parity, real PufferLib buffer lifecycle, and bounded CPU benchmark."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import mujoco
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.env import model_xml
from spatialpole.experiment import freeze_inputs, write_json

ROOT = Path(__file__).resolve().parents[1]
D = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
F = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')


class Library:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path.resolve()))
        specs = {
            'ns_create': ([ct.c_char_p, ct.c_int], ct.c_void_p),
            'ns_set': ([ct.c_void_p, ct.c_int, D], None),
            'ns_step': ([ct.c_void_p, D, ct.c_int], None),
            'ns_read': ([ct.c_void_p, ct.c_int, D, D, F, D], None),
            'ns_cost': ([ct.c_void_p], ct.c_uint64),
            'ns_close': ([ct.c_void_p], None),
            'np_create': ([ct.c_char_p, ct.c_uint, ct.c_double], ct.c_void_p),
            'np_set': ([ct.c_void_p, D], None),
            'np_step': ([ct.c_void_p, F], None),
            'np_read': ([ct.c_void_p, D, D, F, F], None),
            'np_close': ([ct.c_void_p], None),
        }
        for name, (args, result) in specs.items():
            f = getattr(self.lib, name)
            f.argtypes, f.restype = args, result
            setattr(self, name, f)

    def read(self, pointer, n, i=0):
        state, m = np.empty(4 + 7*n), np.empty(9)
        action, tilts = np.empty(2, np.float32), np.empty(n)
        self.ns_read(pointer, i, state, m, action, tilts)
        return state, m, action, tilts

    def puffer_read(self, pointer, n):
        state, m = np.empty(4 + 7*n), np.empty(9)
        obs, event = np.empty(7 + 7*n, np.float32), np.empty(6, np.float32)
        self.np_read(pointer, state, m, obs, event)
        return state, m, obs, event


def exact(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def compare(lib, pointer, n, env):
    state, m, delivered, tilts = lib.read(pointer, n)
    exact(state, env.state())
    exact(delivered, env.data.ctrl.astype(np.float32))
    assert np.max(abs(tilts-env.tilts())) < 5e-14
    assert m[0] == env.steps and m[1] == env.streak and m[2] == env.best_streak
    assert m[3] == env.min_plane_margin and bool(m[4]) == env.warning_seen
    info = env.info()
    assert bool(m[6]) == info['success']
    assert bool(m[5]) == (env.steps >= 1500 or env.min_plane_margin < 0 or env.warning_seen)


p = argparse.ArgumentParser()
p.add_argument('--build', required=True)
p.add_argument('--out', required=True)
p.add_argument('--baseline-root', default='output', help='Directory containing qualify_n1_001 and qualify_n2_001 tapes')
a = p.parse_args()
out, build = Path(a.out), Path(a.build)
freeze_inputs(out, a)
shutil.copytree(ROOT / 'native/puffer_spatial', out / 'native_source')
shutil.copy2(build / 'build.json', out / 'build.json')
cost = 0
records, benchmarks = [], []


def charge(steps):
    global cost
    cost += int(steps)
    write_json(out / 'accounting.json', dict(additional_native_steps=cost, cap=250000))
    assert cost <= 250000


dll_search = os.add_dll_directory(str(Path(mujoco.__file__).parent))
for n in (1, 2, 8):
    lib = Library(build / f'spatial_{n}.dll')
    xml = (out / f'model_{n}.xml').resolve()
    xml.write_text(model_xml(Config(n_links=n)))
    env = SpatialCartpole(Config(n_links=n))
    pointer = lib.ns_create(os.fsencode(xml), 1)
    assert pointer
    try:
        # Both axes, signs, subnormals and radial clipping from asymmetric states.
        for mode in ('hanging', 'upright'):
            for force in ([0.,0.], [17.,-23.], [80.,80.], [-91.,37.], [0.,80.], [75.,-4.], [1e-35,-1e-35]):
                state, _ = env.reset(seed=20261116, options=dict(mode=mode, angle_noise=.02))
                lib.ns_set(pointer, 0, state)
                raw = np.array(force, dtype=np.float64)
                lib.ns_step(pointer, raw, 1)
                env.step_force(raw)
                compare(lib, pointer, n, env)
                charge(4 + env.native_steps)
        # A complete existing qualified trajectory, never a fresh seed.
        full_tape = None
        if n in (1,2):
            full_tape = sorted((ROOT / a.baseline_root / f'qualify_n{n}_001').glob('seed*.npz'))[0]
            with np.load(full_tape) as tape:
                lib.ns_set(pointer, 0, np.ascontiguousarray(tape['states'][0]))
                for t, force in enumerate(tape['actions']):
                    lib.ns_step(pointer, force.astype(np.float64), 1)
                    state, m, action, tilts = lib.read(pointer, n)
                    exact(state, tape['states'][t+1])
                    exact(action, force)
                    assert np.max(abs(tilts-tape['tilts'][t])) < 5e-14
                assert m[0] == 1500 and m[6] == 1
                charge(6000)
        records.append(dict(links=n, single_transition_cases=14,
                            complete_qualified_tape=str(full_tape) if full_tape else None,
                            full_state_and_force_bytes_exact=True, gate_parity=True))
    finally:
        lib.ns_close(pointer)
        env.close()

    # Exercise actual puf_* functions compiled against pinned upstream headers.
    pointer = lib.np_create(os.fsencode(xml), 7, .02)
    env = SpatialCartpole(Config(n_links=n))
    try:
        state, _ = env.reset(seed=20261116, options=dict(mode='hanging', angle_noise=.02))
        lib.np_set(pointer, state)
        for t in range(24):
            action = np.array([.8*np.sin(.37*t), 1.3*np.cos(.23*t)], np.float32)
            lib.np_step(pointer, action)
            env.step(action)
            actual, m, obs, event = lib.puffer_read(pointer, n)
            exact(actual, env.state())
            expected_obs = np.r_[actual[:2]/3., actual[2:env.model.nq],
                                 actual[env.model.nq:env.model.nq+2]/4., actual[env.model.nq+2:]/15.,
                                 env.data.ctrl.astype(np.float32)/np.float32(80), (t+1)/1500.].astype(np.float32)
            exact(obs, expected_obs)
            assert event[1] == 0 and np.isfinite(event[0])
        charge(env.native_steps)
        # Complete stable upright episode tests terminal/logs/autoreset and hold gate.
        lib.np_set(pointer, env.upright_state())
        for _ in range(1500): lib.np_step(pointer, np.zeros(2, np.float32))
        actual, m, obs, event = lib.puffer_read(pointer, n)
        assert event[1] == 1 and event[2] == 1 and event[3] == 1 and abs(event[4]-30.) < 1e-5
        assert m[0] == 0 and event[5] == 1  # next reset already emitted
        assert actual[4] > .99 and np.max(abs(actual[env.model.nq:])) > 0
        lib.np_step(pointer, np.zeros(2, np.float32))
        assert lib.puffer_read(pointer, n)[3][1] == 0  # terminal does not leak
        # A rail exit also autoresets; it must never count as a success.
        state = env.upright_state(); state[0] = 2.999; state[env.model.nq] = 10.
        lib.np_set(pointer, state)
        lib.np_step(pointer, np.zeros(2, np.float32))
        event = lib.puffer_read(pointer, n)[3]
        assert event[1] == 1 and event[2] == 2 and event[3] == 1
        lib.np_step(pointer, np.array([np.nan, 0.], np.float32))
        actual, m, obs, event = lib.puffer_read(pointer, n)
        assert event[1] == 1 and event[2] == 3 and event[3] == 1
        charge(m[7])
        records[-1]['puffer_buffer_observation_terminal_autoreset_log_checks'] = True
    finally:
        lib.np_close(pointer)
        env.close()

    if n != 8: continue
    # Matched repeated transition workload; excludes setup and rendering.
    for count in (1,16,64,256):
        states = []
        template = SpatialCartpole(Config(n_links=n))
        for i in range(count):
            state, _ = template.reset(seed=20261116+i%12, options=dict(mode='hanging', angle_noise=.02))
            states.append(state)
        template.close()
        final_states = []
        for threads in (1,4):
            pointer = lib.ns_create(os.fsencode(xml), count)
            assert pointer
            try:
                for i, state in enumerate(states): lib.ns_set(pointer, i, state)
                start = time.perf_counter()
                for t in range(32):
                    forces = np.tile([5.*np.sin(.1*t), 3.*np.cos(.13*t)], (count,1))
                    lib.ns_step(pointer, forces, threads)
                elapsed = time.perf_counter()-start
                measured = lib.ns_cost(pointer)
                charge(measured)
                assert measured == count*32*4
                final = np.array([lib.read(pointer, n, i)[0] for i in range(count)])
                final_states.append(final)
                benchmarks.append(dict(links=n, environments=count, lane=f'native_threads_{threads}',
                                       control_transitions=count*32, native_steps=measured,
                                       seconds=elapsed, control_steps_per_second=count*32/elapsed))
            finally: lib.ns_close(pointer)
        exact(final_states[0], final_states[1])
        envs = [SpatialCartpole(Config(n_links=n)) for _ in range(count)]
        try:
            for env, state in zip(envs, states): env.set_state(state)
            start = time.perf_counter()
            for t in range(32):
                force = np.array([5.*np.sin(.1*t), 3.*np.cos(.13*t)])
                for env in envs: env.step_force(force)
            elapsed = time.perf_counter()-start
            measured = sum(env.native_steps for env in envs)
            charge(measured)
            exact(np.array([env.state() for env in envs]), final_states[0])
            benchmarks.append(dict(links=n, environments=count, lane='authoritative_python_loop',
                                   control_transitions=count*32, native_steps=measured,
                                   seconds=elapsed, control_steps_per_second=count*32/elapsed))
        finally:
            for env in envs: env.close()
        np.savez_compressed(out/f'benchmark_final_{count}.npz', states=final_states[0])

write_json(out / 'summary.json', dict(records=records, benchmarks=benchmarks,
    additional_native_steps=cost, scope='native CPU plant/adapter parity and matched stepping throughput; no CUDA trainer or learned policy tested',
    puffer_ref='ea7f989371270d2948f96728bda8aec5e7634a4c',
    source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'native_source').iterdir() if p.is_file()}))
write_json(out / 'completed.json', dict(status='completed'))
print(json.dumps(dict(parity='passed', additional_native_steps=cost, benchmarks=benchmarks)), flush=True)
