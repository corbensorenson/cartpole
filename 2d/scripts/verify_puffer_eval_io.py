"""Exercise native evaluation capture before any trained policy is available."""
import argparse
import ctypes as ct
import json
import hashlib
import os
from pathlib import Path
import shutil

import mujoco
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.experiment import freeze_inputs, write_json
from puffer_eval_io import create_plan, import_traces, HEADER
from verify_puffer_native import Library, exact

p = argparse.ArgumentParser(); p.add_argument('--build',required=True); p.add_argument('--out',required=True)
a = p.parse_args(); out = Path(a.out); build = Path(a.build)
freeze_inputs(out,a)
shutil.copytree(Path(__file__).resolve().parents[1]/'native/puffer_spatial',out/'native_source')
shutil.copy2(build/'build.json',out/'build.json')
assert json.loads((build/'build.json').read_text())['evaluation_io']
assert json.loads((build/'build.json').read_text())['native_source_sha256']=={
    p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (out/'native_source').iterdir() if p.suffix in ('.h','.cpp')}
search = os.add_dll_directory(str(Path(mujoco.__file__).parent))
total_actual = total_proof = 0; results=[]
for n in (1,2,8):
    env = SpatialCartpole(Config(n_links=n))
    hanging,_ = env.reset(seed=20261116,options=dict(mode='hanging',angle_noise=.02))
    upright = env.upright_state(); rail = upright.copy(); rail[0]=2.999; rail[env.model.nq]=10.
    cases=[(20261116,.02,0,hanging),(20261017,0.,1,upright),
           (20261018,0.,2,rail),(20261019,0.,2,upright)]
    plan=out/f'plan_n{n}';create_plan(plan,n,cases,'diagnostic')
    traces=out/f'traces_n{n}';traces.mkdir()
    lib=Library(build/f'spatial_{n}.dll')
    native_create=lib.lib.np_create_eval
    native_create.argtypes=[ct.c_char_p]*3;native_create.restype=ct.c_void_p
    pointer=native_create(os.fsencode((plan/'model.xml').resolve()),os.fsencode((plan/'plan.bin').resolve()),os.fsencode(traces.resolve()))
    assert pointer
    try:
        for index,(_,_,_,initial) in enumerate(cases):
            state,m,obs,event=lib.puffer_read(pointer,n)
            exact(state,initial)
            env.set_state(initial)
            assert abs(m[8]-max(env.tilts()))<5e-14  # reset tilts must not be stale
            for t in range(1500):
                if index==0:action=np.array([.35*np.sin(.31*t),.24*np.cos(.17*t)],np.float32)
                elif index==3:action=np.array([np.nan,0.],np.float32)
                else:action=np.zeros(2,np.float32)
                lib.np_step(pointer,action)
                state,m,obs,event=lib.puffer_read(pointer,n)
                if event[1]:break
            assert event[1]==1 and event[2]==index+1
            if index==1:assert t==1499 and event[3]>=1  # synthetic upright, NOT hanging success
        before=m[7]
        for _ in range(5):lib.np_step(pointer,np.zeros(2,np.float32))
        state,m,obs,event=lib.puffer_read(pointer,n)
        assert m[7]==before and event[1]==0 and np.all(obs==0)
        assert len(list(traces.glob('*.sptrace')))==4
        total_actual+=int(m[7])
    finally:
        lib.np_close(pointer);env.close()
    accounting=json.loads((traces/'accounting.json').read_text())
    assert accounting['native_steps']==int(before) and accounting['closed_environments']==1
    assert accounting['control_transitions']*4==int(before)
    report=import_traces(plan,traces,out/f'import_n{n}')
    assert report['actual_native_steps']==int(before)
    assert report['cases']==4 and report['results'][1]['success']
    assert report['results'][3]['policy_failure'] and report['results'][3]['steps']==0
    assert all(not row['success'] for row in report['results'][2:])
    total_proof+=report['additional_native_steps']
    assert total_actual+total_proof<=100000
    results.append(dict(links=n,actual_native_steps=int(before),proof_native_steps=report['additional_native_steps'],
                        diagnostic_only=True,cases=4,actual_tape_and_preobservation_bytes_exact=True,
                        complete_upright_horizon=True,rail_and_nonfinite_rejection=True,eof_no_extra_physics=True))
    results[-1]['native_shutdown_accounting_verified']=True
    write_json(out/'accounting.json',dict(actual_native_steps=total_actual,additional_proof_native_steps=total_proof,
                                         inclusive_native_steps=total_actual+total_proof))
    if n==1:
        # Both rejections happen before the first replay step, so cost is zero.
        negative=[]
        for variant in ('preobservation_corruption','truncated_record'):
            folder=out/variant;shutil.copytree(traces,folder)
            target=folder/'episode0000.sptrace';payload=bytearray(target.read_bytes())
            if variant=='preobservation_corruption':
                offset=HEADER.size+(4+7*n)*8+36
                payload[offset]^=1
            else:payload=payload[:-1]
            target.write_bytes(payload)
            try:import_traces(plan,folder,out/(variant+'_rejected'))
            except AssertionError:negative.append(variant)
            else:raise AssertionError('corrupt trace accepted')
        assert len(negative)==2
write_json(out/'summary.json',dict(results=results,negative_cases=negative,
    inclusive_native_steps=total_actual+total_proof,actual_native_steps=total_actual,
    additional_proof_native_steps=total_proof,trained_policy=False,qualification=False,
    scope='diagnostic evaluation-interface capture/replay/EOF/terminal/video-input checks; no learning or hanging-start certificate'))
write_json(out/'completed.json',dict(status='completed'))
print(json.dumps(dict(status='passed',inclusive_native_steps=total_actual+total_proof)),flush=True)
