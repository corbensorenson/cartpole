"""Check real native reward-offset configuration without changing plant bytes."""
import argparse
import ctypes as ct
import hashlib
import math
import os
from pathlib import Path
import shutil

import mujoco
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.env import model_xml
from spatialpole.experiment import freeze_inputs, write_json
from verify_puffer_native import Library, exact

p=argparse.ArgumentParser();p.add_argument('--build',required=True);p.add_argument('--out',required=True)
args=p.parse_args();out=Path(args.out);build=Path(args.build);freeze_inputs(out,args)
shutil.copytree(Path(__file__).resolve().parents[1]/'native/puffer_spatial',out/'native_source')
directory=os.add_dll_directory(str(Path(mujoco.__file__).parent))
library=Library(build/'spatial_1.dll')
create=library.lib.np_create_reward;create.argtypes=[ct.c_char_p,ct.c_uint,ct.c_double,ct.c_double];create.restype=ct.c_void_p
xml=out/'model.xml';xml.write_text(model_xml(Config(n_links=1)),encoding='utf-8')
env=SpatialCartpole(Config(n_links=1));initial,_=env.reset(seed=20261116,options=dict(mode='hanging',angle_noise=.02))
pointers=[create(os.fsencode(xml.resolve()),73,.02,offset) for offset in (0.,2.)]
assert all(pointers)
states=[];actions=[];rewards=[]
try:
    for pointer in pointers:library.np_set(pointer,initial)
    for step in range(64):
        action=np.array([.02*math.sin(.31*step),.02*math.cos(.17*step)],np.float32)
        for pointer in pointers:library.np_step(pointer,action)
        reads=[library.puffer_read(pointer,1) for pointer in pointers]
        for column in range(3):exact(reads[0][column],reads[1][column])
        exact(reads[0][3][1:],reads[1][3][1:]);assert reads[0][3][1]==0
        state,*_=env.step_force(action.astype(float)*80.);exact(state,reads[0][0])
        delivered=env.deliver(action.astype(float)*80.)
        force2=sum(float(u)**2 for u in delivered)
        reward=math.cos(float(env.tilts()[0]))-.03*sum(float(x)**2 for x in state[:2])-.01*force2/(80.*80.)
        assert reads[0][3][0]==np.float32(reward)
        assert reads[1][3][0]==np.float32(reward+2.)
        states.append(state);actions.append(delivered);rewards.append([reads[0][3][0],reads[1][3][0]])
    actual=int(sum(library.puffer_read(pointer,1)[1][7] for pointer in pointers))
    total=actual+env.native_steps;assert total==768
    np.savez_compressed(out/'comparison.npz',states=states,actions=actions,rewards=rewards,initial=initial)
    write_json(out/'summary.json',dict(passed=True,commands=64,offsets=[0.,2.],
        states_observations_gates_and_forces_byte_identical=True,independent_physics_byte_exact=True,
        reward_formula_and_offset_f32_exact=True,actual_native_steps=actual,proof_native_steps=env.native_steps,
        inclusive_native_steps=total,training=False,qualified=False,
        build_manifest_sha256=hashlib.sha256((build/'build.json').read_bytes()).hexdigest()))
    print('Reward offset check passed; 768 inclusive native integrations',flush=True)
finally:
    for pointer in pointers:library.np_close(pointer)
    env.close();directory.close()
