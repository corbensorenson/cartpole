"""Bounded mechanical continuation; only the exact restored plant can promote."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.control import balance_design,RoutePolicy
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--tape',required=True)
p.add_argument('--out',required=True)
p.add_argument('--step',type=float,default=0.02)
p.add_argument('--minimum-step',type=float,default=0.0003125)
p.add_argument('--max-trials',type=int,default=200)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
shutil.copy2(a.tape,out/'initial_actual_tape.npz')
write_json(out/'tape_input.json',{'source':a.tape,'sha256':hashlib.sha256(Path(a.tape).read_bytes()).hexdigest()})
source,tape = np.load(a.route),np.load(a.tape)
base = Config(**json.loads(str(source['config'])))
target = replace(base,joint_damping=0.0075,cart_armature=0.)
assert target==Config(n_links=base.n_links),'other altered parameters are outside this continuation'
assert Config(**json.loads(str(tape['config'])))==base
assert len(tape['states'])==base.horizon+1 and len(tape['actions'])==base.horizon
H = len(source['actions'])
states,actions,gains = tape['states'][:H+1].copy(),tape['actions'][:H].copy(),source['gains'].copy()
fraction,step,success_run = 0.,a.step,0
rows = []
native_design = actual_steps = proof_steps = 0
start = time.perf_counter()
for trial in range(a.max_trials):
    proposed = min(1.,fraction+step)
    if 1.-proposed<1e-12:
        proposed = 1.
    config = target if proposed==1. else replace(target,
        joint_damping=(1-proposed)*base.joint_damping+proposed*target.joint_damping,
        cart_armature=(1-proposed)*base.cart_armature+proposed*target.cart_armature)
    env = SpatialCartpole(config)
    design = balance_design(env)
    native_design += env.native_steps
    env.close()
    candidate = out/f'trial_{trial:03d}'
    candidate.mkdir()
    np.savez_compressed(candidate/'input_controller.npz',states=states,actions=actions,gains=gains,
                        **design,config=json.dumps(asdict(config)),world_balance=True)
    factory = lambda e: RoutePolicy(e,states,actions,gains,design,True)
    result = episode(config,factory,20261016,'hanging',0.,candidate/'actual.npz')
    actual_steps += result['native_steps']
    proof_steps += result['proof_native_steps']
    accepted = result['success'] and result['replay_exact']
    row = {'trial':trial,'fraction':proposed,'step':step,'accepted':bool(accepted),
           'config':asdict(config),'result':result}
    rows.append(row)
    write_json(candidate/'result.json',row)
    if accepted:
        actual = np.load(candidate/'actual.npz')
        states,actions = actual['states'][:H+1].copy(),actual['actions'][:H].copy()
        fraction = proposed
        np.savez_compressed(candidate/'accepted_controller.npz',states=states,actions=actions,gains=gains,
                            **design,config=json.dumps(asdict(config)),world_balance=True)
        success_run += 1
        if success_run>=2:
            step = min(a.step,step*1.5)
    else:
        step /= 2
        success_run = 0
    summary = {'n_links':base.n_links,'restored_fraction':fraction,'reached_benchmark':fraction==1.,
               'trials':rows,'native_design_steps':native_design,'actual_physics_steps':actual_steps,
               'proof_physics_steps':proof_steps,'wall_seconds':time.perf_counter()-start,
               'scope':'intermediate plants altered; only fraction=1 is the unchanged benchmark'}
    write_json(out/'summary.json',summary)
    print(json.dumps({'trial':trial,'fraction':proposed,'accepted':bool(accepted),'step':step,
                     'hold':result['best_hold_seconds'],'steps':result['steps']}),flush=True)
    if fraction==1.:
        shutil.copy2(candidate/'accepted_controller.npz',out/'route.npz')
        break
    if step<a.minimum_step:
        break
write_json(out/'completed.json',{'status':'completed','reached_benchmark':fraction==1.,'trials':len(rows)})
