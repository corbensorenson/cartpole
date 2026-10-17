"""Frozen tracking repair, with an explicitly canonical plant and actual gates."""
import argparse
from dataclasses import asdict,replace
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.control import RoutePolicy
from spatialpole.tracking_repair import repair_tracking
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--tape')
p.add_argument('--out',required=True)
p.add_argument('--iterations',type=int,default=40)
p.add_argument('--workers',type=int,default=4)
p.add_argument('--physics-budget',type=int,default=15000000)
p.add_argument('--penalty',type=float,default=1000.)
p.add_argument('--sqrt-backward',action='store_true')
p.add_argument('--axis-design',action='store_true')
p.add_argument('--minimum-alpha',type=float,default=.005)
p.add_argument('--small-step-trust-factor',type=float,default=3.)
p.add_argument('--derivative-order',type=int,choices=[2,4],default=2)
p.add_argument('--derivative-eps',type=float)
a = p.parse_args()
if a.derivative_eps is None:
    a.derivative_eps = 1e-3 if a.derivative_order==4 else 1e-5
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
base = Config(**json.loads(str(source['config'])))
config = Config(n_links=base.n_links)
assert replace(base,joint_damping=config.joint_damping,cart_armature=config.cart_armature)==config
H = len(source['actions'])
if a.tape:
    shutil.copy2(a.tape,out/'target_actual_tape.npz')
    write_json(out/'target_input.json',{'source':a.tape,'sha256':hashlib.sha256(Path(a.tape).read_bytes()).hexdigest(),
                                      'target_kind':'recorded actual motion'})
    tape = np.load(a.tape)
    assert Config(**json.loads(str(tape['config']))) == base
    target,target_controls = tape['states'][:H+1],tape['actions'][:H]
else:
    target,target_controls = source['states'],source['actions']
    write_json(out/'target_input.json',{'source':a.route,'target_kind':'nominal proposal; not assumed dynamically feasible'})
env = SpatialCartpole(config)
env.derivative_workers = a.workers
initial,_ = env.reset(seed=20261016,options={'mode':'hanging','angle_noise':0.})
start = time.perf_counter()

def log(row):
    with (out/'optimizer.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def checkpoint(iteration,states,actions,gains):
    (out/'checkpoints').mkdir(exist_ok=True)
    np.savez_compressed(out/'checkpoints'/f'iteration_{iteration:03d}.npz',states=states,
                        actions=actions,gains=gains,config=json.dumps(asdict(config)))

try:
    xs,us,K,design,history,proposal = repair_tracking(env,initial,target,
        target_controls,a.iterations,log,checkpoint,a.physics_budget,a.penalty,a.sqrt_backward,
        a.axis_design,a.minimum_alpha,a.small_step_trust_factor,a.derivative_order,a.derivative_eps)
except Exception as exc:
    write_json(out/'failure.json',{'status':'failed','type':type(exc).__name__,'message':str(exc),
                                  'native_steps':env.native_steps})
    env.close()
    raise
np.savez_compressed(out/'shooting_proposal.npz',**proposal,config=json.dumps(asdict(config)))
np.savez_compressed(out/'route.npz',states=xs,actions=us,gains=K,**design,
                    config=json.dumps(asdict(config)),world_balance=True)
native_design = env.native_steps
env.close()
factory = lambda e: RoutePolicy(e,xs,us,K,design,True)
rows = [episode(config,factory,20261016,'hanging',0.,out/'nominal.npz')]
if rows[0]['success'] and rows[0]['replay_exact']:
    rows.extend(episode(config,factory,20261116+k,'hanging',.02,out/f'validation_{k}.npz')
                for k in range(12))
write_json(out/'summary.json',{'n_links':config.n_links,'passes':sum(r['success'] and r['replay_exact'] for r in rows),
    'results':rows,'design_native_steps':native_design,'optimizer_iterations':len(history),
    'last_optimizer_record':history[-1] if history else None,'wall_seconds':time.perf_counter()-start,
    'scope':'input motion is a geometric target only; actual gates use the unchanged benchmark'})
write_json(out/'completed.json',{'status':'completed'})
print(json.dumps({'passes':sum(r['success'] for r in rows),'results':rows}),flush=True)
