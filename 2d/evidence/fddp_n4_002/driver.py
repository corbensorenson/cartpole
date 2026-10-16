"""Run a frozen defect-aware synthesis experiment and actual episode gates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.fddp import repair_reference
from spatialpole.control import RoutePolicy
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
p.add_argument('--iterations',type=int,default=80)
p.add_argument('--terminal-scale',type=float,default=10.)
p.add_argument('--penalty',type=float,default=1000.)
p.add_argument('--workers',type=int,default=4)
p.add_argument('--physics-budget',type=int,default=25000000)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
start = time.perf_counter()
source = np.load(a.route)
config = Config(**json.loads(str(source['config'])))
env = SpatialCartpole(config)
env.derivative_workers = a.workers
initial,_ = env.reset(seed=20261016,options={'mode':'hanging','angle_noise':0.})
states,actions = source['states'],source['actions']
if len(actions)>config.horizon or a.workers<1 or a.penalty<=0 or a.terminal_scale<=0:
    raise ValueError('invalid repair horizon or numerical parameters')

def log(row):
    with (out/'optimizer.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row)+'\n')
    if row['iteration']%5==0:
        print(json.dumps(row),flush=True)

def checkpoint(iteration,xs,us,gains,gaps):
    (out/'checkpoints').mkdir(exist_ok=True)
    np.savez_compressed(out/'checkpoints'/f'iteration_{iteration:03d}.npz',states=xs,actions=us,
                        gains=gains,gaps=gaps,config=json.dumps(asdict(config)))

try:
    xs,us,K,design,history,proposal = repair_reference(env,initial,states,actions,a.iterations,
        a.terminal_scale,a.penalty,log,checkpoint,a.physics_budget)
except Exception as exc:
    write_json(out/'failure.json',{'status':'failed','type':type(exc).__name__,'message':str(exc),
               'native_steps':env.native_steps,'wall_seconds':time.perf_counter()-start})
    env.close()
    raise
np.savez_compressed(out/'shooting_proposal.npz',**proposal,config=json.dumps(asdict(config)))
np.savez_compressed(out/'route.npz',states=xs,actions=us,gains=K,**design,
                    config=json.dumps(asdict(config)),world_balance=True)
native_design = env.native_steps
env.close()
factory = lambda e: RoutePolicy(e,xs,us,K,design,True)
rows = [episode(config,factory,20261016,'hanging',0.,out/'nominal.npz')]
print('nominal: '+json.dumps(rows[0]),flush=True)
if rows[0]['success'] and rows[0]['replay_exact']:
    for k in range(12):
        rows.append(episode(config,factory,20261116+k,'hanging',0.02,out/f'validation_{k}.npz'))
write_json(out/'summary.json',{'n_links':config.n_links,'passes':sum(r['success'] and r['replay_exact'] for r in rows),
    'results':rows,'design_native_steps':native_design,'optimizer_iterations':len(history),
    'last_optimizer_record':history[-1] if history else None,'wall_seconds':time.perf_counter()-start,
    'scope':'shooting proposal may be infeasible; route is separately rebuilt with zero gaps and actual episodes use float32 forces'})
write_json(out/'completed.json',{'status':'completed','wall_seconds':time.perf_counter()-start})
