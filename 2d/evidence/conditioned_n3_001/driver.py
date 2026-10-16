"""Condition a frozen route's noisy hanging launch without any state resets."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.control import balance_design
from spatialpole.conditioning import ParkedRoutePolicy
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
p.add_argument('--park-seconds',type=float,default=12.)
p.add_argument('--no-align',action='store_true')
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
config = Config(**json.loads(str(source['config'])))
env = SpatialCartpole(config)
reference,_ = env.reset(seed=0,options={'mode':'hanging','angle_noise':0.})
parking = balance_design(env,reference=reference)
park_steps = round(a.park_seconds/config.dt)
if park_steps < 0 or park_steps+len(source['actions']) >= config.horizon:
    raise ValueError('parking plus route must leave a continuation interval')
design = {k:source[k] for k in ('K','P','Q','R','reference')}
values = {k:source[k] for k in source.files}
values.update(park_steps=park_steps,align_gauge=not a.no_align,
              parking_reference=parking['reference'],parking_K=parking['K'],
              parking_P=parking['P'],parking_Q=parking['Q'],parking_R=parking['R'])
np.savez_compressed(out/'route.npz',**values)
factory = lambda e: ParkedRoutePolicy(e,source['states'],source['actions'],source['gains'],design,parking,park_steps,not a.no_align)
rows = [episode(config,factory,20261016,'hanging',0.,out/'nominal.npz')]
print('nominal: '+json.dumps({k:v for k,v in rows[0].items() if k!='policy_metadata'}),flush=True)
seeds = list(range(20261116,20261128))+list(range(20261416,20261448))
for k,seed in enumerate(seeds):
    rows.append(episode(config,factory,seed,'hanging',0.02,out/f'seed{seed}.npz'))
    write_json(out/'summary.json',{'n_links':config.n_links,'passes':sum(x['success'] and x['replay_exact'] for x in rows),
        'results':rows,'parking_closed_loop_radius':parking['closed_loop_radius'],
        'design_native_steps':env.native_steps,'cohort_scope':'development including previously failed raw-route qualification'})
    if k%8==0:
        print(json.dumps({'development_episodes':k+1,'passes':sum(x['success'] and x['replay_exact'] for x in rows)}),flush=True)
write_json(out/'completed.json',{'status':'completed','episodes':len(rows)})
env.close()
