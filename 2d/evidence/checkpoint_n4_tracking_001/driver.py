"""Test a frozen optimizer checkpoint through complete actual episodes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.control import balance_design,RoutePolicy
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
p.add_argument('--episodes',type=int,default=12)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=config.n_links)
env = SpatialCartpole(config)
design = balance_design(env)
native_design = env.native_steps
env.close()
np.savez_compressed(out/'route.npz',states=source['states'],actions=source['actions'],
                    gains=source['gains'],**design,world_balance=True,config=json.dumps(asdict(config)))
factory = lambda e: RoutePolicy(e,source['states'],source['actions'],source['gains'],design,True)
rows = [episode(config,factory,20261016,'hanging',0.,out/'nominal.npz')]
print(json.dumps(rows[0]),flush=True)
if rows[0]['success'] and rows[0]['replay_exact']:
    for k in range(a.episodes):
        rows.append(episode(config,factory,20261116+k,'hanging',.02,out/f'validation_{k}.npz'))
        print(json.dumps({'development':k,'success':rows[-1]['success']}),flush=True)
write_json(out/'summary.json',{'n_links':config.n_links,'passes':sum(r['success'] and r['replay_exact'] for r in rows),
    'results':rows,'design_native_steps':native_design,'scope':'actual checkpoint probe; no independent qualification'})
write_json(out/'completed.json',{'status':'completed'})
