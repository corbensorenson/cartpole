"""Replay a proposed controller on an explicitly altered mechanical model."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.control import balance_design,RoutePolicy
from spatialpole.transfer import spatial_tracking_gains
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
p.add_argument('--damping',type=float,required=True)
p.add_argument('--cart-armature',type=float,required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
original = Config(**json.loads(str(source['config'])))
config = replace(original,joint_damping=a.damping,cart_armature=a.cart_armature)
env = SpatialCartpole(config)
env.derivative_workers = 4
design = balance_design(env)
gains = spatial_tracking_gains(env,source['states'],source['actions'],design)
gains[:,0] = source['gains'][:,0]
np.savez_compressed(out/'route.npz',states=source['states'],actions=source['actions'],gains=gains,
                    **design,config=json.dumps(asdict(config)),world_balance=True)
factory = lambda e: RoutePolicy(e,source['states'],source['actions'],gains,design,True)
result = episode(config,factory,20261016,'hanging',0.,out/'nominal.npz')
write_json(out/'summary.json',{'n_links':config.n_links,'altered_plant':config!=Config(n_links=config.n_links),
    'results':[result],'passes':int(result['success'] and result['replay_exact']),
    'design_native_steps':env.native_steps,'scope':'mechanical-variant component; not a new benchmark link count'})
write_json(out/'completed.json',{'status':'completed'})
print(json.dumps(result),flush=True)
env.close()
