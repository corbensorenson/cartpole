"""Compare count-lift proposals by measured one-command native defects."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import mujoco.rollout
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.transfer import split_reference_states
from spatialpole.physical import directions_and_rates
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
old_cfg = Config(**json.loads(str(source['config'])))
config = replace(old_cfg,n_links=old_cfg.n_links+1)
old,new = SpatialCartpole(old_cfg),SpatialCartpole(config)
states,actions = source['states'],source['actions']
directions,rates = directions_and_rates(old,states)
nv,H = new.model.nv,len(actions)
scales = np.r_[[3.,3.],np.full(nv-2,.15),[4.,4.],np.full(nv-2,15.)]
rows = []
best = None
for index in range(old_cfg.n_links):
    proposal = split_reference_states(old,new,states,index)
    mapping = list(range(old_cfg.n_links))
    mapping.insert(index+1,index)
    actual_d,actual_v = directions_and_rates(new,proposal)
    direction_error = float(np.max(np.abs(actual_d-directions[:,mapping])))
    rate_error = float(np.max(np.abs(actual_v-rates[:,mapping])))
    assert direction_error < 1e-10 and rate_error < 1e-10
    initial = np.c_[np.zeros(H),proposal[:-1]]
    controls = np.repeat(actions[:,None,:],config.frame_skip,axis=1)
    full,_ = mujoco.rollout.rollout(new.model,new.data,initial,controls)
    assert np.all(np.diff(np.c_[np.zeros(H),full[:,:,0]],axis=1)>0)
    defects = np.array([new.difference(f,x) for f,x in zip(full[:,-1,1:],proposal[1:])])
    value = .5*float(np.sum((defects/scales)**2))
    np.savez_compressed(out/f'split_{index}.npz',states=proposal,actions=actions,
                        defects=defects,config=json.dumps(asdict(config)))
    rows.append({'split_index':index,'normalized_defect_value':value,
                 'max_scaled_defect':float(np.max(np.abs(defects/scales))),
                 'direction_preservation_error':direction_error,'direction_rate_preservation_error':rate_error})
    if best is None or value<best[0]:
        best = value,index,proposal
np.savez_compressed(out/'route.npz',states=best[2],actions=actions,config=json.dumps(asdict(config)))
write_json(out/'summary.json',{'n_links':config.n_links,'selected_index':best[1],
    'candidates':rows,'native_physics_steps':H*config.frame_skip*old_cfg.n_links,
    'scope':'geometric state proposal selection by local defects, not an actual episode or solve'})
write_json(out/'completed.json',{'status':'completed'})
print(json.dumps({'selected_index':best[1],'candidates':rows}),flush=True)
old.close()
new.close()
