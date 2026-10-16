"""Measure independent one-command dynamics defects of a trajectory proposal."""
import argparse
import json
from pathlib import Path
import numpy as np
import mujoco.rollout
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
config = Config(**json.loads(str(source['config'])))
env = SpatialCartpole(config)
states,actions = source['states'],source['actions']
initial = np.c_[np.zeros(len(actions)),states[:-1]]
controls = np.repeat(actions[:,None,:],config.frame_skip,axis=1)
full,_ = mujoco.rollout.rollout(env.model,env.data,initial,controls)
steps = int(np.count_nonzero(np.diff(np.c_[np.zeros(len(actions)),full[:,:,0]],axis=1)>0))
assert steps == len(actions)*config.frame_skip
predicted = full[:,-1,1:]
defects = np.array([env.difference(x,y) for x,y in zip(predicted,states[1:])])
nv,n = env.model.nv,config.n_links
groups = {'cart_position_m':np.linalg.norm(defects[:,:2],axis=1),
          'joint_rotation_rad':np.linalg.norm(defects[:,2:nv].reshape(-1,n,3),axis=2),
          'cart_velocity_m_per_s':np.linalg.norm(defects[:,nv:nv+2],axis=1),
          'joint_velocity_rad_per_s':np.linalg.norm(defects[:,nv+2:].reshape(-1,n,3),axis=2)}
metrics = {k:{'max':float(np.max(v)),'rms':float(np.sqrt(np.mean(v*v)))} for k,v in groups.items()}
np.savez_compressed(out/'defects.npz',predicted_states=predicted,reference_states=states,
                    actions=actions,defects=defects,config=source['config'])
write_json(out/'summary.json',{'n_links':n,'commands':len(actions),'native_physics_steps':steps,
           'metrics':metrics,'scope':'independent smooth-force native proposal transitions, not an actual episode or solve'})
write_json(out/'completed.json',{'status':'completed'})
print(json.dumps(metrics),flush=True)
env.close()
