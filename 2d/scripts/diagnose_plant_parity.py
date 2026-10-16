"""Finite local mechanics comparison; altered plants never count as solves."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import mujoco.rollout
import numpy as np
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
states,actions = source['states'],source['actions']
rows = []
for name,damping,slide_armature in (('benchmark',config.joint_damping,0.),('old_damping',0.015,0.),('old_damping_and_slide',0.015,0.0005)):
    env = SpatialCartpole(config)
    env.model.dof_damping[2:] = damping/config.n_links
    env.model.dof_armature[:2] = slide_armature
    initial = np.c_[np.zeros(len(actions)),states[:-1]]
    controls = np.repeat(actions[:,None,:],config.frame_skip,axis=1)
    full,_ = mujoco.rollout.rollout(env.model,env.data,initial,controls)
    assert np.all(np.diff(np.c_[np.zeros(len(actions)),full[:,:,0]],axis=1)>0)
    defects = np.array([env.difference(x,y) for x,y in zip(full[:,-1,1:],states[1:])])
    nv = env.model.nv
    metrics = {'max_cart_position_m':float(np.max(np.linalg.norm(defects[:,:2],axis=1))),
               'max_joint_rotation_rad':float(np.max(np.linalg.norm(defects[:,2:nv].reshape(-1,config.n_links,3),axis=2))),
               'max_cart_rate_m_s':float(np.max(np.linalg.norm(defects[:,nv:nv+2],axis=1))),
               'max_joint_rate_rad_s':float(np.max(np.linalg.norm(defects[:,nv+2:].reshape(-1,config.n_links,3),axis=2)))}
    row = {'name':name,'total_joint_damping':damping,'slide_armature':slide_armature,
           'native_physics_steps':len(actions)*config.frame_skip,'metrics':metrics}
    rows.append(row)
    np.savez_compressed(out/f'{name}.npz',predicted_states=full[:,-1,1:],defects=defects)
    print(json.dumps(row),flush=True)
    env.close()
write_json(out/'summary.json',{'n_links':config.n_links,'comparisons':rows,
    'scope':'finite independent smooth-force transitions, altered parameter diagnostics, not episode results'})
write_json(out/'completed.json',{'status':'completed'})
