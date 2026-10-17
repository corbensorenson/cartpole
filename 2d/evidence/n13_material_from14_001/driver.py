"""Hash-bound planar material-shape proposal at a different native link count."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.transfer import decode_planar_coordinates,regrid_planar_material_states,lift_planar_state
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--source-config',required=True)
p.add_argument('--source-links',type=int,required=True)
p.add_argument('--links',type=int,required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
foreign = out/'foreign_sources'
foreign.mkdir()
identities = {}
for name in ('route','source_config'):
    original = Path(getattr(a,name))
    copy = foreign/(name+'_'+original.name)
    shutil.copy2(original,copy)
    identities[name] = dict(original=str(original),copy=str(copy.relative_to(out)),sha256=hashlib.sha256(copy.read_bytes()).hexdigest())
donor = json.loads(Path(a.route).read_text())
physical = decode_planar_coordinates(donor['search']['nominal_coordinate_states'],
                                     donor['controller']['coordinate_transform'],a.source_links)
lengths = np.array(donor['result']['final_info']['lengths'])
assert len(lengths)==a.source_links and abs(float(np.sum(lengths))-3.)<1e-12
assert abs(float(np.sum(donor['result']['final_info']['masses']))-1.)<1e-12
source_env = SpatialCartpole(Config(n_links=a.source_links))
source_error = 0.
for decoded,tape in zip(physical[1:],donor['result']['trajectory']):
    original = lift_planar_state(source_env,tape['qpos'],tape['qvel'])
    mapped = lift_planar_state(source_env,decoded[:a.source_links+1],decoded[a.source_links+1:])
    source_error = max(source_error,float(np.max(np.abs(source_env.difference(mapped,original)))))
assert source_error<2e-6
source_env.close()
config = Config(n_links=a.links)
target = regrid_planar_material_states(physical,lengths,np.full(a.links,3./a.links))
env = SpatialCartpole(config)
states = np.array([lift_planar_state(env,x[:a.links+1],x[a.links+1:]) for x in target])
actions = np.c_[np.array(donor['controller']['controls'])*80.,np.zeros(len(states)-1)]
assert actions.shape==(len(states)-1,2) and np.max(np.linalg.norm(actions,axis=1))<=80
assert abs(float(donor['controller']['horizon_seconds'])/len(actions)-config.dt)<1e-12
initial,_ = env.reset(seed=20261016,options={'mode':'hanging','angle_noise':0.})
initial_error = float(np.max(np.abs(env.difference(states[0],initial))))
assert initial_error<1e-6
direction_error = 0.
for x,row in zip(states,target):
    env.set_state(x)
    theta = np.cumsum(row[1:a.links+1])
    expected = np.c_[np.sin(theta),np.zeros(a.links),np.cos(theta)]
    direction_error = max(direction_error,float(np.max(np.abs(env.data.xmat[env.link_ids].reshape(-1,3,3)[:,:,2]-expected))))
assert direction_error<1e-10 and env.native_steps==0
np.savez_compressed(out/'route.npz',states=states,actions=actions,config=json.dumps(asdict(config)))
np.savez_compressed(out/'physical_transfer.npz',source=physical,target=target,source_lengths=lengths,target_lengths=np.full(a.links,3./a.links))
write_json(out/'foreign_inputs.json',identities)
write_json(out/'summary.json',dict(n_links=a.links,source_links=a.source_links,source_physical_tape_error=source_error,
    initial_hanging_error=initial_error,coordinate_direction_error=direction_error,design_native_steps=0,
    reference_steps=len(actions),source_nominal_hold_seconds=donor['result']['final_info']['max_upright_streak_seconds'],
    claim='material angle/rate proposal only; canonical target rebuilt, no physical episode or qualified count'))
write_json(out/'completed.json',dict(status='completed'))
env.close()
