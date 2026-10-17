"""Material continuation from a hash-bound qualified spatial reference."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.symmetry import clean_planar_reference
from spatialpole.transfer import decode_planar_native_states,regrid_planar_material_states,lift_planar_state
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--source-manifest',required=True)
p.add_argument('--links',type=int,required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
shutil.copy2(a.source_manifest,out/'source_public_manifest.json')
manifest = json.loads((out/'source_public_manifest.json').read_bytes())
digest = hashlib.sha256((out/'input_route.npz').read_bytes()).hexdigest()
assert digest==manifest['files_sha256'][manifest['controller']]
source = np.load(out/'input_route.npz')
source_config = Config(**json.loads(str(source['config'])))
assert source_config==Config(n_links=manifest['accepted_swingup_links'])
assert asdict(source_config)==manifest['config'] and a.links==source_config.n_links+1
H = int(source['swing_steps'])
assert 0<H<=len(source['actions'])
se = SpatialCartpole(source_config)
cleaned,controls,cleanup = clean_planar_reference(se,source['states'][:H+1],source['actions'][:H])
physical = decode_planar_native_states(se,cleaned)
roundtrip = 0.
for row,state in zip(physical,cleaned):
    rebuilt = lift_planar_state(se,row[:source_config.n_links+1],row[source_config.n_links+1:])
    roundtrip = max(roundtrip,float(np.max(abs(se.difference(rebuilt,state)))))
assert roundtrip<=1e-10 and se.native_steps==0
se.close()
config = Config(n_links=a.links)
assert source_config.dt==config.dt
lengths = np.full(source_config.n_links,source_config.total_length/source_config.n_links)
target_lengths = np.full(a.links,config.total_length/a.links)
target = regrid_planar_material_states(physical,lengths,target_lengths)
env = SpatialCartpole(config)
states = np.array([lift_planar_state(env,row[:a.links+1],row[a.links+1:]) for row in target])
actions = np.array([env.deliver(u) for u in controls],float)
initial,_ = env.reset(seed=20261016,options={'mode':'hanging','angle_noise':0.})
initial_error = float(np.max(abs(env.difference(states[0],initial))))
assert initial_error<=1e-6
directions_error = 0.
for state,row in zip(states,target):
    env.set_state(state)
    theta = np.cumsum(row[1:a.links+1])
    expected = np.c_[np.sin(theta),np.zeros(a.links),np.cos(theta)]
    directions_error = max(directions_error,float(np.max(abs(env.data.xmat[env.link_ids].reshape(-1,3,3)[:,:,2]-expected))))
assert directions_error<=1e-10 and env.native_steps==0
np.savez_compressed(out/'route.npz',states=states,actions=actions,config=json.dumps(asdict(config)))
np.savez_compressed(out/'physical_transfer.npz',source=physical,target=target,source_lengths=lengths,target_lengths=target_lengths)
write_json(out/'reference_cleanup.json',cleanup)
write_json(out/'summary.json',dict(n_links=a.links,source_links=source_config.n_links,
    reference_steps=H,source_controller_sha256=digest,
    source_manifest_sha256=hashlib.sha256((out/'source_public_manifest.json').read_bytes()).hexdigest(),
    source_roundtrip_error=roundtrip,initial_hanging_error=initial_error,
    coordinate_direction_error=directions_error,source_native_steps=0,target_native_steps=env.native_steps,
    reference_cart_max=float(np.max(abs(states[:,:2]))),maximum_force_norm=float(np.max(np.linalg.norm(actions,axis=1))),
    claim='qualified spatial policy reference used as material geometry only; no target physical episode or new qualification'))
write_json(out/'completed.json',dict(status='completed'))
env.close()
