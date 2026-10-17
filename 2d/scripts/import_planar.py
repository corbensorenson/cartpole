"""Frozen planar X feedback + newly synthesized native spatial Y feedback.

This is an explicit transferred-policy experiment, never a physics constraint.
All final success is measured in the new unrestricted ball-joint environment.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import re
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.control import balance_design,RoutePolicy
from spatialpole.transfer import lift_planar_state,decode_planar_coordinates,declared_coordinate_projection,spatial_tracking_gains
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--links',type=int,default=3)
p.add_argument('--out',required=True)
p.add_argument('--route',required=True)
p.add_argument('--full-source',required=True)
p.add_argument('--spec',required=True)
p.add_argument('--source-config',required=True)
p.add_argument('--reference-only',action='store_true')
p.add_argument('--balance-backend',choices=['dare','finite_qr'],default='dare')
p.add_argument('--eigenvalue-digits',type=int,default=0)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
start = time.perf_counter()
foreign = out/'foreign_sources'
foreign.mkdir()
metadata = {}
for name in ('route','full_source','spec','source_config'):
    source = Path(getattr(a,name))
    dest = foreign/source.name
    shutil.copy2(source,dest)
    metadata[name] = {'path':str(source),'copy':str(dest.relative_to(out)),
                      'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
write_json(out/'foreign_inputs.json',metadata)
route = json.loads(Path(a.route).read_text())
full = json.loads(Path(a.full_source).read_text())
coords = np.array(route['search']['nominal_coordinate_states'])
spec_text = Path(a.spec).read_text()
def scale(name):
    match = re.search(r'^\s*'+name+r':\s*([\d.eE+-]+)\s*$',spec_text,re.MULTILINE)
    if match is None or float(match[1]) <= 0:
        raise ValueError('missing positive source scale: '+name)
    return float(match[1])
scales = dict(cart_position=scale('cart_position_abs_max'),angle=scale('absolute_link_angle_abs_max'),
              cart_velocity=scale('cart_velocity_abs_max'),joint_velocity=scale('hinge_velocity_rms_max'))
n = a.links
config = Config(n_links=n)
env = SpatialCartpole(config)
if 'coordinate_transform' not in route['controller']:
    raise ValueError('donor must explicitly declare its coordinate_transform')
source_transform = np.array(route['controller']['coordinate_transform'],float)
physical = decode_planar_coordinates(coords,source_transform,n)
reference = []
for row in physical:
    reference.append(lift_planar_state(env,row[:n+1],row[n+1:]))
reference = np.array(reference)
initial,_ = env.reset(seed=20261016,options={'mode':'hanging','angle_noise':0.})
initial_error = float(np.max(np.abs(env.difference(reference[0],initial))))
if initial_error>1e-6:
    raise ValueError('decoded donor does not start at the declared exact hanging state')
check = np.array([lift_planar_state(env,row['qpos'],row['qvel']) for row in full['result']['trajectory'][:len(reference)-1]])
max_error = max(np.max(np.abs(env.difference(x,y))) for x,y in zip(reference[1:],check))
max_direction_error = 0.
for state,row in zip(reference,physical):
    env.set_state(state)
    directions = env.data.xmat[env.link_ids].reshape(-1,3,3)[:,:,2]
    theta = np.cumsum(row[1:n+1])
    desired = np.c_[np.sin(theta),np.zeros(n),np.cos(theta)]
    max_direction_error = max(max_direction_error,float(np.max(np.abs(directions-desired))))
if max_direction_error>1e-10:
    raise ValueError('decoded physical world directions disagree with native lifted states')
if a.reference_only:
    parent_coords = np.asarray(full['search']['nominal_coordinate_states'])
    if not np.array_equal(coords,parent_coords):
        raise ValueError('historical nominal arrays differ between source and route')
elif max_error > 2e-6:
    raise ValueError(f'foreign coordinate reconstruction disagrees with stored physical tape: {max_error}')
actions = np.c_[np.array(route['controller']['controls'])*80.,np.zeros(len(reference)-1)]
design = balance_design(env,axis_separated=a.balance_backend=='finite_qr',value_backend=a.balance_backend,
                        eigenvalue_digits=a.eigenvalue_digits)
gains = spatial_tracking_gains(env,reference,actions,design)
# Old runtime uses + feedback_gains @ dimensionless error. New runtime uses
# - gains @ tangent error, and force is N rather than normalized action.
old_gains = np.array(route['controller']['feedback_gains'])
gains[:,0,:] = -80.*old_gains@declared_coordinate_projection(env,source_transform)
np.savez_compressed(out/'route.npz',states=reference,actions=actions,gains=gains,
                    **design,config=json.dumps(asdict(config)),world_balance=True)
factory = lambda e: RoutePolicy(e,reference,actions,gains,design,True)
rows = [episode(config,factory,20261016,'hanging',0.,out/'nominal.npz')]
print('nominal: '+json.dumps(rows[0]),flush=True)
if rows[0]['success']:
    for k in range(12):
        rows.append(episode(config,factory,20261116+k,'hanging',0.02,out/f'validation_{k}.npz'))
write_json(out/'summary.json',{'n_links':n,'passes':sum(x['success'] and x['replay_exact'] for x in rows),
    'results':rows,'design_native_steps':env.native_steps,'foreign_reconstruction_error':max_error,
    'source_scales':scales,
    'source_decoding':'inverse of explicit controller.coordinate_transform; scale spec retained as provenance only',
    'source_coordinate_transform_sha256':hashlib.sha256(source_transform.tobytes()).hexdigest(),
    'initial_hanging_error':initial_error,
    'reference_only':a.reference_only,
    'coordinate_direction_error':max_direction_error,
    'source_plant_difference':'old total damping 0.015 vs new 0.0075; old slide armature 0.0005 vs new 0; ball joints and XY cart free',
    'wall_seconds':time.perf_counter()-start})
write_json(out/'completed.json',{'status':'completed','wall_seconds':time.perf_counter()-start})
