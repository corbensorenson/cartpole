"""Hash-bound foreign physical material reference; no foreign policy is loaded."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.transfer import regrid_planar_material_states,lift_planar_state
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--source-root',required=True)
p.add_argument('--links',type=int,required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out,origin = Path(a.out),Path(a.source_root)
freeze_inputs(out,a)
foreign = out/'foreign_sources'
foreign.mkdir()
manifest_bytes = (origin/'frozen-manifest.json').read_bytes()
(foreign/'frozen-manifest.json').write_bytes(manifest_bytes)
manifest = json.loads(manifest_bytes)
reference_name = 'provenance/original/033-b7471fb925f4-reference.npz'
design_name = 'provenance/original/031-b37f09e5f8a0-native-value-design02.json'
names = ['metadata/identity.json','payload/controller.npz','payload/configuration.json',
         'payload/model.xml',reference_name,design_name,'LICENSE','README.md']
identities = {}
for name in names:
    target = foreign/name
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(origin/name,target)
    raw = target.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    pinned = manifest['files'].get(name)
    if pinned is not None:
        assert pinned['sha256']==digest and pinned['bytes']==len(raw),name
    identities[name] = dict(sha256=digest,bytes=len(raw),manifest_bound=pinned is not None)
identity = json.loads((foreign/'metadata/identity.json').read_bytes())
configuration = json.loads((foreign/'payload/configuration.json').read_bytes())
assert identities['payload/controller.npz']['sha256']==identity['controller']['sha256']==configuration['controller']['sha256']
assert identities['payload/model.xml']['sha256']==identity['native_xml_sha256']
assert identities['metadata/identity.json']['sha256']==manifest['original_identity']['sha256']
design = json.loads((foreign/design_name).read_bytes())
assert design['n_links']==17 and design['full_value_horizon_ticks']==700
reference,payload = np.load(foreign/reference_name),np.load(foreign/'payload/controller.npz')
for name in ('states','actions','gains','offsets','native_defects'):
    np.testing.assert_array_equal(reference[name],payload['value_'+name])
states,commands = reference['states'],reference['actions']
assert states.shape==(701,36) and commands.shape==(700,) and np.isfinite(states).all()
assert np.array_equal(commands,commands.astype(np.float32).astype(float)) and np.max(abs(commands))<=1.
model = mj.MjModel.from_xml_path(str(foreign/'payload/model.xml'))
assert model.nq==model.nv==18 and model.nu==1 and model.opt.timestep==.005
assert model.jnt_type[0]==mj.mjtJoint.mjJNT_SLIDE and np.all(model.jnt_type[1:]==mj.mjtJoint.mjJNT_HINGE)
np.testing.assert_array_equal(model.jnt_axis[0],[1.,0.,0.])
np.testing.assert_array_equal(model.jnt_axis[1:],np.repeat([[0.,1.,0.]],17,axis=0))
xml = ET.parse(foreign/'payload/model.xml')
geoms = [xml.find(f'.//geom[@name="link_{i}_geom"]') for i in range(1,18)]
lengths = np.array([float(g.attrib['fromto'].split()[-1]) for g in geoms])
masses = np.array([float(g.attrib['mass']) for g in geoms])
assert abs(np.sum(lengths)-3.)<1e-6 and abs(np.sum(masses)-1.)<1e-8
data,body_ids = mj.MjData(model),[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,f'link_{i}') for i in range(1,18)]
source_direction_error,source_defect_error,source_steps = 0.,0.,0
for t,state in enumerate(states):
    mj.mj_resetData(model,data)
    data.qpos[:],data.qvel[:] = state[:18],state[18:]
    mj.mj_forward(model,data)
    theta = np.cumsum(state[1:18])
    expected = np.c_[np.sin(theta),np.zeros(17),np.cos(theta)]
    source_direction_error = max(source_direction_error,float(np.max(abs(data.xmat[body_ids].reshape(-1,3,3)[:,:,2]-expected))))
    if t<len(commands):
        data.ctrl[0] = float(commands[t])*80.
        for _ in range(4):
            mj.mj_step(model,data)
            source_steps += 1
        defect = np.r_[data.qpos,data.qvel]-states[t+1]
        source_defect_error = max(source_defect_error,float(np.max(abs(defect-reference['native_defects'][t]))))
assert source_direction_error<=1e-10 and source_defect_error<=2e-6 and source_steps<=4000
config = Config(n_links=a.links)
shifted = states.copy()
translation = -float(states[0,0])
shifted[:,0] += translation
target = regrid_planar_material_states(shifted,lengths,np.full(a.links,3./a.links))
env = SpatialCartpole(config)
lifted = np.array([lift_planar_state(env,row[:a.links+1],row[a.links+1:]) for row in target])
raw_actions = np.c_[commands*80.,np.zeros(len(commands))]
actions = np.array([env.deliver(force) for force in raw_actions],float)
initial,_ = env.reset(seed=20261016,options={'mode':'hanging','angle_noise':0.})
initial_error = float(np.max(abs(env.difference(lifted[0],initial))))
assert initial_error<=1e-6
direction_error = 0.
for state,row in zip(lifted,target):
    env.set_state(state)
    theta = np.cumsum(row[1:a.links+1])
    expected = np.c_[np.sin(theta),np.zeros(a.links),np.cos(theta)]
    direction_error = max(direction_error,float(np.max(abs(env.data.xmat[env.link_ids].reshape(-1,3,3)[:,:,2]-expected))))
assert direction_error<=1e-10 and env.native_steps==0
np.savez_compressed(out/'route.npz',states=lifted,actions=actions,config=json.dumps(asdict(config)),swing_steps=450)
np.savez_compressed(out/'physical_transfer.npz',source=states,translated_source=shifted,target=target,
    source_lengths=lengths,source_masses=masses,target_lengths=np.full(a.links,3./a.links))
write_json(out/'foreign_inputs.json',dict(original=str(origin),files=identities,
    manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    credit='Marc / Rage Industries, MIT; external 1D geometric proposal only',
    source_commit='45326b078c54fff1265f5dfa4d48408a74ca4a6c',
    source_url='https://github.com/marctheshark3/pendulum-arena/tree/45326b078c54fff1265f5dfa4d48408a74ca4a6c/benchmark/canonical17/candidate02'))
write_json(out/'summary.json',dict(n_links=a.links,source_links=17,reference_steps=700,bridge_steps=450,
    source_native_steps=source_steps,target_native_steps=env.native_steps,
    source_coordinate_direction_error=source_direction_error,source_native_defect_error=source_defect_error,
    initial_hanging_error=initial_error,coordinate_direction_error=direction_error,
    reference_cart_translation=translation,maximum_cart_reference=float(np.max(abs(lifted[:,:2]))),
    target_action_quantization_change=float(np.max(abs(actions-raw_actions))),
    claim='foreign material moving-target proposal only; no target physical episode, source-policy reproduction or qualified count'))
write_json(out/'completed.json',dict(status='completed'))
env.close()
