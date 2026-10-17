"""Hash-bound selected 5Hz material reference; never import a foreign policy."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import xml.etree.ElementTree as ET
import mujoco as mj
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.transfer import regrid_planar_material_states,lift_planar_state
from spatialpole.experiment import freeze_inputs,write_json

p=argparse.ArgumentParser()
p.add_argument('--source-root',required=True)
p.add_argument('--evidence-archive',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
out,origin=Path(a.out),Path(a.source_root)
freeze_inputs(out,a);foreign=out/'foreign_sources';foreign.mkdir()
names=['synthesis/original-manifest.json','synthesis/export-manifest.json','synthesis/README.md',
    'synthesis/RECIPES.md','synthesis/ANSWERS.md','synthesis/data/16/probe07-controller.npz',
    'synthesis/data/16/orbit5.npz','benchmark/canonical16/frozen-manifest.json',
    'benchmark/canonical16/frozen/LICENSE']
identities={}
def record(name):
    raw=(foreign/name).read_bytes()
    identities[name]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
for name in names:
    target=foreign/name;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(origin/name,target);record(name)
selected=['files.json','metadata/identity.json','payload/configuration.json','payload/model.xml','payload/controller.npz']
archive=Path(a.evidence_archive)
with archive.open('rb') as f:
    archive_digest=hashlib.file_digest(f,'sha256').hexdigest()
found=set()
with tarfile.open(archive,'r|gz') as tar:
    for member in tar:
        relative=member.name.removeprefix('canonical16-evidence/')
        if member.isfile() and member.name=='canonical16-evidence/'+relative and relative in selected:
            assert relative not in found,'duplicate selected archive member'
            found.add(relative);target=foreign/'evidence'/relative;target.parent.mkdir(parents=True,exist_ok=True)
            target.write_bytes(tar.extractfile(member).read());record('evidence/'+relative)
assert found==set(selected),found
original=json.loads((foreign/names[0]).read_text());export=json.loads((foreign/names[1]).read_text())
for manifest in (original,export):
    rows={r['path']:r for r in manifest['files']}
    for name in ('data/16/probe07-controller.npz','data/16/orbit5.npz'):
        assert rows[name]['sha256']==identities['synthesis/'+name]['sha256']
        assert rows[name]['bytes']==identities['synthesis/'+name]['bytes']
frozen=json.loads((foreign/'benchmark/canonical16/frozen-manifest.json').read_text())
identity=json.loads((foreign/'evidence/metadata/identity.json').read_text())
configuration=json.loads((foreign/'evidence/payload/configuration.json').read_text())
assert identities['evidence/metadata/identity.json']['sha256']==frozen['original_identity_sha256']
assert identities['evidence/payload/model.xml']['sha256']==identity['native_xml_sha256']
assert identities['evidence/payload/configuration.json']['sha256']==identity['configuration']['sha256']
assert identities['evidence/payload/controller.npz']['sha256']==identity['controller']['sha256']==configuration['controller']['sha256']
assert identities['benchmark/canonical16/frozen/LICENSE']==frozen['files']['LICENSE']
donor='synthesis/data/16/probe07-controller.npz'
assert identities[donor]['sha256']=='d71eec45605803395859c780a965497dcd2fbf78fb8857f908012a4bb5522507'
assert identities[donor]['sha256']==configuration['design_origin_controller']['sha256']
assert identities[donor]['bytes']==configuration['design_origin_controller']['bytes']
reference=np.load(foreign/donor);released=np.load(foreign/'evidence/payload/controller.npz')
for key in ('states','actions','gains','offsets','native_defects'):
    prefix=reference[key][:651 if key=='states' else 650]
    assert prefix.dtype==released[key].dtype and prefix.shape==released[key].shape and prefix.tobytes()==released[key].tobytes(),key
states,commands=reference['states'],reference['actions']
assert states.shape==(701,34) and commands.shape==(700,) and np.isfinite(states).all()
assert np.array_equal(commands,commands.astype(np.float32).astype(float)) and np.max(abs(commands))<=1
model=mj.MjModel.from_xml_path(str(foreign/'evidence/payload/model.xml'))
assert model.nq==model.nv==17 and model.nu==1 and model.opt.timestep==.005
assert model.opt.integrator==mj.mjtIntegrator.mjINT_RK4 and model.neq==0
assert model.jnt_type[0]==mj.mjtJoint.mjJNT_SLIDE and np.all(model.jnt_type[1:]==mj.mjtJoint.mjJNT_HINGE)
np.testing.assert_array_equal(model.jnt_axis[0],[1.,0.,0.])
np.testing.assert_array_equal(model.jnt_axis[1:],np.tile([0.,1.,0.],(16,1)))
assert not np.any(model.jnt_limited[1:]) and model.actuator_gear[0,0]==1
xml=ET.parse(foreign/'evidence/payload/model.xml')
geoms=[xml.find(f'.//geom[@name="link_{i}_geom"]') for i in range(1,17)]
lengths=np.array([float(g.attrib['fromto'].split()[-1]) for g in geoms])
masses=np.array([float(g.attrib['mass']) for g in geoms])
assert abs(sum(lengths)-3)<1e-6 and abs(sum(masses)-1)<1e-8
data=mj.MjData(model);body_ids=[mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,f'link_{i}') for i in range(1,17)]
direction_error,defect_error,source_steps=0.,0.,0
for t,state in enumerate(states):
    mj.mj_resetData(model,data);data.qpos[:],data.qvel[:]=state[:17],state[17:];mj.mj_forward(model,data)
    angle=np.cumsum(state[1:17]);expected=np.c_[np.sin(angle),np.zeros(16),np.cos(angle)]
    direction_error=max(direction_error,float(np.max(abs(data.xmat[body_ids].reshape(-1,3,3)[:,:,2]-expected))))
    if t<700:
        data.ctrl[0]=float(commands[t])*80
        for _ in range(4):
            mj.mj_step(model,data);source_steps+=1
        actual=np.r_[data.qpos,data.qvel]-states[t+1]
        defect_error=max(defect_error,float(np.max(abs(actual-reference['native_defects'][t]))))
assert direction_error<=1e-10 and defect_error<=2e-6 and source_steps<=4000
config=Config(n_links=14);env=SpatialCartpole(config)
shifted=states.copy();translation=-float(states[0,0]);shifted[:,0]+=translation
target=regrid_planar_material_states(shifted,lengths,np.full(14,3/14))
lifted=np.array([lift_planar_state(env,s[:15],s[15:]) for s in target])
raw=np.c_[commands*80,np.zeros(700)];actions=np.array([env.deliver(u) for u in raw],float)
initial,_=env.reset(seed=20261016,options=dict(mode='hanging',angle_noise=0.))
initial_error=float(np.max(abs(env.difference(lifted[0],initial))))
target_direction_error=0.
for state,row in zip(lifted,target):
    env.set_state(state);angle=np.cumsum(row[1:15]);expected=np.c_[np.sin(angle),np.zeros(14),np.cos(angle)]
    target_direction_error=max(target_direction_error,float(np.max(abs(env.data.xmat[env.link_ids].reshape(-1,3,3)[:,:,2]-expected))))
assert initial_error<=1e-6 and target_direction_error<=1e-10 and env.native_steps==0
np.savez_compressed(out/'route.npz',states=lifted,actions=actions,swing_steps=450,config=json.dumps(asdict(config)))
np.savez_compressed(out/'physical_transfer.npz',source=states,translated_source=shifted,target=target,
    source_lengths=lengths,source_masses=masses,target_lengths=np.full(14,3/14))
write_json(out/'foreign_inputs.json',dict(source_root=str(origin),files=identities,
    archive=dict(path=str(archive),sha256=archive_digest,bytes=archive.stat().st_size),
    source_commit='45326b078c54fff1265f5dfa4d48408a74ca4a6c',
    credit='Marc / Rage Industries, MIT; selected 1D geometry/force proposal only',
    source_url='https://github.com/marctheshark3/pendulum-arena/tree/45326b078c54fff1265f5dfa4d48408a74ca4a6c/synthesis'))
write_json(out/'summary.json',dict(n_links=14,source_links=16,reference_steps=700,bridge_steps=450,
    source_native_steps=source_steps,target_native_steps=0,source_direction_error=direction_error,
    source_stored_defect_agreement=defect_error,released_prefix_bytes_exact=True,
    target_direction_error=target_direction_error,initial_hanging_error=initial_error,
    reference_cart_translation=translation,maximum_cart_reference=float(np.max(abs(lifted[:,:2]))),
    target_action_quantization_change=float(np.max(abs(raw-actions))),
    claim='new full-spatial material proposal; not source-policy reproduction or qualified target count'))
write_json(out/'completed.json',dict(status='completed'));env.close()
