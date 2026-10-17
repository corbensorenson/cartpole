"""Change only the physical tracking metric of the frozen moving quotient law."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import reflection_tracking_values
from spatialpole.world_coordinates import physical_tracking_factor
from spatialpole.experiment import freeze_inputs,write_json

p=argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--quotient-design',required=True)
p.add_argument('--design-controller',required=True)
p.add_argument('--out',required=True)
a=p.parse_args();out=Path(a.out);freeze_inputs(out,a)
for original,name in ((a.quotient_design,'input_quotient_design.npz'),(a.design_controller,'input_design_controller.npz')):
    shutil.copy2(original,out/name)
identities={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
    ('input_route.npz','input_quotient_design.npz','input_design_controller.npz')}
assert identities['input_route.npz']=='3a3f643c4b8af5371c52b6f224393ad2c9c98656b20313ef568ca86274801c17'
assert identities['input_quotient_design.npz']=='2f76a6eb3aa19d9ae164a2aeab071eece9a44075fbe11a59ccb1bb4e63983036'
assert identities['input_design_controller.npz']=='3d81c5c7da04e1c89e9b314e65bf1fef0782ef6f173664a6f2a7f80e8dc15166'
source=np.load(out/'input_route.npz');old=np.load(out/'input_design_controller.npz');q=np.load(out/'input_quotient_design.npz')
for key in old.files:
    if key in source.files:
        np.testing.assert_array_equal(old[key],source[key])
np.testing.assert_array_equal(q['tracking_R'],source['tracking_R'])
np.testing.assert_array_equal(q['reduced_gains']@q['F'][:-1],source['gains'])
np.testing.assert_array_equal(q['reduced_roots']@q['F'],source['roots'])
config=Config(**json.loads(str(source['config'])))
assert config==Config(n_links=14) and int(source['park_steps'])==960
assert bool(source['axial_quotient_planning']) and bool(source['online_gauge']) and not bool(source['native_search'])
assert len(source['actions'])==700 and int(source['swing_steps'])==450
env=SpatialCartpole(config)
L=np.array([physical_tracking_factor(env,s) for s in source['states']])
reduced=L@q['E']
signs=np.r_[np.ones(30),-np.ones(44)]
gram=np.transpose(reduced,(0,2,1))@reduced
cross=float(np.max(abs(gram[:,:30,30:])))
assert cross<1e-8 and np.isfinite(L).all()
models=list(zip(q['reduced_A'],q['reduced_B']))
feed,K,S,r=reflection_tracking_values(models,q['reduced_defects'],reduced[:-1],source['tracking_R'],reduced[-1],signs)
values={key:source[key] for key in source.files}
values.update(feed=feed,gains=K@q['F'][:-1],roots=np.array(S)@q['F'],residuals=r,
    physical_tracking_cost=True,physical_tracking_stage_factors=L)
for key in source.files:
    if key not in ('feed','gains','roots','residuals'):
        assert values[key].dtype==source[key].dtype and values[key].shape==source[key].shape
        assert values[key].tobytes()==source[key].tobytes()
np.savez_compressed(out/'route.npz',**values)
np.savez_compressed(out/'world_cost_design.npz',tracking_L=L,reduced_factors=reduced,
    reduced_A=q['reduced_A'],reduced_B=q['reduced_B'],reduced_defects=q['reduced_defects'],
    E=q['E'],F=q['F'],reduced_gains=K,reduced_roots=S,tracking_R=source['tracking_R'])
assert env.native_steps==0
env.close()
report=dict(n_links=14,design_native_steps=0,input_sha256=identities,
    only_tracking_cost_changed=True,other_controller_arrays_byte_exact=True,
    all_reduced_models_and_defects_exact=True,full_native_coordinates=88,physical_cost_coordinates=74,
    retained_velocity_coordinates=44,maximum_cost_cross_reflection_block=cross,
    maximum_gain=float(np.max(abs(values['gains']))),
    maximum_gain_change=float(np.max(abs(values['gains']-source['gains']))),
    controller_sha256=hashlib.sha256((out/'route.npz').read_bytes()).hexdigest(),
    scope='one fixed physical metric; no actual task or qualification evidence')
write_json(out/'summary.json',report);write_json(out/'completed.json',dict(status='completed'))
print(json.dumps(report),flush=True)
