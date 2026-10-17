"""Three fixed dyadic native input stencils at the exact incumbent reference."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.control import difference_jacobian
from spatialpole.input_derivatives import input_jacobians
from spatialpole.experiment import freeze_inputs,write_json

p=argparse.ArgumentParser();p.add_argument('--route',required=True);p.add_argument('--cache-run',required=True);p.add_argument('--out',required=True)
a=p.parse_args();out=Path(a.out);cache=Path(a.cache_run);freeze_inputs(out,a)
assert (cache/'completed.json').is_file()
for name in ('raw_native_models.npz','affine_controller.npz','inputs.json'):
    shutil.copy2(cache/name,out/('input_'+name))
source=np.load(out/'input_route.npz');old=np.load(out/'input_affine_controller.npz');native=np.load(out/'input_raw_native_models.npz')
for key in ('states','actions','config','tracking_R','reference','K','P','Q','R','defects'):
    # The incumbent QUOTIENT gains/roots intentionally differ from the full
    # value provider. Cache identity concerns its unchanged model centers and
    # costs/balance design; use incumbent gains only in the impact diagnostic.
    np.testing.assert_array_equal(old[key],source[key])
inputs=json.loads((out/'input_inputs.json').read_text())
for name in ('env.py','control.py','symmetry.py'):
    assert hashlib.sha256((out/'source'/name).read_bytes()).hexdigest()==inputs['source_sha256'][name]
config=Config(**json.loads(str(source['config'])))
assert config==Config(n_links=14) and len(source['actions'])==700 and int(source['park_steps'])==960
assert bool(source['axial_quotient_planning']) and not bool(source['native_search'])
env=SpatialCartpole(config);env.derivative_workers=4
states,actions=source['states'],source['actions']
spacings=(1/128,1/32,1/8)
models,bases,probes_inside=[],[],[]
for h in spacings:
    proposals=[]
    for force in actions:
        for axis in range(2):
            for sign in (1.,-1.,2.,-2.):
                u=force.copy();u[axis]+=sign*h
                assert u[axis]-force[axis]==sign*h,'probe increment not exactly represented'
                proposals.append(u)
    maximum=float(np.max(np.linalg.norm(proposals,axis=1)))
    assert maximum<=80.
    raw_B,predicted=input_jacobians(env,states[:-1],actions,eps=h)
    defects=np.array([env.difference(f,x) for f,x in zip(predicted,states[1:])])
    np.testing.assert_array_equal(defects,native['defects'])
    B=np.array([difference_jacobian(env,f,x)@b for f,x,b in zip(predicted,states[1:],raw_B)])
    models.append(B);bases.append(predicted);probes_inside.append(maximum)
    print(json.dumps(dict(force_spacing_N=h,native_steps=env.native_steps,base_defects_exact=True)),flush=True)
assert env.native_steps==75600
scales=native['state_scales'];old_B=native['B'];gain=source['gains'][1:]
summaries=[]
for h,B in zip(spacings,models):
    delta=B-old_B
    force_impact=gain@delta[:-1]
    summaries.append(dict(force_spacing_N=h,max_scaled_B_difference=float(np.max(abs(delta)*80/scales[None,:,None])),
        maximum_next_feedback_input_operator_difference=float(np.max(abs(force_impact))),
        maximum_executed_prefix_operator_difference=float(np.max(abs(force_impact[:540])))))
convergence=[]
for i,j in ((0,1),(1,2),(0,2)):
    delta=models[i]-models[j]
    convergence.append(dict(spacings_N=[spacings[i],spacings[j]],
        max_scaled_B_difference=float(np.max(abs(delta)*80/scales[None,:,None])),
        maximum_next_feedback_operator_difference=float(np.max(abs(gain@delta[:-1])))))
np.savez_compressed(out/'input_models.npz',force_spacings_N=spacings,B=np.array(models),predicted=np.array(bases),
    old_B=old_B,states=states,actions=actions,scales=scales,config=str(source['config']))
write_json(out/'summary.json',dict(n_links=14,native_steps=env.native_steps,state_models_unchanged=True,
    all_base_defects_exact=True,maximum_probe_force_norms=probes_inside,resolutions=summaries,convergence=convergence,
    inputs_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
        ('input_route.npz','input_raw_native_models.npz','input_affine_controller.npz','input_inputs.json')},
    claim='fixed-spacing native input sensitivity diagnostic; not new controller or derivative truth certificate'))
write_json(out/'completed.json',dict(status='completed'));env.close()
print(json.dumps(dict(resolutions=summaries,convergence=convergence)),flush=True)
