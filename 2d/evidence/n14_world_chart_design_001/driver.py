"""Transport a frozen quotient design into physical directional/angular features."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.world_coordinates import native_geometry,world_coordinate_jacobian,world_coordinate_error
from spatialpole.experiment import freeze_inputs,write_json

p=argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--quotient-design',required=True)
p.add_argument('--design-controller',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
out=Path(a.out);freeze_inputs(out,a)
for original,name in ((a.quotient_design,'input_quotient_design.npz'),(a.design_controller,'input_design_controller.npz')):
    shutil.copy2(original,out/name)
source=np.load(out/'input_route.npz');old=np.load(out/'input_design_controller.npz');q=np.load(out/'input_quotient_design.npz')
for key in old.files:
    if key in source.files:
        np.testing.assert_array_equal(old[key],source[key])
config=Config(**json.loads(str(source['config'])))
assert config==Config(n_links=14) and bool(source['axial_quotient_planning']) and not bool(source['native_search'])
assert int(source['park_steps'])==960 and len(source['actions'])==700
env=SpatialCartpole(config);start=time.perf_counter()
rotations,omegas,jacobians=[],[],[]
for state in source['states']:
    R,w=native_geometry(env,state)
    rotations.append(R);omegas.append(w);jacobians.append(world_coordinate_jacobian(env,state))
J=np.array(jacobians);T=J@q['E'];inverse=np.linalg.inv(T)
condition=np.linalg.cond(T)
identity_error=float(np.max(abs(T@inverse-np.eye(74)[None,:,:])))
gauge_error=float(np.max(abs(J@q['D'])))
feature_dual_error=float(np.max(abs(inverse@J-q['F'])))
assert np.isfinite(inverse).all() and identity_error<1e-9 and gauge_error<1e-9
K=q['reduced_gains']@inverse[:-1]
S=q['reduced_roots']@inverse
L=(q['tracking_L'][None,:,:]@q['E'])@inverse
Aw=(T[1:]@q['reduced_A'])@inverse[:-1]
Bw=T[1:]@q['reduced_B']
dw=np.einsum('tij,tj->ti',T[1:],q['reduced_defects'])
linear_gain_error=K@J[:-1]-source['gains']
relative_gain_error=float(np.max(abs(linear_gain_error))/max(1.,np.max(abs(source['gains']))))
assert relative_gain_error<1e-9 and np.isfinite(K).all() and np.isfinite(S).all()
values={key:source[key] for key in source.files}
values.update(world_coordinate_feedback=True,world_gains=K,world_reference_rotations=np.array(rotations),
    world_reference_omega=np.array(omegas),original_roots_provenance_only=True)
np.savez_compressed(out/'route.npz',**values)
np.savez_compressed(out/'world_coordinate_design.npz',J=J,T=T,inverse_T=inverse,
    world_A=Aw,world_B=Bw,world_defects=dw,world_gains=K,world_value_roots=S,
    world_stage_factors=L,world_residuals=source['residuals'],world_feed=source['feed'],R=source['tracking_R'])
assert env.native_steps==0
report=dict(n_links=14,native_integration_steps=0,reference_geometry_forward_queries=701,
    body_angular_velocity_queries=701*14,full_native_coordinates=88,physical_coordinates=74,
    all_velocity_coordinates_retained=44,maximum_coordinate_condition=float(max(condition)),
    maximum_inverse_identity_error=identity_error,maximum_gauge_jacobian_error=gauge_error,
    maximum_feature_dual_error=feature_dual_error,maximum_linear_gain_error=float(np.max(abs(linear_gain_error))),
    relative_linear_gain_error=relative_gain_error,wall_seconds=time.perf_counter()-start,
    controller_sha256=hashlib.sha256((out/'route.npz').read_bytes()).hexdigest(),
    input_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
        ('input_route.npz','input_design_controller.npz','input_quotient_design.npz')},
    scope='same local linear design; changed higher-order physical feature feedback, not nonlinear equivalence')
write_json(out/'summary.json',report);write_json(out/'completed.json',dict(status='completed'))
env.close();print(json.dumps(report),flush=True)
