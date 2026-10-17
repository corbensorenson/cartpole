"""Change only the fixed input derivative table; retain state models and recipe."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.symmetry import project_reflection_models
from spatialpole.quotient import quotient_tracking_values
from spatialpole.experiment import freeze_inputs,write_json

p=argparse.ArgumentParser();p.add_argument('--route',required=True);p.add_argument('--diagnostic',required=True)
p.add_argument('--quotient-design',required=True);p.add_argument('--out',required=True)
a=p.parse_args();out=Path(a.out);diagnostic=Path(a.diagnostic);freeze_inputs(out,a)
assert (diagnostic/'completed.json').is_file()
for name in ('input_models.npz','input_raw_native_models.npz','input_affine_controller.npz','summary.json'):
    shutil.copy2(diagnostic/name,out/('diagnostic_'+name))
shutil.copy2(a.quotient_design,out/'input_quotient_design.npz')
source=np.load(out/'input_route.npz');old=np.load(out/'diagnostic_input_affine_controller.npz')
native=np.load(out/'diagnostic_input_raw_native_models.npz');measure=np.load(out/'diagnostic_input_models.npz')
q=np.load(out/'input_quotient_design.npz');meta=json.loads((out/'diagnostic_summary.json').read_text())
for key in ('states','actions','config','tracking_R','reference','K','P','Q','R','defects'):
    np.testing.assert_array_equal(old[key],source[key])
for key in ('states','actions'):
    np.testing.assert_array_equal(measure[key],source[key])
assert hashlib.sha256((out/'input_route.npz').read_bytes()).hexdigest()==meta['inputs_sha256']['input_route.npz']
assert hashlib.sha256((out/'diagnostic_input_raw_native_models.npz').read_bytes()).hexdigest()==meta['inputs_sha256']['input_raw_native_models.npz']
config=Config(**json.loads(str(source['config'])))
assert config==Config(n_links=14) and bool(source['axial_quotient_planning']) and not bool(source['native_search'])
assert int(source['park_steps'])==960 and len(source['actions'])==700
spacing=1/8
matches=np.flatnonzero(measure['force_spacings_N']==spacing);assert len(matches)==1
B=measure['B'][int(matches[0])]
models,defects,projection=project_reflection_models(list(zip(native['A'],B)),native['defects'],native['reflection_signs'],native['state_scales'],80.)
signs=np.r_[np.ones(30),-np.ones(44)]
frames=[dict(E=E,F=F,signs=signs) for E,F in zip(q['E'],q['F'])]
L=q['tracking_L'];R=source['tracking_R'];np.testing.assert_array_equal(R,q['tracking_R'])
result=quotient_tracking_values(models,defects,[L]*700,R,L,frames)
np.testing.assert_array_equal(np.array([A for A,b in result['reduced_models']]),q['reduced_A'])
np.testing.assert_array_equal(result['reduced_defects'],q['reduced_defects'])
values={key:source[key] for key in source.files}
values.update({key:result[key] for key in ('feed','gains','roots','residuals')})
values.update(input_model_refined=True,input_derivative_spacing_N=spacing)
for key in source.files:
    if key not in ('feed','gains','roots','residuals'):
        np.testing.assert_array_equal(values[key],source[key])
np.savez_compressed(out/'route.npz',**values)
np.savez_compressed(out/'refined_design.npz',A=native['A'],raw_B=B,projected_B=np.array([b for A,b in models]),
    raw_defects=native['defects'],reduced_A=q['reduced_A'],
    reduced_B=np.array([b for A,b in result['reduced_models']]),reduced_defects=result['reduced_defects'],
    reduced_gains=result['reduced_gains'],reduced_roots=result['reduced_roots'])
write_json(out/'summary.json',dict(n_links=14,design_native_steps=0,force_spacing_N=spacing,
    state_models_and_defects_exact=True,all_other_controller_and_parking_arrays_exact=True,
    projection=projection,maximum_gain=float(np.max(abs(result['gains']))),
    maximum_gain_change=float(np.max(abs(result['gains']-source['gains']))),
    input_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
        ('input_route.npz','input_quotient_design.npz','diagnostic_input_models.npz','diagnostic_input_raw_native_models.npz')},
    controller_sha256=hashlib.sha256((out/'route.npz').read_bytes()).hexdigest(),
    claim='single input derivative spacing update; not actual task or qualification evidence'))
write_json(out/'completed.json',dict(status='completed'))
print(json.dumps(dict(design='complete',maximum_gain=float(np.max(abs(result['gains']))))),flush=True)
