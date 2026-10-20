"""State-only six-source guide preparation and measured native-control feedback."""
from pathlib import Path
import hashlib
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import write_json
from spatial_motion_library14 import reference
from spatial_source_bank14 import refine_prepared
from spatial_synthesis_geometry import CorrectorPolicy
from native_control_bellman14 import NativeControlBellman
from stage_six_source_synthesis14 import NUMERICAL_HELPERS
from eleven_synthesis_protocol import read,sha

PROTOCOL='six-source-measured-control-native14-v6'
CORE=tuple(sorted(set(NUMERICAL_HELPERS)|{'native_control_synthesis14.py','native_control_bellman14.py','evaluate_native_control14.py'}))
IDENTITY_KEYS=('protocol','config','algorithm_sha256','source_sha256','template_sha256','solver_receipt_sha256',
               'max_initial_models','max_actor_native_steps','max_audit_native_steps','rule')


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def bundle(folder):
    folder=Path(folder)
    v,m,c,p=[load(folder/name) for name in ('virtual_reference.npz','native_models.npz','coordinates.npz','model_plan.npz')]
    return v,m,c,p['w']


def prepare(config,initial,templates,runtime,out,guide_cache=()):
    assert config==Config(n_links=14)
    out=Path(out);geometry=SpatialCartpole(config)
    try:v,info=reference(geometry,initial,templates);assert geometry.native_steps==0
    finally:geometry.close()
    np.savez_compressed(out/'accepted_initial_proposal.npz',**v);write_json(out/'preparation.json',info)
    if not info['geometry_passed']:return None,dict(mode='geometry_rejected',native_steps=0,guide=None)
    keys=[hashlib.sha256(array.tobytes()).hexdigest() for array in (initial,v['states'],v['actions'])]
    for cached in guide_cache:
        if keys!=cached['exact_input_sha256']:continue
        for filename,digest in cached['proof_files_sha256'].items():assert sha(Path(filename))==digest
        package=Path(__import__('spatialpole').__file__).parent
        assert {path.name:sha(path) for path in package.glob('*.py')}==cached['source_sha256']
        proof=read(Path(cached['audit_summary']));assert proof['passed'] and proof['constructor_byte_exact'] and proof['all_forecasts_cold_byte_exact']
        for name,digest in cached['numeric_helpers_sha256'].items():assert sha(Path(__file__).with_name(name))==digest
        assert sha(Path(runtime)/'receipt.json')==cached['solver_receipt_sha256']
        root=Path(cached['primary']);ps=read(root/'summary.json');rows=ps['results'];assert len(rows)<=4
        folder=root/f"iteration{rows[-1]['iteration']}"
        mode=('fixed_feedback' if ps['plan_found'] else 'model_rejected' if rows[-1].get('model_rejected',False)
              else 'nonzero_residual_rejected' if np.any(load(folder/'model_plan.npz')['w']) else 'native_control')
        return (folder if mode in ('fixed_feedback','native_control') else None),dict(mode=mode,native_steps=0,guide=str(folder),
            inherited_native_steps=ps['inclusive_native_steps'],cache_used=True,cache_inventory=cached['inventory'])
    plan=refine_prepared(config,initial,v,runtime,out/'primary');ps=read(out/'primary/summary.json');rows=ps['results']
    assert 1<=len(rows)<=4 and ps['inclusive_native_steps']<=8766320
    folder=out/'primary'/f"iteration{rows[-1]['iteration']}"
    if plan is not None:mode='fixed_feedback'
    elif rows[-1].get('model_rejected',False):mode='model_rejected'
    elif np.any(load(folder/'model_plan.npz')['w']):mode='nonzero_residual_rejected'
    else:mode='native_control'
    return (folder if mode in ('fixed_feedback','native_control') else None),dict(mode=mode,native_steps=ps['inclusive_native_steps'],guide=str(folder))


def controller(config,folder,mode):
    v,m,c,w=bundle(folder)
    if mode=='fixed_feedback':
        owner=SpatialCartpole(config)
        try:return CorrectorPolicy(owner,v,m['F'],c,w)
        finally:owner.close()
    assert mode=='native_control'
    return NativeControlBellman(config,v,m,c,w)
