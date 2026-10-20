"""Existing successful14 packets plus recovered617; no source-policy changes."""
from pathlib import Path
import numpy as np
from spatialpole import Config
from successful_profiles14 import Profile,ranking
from profile_selector14 import library
from audit_spatial_synthesis_models import load
from eleven_synthesis_protocol import read,sha


ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'output/fourteen_successful_profiles_library_inventory_001.json'
NEW=ROOT/'output/fourteen_delivered_feedback17_witness_inventory_001.json'


def profiles():
    old,m=library(OLD);p=read(NEW);run,audit=Path(p['run']),Path(p['audit'])
    assert p['passed'] and p['task_success'] and not p['qualified'] and p['seed']==20261417
    for path in (run,audit):
        for name,item in p['studies'][str(path)].items():assert sha(path/name)==item['sha256']
    assert sha(run/'actual.npz')==p['actual_sha256'] and sha(run/'summary.json')==p['summary_sha256']
    assert sha(audit/'summary.json')==p['audit_summary_sha256']
    source_inventory=Path(read(run/'inputs.json')['args']['inventory']);source=read(source_inventory);model_run=Path(source['run']);guide=model_run/'model'
    assert source['passed'] and source['full_noise_forecast_found']
    sm=read(model_run/'frozen_manifest.json');source_sha=read(run/'inputs.json')['source_sha256']
    assert source_sha==sm['source_sha256']==m['source_sha256']
    for name in ('spatial_synthesis_geometry.py','native_control_bellman14.py'):
        assert sm['helper_sha256'][name]==sha(Path(__file__).with_name(name))
    actual=load(run/'actual.npz');assert actual['states'].shape==(1501,102) and p['result']['physical_velocities']==44
    files=[NEW,run/'summary.json',run/'frozen_manifest.json',run/'inputs.json',run/'actual.npz',audit/'summary.json',
        source_inventory,model_run/'summary.json',model_run/'frozen_manifest.json',run/'native_queries.npz']
    files += [guide/name for name in ('virtual_reference.npz','native_models.npz','coordinates.npz','model_plan.npz')]
    meta=dict(seed=20261417,mode='native',switch=1500,guide=str(guide),after=None,inventory=str(NEW),run=str(run),audit=str(audit),
        actual=str(run/'actual.npz'),prefix_trace=str(run/'native_queries.npz'),future_trace=None,
        initial_state=actual['states'][0].tolist(),input_sha256={str(path.resolve()):sha(path) for path in files},source_sha256=source_sha,result=p['result'])
    combined=old+[meta];assert len(combined)==15 and len({initial_key(p) for p in combined})==15
    return combined,dict(old_library_inventory_sha256=sha(OLD),new_witness_inventory_sha256=sha(NEW),
        old_library_manifest=m,source_sha256=source_sha,profile_implementation_sha256=sha(Path(__file__).with_name('successful_profiles14.py')))


def initial_key(meta):
    value=np.array(meta['initial_state'],np.float64);assert value.shape==(102,) and np.isfinite(value).all()
    return value.tobytes()


def exact_index(observed,bank):
    assert observed.shape==(102,) and observed.dtype==np.float64 and np.isfinite(observed).all()
    matches=[i for i,p in enumerate(bank) if observed.tobytes()==initial_key(p)]
    assert len(matches)<=1
    return matches[0] if matches else None


def choose(geometry,observed,bank):
    assert geometry.config==Config(n_links=14)
    exact=exact_index(observed,bank)
    return exact,ranking(geometry,observed,bank) if exact is None else [dict(index=exact,seed=bank[exact]['seed'],initial_cost=0.)]


def identity(bank):
    result=[]
    for meta in bank:
        fields={}
        for prefix,path,names in (
            ('guide',meta['guide'],('virtual_reference.npz','native_models.npz','coordinates.npz','model_plan.npz')),
            ('after',meta['after'],('execution_reference.npz','execution_geometry.npz','execution_coordinates.npz','execution_plan.npz'))):
            if path is not None:
                for name in names:fields[prefix+'/'+name]=sha(Path(path)/name)
        import hashlib
        result.append(dict(seed=meta['seed'],mode=meta['mode'],switch=meta['switch'],
            initial_state_sha256=hashlib.sha256(initial_key(meta)).hexdigest(),packet_sha256=fields))
    return result
