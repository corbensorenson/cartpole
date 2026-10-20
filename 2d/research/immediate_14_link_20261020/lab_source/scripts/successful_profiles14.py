"""Complete existing immediate14 feedback profiles; no reference blending."""
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.native_mpc import exact_bytes
from spatialpole.world_coordinates import native_geometry,world_coordinate_error
from spatial_synthesis_geometry import CorrectorPolicy
from native_control_bellman14 import NativeControlBellman
from multiplicative_control_bellman14 import MultiplicativeControlBellman
from audit_spatial_synthesis_models import load
from eleven_synthesis_protocol import read,sha
from prepare_spatial_motion_blend14 import SCALE
from witness_multiplicative_critic14 import inputs as inputs25
from witness_repaired_uncertain27 import inputs as inputs27


COORDINATES=('S','K','feed','value_residual','Cinv')
SPECS=[(20261016,'fourteen_uncertain_refresh_v10_nominal_success_inventory_001.json')]
SPECS += [(20261100+i,f'fourteen_uncertain_refresh_v10_seed{i}_success_inventory_001.json') for i in range(16,22)]
SPECS += [(20261122,'fourteen_six_source_motion22_success_inventory_001.json'),
    (20261123,'fourteen_uncertain_refresh_v9_seed23_success_inventory_001.json'),
    (20261124,'fourteen_native_control_jacobian24_success_inventory_001.json'),
    (20261125,'fourteen_multiplicative_critic25_witness_inventory_001.json'),
    (20261126,'fourteen_uncertain_refresh_v7_seed26_success_inventory_001.json'),
    (20261127,'fourteen_uncertain_refresh_v8_saved27_witness_inventory_001.json'),
    (20261416,'fourteen_uncertain_refresh_v10_additional16_success_inventory_001.json')]


def packet(folder,execution=False):
    folder=Path(folder)
    if execution:
        names=('execution_reference.npz','execution_geometry.npz','execution_coordinates.npz','execution_plan.npz')
    else:names=('virtual_reference.npz','native_models.npz','coordinates.npz','model_plan.npz')
    v=load(folder/names[0])
    with np.load(folder/names[1]) as z:F=z['F']
    with np.load(folder/names[2]) as z:c={k:z[k] for k in COORDINATES}
    if execution:
        with np.load(folder/names[2]) as z:c['uncertainty_weight']=z['uncertainty_weight']
    with np.load(folder/names[3]) as z:w=z['w']
    assert v['states'].shape==(1501,102) and F.shape==(1501,74,88) and not np.any(w)
    return v,dict(F=F),c,w


def source(seed,name):
    inventory=(Path('output')/name).resolve();parent=read(inventory)
    assert parent['passed'] and parent['task_success'] and not parent['qualified']
    run,audit=Path(parent['run']),Path(parent['audit'])
    assert (run/'completed.json').exists() and (audit/'completed.json').exists()
    assert sha(run/'summary.json')==parent['summary_sha256'] and sha(audit/'summary.json')==parent['audit_summary_sha256']
    s,m,g=read(run/'summary.json'),read(run/'frozen_manifest.json'),read(audit/'summary.json')
    assert s['task_success'] and g['passed'] and m['config']==Config(n_links=14).__dict__
    tape=load(run/'actual.npz')
    assert sha(run/'actual.npz')==parent['actual_sha256'] and len(tape['actions'])==1500
    assert int(m['seed'])==seed
    mode='fixed' if seed in [20261016]+list(range(20261116,20261123)) else 'native'
    after=None;switch=1500;prefix_trace=None;future_trace=None;extra=[]
    if seed==20261122:
        planning=run/'planning';steps=read(planning/'summary.json')['results'];guide=planning/f"iteration{steps[-1]['iteration']}"
    elif seed==20261124:guide=Path(m['source_guide'])
    elif seed==20261125:
        ci=Path(read(run/'inputs.json')['args']['component_inventory']);after,old,guide,_=inputs25(ci)
        switch=m['handoff_phase'];extra += [ci,after/'summary.json',after/'frozen_manifest.json']
    elif seed==20261127:
        ci=Path(read(run/'inputs.json')['args']['component_inventory']);_,forecast,after,stage,stage_manifest,old=inputs27(ci)
        guide=Path(stage_manifest['original_guide']);switch=s['event']['phase']
        extra += [ci,forecast/'summary.json',stage/'frozen_manifest.json']
    else:
        guide=Path(s['guide'])
        if s.get('event') is not None:
            switch=s['event']['phase'];after=Path(s['refresh_report']['forecast_folder'])
    if mode=='native':
        prefix_trace=run/('native_queries.npz' if seed==20261124 else 'prefix_queries.npz')
        assert prefix_trace.exists()
        if after is not None:future_trace=run/'future_queries.npz';assert future_trace.exists()
    files=[inventory,run/'summary.json',run/'frozen_manifest.json',run/'inputs.json',audit/'summary.json',run/'actual.npz']+extra
    files += [guide/name for name in ('virtual_reference.npz','native_models.npz','coordinates.npz','model_plan.npz')]
    if after is not None:
        files += [after/name for name in ('execution_reference.npz','execution_geometry.npz','execution_coordinates.npz','execution_plan.npz')]
    files += [path for path in (prefix_trace,future_trace) if path is not None]
    for key,filename in (('prefix_queries_sha256','prefix_queries.npz'),('future_queries_sha256','future_queries.npz'),('native_query_trace_sha256','native_queries.npz')):
        if key in parent:assert sha(run/filename)==parent[key]
    kernels=('spatial_synthesis_geometry.py',)+(() if mode=='fixed' else ('native_control_bellman14.py',))
    if after is not None:kernels+=('multiplicative_control_bellman14.py',)
    for name in kernels:
        assert m['helper_sha256'][name]==sha(Path(__file__).with_name(name))
    result=s.get('result',parent.get('result'))
    assert result['success'] and result['steps']==1500 and result['replay_exact']
    return dict(seed=seed,mode=mode,switch=switch,guide=str(guide),after=None if after is None else str(after),
        inventory=str(inventory),run=str(run),audit=str(audit),actual=str(run/'actual.npz'),
        prefix_trace=None if prefix_trace is None else str(prefix_trace),future_trace=None if future_trace is None else str(future_trace),
        initial_state=tape['states'][0].tolist(),input_sha256={str(p.resolve()):sha(p) for p in files},
        source_sha256=m['source_sha256'],result=result)


class Profile:
    def __init__(self,config,meta):
        assert config==Config(n_links=14)
        self.meta=meta;self.geometry=SpatialCartpole(config)
        before=packet(meta['guide'])
        self.before=(CorrectorPolicy(self.geometry,before[0],before[1]['F'],before[2],before[3]) if meta['mode']=='fixed'
            else NativeControlBellman(config,*before))
        self.after=None if meta['after'] is None else MultiplicativeControlBellman(config,*packet(meta['after'],True))
    def __call__(self,state,phase):
        return (self.before if phase<self.meta['switch'] else self.after)(state,phase)
    def chosen(self,phase):return self.before if phase<self.meta['switch'] else self.after
    @property
    def predictor_native_steps(self):
        return sum(p.predictor.native_steps for p in (self.before,self.after) if p is not None and hasattr(p,'predictor'))
    def close(self):
        self.before.close()
        if self.after is not None:self.after.close()
        assert self.geometry.native_steps==0;self.geometry.close()


def ranking(geometry,observed,profiles):
    nominal=np.array(next(p for p in profiles if p['seed']==20261016)['initial_state'])
    R,w=native_geometry(geometry,nominal)
    target=world_coordinate_error(geometry,observed,nominal,R,w)/SCALE
    rows=[]
    for i,p in enumerate(profiles):
        x=np.array(p['initial_state'])
        cost=0. if np.array_equal(x,observed) else float(np.sum((world_coordinate_error(geometry,x,nominal,R,w)/SCALE-target)**2))
        rows.append(dict(index=i,seed=p['seed'],initial_cost=cost))
    return sorted(rows,key=lambda r:(r['initial_cost'],r['seed']))
