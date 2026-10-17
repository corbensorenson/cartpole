"""Frozen axial-quotient redesign using an exactly bound native model cache."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import AffineValuePolicy
from spatialpole.quotient import axial_quotient_frame,quotient_tracking_values
from spatialpole.symmetry import clean_planar_reference,project_reflection_models
from spatialpole.experiment import freeze_inputs,write_json,episode

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--cache-run',required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out,cache = Path(a.out),Path(a.cache_run)
assert (cache/'completed.json').is_file(),'model-cache provider must be terminal'
assert hashlib.sha256(Path(a.route).read_bytes()).digest()==hashlib.sha256((cache/'affine_controller.npz').read_bytes()).digest()
freeze_inputs(out,a)
provenance = out/'cache_provenance'
provenance.mkdir()
identities={}
for name in ('raw_native_models.npz','input_route.npz','inputs.json','reference_cleanup.json','model_projection.json','driver.py','pyproject.toml','requirements-lock.txt'):
    shutil.copy2(cache/name,provenance/name)
    identities[name] = hashlib.sha256((provenance/name).read_bytes()).hexdigest()
source_inputs = json.loads((provenance/'inputs.json').read_text())
for name,expected in source_inputs['input_artifacts_sha256'].items():
    assert identities[name]==expected
(provenance/'source').mkdir()
for name,expected in source_inputs['source_sha256'].items():
    shutil.copy2(cache/'source'/name,provenance/'source'/name)
    actual = hashlib.sha256((provenance/'source'/name).read_bytes()).hexdigest()
    assert actual==expected
    identities['source/'+name] = actual
for name in ('env.py','control.py','symmetry.py'):
    assert hashlib.sha256((out/'source'/name).read_bytes()).hexdigest()==source_inputs['source_sha256'][name]
assert identities['input_route.npz']==source_inputs['input_artifacts_sha256']['input_route.npz']
assert source_inputs['args']['reflection_model'] and source_inputs['args']['planar_reference']
assert source_inputs['args']['tracking_metric']=='relative' and source_inputs['args']['terminal_scale']==0.
source = np.load(out/'input_route.npz')
raw_reference = np.load(provenance/'input_route.npz')
native = np.load(provenance/'raw_native_models.npz')
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=14)==Config(**json.loads(str(native['config'])))
assert len(source['actions'])==700 and bool(source['reflection_projected_model'])
env = SpatialCartpole(config)
actions = np.array([env.deliver(u) for u in raw_reference['actions']],float)
states,actions,cleanup = clean_planar_reference(env,raw_reference['states'],actions)
np.testing.assert_array_equal(states,source['states'])
np.testing.assert_array_equal(actions,source['actions'])
models,defects,projection = project_reflection_models(list(zip(native['A'],native['B'])),
    native['defects'],native['reflection_signs'],native['state_scales'],config.force_limit)
np.testing.assert_array_equal(defects,source['defects'])
frames = [axial_quotient_frame(env,x,planar=True) for x in states]
scales = native['state_scales']
gauge_error = max(float(np.max(abs(A@frames[t]['D']-frames[t+1]['D'])/scales[:,None]))
    for t,(A,B) in enumerate(models))
record = dict(cache_sha256=identities,reference_arrays_exact=True,cleanup=cleanup,projection=projection,
    full_state_coordinates=2*env.model.nv,planning_coordinates=frames[0]['E'].shape[1],
    planning_even=int(sum(frames[0]['signs']>0)),planning_odd=int(sum(frames[0]['signs']<0)),
    retained_velocity_coordinates=env.model.nv,max_normalized_gauge_flow_error=gauge_error,
    gauge_flow_tolerance=1e-5,scope='quotient planning only; full native joints/velocities/plant unchanged')
write_json(out/'cache_and_quotient_preflight.json',record)
assert gauge_error<=1e-5,record
nv = env.model.nv
weights = np.r_[[4.,4.],np.full(nv-2,40.),np.full(nv,.05)]
for j in range(config.n_links):
    weights[4+3*j],weights[nv+4+3*j] = .1,.005
L = np.diag(np.sqrt(weights))
R = source['tracking_R']
result = quotient_tracking_values(models,defects,[L]*len(actions),R,L,frames)
design = {k:source[k] for k in ('reference','K','P','Q','R')}
values = {k:source[k] for k in source.files}
values.update({k:result[k] for k in ('feed','gains','roots','residuals')})
values.update(axial_quotient_planning=True,quotient_coordinates=frames[0]['E'].shape[1],
              swing_steps=450,native_search=False)
np.savez_compressed(out/'affine_controller.npz',**values)
np.savez_compressed(out/'quotient_design.npz',**{k:np.array([f[k] for f in frames]) for k in ('C','D','G','E','F')},
    reduced_A=np.array([A for A,B in result['reduced_models']]),
    reduced_B=np.array([B for A,B in result['reduced_models']]),
    reduced_defects=result['reduced_defects'],reduced_gains=result['reduced_gains'],
    reduced_roots=result['reduced_roots'],tracking_L=L,tracking_R=R)
assert env.native_steps==0
env.close()
factory = lambda e: AffineValuePolicy(e,states,actions,result['feed'],result['gains'],result['roots'],result['residuals'],R,design)
rows=[]
for index,seed in enumerate([20261016]+list(range(20261116,20261128))):
    row = episode(config,factory,seed,'hanging',0. if index==0 else .02,
        out/('nominal.npz' if index==0 else f'seed{seed}.npz'))
    rows.append(row)
    print(json.dumps(dict(seed=seed,success=row['success'],hold=row['best_hold_seconds'],steps=row['steps'],replay_exact=row['replay_exact'])),flush=True)
    if index==0 and not (row['success'] and row['replay_exact']):
        break
inclusive = sum(r['native_steps']+r['proof_native_steps'] for r in rows)
assert inclusive<=180000
write_json(out/'summary.json',dict(n_links=config.n_links,results=rows,passes=sum(r['success'] and r['replay_exact'] for r in rows),
    design_native_steps=0,inclusive_native_steps=inclusive,max_feedback_gain=float(np.max(abs(result['gains']))),
    scope='reused development only; no reserved14 data'))
write_json(out/'completed.json',dict(status='completed'))
