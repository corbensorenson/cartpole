"""Matched full-native comparison of spatial affine-value controller variants."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.control import balance_design,linearize_many,difference_jacobian
from spatialpole.affine_value import affine_tracking_values,AffineValuePolicy
from spatialpole.tracking_repair import positive_value_factor
from spatialpole.experiment import freeze_inputs,write_json,episode

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
p.add_argument('--hybrid-document',required=True)
p.add_argument('--extend-to-episode',action='store_true')
p.add_argument('--design-guard',type=int,default=0)
p.add_argument('--terminal-scale',type=float,default=10.)
p.add_argument('--arms',nargs='+',choices=['zero_offset','affine','affine_search'],default=['zero_offset','affine','affine_search'])
p.add_argument('--balance-backend',choices=['dare','finite_qr'],default='dare')
p.add_argument('--planar-reference',action='store_true',help='remove only sub-tolerance off-plane proposal roundoff')
p.add_argument('--tracking-metric',choices=['relative','directions'],default='relative')
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
shutil.copy2(a.hybrid_document,out/'corben-marc-hybrid.md')
write_json(out/'document_identity.json',dict(path=a.hybrid_document,
    sha256=hashlib.sha256((out/'corben-marc-hybrid.md').read_bytes()).hexdigest()))
source = np.load(a.route)
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=config.n_links)
env = SpatialCartpole(config)
env.derivative_workers = 4
states = source['states'].copy()
actions = np.array([env.deliver(u) for u in source['actions']],float)
if a.planar_reference:
    from spatialpole.symmetry import clean_planar_reference
    states,actions,record = clean_planar_reference(env,states,actions)
    write_json(out/'reference_cleanup.json',record)
models,predicted = linearize_many(env,states[:-1],actions,eps=.001,order=4,return_next=True)
if a.extend_to_episode:
    count = config.horizon+a.design_guard-len(actions)
    assert count>0
    # A declared exact stationary upright destination; the single bridge
    # mismatch is retained as an affine defect. All identical tail linearizations
    # share exactly the same native state/action center, with no stale cache.
    upright = env.upright_state()
    tail_model,tail_prediction = linearize_many(env,upright[None,:],np.zeros((1,2)),
        eps=.001,order=4,return_next=True)
    models += [tail_model[0]]*(count-1)
    bridge_model,bridge_prediction = linearize_many(env,states[-1:],np.zeros((1,2)),
        eps=.001,order=4,return_next=True)
    models.insert(len(source['actions']),bridge_model[0])
    predicted = np.vstack((predicted,bridge_prediction,np.repeat(tail_prediction,count-1,axis=0)))
    states = np.vstack((states,np.repeat(upright[None,:],count,axis=0)))
    actions = np.vstack((actions,np.zeros((count,2))))
defects = np.array([env.difference(f,x) for f,x in zip(predicted,states[1:])])
# Native derivatives are based at F; transport them into the next reference's
# logarithmic error chart. This is not the virtual-gap forward map transport.
models = [(difference_jacobian(env,f,x)@A,difference_jacobian(env,f,x)@B)
          for (A,B),f,x in zip(models,predicted,states[1:])]
design = balance_design(env,axis_separated=True,value_backend=a.balance_backend)
nv,nx = env.model.nv,2*env.model.nv
Q = np.diag(np.r_[[4.,4.],np.full(nv-2,40.),np.full(nv,.05)])
if a.tracking_metric=='relative':
    for j in range(config.n_links):
        Q[4+3*j,4+3*j] = .1
        Q[nv+4+3*j,nv+4+3*j] = .005
    L = np.sqrt(Q)
    factors,terminal_local = [L]*len(actions),L
else:
    from spatialpole.physical import direction_tracking_factors
    if a.extend_to_episode:
        original_horizon = len(source['actions'])
        assert np.array_equal(states[original_horizon+1:],np.repeat(states[-1:],len(states)-original_horizon-1,axis=0))
        # Only identical upright centers share a factor; all swing and bridge
        # centers get their own fresh physical kinematic Jacobian.
        queried = np.vstack((states[:original_horizon+1],states[-1:]))
        physical_factors = direction_tracking_factors(env,queried,np.diag(Q))
        terminal_local = physical_factors[-1]
        factors = list(physical_factors[:-1])+[terminal_local]*(len(actions)-original_horizon-1)
    else:
        physical_factors = direction_tracking_factors(env,states,np.diag(Q))
        factors,terminal_local = list(physical_factors[:-1]),physical_factors[-1]
R = np.eye(2)*.001
root = np.sqrt(a.terminal_scale)*design['value_factor'] if 'value_factor' in design else positive_value_factor(a.terminal_scale*design['P'])[0]
feed,gains,roots,residuals = affine_tracking_values(models,defects,factors,R,np.vstack((root,terminal_local)))
if a.planar_reference:
    from spatialpole.symmetry import reflection_signs
    parity = reflection_signs(env)
    write_json(out/'reflection_gain_diagnostic.json',dict(
        max_transverse_gain_from_in_plane_state=float(np.max(np.abs(gains[:,1,parity>0]))),
        max_in_plane_gain_from_transverse_state=float(np.max(np.abs(gains[:,0,parity<0]))),
        max_transverse_feed=float(np.max(np.abs(feed[:,1]))),
        scope='fresh full native models; no derivative symmetry projection or physical joint constraint'))
np.savez_compressed(out/'affine_controller.npz',states=states,actions=actions,feed=feed,
    gains=gains,roots=roots,residuals=residuals,tracking_R=R,defects=defects,
    **design,world_balance=True,affine_value=True,native_search=False,swing_steps=len(source['actions']),
    tracking_metric=a.tracking_metric,
    config=json.dumps(asdict(config)))
if a.tracking_metric=='directions':
    np.savez_compressed(out/'observable_tracking_factors.npz',factors=np.array(factors),terminal=terminal_local,weights=np.diag(Q))
design_steps = env.native_steps
env.close()
results = {}
for arm in a.arms:
    directory = out/arm
    directory.mkdir()
    rows = []
    for index,seed in enumerate([20261016]+list(range(20261116,20261128))):
        owned = []
        def factory(e):
            policy = AffineValuePolicy(e,states,actions,np.zeros_like(feed) if arm=='zero_offset' else feed,
                gains,roots,residuals,R,design,search=arm=='affine_search')
            owned.append(policy)
            return policy
        try:
            row = episode(config,factory,seed,'hanging',0. if index==0 else .02,
                          directory/('nominal.npz' if index==0 else 'development_'+str(seed)+'.npz'))
            trace = owned[0].trace
            if trace:
                candidates = [(t['step'],j,c) for t in trace for j,c in enumerate(t['candidates'])]
                np.savez_compressed(directory/('search_'+str(seed)+'.npz'),
                    step=[x[0] for x in candidates],candidate=[x[1] for x in candidates],
                    actions=[x[2]['action'] for x in candidates],costs=[x[2]['cost'] for x in candidates],
                    valid=[x[2]['valid'] for x in candidates],successors=[x[2]['successor'] for x in candidates],
                    plane_margin=[x[2]['plane_margin'] for x in candidates],
                    selected=[t['selected'] for t in trace])
            rows.append(row)
            print(json.dumps(dict(arm=arm,seed=seed,success=row['success'],steps=row['steps'],
                best_hold_seconds=row['best_hold_seconds'],predictor_native_steps=row['policy_metadata']['predictor_native_steps'])),flush=True)
        finally:
            for policy in owned:
                policy.close()
        if index==0 and not (row['success'] and row['replay_exact']):
            break
    results[arm] = dict(passes=sum(r['success'] and r['replay_exact'] for r in rows),results=rows)
    write_json(directory/'summary.json',results[arm])
total = design_steps+sum(r['native_steps']+r['proof_native_steps']+r['policy_metadata']['predictor_native_steps']
    for arm in results.values() for r in arm['results'])
assert total<=2500000
write_json(out/'summary.json',dict(n_links=config.n_links,arms=results,design_native_steps=design_steps,
    inclusive_native_steps=total,claim='matched development diagnostics only; no reserved seeds or new qualified count'))
write_json(out/'completed.json',dict(status='completed'))
