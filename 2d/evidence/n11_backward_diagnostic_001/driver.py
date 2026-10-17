"""Diagnose a failed tracking backward pass without accepting any swing-up."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.control import balance_design,linearize_many
from spatialpole.fddp import affine_backward,least_squares_backward,reference_gaps,gap_transport,gap_rollout
from spatialpole.tracking_repair import positive_value_factor
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=config.n_links)
e = SpatialCartpole(config)
e.derivative_workers = 4
initial,_ = e.reset(seed=20261016,options={'mode':'hanging','angle_noise':0.})
xs,us = source['states'].copy(),source['actions'].copy()
xs[0] = initial
H,nv,nx = len(us),e.model.nv,2*e.model.nv
design = balance_design(e)
Q = np.diag(np.r_[[4.,4.],np.full(nv-2,40.),np.full(nv,.05)])
for j in range(config.n_links):
    Q[4+3*j,4+3*j] = .1
    Q[nv+4+3*j,nv+4+3*j] = .005
R,L = .001*np.eye(2),np.sqrt(Q)
P = 10*design['P']+Q
root,minimum = positive_value_factor(10*design['P'])
terminal = np.vstack((root,L))
models,predicted = linearize_many(e,xs[:-1],us,return_next=True)
offsets = np.array([e.difference(f,x) for f,x in zip(predicted,xs[1:])])
models = [(gap_transport(e,c)@A,gap_transport(e,c)@B) for (A,B),c in zip(models,offsets)]
gaps = reference_gaps(e,xs,predicted)
np.savez_compressed(out/'linear_models.npz',A=np.array([m[0] for m in models]),
    B=np.array([m[1] for m in models]),offsets=offsets,gaps=gaps,P=P,terminal_factor=terminal,
    Q=Q,R=R,states=xs,actions=us,config=json.dumps(asdict(config)))
rows = []
for backend in ('normal','qr'):
    for trust in (1.,1e6,1e9):
        if backend=='normal':
            feed,K,active = affine_backward(models,offsets,us,np.zeros((H,nx)),
                np.repeat(Q[None],H,axis=0),np.zeros(nx),P,R,0.,config.force_limit,
                trust*Q,trust*.001,action_reference=us)
        else:
            feed,K,active = least_squares_backward(models,offsets,us,[L]*H,
                [np.zeros(nx)]*H,terminal,np.zeros(len(terminal)),R,config.force_limit,
                np.sqrt(trust)*L,trust*.001,action_reference=us)
        label = backend+'_trust'+str(int(trust))
        np.savez_compressed(out/(label+'_backward.npz'),feed=feed,gains=K)
        forward = []
        for alpha in (0.,1e-8,.005,1.):
            trial,actions,_ = gap_rollout(e,initial,xs,us,gaps,feed,K,alpha)
            row = {'alpha':alpha,'finite':trial is not None}
            if trial is not None:
                errors = np.array([e.difference(x,y) for x,y in zip(trial,xs)])
                per_step = np.max(np.abs(errors),axis=1)
                exceeds = np.flatnonzero(per_step>1e-6)
                row.update(max_reference_error=float(np.max(per_step)),
                    first_error_over_1e6_step=int(exceeds[0]) if len(exceeds) else None,
                    max_cart=float(np.max(np.abs(trial[:,:2]))),
                    terminal_cost=float(.5*np.sum((terminal@errors[-1])**2)))
                np.savez_compressed(out/(label+'_alpha'+str(alpha)+'.npz'),
                                    states=trial,actions=actions,error=errors)
            forward.append(row)
        record = dict(backend=backend,trust=trust,max_gain=float(np.max(np.abs(K))),
                      max_feed=float(np.max(np.abs(feed))),bounded_steps=int(active),forward=forward)
        rows.append(record)
        print(json.dumps(record),flush=True)
        write_json(out/'summary.json',{'n_links':config.n_links,'results':rows,
            'terminal_projected_min_eigenvalue':minimum,'design_native_steps':e.native_steps,
            'scope':'backward/virtual-gap diagnostics only; no qualification or accepted physical motion'})
e.close()
write_json(out/'completed.json',{'status':'completed'})
