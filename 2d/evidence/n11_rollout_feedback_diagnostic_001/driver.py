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
p.add_argument('--cached-models',action='store_true')
p.add_argument('--axis-design',action='store_true')
p.add_argument('--fine-steps',action='store_true')
p.add_argument('--backends',nargs='+',choices=['normal','qr'],default=['normal','qr'])
p.add_argument('--trust',nargs='+',type=float,default=[1.,1e6,1e9])
p.add_argument('--derivative-order',type=int,choices=[2,4],default=2)
p.add_argument('--derivative-eps',type=float,default=1e-5)
p.add_argument('--terminal-value-scale',type=float,default=10.)
p.add_argument('--free-feedback',action='store_true')
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
design = balance_design(e,axis_separated=a.axis_design)
Q = np.diag(np.r_[[4.,4.],np.full(nv-2,40.),np.full(nv,.05)])
for j in range(config.n_links):
    Q[4+3*j,4+3*j] = .1
    Q[nv+4+3*j,nv+4+3*j] = .005
R,L = .001*np.eye(2),np.sqrt(Q)
if not np.isfinite(a.terminal_value_scale) or a.terminal_value_scale<0:
    raise ValueError('terminal value scale must be finite and nonnegative')
P = a.terminal_value_scale*design['P']+Q
root,minimum = positive_value_factor(a.terminal_value_scale*design['P'])
terminal = np.vstack((root,L))
if a.cached_models:
    models = list(zip(source['A'],source['B']))
    offsets,gaps = source['offsets'],source['gaps']
else:
    models,predicted = linearize_many(e,xs[:-1],us,return_next=True,
                                     eps=a.derivative_eps,order=a.derivative_order)
    offsets = np.array([e.difference(f,x) for f,x in zip(predicted,xs[1:])])
    models = [(gap_transport(e,c)@A,gap_transport(e,c)@B) for (A,B),c in zip(models,offsets)]
    gaps = reference_gaps(e,xs,predicted)
np.savez_compressed(out/'linear_models.npz',A=np.array([m[0] for m in models]),
    B=np.array([m[1] for m in models]),offsets=offsets,gaps=gaps,P=P,terminal_factor=terminal,
    Q=Q,R=R,states=xs,actions=us,config=json.dumps(asdict(config)))
rows = []
scales = np.r_[[3.,3.],np.full(nv-2,.15),[4.,4.],np.full(nv-2,15.)]
baseline_defect = .5*np.sum((offsets/scales)**2)
for backend in a.backends:
    for trust in a.trust:
        if backend=='normal':
            feed,K,active = affine_backward(models,offsets,us,np.zeros((H,nx)),
                np.repeat(Q[None],H,axis=0),np.zeros(nx),P,R,0.,config.force_limit,
                trust*Q,trust*.001,action_reference=us)
        else:
            feed,K,active = least_squares_backward(models,offsets,us,[L]*H,
                [np.zeros(nx)]*H,terminal,np.zeros(len(terminal)),R,config.force_limit,
                np.sqrt(trust)*L,trust*.001,action_reference=us)
        if a.free_feedback:
            _,K,_ = least_squares_backward(models,np.zeros_like(offsets),np.zeros_like(us),
                [L]*H,[np.zeros(nx)]*H,terminal,np.zeros(len(terminal)),R,1e30,
                np.sqrt(trust)*L,trust*.001,action_reference=np.zeros_like(us))
        label = backend+'_trust'+str(int(trust))
        np.savez_compressed(out/(label+'_backward.npz'),feed=feed,gains=K)
        forward = []
        alphas = (0.,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,.005,1.) if a.fine_steps else (0.,1e-8,.005,1.)
        for alpha in alphas:
            trial,actions,fx = gap_rollout(e,initial,xs,us,gaps,feed,K,alpha)
            row = {'alpha':alpha,'finite':trial is not None}
            if trial is not None:
                errors = np.array([e.difference(x,y) for x,y in zip(trial,xs)])
                per_step = np.max(np.abs(errors),axis=1)
                exceeds = np.flatnonzero(per_step>1e-6)
                terminal_cost = float(.5*np.sum((terminal@errors[-1])**2))
                du = actions-us
                cost = float(.5*np.einsum('bi,ij,bj->',errors[:-1],Q,errors[:-1])
                    +.5*np.einsum('bi,ij,bj->',du,R,du)+terminal_cost
                    +1000*np.sum(np.maximum(np.abs(trial[:,:2])-2.7,0.)**2))
                defect = .5*np.sum((np.array([e.difference(f,x) for f,x in zip(fx,trial[1:])])/scales)**2)
                row.update(max_reference_error=float(np.max(per_step)),
                    first_error_over_1e6_step=int(exceeds[0]) if len(exceeds) else None,
                    max_cart=float(np.max(np.abs(trial[:,:2]))),
                    terminal_cost=terminal_cost,cost=cost,defect_value=float(defect),
                    would_accept_at_penalty={str(rho):bool(cost+rho*defect<rho*baseline_defect-1e-10)
                                             for rho in (1e5,1e8,1e11)})
                np.savez_compressed(out/(label+'_alpha'+str(alpha)+'.npz'),
                                    states=trial,actions=actions,error=errors)
            forward.append(row)
        record = dict(backend=backend,trust=trust,free_feedback=a.free_feedback,max_gain=float(np.max(np.abs(K))),
                      max_feed=float(np.max(np.abs(feed))),bounded_steps=int(active),forward=forward)
        rows.append(record)
        print(json.dumps(record),flush=True)
        write_json(out/'summary.json',{'n_links':config.n_links,'results':rows,
            'terminal_projected_min_eigenvalue':minimum,'design_native_steps':e.native_steps,
            'scope':'backward/virtual-gap diagnostics only; no qualification or accepted physical motion'})
e.close()
write_json(out/'completed.json',{'status':'completed'})
