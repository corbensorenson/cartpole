"""Native axis model: stationary DARE versus finite residual-QR designs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.control import balance_design,linearize
from spatialpole.affine_value import affine_tracking_values
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--links',type=int,required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
env = SpatialCartpole(Config(n_links=a.links))
design = balance_design(env,axis_separated=True)
A,B = linearize(env,env.upright_state(),np.zeros(2))
n,nv = a.links,env.model.nv
ix = np.array([0]+[3+3*j for j in range(n)])
iy = np.array([1]+[2+3*j for j in range(n)])
ix,iy = np.r_[ix,ix+nv],np.r_[iy,iy+nv]
signs = np.tile(np.r_[1.,-np.ones(n)],2)
ax = (A[np.ix_(ix,ix)]+signs[:,None]*A[np.ix_(iy,iy)]*signs[None,:])/2
bx = (B[ix,0:1]+signs[:,None]*B[iy,1:2])/2
Q,R = design['Q'][np.ix_(ix,ix)],design['R'][:1,:1]
np.savez_compressed(out/'native_model.npz',A=A,B=B,ax=ax,bx=bx,Q=Q,R=R,
    stationary_K=design['K'],stationary_P=design['P'],config=json.dumps(asdict(env.config)))
rows = [dict(backend='dare',closed_loop_radius=design['closed_loop_radius'],
    gain_max=float(np.max(np.abs(design['K']))),value_min_eigenvalue=float(np.linalg.eigvalsh(design['P'][np.ix_(ix,ix)])[0]))]
for H in (500,1500,3000):
    start = time.perf_counter()
    feed,gains,roots,residuals = affine_tracking_values([(ax,bx)]*H,np.zeros((H,len(ax))),
        [np.sqrt(Q)]*H,R,np.zeros_like(Q))
    K,S = gains[0],roots[0]
    P = S.T@S
    row = dict(backend='finite_qr',horizon=H,
        closed_loop_radius=float(np.max(np.abs(np.linalg.eigvals(ax-bx@K)))),
        gain_max=float(np.max(np.abs(K))),value_min_eigenvalue=float(np.linalg.eigvalsh(P)[0]),
        wall_seconds=time.perf_counter()-start)
    np.savez_compressed(out/('qr_'+str(H)+'.npz'),K=K,S=S,P=P)
    rows.append(row)
    print(json.dumps(row),flush=True)
assert env.native_steps<=2000
write_json(out/'summary.json',dict(n_links=a.links,results=rows,design_native_steps=env.native_steps,
    claim='upright linear design diagnostic only; no physical episode or new solved count'))
write_json(out/'completed.json',dict(status='completed'))
env.close()
