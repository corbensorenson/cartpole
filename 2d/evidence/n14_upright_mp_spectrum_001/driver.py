"""High-precision spectra of EXACT saved native models and rounded gains."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import mpmath as mp
import numpy as np
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--gains-directory',required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(out/'input_route.npz')
config = json.loads(str(source['config']))
n,nv = config['n_links'],2+3*config['n_links']
ix = np.array([0]+[3+3*j for j in range(n)])
iy = np.array([1]+[2+3*j for j in range(n)])
ix,iy = np.r_[ix,ix+nv],np.r_[iy,iy+nv]
signs = np.tile(np.r_[1.,-np.ones(n)],2)
A,B = source['A'],source['B']
cross = max(float(np.max(np.abs(A[np.ix_(ix,iy)]))),
            float(np.max(np.abs(A[np.ix_(iy,ix)]))),
            float(np.max(np.abs(B[ix,1]))),float(np.max(np.abs(B[iy,0]))))
frozen = out/'frozen_gains'
frozen.mkdir()
identities,gains = {},{}
for H in (500,1500,3000):
    origin = Path(a.gains_directory)/f'qr_{H}.npz'
    target = frozen/origin.name
    shutil.copy2(origin,target)
    identities[target.name] = hashlib.sha256(target.read_bytes()).hexdigest()
    gains[H] = np.load(target)['K']
write_json(out/'oracle_inputs.json',dict(mpmath=mp.__version__,gains_sha256=identities,
    exact_native_cross_axis_max=cross,native_steps=0,
    arithmetic='exact stored binary64 values converted before high-precision matrix products'))
started,rows = time.perf_counter(),[]

def spectrum(label,H,precision,ma,mb,mk):
    with mp.workdps(precision):
        M = mp.matrix(ma.tolist())-mp.matrix(mb.tolist())*mp.matrix(mk.tolist())
        values = mp.eig(M,left=False,right=False)
        radius = max(abs(v) for v in values)
        row = dict(model=label,horizon=H,decimal_digits=precision,radius=float(radius),
            radius_decimal=mp.nstr(radius,precision-5),
            eigenvalues=[dict(real=mp.nstr(mp.re(v),precision-5),imag=mp.nstr(mp.im(v),precision-5)) for v in values],
            wall_seconds=time.perf_counter()-started)
    rows.append(row)
    write_json(out/'partial.json',dict(results=rows))
    print(json.dumps({k:v for k,v in row.items() if k!='eigenvalues'}),flush=True)
    assert time.perf_counter()-started<=900

def native(precision):
    K = gains[1500]
    if cross==0.:
        spectrum('native_x',1500,precision,A[np.ix_(ix,ix)],B[ix,0:1],K)
        spectrum('native_y',1500,precision,A[np.ix_(iy,iy)],B[iy,1:2],K*signs[None,:])
    else:
        ids = np.r_[ix,iy]
        full = np.zeros((2,len(A)))
        full[0,ix],full[1,iy] = K[0],K[0]*signs
        spectrum('native_full',1500,precision,A[np.ix_(ids,ids)],B[ids],full[:,ids])

for H in (500,1500,3000):
    spectrum('averaged_axis',H,80,source['ax'],source['bx'],gains[H])
native(80)
spectrum('averaged_axis',1500,120,source['ax'],source['bx'],gains[1500])
native(120)
with mp.workdps(120):
    for row in [r for r in rows if r['decimal_digits']==120]:
        earlier = next(r for r in rows if r['model']==row['model'] and r['horizon']==row['horizon'] and r['decimal_digits']==80)
        assert abs(mp.mpf(row['radius_decimal'])-mp.mpf(earlier['radius_decimal']))<mp.mpf('1e-30')
write_json(out/'summary.json',dict(n_links=n,results=rows,native_steps=0,precision_agreement=True,
    wall_seconds=time.perf_counter()-started,
    claim='spectrum oracle for rounded identified linear model and frozen F64 gains; no physical episode or qualification'))
write_json(out/'completed.json',dict(status='completed'))
