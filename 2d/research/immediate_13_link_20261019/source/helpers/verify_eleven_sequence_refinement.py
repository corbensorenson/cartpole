"""Two algebraic numerical checks on an exact rejected sequence problem."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import affine_tracking_values
from spatialpole.experiment import freeze_inputs,write_json
from donor_conic_step import solver_module
from eleven_sequence_qp import reduced_cache
from trajectory_expert_support import sha


def load(path):
    with np.load(path) as data:return {k:data[k] for k in data.files}


def check(P,A,b,x,y,s,H,nx):
    px,aty=P@x,A.T@y
    primal=float(np.max(abs(A@x+s-b)))
    stationarity=float(np.max(abs(px+aty))/max(1.,float(np.max(abs(px))),float(np.max(abs(aty)))))
    cone=max(float(np.max(abs(s[:H*nx]))),float(-s[H*nx:H*nx+4*H].min()),float(-y[H*nx:H*nx+4*H].min()),0.)
    for i in range(H):
        start=H*nx+4*H+3*i
        for v in (s[start:start+3],y[start:start+3]):cone=max(cone,float(np.linalg.norm(v[1:])-v[0]))
    objective=.5*float(x@px);gap=abs(float(x@px+b@y))/max(1.,2*objective)
    comp=abs(float(s@y))/max(1.,2*objective)
    return dict(passed=bool(max(primal,stationarity,cone,gap,comp)<1e-6),primal=primal,
        stationarity=stationarity,cone_error=cone,relative_gap=gap,complementarity=comp,objective=objective)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--runtime',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run,out=Path(a.run),Path(a.out);freeze_inputs(out,a)
    assert (run/'completed.json').exists() and json.loads((run/'solver_rejection.json').read_text())['phase']==185
    z=load(run/'certificate_0185.npz');P=sparse.load_npz(run/'certificate_0185_P.npz');A=sparse.load_npz(run/'certificate_0185_A.npz');b=z['b']
    cache=load(run/'proof_input_cache.npz');v=load(run/'input_route.npz');r=reduced_cache(cache)
    t=185;H,nx=int(z['H']),int(z['nx']);assert (H,nx)==(415,59)
    geometry=SpatialCartpole(Config(n_links=11))
    try:
        feed,gains,roots,residuals=affine_tracking_values(list(zip(r['A'][t:],r['B'][t:])),r['d'][t:],r['factors'][t:-1],v['tracking_R'],10.*r['factors'][-1])
        errors=[z['initial']];du=[]
        for j,(aa,bb,dd) in enumerate(zip(r['A'][t:],r['B'][t:],r['d'][t:])):
            delta=feed[j]-gains[j]@errors[-1];du.append(delta);errors.append(aa@errors[-1]+bb@delta+dd)
        errors,du=np.array(errors),np.array(du);x=np.r_[errors[1:].ravel(),(du/80.).ravel()]
        costates=np.array([S.T@(S@e+d) for S,e,d in zip(roots,errors,residuals)])
        y=np.r_[-costates[1:].ravel(),np.zeros(7*H)];s=b-A@x
        qr=check(P,A,b,x,y,s,H,nx)
        qr['maximum_raw_force']=float(np.linalg.norm(v['actions'][t:600]+du,axis=1).max())
        np.savez_compressed(out/'square_root.npz',decision=x,dual=y,slack=s,errors=errors,delta_controls=du,feed=feed,gains=gains)
        write_json(out/'square_root_result.json',qr);print(json.dumps(dict(method='square_root',**qr)),flush=True)
        c=solver_module(a.runtime);settings=c.DefaultSettings();settings.verbose=False;settings.direct_solve_method='qdldl'
        settings.max_iter=100;settings.time_limit=30.;settings.static_regularization_enable=False
        settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-9
        cones=[c.ZeroConeT(H*nx),c.NonnegativeConeT(4*H)]+[c.SecondOrderConeT(3) for _ in range(H)]
        solver=c.DefaultSolver(sparse.triu(P,format='csc'),np.zeros(P.shape[0]),A,b,cones,settings);sol=solver.solve()
        x,y,s=np.array(sol.x),np.array(sol.z),np.array(sol.s)
        direct=check(P,A,b,x,y,s,H,nx);direct.update(status=str(sol.status),seconds=sol.solve_time,iterations=sol.iterations)
        direct['passed']=bool(direct['passed'] and str(sol.status) in ('Solved','AlmostSolved'))
        np.savez_compressed(out/'without_static_regularization.npz',decision=x,dual=y,slack=s)
        assert geometry.native_steps==0
        write_json(out/'summary.json',dict(completed=True,native_steps=0,square_root=qr,without_static_regularization=direct,
            source_certificate_sha256=sha(run/'certificate_0185.npz'),qualified=False,reserved_cases_consumed=0,
            scope='same rejected matrix problem, no native forecasts or episodes'))
        write_json(out/'completed.json',dict(status='completed'));print(json.dumps(dict(method='without_static_regularization',**direct)),flush=True)
    finally:geometry.close()


if __name__=='__main__':main()
