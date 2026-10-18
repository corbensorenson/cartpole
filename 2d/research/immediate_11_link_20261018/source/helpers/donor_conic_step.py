"""Sparse conic least-squares step with actuator disks enforced during solve."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy import sparse


def solver_module(runtime):
    runtime=Path(runtime);receipt=json.loads((runtime/'receipt.json').read_text())
    assert receipt['clarabel_version']=='0.11.1'
    for name,digest in receipt['files_sha256'].items():
        assert hashlib.sha256((runtime/name).read_bytes()).hexdigest()==digest
    sys.path.insert(0,str(runtime/'site'))
    import clarabel
    assert clarabel.__version__=='0.11.1'
    return clarabel


def solve_step(J,residual,actions,ridge,radius,runtime,method='auto',node_bounds=None,accuracy=1e-8,certificate_path=None):
    clarabel=solver_module(runtime);J=J.tocsc();m,n=J.shape;H=len(actions);nodes=n-actions.size
    assert actions.shape==(H,2) and nodes>=0
    P=sparse.diags(np.r_[np.full(n,ridge*ridge),np.ones(m)],format='csc')
    eq=sparse.hstack([-J,sparse.eye(m,format='csc')],format='csc')
    identity=sparse.eye(nodes,n+m,format='csc');box=sparse.vstack([identity,-identity],format='csc')
    rows=[];cols=[];values=[];bcone=[]
    for kind in range(2):
        for t in range(H):
            row=3*(kind*H+t)
            rows.extend([row+1,row+2]);cols.extend([nodes+2*t,nodes+2*t+1]);values.extend([-1.,-1.])
            bcone.extend([radius,0.,0.] if kind==0 else [1.,float(actions[t,0])/80.,float(actions[t,1])/80.])
    cone=sparse.coo_matrix((values,(rows,cols)),shape=(6*H,n+m)).tocsc()
    lower,upper=np.full(nodes,-radius),np.full(nodes,radius)
    if node_bounds is not None:
        lower=np.maximum(lower,node_bounds[0]);upper=np.minimum(upper,node_bounds[1])
        assert lower.shape==upper.shape==(nodes,) and np.all(lower<=upper)
    A=sparse.vstack([eq,box,cone],format='csc');b=np.r_[residual,upper,-lower,bcone]
    cones=[clarabel.ZeroConeT(m)]
    if nodes:cones.append(clarabel.NonnegativeConeT(2*nodes))
    cones.extend(clarabel.SecondOrderConeT(3) for _ in range(2*H))
    settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=100;settings.time_limit=120.
    settings.direct_solve_method=method
    settings.tol_gap_abs=accuracy;settings.tol_gap_rel=accuracy;settings.tol_feas=accuracy
    if hasattr(settings,'max_threads'):settings.max_threads=4
    solver=clarabel.DefaultSolver(P,np.zeros(n+m),A,b,cones,settings);solution=solver.solve()
    decision=np.array(solution.x);dual=np.array(solution.z);slack=np.array(solution.s)
    assert np.isfinite(decision).all() and np.isfinite(dual).all() and np.isfinite(slack).all()
    primal=float(np.max(abs(A@decision+slack-b)));stationarity=P@decision+A.T@dual
    scale=max(1.,float(np.max(abs(P@decision))),float(np.max(abs(A.T@dual))))
    dual_error=float(np.max(abs(stationarity))/scale)
    cone_violation=max(float(np.max(abs(slack[:m]))),0.)
    if nodes:cone_violation=max(cone_violation,float(-slack[m:m+2*nodes].min()),float(-dual[m:m+2*nodes].min()))
    start=m+2*nodes
    for i in range(2*H):
        for vector in (slack[start+3*i:start+3*i+3],dual[start+3*i:start+3*i+3]):
            cone_violation=max(cone_violation,float(np.linalg.norm(vector[1:])-vector[0]))
    complementarity=float(abs(slack@dual));objective=.5*float(decision@(P@decision))
    objective_gap=float(abs(2*objective+b@dual))
    status=str(solution.status)
    assert status in ('Solved','AlmostSolved'),status
    assert max(primal,dual_error,cone_violation,complementarity,objective_gap)<1e-6
    if certificate_path is not None:
        path=Path(certificate_path)
        sparse.save_npz(path.with_name(path.stem+'_A.npz'),A)
        np.savez_compressed(path,decision=decision,dual=dual,slack=slack,b=b,diagonal=P.diagonal(),n=n,m=m,H=H)
    step=decision[:n];raw_controls=actions.astype(float)+80.*step[nodes:].reshape(H,2)
    assert np.max(np.linalg.norm(raw_controls,axis=1))<=80.+1e-5
    assert np.max(np.linalg.norm(step[nodes:].reshape(H,2),axis=1))<=radius+1e-6
    if nodes:assert np.max(abs(step[:nodes]))<=radius+1e-6
    r=residual+J@step
    report=dict(status=status,iterations=solution.iterations,solve_seconds=solution.solve_time,
        primal_residual=primal,normalized_stationarity_error=dual_error,cone_violation=cone_violation,
        complementarity=complementarity,objective_gap=objective_gap,regularized_objective=objective,
        model_cost=.5*float(r@r),maximum_raw_force=float(np.linalg.norm(raw_controls,axis=1).max()),
        ridge=ridge,radius=radius,accuracy=accuracy,variables=n+m,constraints=len(b),matrix_nonzeros=A.nnz,
        linear_solver=solver.get_info().linsolver.name,linear_solver_threads=solver.get_info().linsolver.threads)
    return step,report
