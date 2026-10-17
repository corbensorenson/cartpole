"""One bounded sparse shooting correction from cached native local models."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear,OptimizeResult
from spatialpole import Config,SpatialCartpole
from spatialpole.fddp import gap_transport
from spatialpole.direct_shooting import shooting_defects,shooting_jacobian,apply_shooting_step,solve_augmented_ridge
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
p.add_argument('--regularization',type=float,default=.001)
p.add_argument('--max-solver-iterations',type=int,default=20)
p.add_argument('--max-lsmr',type=int,default=300)
p.add_argument('--linear-solver',choices=['lsmr','augmented'],default='lsmr')
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=config.n_links)
e = SpatialCartpole(config)
e.derivative_workers = 4
xs,us = source['states'].copy(),source['actions'].copy()
H,nv,nx = len(us),e.model.nv,2*e.model.nv
scales = np.r_[[3.,3.],np.full(nv-2,.15),[4.,4.],np.full(nv-2,15.)]
# The cached models include virtual-gap transport. Undo its orthogonal blocks
# before taking the actual manifold defect derivative.
models = [(gap_transport(e,c).T@A,gap_transport(e,c).T@B)
          for A,B,c in zip(source['A'],source['B'],source['offsets'])]
residual,predicted = shooting_defects(e,xs,us,scales)
J = shooting_jacobian(e,xs,predicted,models,scales,fix_terminal=True)
M = sparse.vstack((J,a.regularization*sparse.eye(J.shape[1],format='csr')),format='csr')
rhs = np.r_[-residual.ravel(),np.zeros(J.shape[1])]
lower,upper = np.full(J.shape[1],-.1),np.full(J.shape[1],.1)
lower[(H-1)*nx:] = np.maximum(-.25,(-config.force_limit-us).ravel()/config.force_limit)
upper[(H-1)*nx:] = np.minimum(.25,(config.force_limit-us).ravel()/config.force_limit)
sparse.save_npz(out/'defect_jacobian.npz',J)
np.savez_compressed(out/'initial.npz',states=xs,actions=us,residual=residual,predicted=predicted,
                    lower=lower,upper=upper,scales=scales,config=source['config'])
baseline = .5*float(np.sum(residual**2))
print(json.dumps({'stage':'sparse bounded solve','shape':M.shape,'nonzeros':M.nnz,
                  'baseline_defect_value':baseline}),flush=True)
started = time.perf_counter()
backend = 'lsmr'
solution = None
if a.linear_solver=='augmented':
    step = solve_augmented_ridge(J,residual.ravel(),a.regularization)
    violations = int(np.count_nonzero((step<lower-1e-12)|(step>upper+1e-12)))
    print(json.dumps({'stage':'augmented solve','bound_violations':violations}),flush=True)
    if not violations:
        step = np.clip(step,lower,upper)
        linear_residual = M@step-rhs
        solution = OptimizeResult(x=step,status=3,message='accurate unconstrained ridge solution inside bounds',
            nit=0,optimality=float(np.max(np.abs(M.T@linear_residual))),cost=.5*float(linear_residual@linear_residual))
        backend = 'augmented'
if solution is None:
    solution = lsq_linear(M,rhs,bounds=(lower,upper),tol=1e-8,lsmr_tol=1e-10,
        max_iter=a.max_solver_iterations,lsmr_maxiter=a.max_lsmr,verbose=2)
np.savez_compressed(out/'linear_step.npz',step=solution.x)
solver = dict(backend=backend,status=int(solution.status),message=solution.message,iterations=int(solution.nit),
              optimality=float(solution.optimality),cost=float(solution.cost),
              max_scaled_step=float(np.max(np.abs(solution.x))),wall_seconds=time.perf_counter()-started)
write_json(out/'solver.json',solver)
rows = []
for alpha in (1.,.5,.25,.1,.05,.01):
    states,actions = apply_shooting_step(e,xs,us,solution.x,scales,alpha,fix_terminal=True)
    errors,_ = shooting_defects(e,states,actions,scales)
    state_change = np.array([e.difference(x,y) for x,y in zip(states[1:-1],xs[1:-1])])/scales
    control_change = (actions-us)/config.force_limit
    value = .5*float(np.sum(errors**2))
    merit = value+.5*a.regularization**2*(float(np.sum(state_change**2))+float(np.sum(control_change**2)))
    row = dict(alpha=alpha,defect_value=value,max_scaled_defect=float(np.max(np.abs(errors))),
               merit=merit,improves=bool(merit<baseline),max_cart=float(np.max(np.abs(states[:,:2]))),
               max_force=float(np.max(np.linalg.norm(actions,axis=1))))
    rows.append(row)
    np.savez_compressed(out/('alpha'+str(alpha)+'.npz'),states=states,actions=actions,
                        residual=errors,config=source['config'])
    print(json.dumps(row),flush=True)
write_json(out/'summary.json',dict(n_links=config.n_links,results=rows,solver=solver,
    baseline_defect_value=baseline,design_native_steps=e.native_steps,
    scope='simultaneous full-spatial shooting proposals only; no accepted physical episode'))
e.close()
write_json(out/'completed.json',{'status':'completed'})
