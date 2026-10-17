"""Iterative full-spatial sparse feasibility repair, then actual controller gates."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear,OptimizeResult
from spatialpole import Config,SpatialCartpole
from spatialpole.control import balance_design,linearize_many,RoutePolicy
from spatialpole.direct_shooting import shooting_defects,shooting_jacobian,apply_shooting_step,solve_augmented_ridge
from spatialpole.tracking_repair import positive_value_factor
from spatialpole.fddp import least_squares_backward,gap_rollout
from spatialpole.experiment import freeze_inputs,write_json,episode

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
p.add_argument('--iterations',type=int,default=20)
p.add_argument('--physics-budget',type=int,default=12000000)
p.add_argument('--regularization',type=float,default=.001)
p.add_argument('--max-lsmr',type=int,default=1000)
p.add_argument('--max-solver-iterations',type=int,default=20)
p.add_argument('--derivative-order',type=int,choices=[2,4],default=4)
p.add_argument('--derivative-eps',type=float,default=.001)
p.add_argument('--linear-solver',choices=['lsmr','augmented'],default='lsmr')
p.add_argument('--balance-backend',choices=['dare','finite_qr'],default='dare')
p.add_argument('--eigenvalue-digits',type=int,default=0)
p.add_argument('--free-terminal',action='store_true')
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=config.n_links)
e = SpatialCartpole(config)
e.derivative_workers = 4
xs,us = source['states'].copy(),source['actions'].copy()
initial,_ = e.reset(seed=20261016,options={'mode':'hanging','angle_noise':0.})
xs[0] = initial
H,nv,nx = len(us),e.model.nv,2*e.model.nv
fixed_terminal = not a.free_terminal
variable_nodes = H-1 if fixed_terminal else H
state_variables = variable_nodes*nx
variable_slice = slice(1,-1) if fixed_terminal else slice(1,None)
scales = np.r_[[3.,3.],np.full(nv-2,.15),[4.,4.],np.full(nv-2,15.)]
history = []
regularization = a.regularization
started = time.perf_counter()

def stage(name,**details):
    row = dict(stage=name,native_steps=e.native_steps,wall_seconds=time.perf_counter()-started,**details)
    with (out/'stages.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

for iteration in range(a.iterations):
    # Leave enough for the final derivative/gain design and native rebuild.
    estimate = (1+a.derivative_order*(nx+2))*H*config.frame_skip
    if e.native_steps+2*estimate+10*H*config.frame_skip>a.physics_budget:
        break
    stage('native_model_start',iteration=iteration)
    models,predicted = linearize_many(e,xs[:-1],us,eps=a.derivative_eps,
                                      return_next=True,order=a.derivative_order)
    residual = np.array([e.difference(f,x) for f,x in zip(predicted,xs[1:])])/scales
    baseline = .5*float(np.sum(residual**2))
    maximum = float(np.max(np.abs(residual)))
    stage('native_model_complete',iteration=iteration,defect_value=baseline,max_scaled_defect=maximum)
    if maximum<1e-11:
        break
    J = shooting_jacobian(e,xs,predicted,models,scales,fix_terminal=fixed_terminal)
    M = sparse.vstack((J,regularization*sparse.eye(J.shape[1],format='csr')),format='csr')
    rhs = np.r_[-residual.ravel(),np.zeros(J.shape[1])]
    lower,upper = np.full(J.shape[1],-.1),np.full(J.shape[1],.1)
    lower[state_variables:] = np.maximum(-.25,(-config.force_limit-us).ravel()/config.force_limit)
    upper[state_variables:] = np.minimum(.25,(config.force_limit-us).ravel()/config.force_limit)
    solve_started = time.perf_counter()
    solution,backend,bound_violations = None,'lsmr',None
    if a.linear_solver=='augmented':
        stage('augmented_solve_start',iteration=iteration)
        step = solve_augmented_ridge(J,residual.ravel(),regularization)
        bound_violations = int(np.count_nonzero((step<lower-1e-12)|(step>upper+1e-12)))
        stage('augmented_solve_complete',iteration=iteration,bound_violations=bound_violations)
        if not bound_violations:
            step = np.clip(step,lower,upper)
            error = M@step-rhs
            solution = OptimizeResult(x=step,status=3,nit=0,
                optimality=float(np.max(np.abs(M.T@error))))
            backend = 'augmented'
    if solution is None:
        stage('bounded_lsmr_start',iteration=iteration)
        solution = lsq_linear(M,rhs,bounds=(lower,upper),tol=1e-8,lsmr_tol=1e-10,
            max_iter=a.max_solver_iterations,lsmr_maxiter=a.max_lsmr)
    solve_wall = time.perf_counter()-solve_started
    accepted,alpha_used,trials = False,0.,[]
    for alpha in (1.,.5,.25,.1,.05,.01,.001):
        if e.native_steps+2*H*config.frame_skip>a.physics_budget:
            break
        states,actions = apply_shooting_step(e,xs,us,solution.x,scales,alpha,fix_terminal=fixed_terminal)
        errors,_ = shooting_defects(e,states,actions,scales)
        change = np.array([e.difference(x,y) for x,y in zip(states[variable_slice],xs[variable_slice])])/scales
        du = (actions-us)/config.force_limit
        value = .5*float(np.sum(errors**2))
        merit = value+.5*regularization**2*(float(np.sum(change**2))+float(np.sum(du**2)))
        trials.append(dict(alpha=alpha,defect_value=value,max_scaled_defect=float(np.max(np.abs(errors))),
                           merit=merit,max_cart=float(np.max(np.abs(states[:,:2])))))
        if merit<baseline-1e-20:
            xs,us = states,actions
            baseline,maximum = value,float(np.max(np.abs(errors)))
            accepted,alpha_used = True,alpha
            break
    if not accepted:
        regularization *= 10
    elif iteration%5==4:
        regularization = max(regularization/3,1e-7)
    row = dict(iteration=iteration,defect_value=baseline,max_scaled_defect=maximum,
        accepted=accepted,alpha=alpha_used,regularization=regularization,
        solver_status=int(solution.status),solver_optimality=float(solution.optimality),
        solver_iterations=int(solution.nit),solver_wall_seconds=solve_wall,
        solver_backend=backend,augmented_bound_violations=bound_violations,
        native_steps=e.native_steps,max_cart=float(np.max(np.abs(xs[:,:2]))),trials=trials)
    history.append(row)
    with (out/'optimizer.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
    (out/'checkpoints').mkdir(exist_ok=True)
    np.savez_compressed(out/'checkpoints'/('iteration_'+str(iteration).zfill(3)+'.npz'),
                        states=xs,actions=us,config=json.dumps(asdict(config)))
np.savez_compressed(out/'shooting_proposal.npz',states=xs,actions=us,config=json.dumps(asdict(config)),
                    fixed_terminal=fixed_terminal)
stage('final_feedback_design_start',optimizer_iterations=len(history))
design = balance_design(e,axis_separated=True,value_backend=a.balance_backend,
                        eigenvalue_digits=a.eigenvalue_digits)
models = linearize_many(e,xs[:-1],us,eps=a.derivative_eps,order=a.derivative_order)
Q = np.diag(np.r_[[4.,4.],np.full(nv-2,40.),np.full(nv,.05)])
for j in range(config.n_links):
    Q[4+3*j,4+3*j] = .1
    Q[nv+4+3*j,nv+4+3*j] = .005
L,R = np.sqrt(Q),.001*np.eye(2)
root = np.sqrt(10)*design['value_factor'] if 'value_factor' in design else positive_value_factor(10*design['P'])[0]
terminal = np.vstack((root,L))
_,K,_ = least_squares_backward(models,np.zeros((H,nx)),np.zeros_like(us),[L]*H,
    [np.zeros(nx)]*H,terminal,np.zeros(len(terminal)),R,1e30)
rebuilt,controls,_ = gap_rollout(e,initial,xs,us,np.zeros((H,nx)),np.zeros_like(us),K,1.)
native_design = e.native_steps
e.close()
if rebuilt is None:
    write_json(out/'failure.json',dict(message='gap-free native rebuild diverged',design_native_steps=native_design))
    raise RuntimeError('gap-free native rebuild diverged')
np.savez_compressed(out/'route.npz',states=rebuilt,actions=controls,gains=K,**design,
                    config=json.dumps(asdict(config)),world_balance=True)
factory = lambda env: RoutePolicy(env,rebuilt,controls,K,design,True)
rows = [episode(config,factory,20261016,'hanging',0.,out/'nominal.npz')]
if rows[0]['success'] and rows[0]['replay_exact']:
    rows.extend(episode(config,factory,20261116+k,'hanging',.02,out/('validation_'+str(k)+'.npz')) for k in range(12))
write_json(out/'summary.json',dict(n_links=config.n_links,
    passes=sum(r['success'] and r['replay_exact'] for r in rows),results=rows,
    design_native_steps=native_design,optimizer_iterations=len(history),
    fixed_terminal=fixed_terminal,
    last_optimizer_record=history[-1] if history else None,wall_seconds=time.perf_counter()-started,
    scope='simultaneous full-spatial proposals; acceptance only from actual unchanged float32 native episodes'))
write_json(out/'completed.json',{'status':'completed'})
print(json.dumps({'passes':sum(r['success'] for r in rows),'episodes':len(rows)}),flush=True)
