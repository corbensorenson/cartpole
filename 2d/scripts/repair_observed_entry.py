"""Repair a full spatial suffix from a retained actual development entry state."""
import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import affine_policy_from_payload
from spatialpole.aligned_value import aligned_next_value
from spatialpole.conditioning import axial_gauge_state
from spatialpole.control import linearize_many,difference_jacobian
from spatialpole.direct_shooting import (shooting_jacobian,shooting_defects,
    apply_shooting_step,solve_augmented_ridge)
from spatialpole.quotient import axial_quotient_frame,quotient_tracking_values
from spatialpole.experiment import freeze_inputs,write_json,episode

p=argparse.ArgumentParser()
p.add_argument('--route',required=True);p.add_argument('--tape',required=True)
p.add_argument('--out',required=True)
p.add_argument('--center',type=int,choices=[20261431,20261437],default=20261431)
a=p.parse_args();out=Path(a.out);freeze_inputs(out,a)
shutil.copy2(a.tape,out/'input_actual_tape.npz')
identity={n:hashlib.sha256((out/n).read_bytes()).hexdigest() for n in ('input_route.npz','input_actual_tape.npz')}
assert identity['input_route.npz']=='3a3f643c4b8af5371c52b6f224393ad2c9c98656b20313ef568ca86274801c17'
expected_tapes={20261431:'70c8b69b9d47f6687837436583c949fc2bdab1360ceb39c1f9077d51b70dd426',
    20261437:'6b14a76a25c0c766ef3f906adbf5b2214a28006b8ae5940ce5b4fd5eb0514081'}
assert identity['input_actual_tape.npz']==expected_tapes[a.center]
source=np.load(out/'input_route.npz');tape=np.load(out/'input_actual_tape.npz')
config=Config(**json.loads(str(source['config'])))
assert config==Config(n_links=14)
seam,park,design,budget=150,960,700,8000000
assert int(source['park_steps'])==park and len(source['actions'])==design
e=SpatialCartpole(config);e.derivative_workers=4;start=time.perf_counter()

def stage(name,**fields):
    row=dict(stage=name,native_steps=e.native_steps,wall_seconds=time.perf_counter()-start)
    row.update(fields)
    with (out/'stages.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)

def exact(actual,expected):
    assert actual.dtype==expected.dtype and actual.shape==expected.shape
    assert actual.tobytes()==expected.tobytes()

try:
    state,_=e.reset(seed=a.center,options=dict(mode='hanging',angle_noise=.02))
    exact(state,tape['states'][0]);policy=affine_policy_from_payload(e,source)
    for t in range(park+seam):
        raw=policy(state,t);delivered=e.deliver(raw)
        exact(raw,tape['raw_actions'][t]);exact(delivered,tape['actions'][t])
        state,_,terminated,truncated,_=e.step_force(delivered)
        exact(state,tape['states'][t+1]);assert not terminated and not truncated
    policy.close()
    stage('actual_prefix_byte_exact',commands=park+seam)
    _,_,angles=aligned_next_value(e,source['states'][seam],np.eye(88),state)
    initial=axial_gauge_state(e,state,-angles)
    xs,us=source['states'][seam:].copy(),source['actions'][seam:].copy()
    xs[0]=initial
    target=xs.copy();target_actions=us.copy()
    H,nv,nx=len(us),e.model.nv,2*e.model.nv
    scales=np.r_[[3.,3.],np.full(nv-2,.15),[4.,4.],np.full(nv-2,15.)]
    np.savez_compressed(out/'initial_proposal.npz',states=xs,actions=us,actual_entry=state,
        proposal_entry=initial,axial_frame_angles=angles,config=source['config'])
    (out/'model.xml').write_text(e.xml,encoding='utf-8')
    regularization=.001;history=[]
    estimate=(1+4*(nx+2))*H*config.frame_skip
    for iteration in range(6):
        assert e.native_steps+2*estimate+10*H*config.frame_skip<budget
        stage('native_model_start',iteration=iteration)
        models,predicted=linearize_many(e,xs[:-1],us,eps=.001,order=4,return_next=True)
        errors=np.array([e.difference(f,x) for f,x in zip(predicted,xs[1:])])/scales
        baseline=.5*float(np.sum(errors**2));maximum=float(np.max(abs(errors)))
        stage('native_model_complete',iteration=iteration,defect_value=baseline,max_scaled_defect=maximum)
        if maximum<1e-10:
            break
        J=shooting_jacobian(e,xs,predicted,models,scales,fix_terminal=False)
        lower,upper=np.full(J.shape[1],-.1),np.full(J.shape[1],.1)
        lower[H*nx:]=np.maximum(-.25,(-config.force_limit-us).ravel()/config.force_limit)
        upper[H*nx:]=np.minimum(.25,(config.force_limit-us).ravel()/config.force_limit)
        stage('augmented_solve_start',iteration=iteration)
        step=solve_augmented_ridge(J,errors.ravel(),regularization)
        violations=int(np.count_nonzero((step<lower-1e-12)|(step>upper+1e-12)))
        status,backend,solver_iterations=3,'augmented',0
        if violations:
            M=sparse.vstack((J,regularization*sparse.eye(J.shape[1])),format='csr')
            fit=lsq_linear(M,np.r_[-errors.ravel(),np.zeros(J.shape[1])],bounds=(lower,upper),
                tol=1e-8,lsmr_tol=1e-10,max_iter=20,lsmr_maxiter=1000)
            step,status,backend,solver_iterations=fit.x,int(fit.status),'bounded_lsmr',int(fit.nit)
        else:
            step=np.clip(step,lower,upper)
        stage('solve_complete',iteration=iteration,augmented_bound_violations=violations,backend=backend,status=status)
        accepted=False;trials=[]
        for alpha in (1.,.5,.25,.1,.05,.01,.001):
            states,actions=apply_shooting_step(e,xs,us,step,scales,alpha,fix_terminal=False)
            actions=np.array([e.deliver(u) for u in actions],float)
            residual,_=shooting_defects(e,states,actions,scales)
            movement=np.array([e.difference(x,y) for x,y in zip(states[1:],xs[1:])])/scales
            value=.5*float(np.sum(residual**2))
            merit=value+.5*regularization**2*(float(np.sum(movement**2))+float(np.sum(((actions-us)/80.)**2)))
            trials.append(dict(alpha=alpha,defect_value=value,merit=merit,
                max_scaled_defect=float(np.max(abs(residual))),max_reference_cart=float(np.max(abs(states[:,:2])))))
            if merit<baseline-1e-24:
                xs,us=states,actions;accepted=True;break
        row=dict(iteration=iteration,accepted=accepted,regularization=regularization,
            solver_backend=backend,solver_status=status,solver_iterations=solver_iterations,
            augmented_bound_violations=violations,trials=trials,native_steps=e.native_steps)
        history.append(row)
        with (out/'optimizer.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(row)+'\n')
        (out/'checkpoints').mkdir(exist_ok=True)
        np.savez_compressed(out/'checkpoints'/f'iteration_{iteration:03d}.npz',states=xs,actions=us,config=source['config'])
        stage('iteration_complete',**row)
        if not accepted:
            regularization*=10
    np.savez_compressed(out/'shooting_proposal.npz',states=xs,actions=us,config=source['config'])
    stage('final_native_model_start')
    raw,predicted=linearize_many(e,xs[:-1],us,eps=.001,order=4,return_next=True)
    defects=np.array([e.difference(f,x) for f,x in zip(predicted,xs[1:])])
    models=[(difference_jacobian(e,f,x)@A,difference_jacobian(e,f,x)@B)
        for (A,B),f,x in zip(raw,predicted,xs[1:])]
    frames=[axial_quotient_frame(e,x) for x in xs]
    gauge_error=max(float(np.max(abs(A@frames[t]['D']-frames[t+1]['D'])/scales[:,None]))
        for t,(A,B) in enumerate(models))
    write_json(out/'quotient_guard.json',dict(max_normalized_gauge_flow_error=gauge_error,tolerance=1e-5))
    assert gauge_error<=1e-5
    weights=np.r_[[4.,4.],np.full(nv-2,40.),np.full(nv,.05)]
    for j in range(config.n_links):
        weights[4+3*j],weights[nv+4+3*j]=.1,.005
    L=np.diag(np.sqrt(weights))
    result=quotient_tracking_values(models,defects,[L]*H,source['tracking_R'],L,frames)
    values={k:source[k].copy() for k in source.files}
    values['states'][seam:]=xs;values['actions'][seam:]=us
    values['defects'][seam:]=defects
    for key in ('feed','gains','roots','residuals'):
        values[key][seam:]=result[key]
    values.update(observed_entry_patch=True,patch_route_step=seam,reflection_projected_model=False,
        nonplanar_quotient_planning=True)
    for key in ('states','actions','feed','gains','roots','residuals','defects'):
        exact(values[key][:seam],source[key][:seam])
    np.savez_compressed(out/'route.npz',**values)
    np.savez_compressed(out/'final_models.npz',raw_A=np.array([A for A,B in raw]),raw_B=np.array([B for A,B in raw]),
        A=np.array([A for A,B in models]),B=np.array([B for A,B in models]),defects=defects,
        E=np.array([f['E'] for f in frames]),F=np.array([f['F'] for f in frames]),
        reduced_A=np.array([A for A,B in result['reduced_models']]),reduced_B=np.array([B for A,B in result['reduced_models']]),
        reduced_defects=result['reduced_defects'],reduced_gains=result['reduced_gains'],reduced_roots=result['reduced_roots'])
    assert e.native_steps<=budget
    report=dict(input_sha256=identity,seam_route_command=seam,seam_global_command=park+seam,
        actual_prefix_byte_exact=True,design_native_steps=e.native_steps,iterations=len(history),
        final_max_scaled_defect=float(np.max(abs(defects/scales))),
        maximum_gain=float(np.max(abs(result['gains']))),gauge_error=gauge_error,
        maximum_proposal_state_change_scaled=float(np.max(abs(np.array([e.difference(x,y) for x,y in zip(xs,target)])/scales))),
        maximum_proposal_force_change_N=float(np.max(abs(us-target_actions))),
        controller_sha256=hashlib.sha256((out/'route.npz').read_bytes()).hexdigest())
    write_json(out/'design_summary.json',report)
    stage('design_complete',**report)
except Exception as exc:
    write_json(out/'failure.json',dict(type=type(exc).__name__,message=str(exc),native_steps=e.native_steps))
    e.close();raise
e.close()

rows=[];factory=lambda env:affine_policy_from_payload(env,values)
cohort=(a.center,20261016,20261116,20261117,20261118,20261120,20261121,
    20261437 if a.center==20261431 else 20261431)
for seed in cohort:
    row=episode(config,factory,seed,'hanging',0. if seed==20261016 else .02,out/f'seed{seed}.npz')
    if seed==a.center:
        actual=np.load(out/f'seed{seed}.npz')
        for key,end in (('states',park+seam+1),('actions',park+seam),('raw_actions',park+seam)):
            exact(actual[key][:end],tape[key][:end])
    rows.append(row)
    inclusive=sum(r['native_steps']+r['proof_native_steps'] for r in rows)
    assert inclusive<=100000
    write_json(out/'summary.json',dict(n_links=14,passes=sum(r['success'] and r['replay_exact'] for r in rows),
        results=rows,design_native_steps=report['design_native_steps'],pilot_native_steps=inclusive,
        inclusive_native_steps=report['design_native_steps']+inclusive,controller_sha256=report['controller_sha256'],
        scope='observed development entry repair; no qualification'))
    print(json.dumps(dict(seed=seed,success=row['success'],hold=row['best_hold_seconds'],steps=row['steps'],replay_exact=row['replay_exact'])),flush=True)
write_json(out/'completed.json',dict(status='completed'))
