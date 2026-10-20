"""Independent model, certificate, delivery and cold audits of common planning."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import freeze_inputs,write_json
from spatialpole.native_mpc import exact_bytes
from spatialpole.direct_shooting import native_successors
from spatialpole.world_coordinates import physical_tracking_factor
from spatial_synthesis_geometry import error,derivative,defects,CorrectorPolicy
from general_delivered_mean14 import construct,algebra
from eleven_value_sequence import coordinates,forward,mapped_error,physical_cost
from delivered_candidate14 import candidate
from delivered_reference_bank14 import choose
from delivered_planner14 import geometry_gate,MAX_MODELS,ACTOR_NATIVE,AUDIT_NATIVE
from audit_spatial_synthesis_models import load,certificate
from audit_native_control_queries14 import verify
from stage_six_source_synthesis14 import helper_closure
from eleven_synthesis_protocol import read,sha


def same_dict(actual,expected):
    assert actual.keys()==expected.keys()
    for key in actual:exact_bytes(actual[key],expected[key])


def value_plan(folder,v,r,c,initial):
    plan=load(folder/'model_plan.npz')
    controls,errors,w=certificate(folder/'certificate.npz',c,initial,v['states'],v['actions'].astype(float))
    np.testing.assert_allclose(controls,plan['controls'],atol=2e-12,rtol=1e-13)
    exact_bytes(errors,plan['errors']);exact_bytes(w,plan['w'])
    mapped=mapped_error(r,errors,controls-v['actions']);assert mapped<1e-6
    score=physical_cost(r,errors,controls-v['actions'],np.eye(2)*.001,terminal_scale=1.)
    _,be,bu=forward(c,initial,np.zeros((1500,2)))
    completion=abs(score-physical_cost(r,be,bu,np.eye(2)*.001,terminal_scale=1.)-.5*np.sum(w*w))/max(1.,abs(score))
    assert completion<1e-6
    return plan,dict(passed=True,mapped_error=mapped,completion_error=completion,model_cost=score)


def mean(folder,v,r,c,initial,w,observed,geometry,native,changed_first):
    before=native.native_steps;s=read(folder/'summary.json')
    rebuilt,data=construct(geometry,v,r,c,initial,w,observed)
    same_dict(rebuilt,load(folder/'virtual_proposal.npz'));same_dict(data,load(folder/'projection.npz'))
    assert algebra(geometry,v,r,c,data)==s['algebra']
    gate=geometry_gate(geometry,rebuilt,data);assert gate==s['geometry']
    if gate['passed']:
        baseline=r['predicted'].copy()
        if changed_first:baseline[0]=native.transition(observed,v['actions'][0])
        native.min_plane_margin=3.;native.warning_seen=False
        predicted=np.array([native.transition(x,u) for x,u in zip(rebuilt['states'][:-1],rebuilt['actions'])])
        score,maximum,residual=defects(geometry,predicted,rebuilt['states'][1:])
        base,bmax,_=defects(geometry,baseline,v['states'][1:])
        same_dict(dict(baseline_predicted=baseline,predicted=predicted,residual=residual),load(folder/'interval_check.npz'))
        merit=dict(passed=bool(score<=.8*base and native.min_plane_margin>=0 and not native.warning_seen),
            baseline_physical_defect_cost=base,baseline_physical_defect_max=bmax,physical_defect_cost=score,physical_defect_max=maximum,
            physical_cost_ratio=score/base if base else None,native_minimum_plane_margin=native.min_plane_margin,physics_warning=native.warning_seen)
        assert merit==s['native_merit'];admitted=merit['passed']
    else:assert s['native_merit'] is None;admitted=False
    assert s['admitted']==admitted and s['changed_first_baseline']==changed_first
    assert native.native_steps-before==s['native_steps']==(6000+4*int(changed_first) if gate['passed'] else 0)
    return rebuilt if admitted else None,dict(passed=True,admitted=admitted,native_steps=native.native_steps-before)


def model(folder,v,geometry,native,out,index):
    before=native.native_steps;r,c=[load(folder/name) for name in ('native_models.npz','coordinates.npz')]
    same_dict(v,load(folder/'virtual_reference.npz'))
    exact_bytes(v['states'],r['reference_states']);exact_bytes(v['actions'],r['reference_actions'])
    centers=native_successors(native,v['states'][:-1],v['actions']);exact_bytes(centers,r['predicted'])
    ds=np.array([error(geometry,v,r['F'],x,t+1) for t,x in enumerate(centers)]);exact_bytes(ds,r['d'])
    np.testing.assert_allclose(r['successor_error_jacobian']@r['raw_A']@r['E'][:-1],r['A'],atol=1e-12,rtol=1e-13)
    np.testing.assert_allclose(r['successor_error_jacobian']@r['raw_B'],r['B'],atol=1e-12,rtol=1e-13)
    factors=np.array([physical_tracking_factor(geometry,x) for x in v['states']])@r['E'];exact_bytes(factors,r['factors'])
    assert all(np.linalg.matrix_rank(E[44:])==44 for E in r['E'])
    assert all(np.linalg.matrix_rank(J[:,44:])==44 for J in r['successor_error_jacobian'])
    same_dict(coordinates(r,np.eye(2)*.001,terminal_scale=1.),c)
    plan,value=value_plan(folder,v,r,c,np.zeros(74));s=read(folder/'design_summary.json')
    assert s['passed'] and s['native_steps']==2166080 and s['zero_residual']==(not np.any(plan['w']))
    assert abs(value['model_cost']-s['model_cost'])<1e-9
    checks=[];rng=np.random.default_rng(47191+index)
    for t in (0,160,205,325,1000):
        jac=float(np.max(abs(derivative(geometry,v,r['F'],centers[t],t+1,5e-6)-r['successor_error_jacobian'][t])));assert jac<1e-6
        d=rng.normal(size=74);d/=np.linalg.norm(d);u=v['actions'][t].astype(float);du=-u/max(1.,float(np.linalg.norm(u)));eps=1e-4
        xs=np.array([geometry.integrate(v['states'][t],k*eps*(r['E'][t]@d)) for k in range(1,5)])
        us=np.array([u+k*eps*du for k in range(1,5)]);successors=native_successors(native,xs,us)
        ys=np.array([error(geometry,v,r['F'],x,t+1) for x in successors])
        measured=(-25*ds[t]+48*ys[0]-36*ys[1]+16*ys[2]-3*ys[3])/(12*eps)
        expected=r['A'][t]@d+r['B'][t]@du;discrepancy=float(np.max(abs(measured-expected)));assert discrepancy<1e-5
        checks.append(dict(phase=t,directional_error=discrepancy,jacobian_error=jac))
        np.savez_compressed(out/f'model{index}_probe{t}.npz',states=xs,actions=us,successors=successors,measured=measured,expected=expected)
    assert native.native_steps-before==6080
    return r,c,plan,dict(**value,checks=checks,native_steps=6080,actor_native_steps=2166080)


def forecast(folder,v,r,c,plan,result,kind,geometry,native,out):
    tape=load(folder/'native_forecast.npz');cold=SpatialCartpole(geometry.config)
    exact_bytes(tape['states'][0],v['states'][0]);cold.set_state(v['states'][0]);before=native.native_steps
    try:
        if kind=='native_control':
            trace=load(folder/'native_queries.npz');queries=verify(trace,v,r,c,plan,tape,geometry,native,cold,out)
            actor=4*len(trace['phase'])+8*len(tape['actions'])
        else:
            queries=None;policy=CorrectorPolicy(geometry,v,r['F'],c,plan['w'])
            try:
                for t,u in enumerate(tape['actions']):
                    raw=policy(tape['states'][t],t);exact_bytes(raw,tape['raw_actions'][t]);exact_bytes(geometry.deliver(raw),u)
                    x,_,done,truncated,info=cold.step_force(u);exact_bytes(x,tape['states'][t+1]);exact_bytes(cold.tilts(),tape['tilts'][t])
                    assert t==len(tape['actions'])-1 or not (done or truncated)
            finally:policy.close()
            actor=8*len(tape['actions'])
        info=cold.info();feasible=bool(len(tape['actions'])==1500 and info['success'] and info['plane_margin']>=0 and not info['physics_warning'])
        assert result['cold_replay_byte_exact'] and not result['benchmark_success']
        assert result['continuation_feasible']==feasible and result['commands']==len(tape['actions'])
        for key in ('best_hold_seconds','plane_margin','physics_warning'):assert result[key]==info[key]
        assert result['inclusive_native_steps']==actor and cold.native_steps==4*len(tape['actions'])
        return tape,dict(passed=True,kind=kind,feasible=feasible,queries=queries,
            native_steps=native.native_steps-before+cold.native_steps,cold_native_steps=cold.native_steps,actor_native_steps=actor)
    except BaseException:
        write_json(out/'forecast_audit_failure.json',dict(native_steps=native.native_steps-before+cold.native_steps,cold_native_steps=cold.native_steps));raise
    finally:cold.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--amendment',required=True);a=p.parse_args()
    run,out=Path(a.run),Path(a.out);assert (run/'completed.json').exists()
    s,m=read(run/'summary.json'),read(run/'frozen_manifest.json');bank,runtime,cm=candidate(m['candidate'])
    assert sha(Path(m['candidate'])/'frozen_manifest.json')==m['candidate_manifest_sha256']
    assert s['controller_identity']==m['controller_identity']==cm['controller_identity'] and m['seed']==20261418
    assert m['noise']==.02 and m['conditioning_steps']==0 and not s['task_success'] and not s['qualified']
    for name,digest in cm['audit_helper_sha256'].items():
        assert sha(Path(m['candidate'])/'audit_helpers'/name)==digest==sha(Path(__file__).with_name(name))
    amendment=read(a.amendment);prior_failed=amendment['prior_failed_audit_native_steps']
    assert sha(Path(__file__))==amendment['target_sha256'] and sha(Path(amendment['source']))==amendment['source_sha256']
    assert sha(Path(amendment['failed_audit'])/'failure.json')==amendment['failure_sha256'] and prior_failed==6004
    freeze_inputs(out,a);assert read(out/'inputs.json')['source_sha256']==cm['source_sha256']
    shutil.copyfile(a.amendment,out/'audit_amendment.json')
    for name in helper_closure({'audit_delivered_case14_margin.py','audit_delivered_case14.py'}):shutil.copyfile(Path(__file__).with_name(name),out/name)
    geometry=SpatialCartpole(Config(n_links=14));native=SpatialCartpole(geometry.config);started=time.monotonic()
    native.derivative_workers=4;closed=0;actor=0;rows=[]
    try:
        observed,_=geometry.reset(seed=m['seed'],options=dict(mode='hanging',angle_noise=.02))
        exact_bytes(observed,load(run/'observed_state.npz')['state']);exact_bytes(observed,load(Path(m['candidate'])/'registered_initial.npz')['state'])
        planning=run/'planning';exact_bytes(observed,load(planning/'observed_state.npz')['state'])
        exact,order=choose(geometry,observed,bank);assert read(planning/'selection.json')==dict(exact_index=exact,ranking=order)
        assert s['selected_profile_index']==order[0]['index']
        if exact is not None:
            assert s['mode']=='cached_profile' and s['plan_found'] and s['native_steps']==0
            found=True
        else:
            source=Path(bank[order[0]['index']]['guide']);v,r,c=[load(source/name) for name in ('virtual_reference.npz','native_models.npz','coordinates.npz')]
            initial=error(geometry,v,r['F'],observed,0);exact_bytes(initial,load(planning/'warm_plan/model_plan.npz')['initial_error'])
            plan,warmvalue=value_plan(planning/'warm_plan',v,r,c,initial)
            v,warm=mean(planning/'warm_mean',v,r,c,initial,plan['w'],observed,geometry,native,True);actor+=warm['native_steps']
            found=False
            for index,row in enumerate(s['models']):
                assert v is not None and index==row['index'] and index<MAX_MODELS
                stage=planning/f'iteration{index}';r,c,plan,proof=model(stage,v,geometry,native,out,index);actor+=2166080
                forecasts=[]
                if not np.any(plan['w']):
                    assert row['fixed'] is not None
                    tape,f=forecast(stage,v,r,c,plan,row['fixed'],'fixed',geometry,native,out)
                    forecasts.append(f);closed+=f['cold_native_steps'];actor+=f['actor_native_steps'];found=f['feasible']
                    if not found:
                        assert row['native_control'] is not None
                        tape,f=forecast(stage/'native_control',v,r,c,plan,row['native_control'],'native_control',geometry,native,out)
                        forecasts.append(f);closed+=f['cold_native_steps'];actor+=f['actor_native_steps'];found=f['feasible']
                    else:assert row['native_control'] is None
                else:assert row['fixed'] is None and row['native_control'] is None
                rows.append(dict(index=index,model=proof,forecasts=forecasts))
                if found:
                    assert row['update'] is None and s['selected_iteration']==index and index==len(s['models'])-1
                    assert s['mode']==forecasts[-1]['kind'];same_dict(tape,load(planning/'accepted_anchor.npz'));break
                if row['update'] is not None:
                    v,update=mean(stage/'next_mean',v,r,c,np.zeros(74),plan['w'],observed,geometry,native,False)
                    actor+=update['native_steps'];rows[-1]['update']=update
                else:assert index==MAX_MODELS-1
            assert found==s['plan_found']
            if not found:
                if not s['models']:assert v is None and s['stage']=='warm_mean_rejected'
                elif v is None:assert s['stage']=='updated_mean_rejected'
                else:assert len(rows)==MAX_MODELS and s['stage']=='model_budget_exhausted'
        assert actor==s['native_steps']<=ACTOR_NATIVE
        assert native.native_steps+closed+prior_failed<=AUDIT_NATIVE and geometry.native_steps==0 and time.monotonic()-started<7200
        result=dict(passed=True,links=14,qualified=False,task_success=False,plan_found=found,seed=m['seed'],controller_identity=cm['controller_identity'],
            native_steps=native.native_steps+closed,prior_failed_audit_native_steps=prior_failed,inclusive_audit_native_steps=native.native_steps+closed+prior_failed,actor_native_steps=actor,models=rows,source_summary_sha256=sha(run/'summary.json'),
            source_manifest_sha256=sha(run/'frozen_manifest.json'),original_reset_episodes=0,reserved_cases_consumed=0,
            scope='independent planning proof; canonical original-task episode required after a positive plan')
        write_json(out/'summary.json',result);write_json(out/'completed.json',dict(status='completed'));print(result,flush=True)
    except BaseException as exc:
        failure=out/'forecast_audit_failure.json';extra=read(failure)['cold_native_steps'] if failure.exists() else 0
        write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=native.native_steps+closed+extra,accounting_complete=True));raise
    finally:native.close();geometry.close()


if __name__=='__main__':main()
