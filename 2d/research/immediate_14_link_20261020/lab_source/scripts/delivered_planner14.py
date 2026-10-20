"""Bounded observed-state planning with original actuator-delivery projection."""
from pathlib import Path
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import write_json
from spatialpole.native_mpc import exact_bytes
from spatialpole.direct_shooting import native_successors
from spatial_synthesis_geometry import error,defects,CorrectorPolicy
from spatial_suffix_feedback import build_model
from spatial_observed_synthesis import forecast,ForecastFailure
from build_initial_path17_model14 import native_forecast
from eleven_value_sequence import coordinates,solve_value_sequence,mapped_error,physical_cost,forward
from general_delivered_mean14 import construct,algebra
from delivered_reference_bank14 import choose,Profile
from native_control_bellman14 import NativeControlBellman
from audit_spatial_synthesis_models import load


PROTOCOL='observed-initial-original-delivery-planner14-v1'
MAX_MODELS=4
ACTOR_NATIVE=9072324
AUDIT_NATIVE=384324


def controller(config,report,folder,bank):
    assert config==Config(n_links=14) and report['plan_found']
    if report['mode']=='cached_profile':return Profile(config,bank[report['selected_profile_index']])
    stage=Path(folder)/f"iteration{report['selected_iteration']}"
    v,c,p=[load(stage/name) for name in ('virtual_reference.npz','coordinates.npz','model_plan.npz')]
    with np.load(stage/'native_models.npz') as data:F=data['F']
    if report['mode']=='native_control':return NativeControlBellman(config,v,dict(F=F),c,p['w'])
    assert report['mode']=='fixed'
    owner=SpatialCartpole(config)
    try:return CorrectorPolicy(owner,v,F,c,p['w'])
    finally:owner.close()


def geometry_gate(geometry,proposal,data):
    tail=[]
    for state in proposal['states'][-251:]:geometry.set_state(state);tail.append(geometry.tilts())
    result=dict(virtual_plane_margin=float(3.-np.max(abs(proposal['states'][:,:2]))),
        last251_virtual_max_tilt=float(np.max(tail)),maximum_raw_request=float(np.linalg.norm(data['raw_controls'],axis=1).max()),
        maximum_delivered_force=float(np.linalg.norm(proposal['actions'].astype(float),axis=1).max()))
    result['passed']=bool(result['virtual_plane_margin']>=0 and result['last251_virtual_max_tilt']<.15 and result['maximum_delivered_force']<=80.)
    return result


def mean_screen(geometry,native,v,r,c,e0,w,observed,out,changed_first):
    out.mkdir();before=native.native_steps
    proposal,data=construct(geometry,v,r,c,e0,w,observed);proof=algebra(geometry,v,r,c,data)
    np.savez_compressed(out/'virtual_proposal.npz',**proposal);np.savez_compressed(out/'projection.npz',**data)
    gate=geometry_gate(geometry,proposal,data);merit=None
    if gate['passed']:
        baseline=r['predicted'].copy()
        if changed_first:baseline[0]=native.transition(observed,v['actions'][0])
        native.min_plane_margin=3.;native.warning_seen=False
        predicted=np.array([native.transition(x,u) for x,u in zip(proposal['states'][:-1],proposal['actions'])])
        base,bmax,_=defects(geometry,baseline,v['states'][1:]);score,maximum,residual=defects(geometry,predicted,proposal['states'][1:])
        np.savez_compressed(out/'interval_check.npz',baseline_predicted=baseline,predicted=predicted,residual=residual)
        merit=dict(passed=bool(score<=.8*base and native.min_plane_margin>=0 and not native.warning_seen),
            baseline_physical_defect_cost=base,baseline_physical_defect_max=bmax,physical_defect_cost=score,physical_defect_max=maximum,
            physical_cost_ratio=score/base if base else None,native_minimum_plane_margin=native.min_plane_margin,physics_warning=native.warning_seen)
    admitted=bool(merit and merit['passed']);used=native.native_steps-before
    assert used==(6000+4*int(changed_first) if gate['passed'] else 0)
    report=dict(passed=True,algebra=proof,geometry=gate,native_merit=merit,admitted=admitted,native_steps=used,changed_first_baseline=changed_first)
    write_json(out/'summary.json',report);write_json(out/'completed.json',dict(status='completed'))
    return proposal if admitted else None,report


def design_model(v,geometry,native,runtime,out,index):
    out.mkdir();np.savez_compressed(out/'virtual_reference.npz',**v);before=native.native_steps
    build_model(native,geometry,v,out);assert native.native_steps-before==2166000
    r=load(out/'native_models.npz');F=r['F'];checks=[];rng=np.random.default_rng(140251+index)
    for t in (0,160,205,325,1000):
        d=rng.normal(size=74);d/=np.linalg.norm(d);u=v['actions'][t].astype(float);du=-u/max(1.,float(np.linalg.norm(u)));eps=1e-4
        xs=np.array([geometry.integrate(v['states'][t],k*eps*(r['E'][t]@d)) for k in range(1,5)])
        us=np.array([u+k*eps*du for k in range(1,5)]);next_states=native_successors(native,xs,us)
        ys=np.array([error(geometry,v,F,x,t+1) for x in next_states]);measured=(-25*r['d'][t]+48*ys[0]-36*ys[1]+16*ys[2]-3*ys[3])/(12*eps)
        expected=r['A'][t]@d+r['B'][t]@du;discrepancy=float(np.max(abs(measured-expected)));assert discrepancy<1e-5
        checks.append(dict(phase=t,error=discrepancy));np.savez_compressed(out/f'probe{t}.npz',states=xs,actions=us,successors=next_states,measured=measured,expected=expected)
    c=coordinates(r,np.eye(2)*.001,terminal_scale=1.);np.savez_compressed(out/'coordinates.npz',**c)
    u,e,w,proof=solve_value_sequence(c,np.zeros(74),v['states'],v['actions'].astype(float),runtime,out/'certificate.npz',time_limit=120.)
    assert mapped_error(r,e,u-v['actions'])<1e-6
    score=physical_cost(r,e,u-v['actions'],np.eye(2)*.001,terminal_scale=1.)
    _,be,bu=forward(c,np.zeros(74),np.zeros((1500,2)));base=physical_cost(r,be,bu,np.eye(2)*.001,terminal_scale=1.)
    assert abs(score-base-.5*np.sum(w*w))/max(1.,abs(score))<1e-6
    np.savez_compressed(out/'model_plan.npz',controls=u,errors=e,w=w)
    report=dict(passed=True,checks=checks,proof=proof,model_cost=score,zero_residual=bool(not np.any(w)),native_steps=native.native_steps-before)
    write_json(out/'design_summary.json',report);assert report['native_steps']==2166080
    return r,c,w,report


def prepare(observed,bank,runtime,out,wall_seconds=15000):
    """No seed/cohort label or future observation is an input."""
    assert observed.shape==(102,) and observed.dtype==np.float64 and np.isfinite(observed).all()
    out=Path(out);out.mkdir();geometry=SpatialCartpole(Config(n_links=14));native=SpatialCartpole(geometry.config)
    started=time.monotonic();closed=0;rows=[];mode=None;selected=None;warm=None
    def total():return native.native_steps+closed
    def checkpoint(stage):
        assert total()<=ACTOR_NATIVE and time.monotonic()-started<wall_seconds
        write_json(out/'progress.json',dict(stage=stage,known_native_steps=total(),completed_models=len(rows)));print(stage,flush=True)
    def finish(found,stage,**extra):
        report=dict(passed=True,links=14,qualified=False,task_success=False,plan_found=found,stage=stage,mode=mode,
            selected_profile_index=selected,models=rows,warm=warm,native_steps=total(),native_design_and_mean_steps=native.native_steps,
            forecast_native_steps=closed,original_reset_episodes=0,new_noisy_samples=0,reserved_cases_consumed=0,**extra)
        assert total()<=ACTOR_NATIVE and geometry.native_steps==0 and time.monotonic()-started<wall_seconds
        write_json(out/'summary.json',report);write_json(out/'completed.json',dict(status='completed'));return report
    try:
        exact,order=choose(geometry,observed,bank);selected=order[0]['index'];write_json(out/'selection.json',dict(exact_index=exact,ranking=order))
        np.savez_compressed(out/'observed_state.npz',state=observed)
        if exact is not None:
            mode='cached_profile';return finish(True,'exact_existing_profile',source_profile_seed=bank[exact]['seed'],source_profile=bank[exact])
        source=Path(bank[selected]['guide']);v,r,c=[load(source/name) for name in ('virtual_reference.npz','native_models.npz','coordinates.npz')]
        exact_bytes(v['states'],r['reference_states']);exact_bytes(v['actions'],r['reference_actions'])
        initial_error=error(geometry,v,r['F'],observed,0);folder=out/'warm_plan';folder.mkdir()
        checkpoint('warm full-horizon observed-state value solve')
        u,e,w,proof=solve_value_sequence(c,initial_error,v['states'],v['actions'].astype(float),runtime,folder/'certificate.npz',time_limit=120.)
        np.savez_compressed(folder/'model_plan.npz',controls=u,errors=e,w=w,initial_error=initial_error)
        write_json(folder/'summary.json',dict(passed=True,proof=proof,source_guide=str(source)))
        v,warm=mean_screen(geometry,native,v,r,c,initial_error,w,observed,out/'warm_mean',True)
        if v is None:return finish(False,'warm_mean_rejected')
        for index in range(MAX_MODELS):
            checkpoint(f'building native model {index+1}/{MAX_MODELS}');folder=out/f'iteration{index}'
            r,c,w,design=design_model(v,geometry,native,runtime,folder,index)
            row=dict(index=index,design=design,fixed=None,native_control=None,update=None);rows.append(row)
            if not np.any(w):
                checkpoint(f'fixed native forecast {index+1}')
                tape,result=forecast(geometry.config,observed,lambda env:CorrectorPolicy(env,v,r['F'],c,w),folder/'native_forecast.npz')
                closed+=result['inclusive_native_steps'];row['fixed']=result
                if result['continuation_feasible']:
                    mode='fixed';np.savez_compressed(out/'accepted_anchor.npz',**tape);write_json(folder/'summary.json',row)
                    return finish(True,'native_forecast',selected_iteration=index)
                fallback=folder/'native_control';fallback.mkdir();checkpoint(f'native measured-force forecast {index+1}')
                tape,result=native_forecast(geometry.config,v,r,c,w,fallback);closed+=result['inclusive_native_steps'];row['native_control']=result
                if result['continuation_feasible']:
                    mode='native_control';np.savez_compressed(out/'accepted_anchor.npz',**tape);write_json(folder/'summary.json',row)
                    return finish(True,'native_forecast',selected_iteration=index)
            if index==MAX_MODELS-1:write_json(folder/'summary.json',row);break
            checkpoint(f'original-delivery mean update {index+1}')
            next_v,update=mean_screen(geometry,native,v,r,c,np.zeros(74),w,observed,folder/'next_mean',False)
            row['update']=update;write_json(folder/'summary.json',row)
            if next_v is None:return finish(False,'updated_mean_rejected')
            v=next_v
        return finish(False,'model_budget_exhausted')
    except BaseException as exc:
        if isinstance(exc,ForecastFailure):closed+=exc.native_steps
        write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=total(),accounting_complete=True,models=rows));raise
    finally:geometry.close();native.close()
