"""One full changed-mean model and bounded quarter-anchor native continuations."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import freeze_inputs,write_json
from spatialpole.native_mpc import exact_bytes
from spatialpole.direct_shooting import native_successors
from spatial_synthesis_geometry import error,CorrectorPolicy
from spatial_suffix_feedback import build_model
from spatial_observed_synthesis import forecast,ForecastFailure
from eleven_value_sequence import coordinates,solve_value_sequence,mapped_error,physical_cost,forward
from native_control_bellman14 import NativeControlBellman
from stage_six_source_synthesis14 import helper_closure
from audit_spatial_synthesis_models import load
from eleven_synthesis_protocol import read,sha


def inputs(inventory):
    p=read(inventory);run,audit=Path(p['run']),Path(p['audit'])
    assert p['passed'] and p['new_native_model_admitted'] and not p['task_success'] and p['links']==14
    assert sha(run/'summary.json')==p['summary_sha256'] and sha(audit/'summary.json')==p['audit_summary_sha256']
    for path in (run,audit):
        for name,entry in p['studies'][str(path)].items():assert sha(path/name)==entry['sha256']
    args=read(run/'inputs.json');package=Path(__file__).resolve().parents[1]/'src/spatialpole'
    assert all(sha(package/name)==digest for name,digest in args['source_sha256'].items())
    manifest=read(run/'frozen_manifest.json');assert manifest['fraction']==.25
    for path,digest in manifest['input_sha256'].items():assert sha(Path(path))==digest
    original=read(Path('output/fourteen_uncertain_refresh_v10_additional17_negative_inventory_001.json'))
    old=read(Path(original['run'])/'frozen_manifest.json');runtime=Path(old['runtime'])
    assert sha(runtime/'receipt.json')==old['solver_receipt_sha256']
    return load(run/'virtual_proposal.npz'),runtime,dict(parent_inventory_sha256=sha(inventory),
        parent_summary_sha256=sha(run/'summary.json'),parent_audit_sha256=sha(audit/'summary.json'),
        virtual_proposal_sha256=sha(run/'virtual_proposal.npz'),solver_receipt_sha256=sha(runtime/'receipt.json'))


def native_forecast(config,v,r,c,w,out):
    env=SpatialCartpole(config);cold=SpatialCartpole(config);policy=NativeControlBellman(config,v,r,c,w)
    state=v['states'][0].copy();env.set_state(state);states=[state.copy()];actions=[];raws=[];tilts=[]
    try:
        for t in range(1500):
            before,clock=env.state().tobytes(),env.data.time;raw=policy(state,t)
            assert env.state().tobytes()==before and env.data.time==clock
            delivered=env.deliver(raw);state,_,done,truncated,info=env.step_force(delivered)
            states.append(state);actions.append(delivered);raws.append(raw);tilts.append(env.tilts())
            if t%100==0:write_json(out/'forecast_progress.json',dict(commands=t+1,predictor_native_steps=policy.predictor.native_steps,actual_native_steps=env.native_steps))
            if done or truncated:break
        state=states[0].copy()
        for t,u in enumerate(actions):state=cold.transition(state,u);exact_bytes(state,states[t+1]);exact_bytes(cold.tilts(),tilts[t])
        tape=dict(states=np.array(states),actions=np.array(actions,np.float32),raw_actions=np.array(raws),tilts=np.array(tilts),config=np.array(json.dumps(config.__dict__)))
        np.savez_compressed(out/'native_forecast.npz',**tape);policy.save(out/'native_queries.npz')
        feasible=bool(len(actions)==1500 and info['success'] and info['plane_margin']>=0 and not info['physics_warning'])
        return tape,dict(continuation_feasible=feasible,commands=len(actions),best_hold_seconds=info['best_hold_seconds'],
            plane_margin=info['plane_margin'],physics_warning=info['physics_warning'],cold_replay_byte_exact=True,
            rollout_native_steps=env.native_steps,cold_native_steps=cold.native_steps,predictor_native_steps=policy.predictor.native_steps,
            inclusive_native_steps=env.native_steps+cold.native_steps+policy.predictor.native_steps,benchmark_success=False)
    except BaseException as exc:
        cost=env.native_steps+cold.native_steps+policy.predictor.native_steps
        write_json(out/'forecast_failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=cost,accounting_complete=True))
        raise ForecastFailure(cost,str(exc)) from exc
    finally:policy.close();env.close();cold.close()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inventory',required=True);parser.add_argument('--out',required=True);args=parser.parse_args()
    inventory,out=Path(args.inventory),Path(args.out);v,runtime,binding=inputs(inventory);freeze_inputs(out,args)
    helpers=helper_closure({'build_initial_path17_model14.py'})
    for name in helpers:shutil.copyfile(Path(__file__).with_name(name),out/name)
    manifest=dict(protocol='directed-quarter17-14-one-native-model-v1',config=Config(n_links=14).__dict__,fraction=.25,
        **binding,source_sha256=read(out/'inputs.json')['source_sha256'],helper_sha256={name:sha(out/name) for name in helpers},
        max_models=1,max_native_steps=2262080,max_wall_seconds=3600,audit_max_native_steps=90080,audit_max_wall_seconds=1800,
        fallback_rule='native control only after failed fixed forecast and zero residual; no new model',
        original_reset_episodes=0,new_noisy_samples=0,reserved_cases_consumed=0)
    write_json(out/'frozen_manifest.json',manifest);write_json(out/'worker.json',dict(pid=os.getpid()))
    native=SpatialCartpole(Config(n_links=14));geometry=SpatialCartpole(native.config);closed=0;started=time.monotonic();rows=[]
    def progress(stage):
        write_json(out/'progress.json',dict(stage=stage,known_native_steps=native.native_steps+closed));print(stage,flush=True)
    try:
        folder=out/'model';folder.mkdir();np.savez_compressed(folder/'virtual_reference.npz',**v)
        progress('building one full native model');r,F,E,predicted=build_model(native,geometry,v,folder)
        assert native.native_steps==2166000;progress('native model complete; directional checks')
        rng=np.random.default_rng(140251);checks=[]
        for t in (0,160,205,325,1000):
            direction=rng.normal(size=74);direction/=np.linalg.norm(direction);u=v['actions'][t].astype(float);du=-u/max(1.,float(np.linalg.norm(u)));eps=1e-4
            xs=np.array([geometry.integrate(v['states'][t],k*eps*(E[t]@direction)) for k in range(1,5)])
            us=np.array([u+k*eps*du for k in range(1,5)]);following=native_successors(native,xs,us)
            ys=np.array([error(geometry,v,F,x,t+1) for x in following]);measured=(-25*r['d'][t]+48*ys[0]-36*ys[1]+16*ys[2]-3*ys[3])/(12*eps)
            expected=r['A'][t]@direction+r['B'][t]@du;discrepancy=float(np.max(abs(measured-expected)));assert discrepancy<1e-5
            checks.append(dict(phase=t,error=discrepancy));np.savez_compressed(folder/f'probe{t}.npz',states=xs,actions=us,successors=following,measured=measured,expected=expected)
        c=coordinates(r,np.eye(2)*.001,terminal_scale=1.);np.savez_compressed(folder/'coordinates.npz',**c)
        u,e,w,proof=solve_value_sequence(c,np.zeros(74),v['states'],v['actions'].astype(float),runtime,folder/'certificate.npz',time_limit=120.)
        assert mapped_error(r,e,u-v['actions'])<1e-6
        score=physical_cost(r,e,u-v['actions'],np.eye(2)*.001,terminal_scale=1.)
        _,be,bu=forward(c,np.zeros(74),np.zeros((1500,2)));base=physical_cost(r,be,bu,np.eye(2)*.001,terminal_scale=1.)
        assert abs(score-base-.5*np.sum(w*w))/max(1.,abs(score))<1e-6
        np.savez_compressed(folder/'model_plan.npz',controls=u,errors=e,w=w)
        model_admitted=bool(not np.any(w));anchor=None
        write_json(folder/'design_summary.json',dict(passed=True,checks=checks,proof=proof,model_cost=score,zero_residual=model_admitted))
        if model_admitted:
            progress('fixed feedback native forecast')
            tape,result=forecast(native.config,v['states'][0],lambda env:CorrectorPolicy(env,v,F,c,w),folder/'native_forecast.npz')
            closed+=result['inclusive_native_steps'];rows.append(dict(kind='fixed',result=result));write_json(folder/'summary.json',dict(model_cost=score,result=result))
            if result['continuation_feasible']:anchor=tape
            else:
                fallback=out/'native_control';fallback.mkdir();progress('fixed forecast failed; one native control forecast')
                tape,result=native_forecast(native.config,v,r,c,w,fallback);closed+=result['inclusive_native_steps']
                rows.append(dict(kind='native_control',result=result));write_json(fallback/'summary.json',result)
                if result['continuation_feasible']:anchor=tape
        if anchor is not None:np.savez_compressed(out/'accepted_anchor.npz',**anchor)
        assert native.native_steps==2166080 and geometry.native_steps==0 and native.native_steps+closed<=2262080 and time.monotonic()-started<3600
        result=dict(passed=True,links=14,qualified=False,task_success=False,internal_anchor_found=anchor is not None,
            fraction=.25,model_admitted=model_admitted,forecasts=rows,model_native_steps=native.native_steps,
            forecast_native_steps=closed,native_steps=native.native_steps+closed,original_reset_episodes=0,new_noisy_samples=0,reserved_cases_consumed=0,
            scope='one full native changed-mean model and internal quarter-anchor continuations; no original noisy reset or qualification claim')
        write_json(out/'summary.json',result);write_json(out/'completed.json',dict(status='completed'));print(result,flush=True)
    except BaseException as exc:
        if isinstance(exc,ForecastFailure):closed+=exc.native_steps
        write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=native.native_steps+closed,accounting_complete=True));raise
    finally:native.close();geometry.close()


if __name__=='__main__':main()
