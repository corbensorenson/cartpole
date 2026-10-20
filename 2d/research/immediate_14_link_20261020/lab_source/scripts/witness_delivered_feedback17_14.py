"""One canonical original-reset witness, with exact audited predictor reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import freeze_inputs,write_json
from spatialpole.native_mpc import exact_bytes
from spatial_synthesis_geometry import CorrectorPolicy
from native_control_bellman14 import NativeControlBellman
from witness_multiplicative_critic14 import AuditedQueryCache,trace_parity
from audit_spatial_synthesis_models import load
from stage_six_source_synthesis14 import helper_closure
from eleven_synthesis_protocol import read,sha


def protected(env):
    return (env.state().tobytes(),env.data.time,env.steps,env.streak,env.best_streak,
        env.min_plane_margin,env.warning_seen,env.native_steps,env.data.ctrl.tobytes())


def inputs(inventory):
    p=read(inventory);run,audit=Path(p['run']),Path(p['audit'])
    assert p['passed'] and p['full_noise_forecast_found'] and not p['qualified'] and not p['task_success']
    assert p['fraction']==1. and p['target_seed']==20261417
    for path in (run,audit):
        for name,item in p['studies'][str(path)].items():assert sha(path/name)==item['sha256']
    assert sha(run/'summary.json')==p['summary_sha256'] and sha(audit/'summary.json')==p['audit_summary_sha256']
    m=read(run/'frozen_manifest.json');assert m['config']==Config(n_links=14).__dict__
    package=Path(__file__).resolve().parents[1]/'src/spatialpole'
    assert all(sha(package/name)==digest for name,digest in m['source_sha256'].items())
    for name in ('spatial_synthesis_geometry.py','native_control_bellman14.py'):
        assert sha(Path(__file__).with_name(name))==m['helper_sha256'][name]
    row=read(run/'summary.json')['forecasts'][-1];mode=row['kind'];assert row['result']['continuation_feasible']
    folder=run/'model';v,c,plan=[load(folder/name) for name in ('virtual_reference.npz','coordinates.npz','model_plan.npz')]
    with np.load(folder/'native_models.npz') as data:F=data['F']
    trace=None;forecast=run/('native_control' if mode=='native_control' else 'model')/'native_forecast.npz'
    tape=load(forecast);assert tape['states'].shape==(1501,102) and len(tape['actions'])==1500
    exact_bytes(v['states'][0],tape['states'][0]);assert not np.any(plan['w'])
    if mode=='native_control':
        trace=load(run/'native_control/native_queries.npz')
        proof=read(audit/'summary.json')['native_queries'];assert proof['all_native_queries_byte_exact'] and proof['calls']==1500
    paths=[inventory,run/'summary.json',audit/'summary.json',run/'frozen_manifest.json',forecast]
    paths += [folder/name for name in ('virtual_reference.npz','native_models.npz','coordinates.npz','model_plan.npz')]
    if trace is not None:paths.append(run/'native_control/native_queries.npz')
    hashes={str(path.resolve()):sha(path) for path in paths}
    identity=dict(config=Config(n_links=14).__dict__,mode=mode,files=hashes,
        numeric_helpers={name:sha(Path(__file__).with_name(name)) for name in ('spatial_synthesis_geometry.py','native_control_bellman14.py')})
    cid=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    return v,F,c,plan,tape,trace,mode,hashes,cid


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--inventory');parser.add_argument('--run');parser.add_argument('--out',required=True);args=parser.parse_args()
    original=Path(args.run) if args.run else None
    inventory=Path(read(original/'inputs.json')['args']['inventory'] if original else args.inventory);out=Path(args.out)
    v,F,c,plan,expected,trace,mode,hashes,cid=inputs(inventory);freeze_inputs(out,args)
    helpers=helper_closure({'witness_delivered_feedback17_14.py'})
    for name in helpers:shutil.copyfile(Path(__file__).with_name(name),out/name)
    manifest=dict(protocol='canonical-delivered-feedback17-14-witness-v1',config=Config(n_links=14).__dict__,seed=20261417,
        noise=.02,conditioning_steps=0,input_sha256=hashes,controller_identity=cid,mode=mode,max_native_steps=12000,max_wall_seconds=180,
        exact_input_only_predictor_cache=True,actual_steps_are_native=True,new_noisy_samples=0,reserved_cases_consumed=0,
        helper_sha256={name:sha(out/name) for name in helpers})
    write_json(out/'frozen_manifest.json',manifest)
    env=SpatialCartpole(Config(n_links=14));cold=SpatialCartpole(env.config);policy=None;cache=None;started=time.monotonic()
    states=[];raws=[];actions=[];tilts=[]
    metadata=dict(park_seconds=0.,conditioning_steps=0,route_seconds=5.84,physical_velocities=44,
        controller_identity=cid,development_only=True,hardware_realtime_claim=False)
    def save(name):
        np.savez_compressed(out/name,states=np.array(states).reshape(-1,102),raw_actions=np.array(raws).reshape(-1,2),
            actions=np.array(actions,np.float32).reshape(-1,2),tilts=np.array(tilts).reshape(-1,14),
            config=np.array(json.dumps(env.config.__dict__)),policy_metadata=np.array(json.dumps(metadata)))
    try:
        state,_=env.reset(seed=20261417,options=dict(mode='hanging',angle_noise=.02));exact_bytes(state,v['states'][0]);states.append(state.copy())
        before=protected(env)
        policy=(NativeControlBellman(env.config,v,dict(F=F),c,plan['w']) if mode=='native_control'
            else CorrectorPolicy(env,v,F,c,plan['w']))
        if trace is not None:cache=AuditedQueryCache(policy.predictor,expected,trace,0,1500)
        assert protected(env)==before
        for t in range(1500):
            exact_bytes(state,expected['states'][t]);before=protected(env);raw=policy(state,t);assert protected(env)==before
            exact_bytes(raw,expected['raw_actions'][t]);force=env.deliver(raw);exact_bytes(force,expected['actions'][t])
            state,_,done,truncated,info=env.step_force(force)
            exact_bytes(state,expected['states'][t+1]);exact_bytes(env.tilts(),expected['tilts'][t])
            states.append(state.copy());raws.append(raw.copy());actions.append(force);tilts.append(env.tilts())
            if t%100==0:
                save('partial_actual.npz');write_json(out/'progress.json',dict(commands=t+1,best_hold_seconds=info['best_hold_seconds'],actual_native_steps=env.native_steps))
            assert t==1499 or not (done or truncated)
        replay=states[0].copy()
        for t,u in enumerate(actions):replay=cold.transition(replay,u);exact_bytes(replay,states[t+1]);exact_bytes(cold.tilts(),tilts[t])
        assert cold.min_plane_margin==env.min_plane_margin and cold.warning_seen==env.warning_seen
        save('actual.npz');queries=0
        if trace is not None:
            policy.save(out/'native_queries.npz');actual_trace=load(out/'native_queries.npz');trace_parity(actual_trace,trace,0,1500)
            queries=len(cache.used);assert queries==len(trace['phase']) and policy.predictor.native_steps==0
        result=dict(seed=20261417,initialization='hanging',angle_noise=.02,**env.info(),steps=env.steps,replay_exact=True,
            park_seconds=0.,physical_velocities=44,max_force_norm=float(np.linalg.norm(np.array(actions).astype(float),axis=1).max()),hardware_realtime_claim=False)
        success=bool(result['success'] and result['steps']==1500 and result['best_hold_seconds']>=5 and result['plane_margin']>=0 and not result['physics_warning'])
        assert success and env.native_steps==cold.native_steps==6000 and time.monotonic()-started<180
        metrics=dict(passed=True,links=14,qualified=False,task_success=success,controller_identity=cid,result=result,native_steps=12000,
            actual_native_steps=env.native_steps,cold_native_steps=cold.native_steps,new_predictor_native_steps=0,
            inherited_predictor_query_intervals=queries,all_forecast_arrays_byte_exact=True,actual_callback_state_time_counters_unchanged=True,
            original_seed_reused=True,new_noisy_samples=0,reserved_cases_consumed=0,
            scope='one retained original full-noise reset development witness; no fresh qualification or extra-gust robustness claim')
        if original:
            assert (original/'completed.json').exists() and read(original/'frozen_manifest.json')==manifest and read(original/'summary.json')==metrics
            actual,old=load(out/'actual.npz'),load(original/'actual.npz')
            for key in actual:exact_bytes(actual[key],old[key])
            if trace is not None:
                old=load(original/'native_queries.npz')
                for key in actual_trace:exact_bytes(actual_trace[key],old[key])
            metrics.update(source_summary_sha256=sha(original/'summary.json'),source_actual_sha256=sha(original/'actual.npz'),
                canonical_reset_actual_controls_states_scores_cache_and_cold_independently_byte_exact=True)
        write_json(out/'summary.json',metrics);write_json(out/'completed.json',dict(status='completed'));print(metrics,flush=True)
    except BaseException as exc:
        save('partial_actual_failure.npz')
        if policy is not None and trace is not None:policy.save(out/'partial_native_queries_failure.npz')
        write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),actual_native_steps=env.native_steps,cold_native_steps=cold.native_steps,
            predictor_native_steps=getattr(getattr(policy,'predictor',None),'native_steps',0),accounting_complete=True));raise
    finally:
        if policy is not None:policy.close()
        env.close();cold.close()


if __name__=='__main__':main()
