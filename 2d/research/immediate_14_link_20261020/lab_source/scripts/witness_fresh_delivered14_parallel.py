"""Canonical original-reset witness of an independently audited common plan."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import freeze_inputs,write_json
from spatialpole.native_mpc import exact_bytes
from delivered_candidate14 import candidate
from delivered_fresh_context14_parallel import cohort
from delivered_planner14 import controller
from witness_delivered_feedback17_14 import protected
from witness_multiplicative_critic14 import AuditedQueryCache,trace_parity
from audit_spatial_synthesis_models import load
from stage_six_source_synthesis14 import helper_closure
from eleven_synthesis_protocol import read,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory');p.add_argument('--reproduce');p.add_argument('--out',required=True);a=p.parse_args()
    original=Path(a.reproduce) if a.reproduce else None
    inventory=Path(read(original/'frozen_manifest.json')['inventory'] if original else a.inventory)
    receipt=read(inventory);run,audit=Path(receipt['run']),Path(receipt['audit'])
    assert receipt['passed'] and receipt['plan_found'] and not receipt['qualified'] and not receipt['task_success']
    for folder in (run,audit):
        for name,item in receipt['studies'][str(folder)].items():assert sha(folder/name)==item['sha256']
    m=read(run/'frozen_manifest.json');bank,runtime,cm,fm,development=cohort(m['cohort']);report=read(run/'planning/summary.json')
    assert sha(Path(m['cohort'])/'frozen_manifest.json')==m['cohort_manifest_sha256']==receipt['cohort_manifest_sha256']
    assert report['plan_found'] and report['mode'] in ('fixed','native_control')
    stage=run/'planning'/f"iteration{report['selected_iteration']}";mode=report['mode']
    forecast=stage/('native_control' if mode=='native_control' else '')/'native_forecast.npz'
    expected=load(forecast);assert expected['states'].shape==(1501,102)
    trace=load(stage/'native_control/native_queries.npz') if mode=='native_control' else None
    if trace is not None:
        proof=read(audit/'summary.json')['models'][report['selected_iteration']]['forecasts'][-1]['queries']
        assert proof['all_native_queries_byte_exact'] and proof['calls']==1500
    out=Path(a.out);freeze_inputs(out,a)
    helpers=helper_closure({'witness_fresh_delivered14_parallel.py'})
    for name in helpers:shutil.copyfile(Path(__file__).with_name(name),out/name)
    manifest=dict(protocol='common-original-delivery14-fresh-canonical-witness-v1',qualification_attempt=True,cohort_manifest_sha256=m['cohort_manifest_sha256'],config=Config(n_links=14).__dict__,seed=m['seed'],noise=.02,
        conditioning_steps=0,controller_identity=cm['controller_identity'],inventory=str(inventory),inventory_sha256=sha(inventory),
        source_manifest_sha256=sha(run/'frozen_manifest.json'),mode=mode,max_native_steps=12000,max_wall_seconds=240,
        exact_input_only_predictor_cache=True,actual_steps_are_native=True,new_noisy_samples=0,reserved_cases_consumed=0,
        helper_sha256={name:sha(out/name) for name in helpers})
    write_json(out/'frozen_manifest.json',manifest)
    env=SpatialCartpole(Config(n_links=14));cold=SpatialCartpole(env.config);policy=None;cache=None;started=time.monotonic()
    states=[];raws=[];actions=[];tilts=[]
    metadata=dict(park_seconds=0.,conditioning_steps=0,physical_velocities=44,controller_identity=cm['controller_identity'],
        development_only=False,qualification_attempt=True,hardware_realtime_claim=False)
    def save(name):
        np.savez_compressed(out/name,states=np.array(states).reshape(-1,102),raw_actions=np.array(raws).reshape(-1,2),
            actions=np.array(actions,np.float32).reshape(-1,2),tilts=np.array(tilts).reshape(-1,14),
            config=np.array(json.dumps(env.config.__dict__)),policy_metadata=np.array(json.dumps(metadata)))
    try:
        state,_=env.reset(seed=m['seed'],options=dict(mode='hanging',angle_noise=.02));exact_bytes(state,expected['states'][0]);states.append(state.copy())
        before=protected(env);policy=controller(env.config,report,run/'planning',bank)
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
        result=dict(seed=m['seed'],initialization='hanging',angle_noise=.02,**env.info(),steps=env.steps,replay_exact=True,
            park_seconds=0.,physical_velocities=44,max_force_norm=float(np.linalg.norm(np.array(actions).astype(float),axis=1).max()),hardware_realtime_claim=False)
        success=bool(result['success'] and result['steps']==1500 and result['best_hold_seconds']>=5 and result['plane_margin']>=0 and not result['physics_warning'])
        assert success and env.native_steps==cold.native_steps==6000 and time.monotonic()-started<240
        metrics=dict(passed=True,links=14,qualified=False,task_success=success,controller_identity=cm['controller_identity'],result=result,native_steps=12000,
            actual_native_steps=6000,cold_native_steps=6000,new_predictor_native_steps=0,inherited_predictor_query_intervals=queries,
            all_forecast_arrays_byte_exact=True,actual_callback_state_time_counters_unchanged=True,
            original_seed_reused=True,new_noisy_samples=0,reserved_cases_consumed=0,
            scope='canonical fresh-case original-reset witness under frozen common planner; full32 qualification still required')
        if original:
            assert (original/'completed.json').exists() and read(original/'frozen_manifest.json')==manifest and read(original/'summary.json')==metrics
            same=load(original/'actual.npz')
            for key,value in load(out/'actual.npz').items():exact_bytes(value,same[key])
            if trace is not None:
                old=load(original/'native_queries.npz')
                for key,value in actual_trace.items():exact_bytes(value,old[key])
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
