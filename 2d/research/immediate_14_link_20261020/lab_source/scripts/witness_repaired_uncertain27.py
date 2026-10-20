"""Full causal original-reset27 witness, reusing all audited exact-input work."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.experiment import freeze_inputs, write_json
from spatialpole.native_mpc import exact_bytes
from audit_spatial_synthesis_models import load
from native_control_synthesis14 import bundle
from native_critic_execution14 import snapshot, read_execution
from uncertain_refresh14_v8 import MeasuredUncertaintyPolicy
from witness_multiplicative_critic14 import AuditedQueryCache, trace_parity
from eleven_synthesis_protocol import read, sha
from stage_six_source_synthesis14 import helper_closure


def inputs(inventory):
    p = read(inventory)
    assert p['passed'] and p['component_plan_found'] and not p['task_success']
    run,component = Path(p['run']),Path(p['component'])
    assert sha(run/'summary.json')==p['summary_sha256'] and sha(Path(p['audit'])/'summary.json')==p['audit_summary_sha256']
    assert sha(component/'adaptive_forecast.npz')==p['forecast_sha256'] and sha(component/'adaptive_native_queries.npz')==p['queries_sha256']
    stage = Path(read(run/'frozen_manifest.json')['stage'])
    sm = read(stage/'frozen_manifest.json')
    old = Path(sm['original_run'])
    assert sm['controller_identity']==p['controller_identity']
    for name,value in sm['helper_sha256'].items():
        assert sha(stage/name)==value==sha(Path(__file__).with_name(name))
    for path,value in sm['input_sha256'].items():
        assert sha(Path(path))==value
    return p,run,component,stage,sm,old


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--component-inventory',required=True)
    p.add_argument('--out',required=True)
    a = p.parse_args()
    inventory,out = Path(a.component_inventory),Path(a.out)
    parent,run,component,stage,sm,old = inputs(inventory)
    source = bundle(Path(sm['original_guide']))
    packet = read_execution(component)
    prefix = load(old/'partial_actual_failure.npz')
    pq = load(old/'prefix_queries.npz')
    future,fq = load(component/'adaptive_forecast.npz'),load(component/'adaptive_native_queries.npz')
    event = parent['event']
    phase = event['phase']
    freeze_inputs(out,a)
    assert read(out/'inputs.json')['source_sha256']==sm['source_sha256']
    helpers = helper_closure({'witness_repaired_uncertain27.py','uncertain_refresh14_v8.py'})
    for name in helpers:
        shutil.copyfile(Path(__file__).with_name(name),out/name)
    write_json(out/'frozen_manifest.json',dict(protocol='corrected-uncertainty14-causal-originalreset27-witness-v1',
        controller_identity=parent['controller_identity'],config=Config(n_links=14).__dict__,source_sha256=sm['source_sha256'],
        component_inventory_sha256=sha(inventory),helper_sha256={name:sha(out/name) for name in helpers},
        max_native_steps=12000,max_audit_native_steps=6000,aggregate_actor_native_before_witness=10391748,
        aggregate_audit_native_before_witness=111232,seed=20261127,noise=.02,
        handoff_selected_from_actual_error=True,qualified=False,new_noisy_samples=0,reserved_cases_consumed=0))
    env,cold = SpatialCartpole(Config(n_links=14)),SpatialCartpole(Config(n_links=14))
    requests = []
    def builder(observed,observed_event):
        exact_bytes(observed,load(component/'observed_state.npz')['state'])
        assert observed_event==event
        requests.append(observed_event)
        return packet,dict(passed=True,native_steps=0,cache_used=True,component_plan_found=True,event=event,
            forecast_folder=str(component),source_manifest_sha256=sha(run/'frozen_manifest.json'),
            constructor_reused_only_after_exact_observed_state_event_source_and_controller_proof=True)
    policy = MeasuredUncertaintyPolicy(env.config,source,builder)
    cache = AuditedQueryCache(policy.prefix.predictor,prefix,pq,0,phase)
    states,actions,raw,tilts = [],[],[],[]
    started = time.monotonic()
    def save(name):
        np.savez_compressed(out/name,states=np.asarray(states).reshape(-1,102),actions=np.asarray(actions,np.float32).reshape(-1,2),
            raw_actions=np.asarray(raw).reshape(-1,2),tilts=np.asarray(tilts).reshape(-1,14),config=np.array(json.dumps(env.config.__dict__)))
        policy.prefix.save(out/'prefix_queries.npz')
        if policy.suffix is not None:
            policy.suffix.save(out/'future_queries.npz')
            np.savez_compressed(out/'query_reuse.npz',prefix_indices=np.array(cache.used),future_indices=np.array(policy.future_cache.used))
    try:
        state,_ = env.reset(seed=20261127,options=dict(mode='hanging',angle_noise=.02))
        exact_bytes(state,prefix['states'][0])
        states.append(state.copy())
        for t in range(1500):
            if time.monotonic()-started>360.:
                raise TimeoutError('Witness exceeded frozen360s cap')
            protected = snapshot(env)
            request = policy(state,t)
            assert snapshot(env)==protected
            expected = prefix if t<phase else future
            local = t if t<phase else t-phase
            exact_bytes(request,expected['raw_actions'][local])
            force = env.deliver(request)
            state,_,done,truncated,info = env.step_force(force)
            exact_bytes(force,expected['actions'][local])
            exact_bytes(state,expected['states'][local+1])
            exact_bytes(state,policy.selected['successor'])
            policy.observe(state)
            assert policy.streak==env.streak and policy.best==env.best_streak and policy.margin==env.min_plane_margin and policy.warning==env.warning_seen
            states.append(state.copy())
            actions.append(force)
            raw.append(request)
            tilts.append(env.tilts())
            if t%250==0:
                print(dict(commands=t+1,best_hold=info['best_hold_seconds'],causal_event=policy.phase),flush=True)
            if done or truncated:
                break
        state = states[0].copy()
        for t,u in enumerate(actions):
            state = cold.transition(state,u)
            exact_bytes(state,states[t+1])
            exact_bytes(cold.tilts(),tilts[t])
        assert cold.min_plane_margin==env.min_plane_margin and cold.warning_seen==env.warning_seen
        save('actual.npz')
        assert len(requests)==1 and policy.event==event and policy.phase==phase
        trace_parity(load(out/'prefix_queries.npz'),pq,0,phase)
        trace_parity(load(out/'future_queries.npz'),fq,phase,1500)
        assert policy.prefix.predictor.native_steps==policy.suffix.predictor.native_steps==0
        total = env.native_steps+cold.native_steps
        assert total==12000 and total+10391748<=11100000
        result = dict(seed=20261127,initialization='hanging',angle_noise=.02,**env.info(),steps=len(actions),replay_exact=True,
            park_seconds=0.,physical_velocities=44,hardware_realtime_claim=False,
            max_force_norm=float(np.linalg.norm(np.asarray(actions).astype(float),axis=1).max()))
        write_json(out/'summary.json',dict(passed=True,links=14,qualified=False,task_success=bool(len(actions)==1500 and env.info()['success']),
            controller_identity=parent['controller_identity'],result=result,event=event,native_steps=total,predictor_native_steps=0,
            inherited_prefix_queries=len(cache.used),inherited_future_queries=len(policy.future_cache.used),
            all_regenerated_queries_and_actual_sources_byte_exact=True,generic_first_actual_error_trigger_checked=True,
            no_model_or_preparation_rebuild=True,original_reset_episodes=1,new_noisy_samples=0,reserved_cases_consumed=0,
            scope='full causal corrected14 original-reset27 native/cold witness; exact-input audited constructor/predictor transfer, no fixed-time dispatch or qualified14 claim'))
        write_json(out/'completed.json',dict(status='completed'))
        print(read(out/'summary.json'),flush=True)
    except BaseException as exc:
        save('partial_actual_failure.npz')
        write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=env.native_steps+cold.native_steps,accounting_complete=True))
        raise
    finally:
        policy.close()
        env.close()
        cold.close()


if __name__=='__main__':
    main()
