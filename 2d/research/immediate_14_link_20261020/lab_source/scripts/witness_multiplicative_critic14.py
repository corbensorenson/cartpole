"""A continuous native original-reset witness with exact audited predictor reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.experiment import freeze_inputs, write_json
from spatialpole.native_mpc import exact_bytes
from audit_spatial_synthesis_models import load
from native_control_bellman14 import NativeControlBellman
from multiplicative_control_bellman14 import MultiplicativeControlBellman
from native_critic_execution14 import snapshot, read_execution
from eleven_synthesis_protocol import read, sha
from stage_six_source_synthesis14 import helper_closure


def inputs(inventory):
    parent = read(inventory)
    future = Path(parent['run'])
    assert parent['passed'] and parent['component_plan_found'] and not parent['task_success']
    assert sha(future/'summary.json')==parent['summary_sha256'] and sha(Path(parent['audit'])/'summary.json')==parent['audit_summary_sha256']
    assert sha(future/'adaptive_forecast.npz')==parent['forecast_sha256'] and sha(future/'adaptive_native_queries.npz')==parent['queries_sha256']
    fm = read(future/'frozen_manifest.json')
    refresh = Path(fm['source_folder'])
    mean_inventory = Path(read(refresh/'inputs.json')['args']['mean_inventory'])
    mean = Path(read(mean_inventory)['run'])
    screen_inventory = Path(read(mean/'inputs.json')['args']['screen_inventory'])
    screen = Path(read(screen_inventory)['run'])
    failure_inventory = Path(read(screen/'inputs.json')['args']['failure_inventory'])
    old_parent = read(failure_inventory)
    old = Path(old_parent['run'])
    assert sha(old/'summary.json')==old_parent['summary_sha256'] and sha(Path(old_parent['audit'])/'summary.json')==old_parent['audit_summary_sha256']
    assert sha(old/'actual.npz')==old_parent['actual_sha256'] and sha(old/'native_queries.npz')==old_parent['query_trace_sha256']
    guide = Path(read(refresh/'frozen_manifest.json')['source_old_folder'])
    assert read(old/'summary.json')['guide']==str(guide)
    assert sha(old/'native_control_bellman14.py')==sha(Path(__file__).with_name('native_control_bellman14.py'))
    return future,old,guide,failure_inventory


class AuditedQueryCache:
    def __init__(self,predictor,tape,trace,start,stop):
        self.predictor,self.tape,self.trace = predictor,tape,trace
        self.start,self.stop = start,stop
        self.lookup,self.used = {},[]
        for i,phase in enumerate(trace['phase']):
            phase = int(phase)
            if not start<=phase<stop:
                continue
            key = (tape['states'][phase-start].tobytes(),trace['force'][i].tobytes())
            if key in self.lookup:
                j = self.lookup[key]
                exact_bytes(trace['successor'][i],trace['successor'][j])
                assert trace['margin'][i]==trace['margin'][j] and trace['warning'][i]==trace['warning'][j]
            else:
                self.lookup[key] = i
        predictor.transition = self.query

    def query(self,state,force,quantize=True):
        assert quantize is False
        key = (state.tobytes(),np.asarray(force,float).tobytes())
        if key not in self.lookup:
            raise RuntimeError('Unseen predictor input; frozen witness forbids new probes')
        i = self.lookup[key]
        self.used.append(i)
        self.predictor.min_plane_margin = float(self.trace['margin'][i])
        self.predictor.warning_seen = bool(self.trace['warning'][i])
        return self.trace['successor'][i].copy()


def trace_parity(actual,source,phase_start,phase_stop):
    qmask = (source['phase']>=phase_start)&(source['phase']<phase_stop)
    cmask = (source['call_phase']>=phase_start)&(source['call_phase']<phase_stop)
    query_keys = {'phase','kind','force','successor','error','margin','warning','valid','relative_cost'}
    offset = int(np.flatnonzero(qmask)[0]) if np.any(qmask) else 0
    for key,value in actual.items():
        expected = source[key][qmask if key in query_keys else cmask]
        if key in ('baseline','selected'):
            expected = expected-offset
        exact_bytes(value,expected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--component-inventory',required=True)
    p.add_argument('--out',required=True)
    a = p.parse_args()
    inventory,out = Path(a.component_inventory),Path(a.out)
    future,old,guide,failure_inventory = inputs(inventory)
    fm = read(future/'frozen_manifest.json')
    v,m,c,plan = [load(guide/name) for name in ('virtual_reference.npz','native_models.npz','coordinates.npz','model_plan.npz')]
    old_tape,old_trace = load(old/'actual.npz'),load(old/'native_queries.npz')
    ft,fq = load(future/'adaptive_forecast.npz'),load(future/'adaptive_native_queries.npz')
    execution = read_execution(future)
    freeze_inputs(out,a)
    assert read(out/'inputs.json')['source_sha256']==fm['source_sha256']
    helpers = helper_closure({'witness_multiplicative_critic14.py','native_control_bellman14.py','multiplicative_control_bellman14.py'})
    for name in helpers:
        shutil.copyfile(Path(__file__).with_name(name),out/name)
    manifest = dict(protocol='empirical-multiplicative-native14-fullreset-witness-v1',config=Config(n_links=14).__dict__,
        seed=20261125,noise=.02,handoff_phase=370,component_inventory_sha256=sha(inventory),prefix_failure_inventory_sha256=sha(failure_inventory),
        source_sha256=fm['source_sha256'],helper_sha256={name:sha(out/name) for name in helpers},
        future_manifest_sha256=sha(future/'frozen_manifest.json'),max_native_steps=12000,max_audit_native_steps=6000,max_wall_seconds=360,
        old_guide_files_sha256={name:sha(guide/name) for name in ('virtual_reference.npz','native_models.npz','coordinates.npz','model_plan.npz')},
        new_noisy_samples=0,reserved_cases_consumed=0,qualified=False,query_cache_exact_inputs_only=True,cache_miss_stops=True,
        scope='fixed retained-case phased controller witness; no generic measured-trigger or qualification claim')
    identity = hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
    write_json(out/'frozen_manifest.json',dict(**manifest,controller_identity=identity))
    env,cold = SpatialCartpole(Config(n_links=14)),SpatialCartpole(Config(n_links=14))
    before = NativeControlBellman(env.config,v,m,c,plan['w'])
    after = MultiplicativeControlBellman(env.config,*execution)
    prefix_cache = AuditedQueryCache(before.predictor,old_tape,old_trace,0,370)
    future_cache = AuditedQueryCache(after.predictor,ft,fq,370,1500)
    states,actions,raw,tilts = [],[],[],[]
    started = time.monotonic()
    def save(name):
        np.savez_compressed(out/name,states=np.asarray(states).reshape(-1,102),actions=np.asarray(actions,np.float32).reshape(-1,2),
            raw_actions=np.asarray(raw).reshape(-1,2),tilts=np.asarray(tilts).reshape(-1,14),config=np.array(json.dumps(env.config.__dict__)))
        before.save(out/'prefix_queries.npz')
        after.save(out/'future_queries.npz')
        np.savez_compressed(out/'query_reuse.npz',prefix_indices=np.array(prefix_cache.used),future_indices=np.array(future_cache.used))
    try:
        state,_ = env.reset(seed=20261125,options=dict(mode='hanging',angle_noise=.02))
        exact_bytes(state,old_tape['states'][0])
        states.append(state.copy())
        for t in range(1500):
            if time.monotonic()-started>360.:
                raise TimeoutError('Witness exceeded frozen360s cap')
            policy = before if t<370 else after
            protected = snapshot(env)
            request = policy(state,t)
            assert snapshot(env)==protected
            expected_tape = old_tape if t<370 else ft
            local = t if t<370 else t-370
            exact_bytes(request,expected_tape['raw_actions'][local])
            force = env.deliver(request)
            state,_,done,truncated,info = env.step_force(force)
            exact_bytes(force,expected_tape['actions'][local])
            exact_bytes(state,expected_tape['states'][local+1])
            exact_bytes(state,policy.selected['successor'])
            states.append(state.copy())
            actions.append(force)
            raw.append(request)
            tilts.append(env.tilts())
            if t==369:
                event = fm['event']
                assert env.steps==370 and env.streak==event['prefix_streak'] and env.best_streak==event['prefix_best_streak']
                assert env.min_plane_margin==event['prefix_plane_margin'] and env.warning_seen==event['prefix_warning']
            if t%250==0:
                print(dict(commands=t+1,best_hold=info['best_hold_seconds'],new_predictor_native=before.predictor.native_steps+after.predictor.native_steps),flush=True)
            if done or truncated:
                break
        state = states[0].copy()
        for t,u in enumerate(actions):
            state = cold.transition(state,u)
            exact_bytes(state,states[t+1])
            exact_bytes(cold.tilts(),tilts[t])
        assert cold.min_plane_margin==env.min_plane_margin and cold.warning_seen==env.warning_seen
        save('actual.npz')
        trace_parity(load(out/'prefix_queries.npz'),old_trace,0,370)
        trace_parity(load(out/'future_queries.npz'),fq,370,1500)
        assert before.predictor.native_steps==after.predictor.native_steps==0
        total = env.native_steps+cold.native_steps
        assert total<=12000
        result = dict(seed=20261125,initialization='hanging',angle_noise=.02,**env.info(),steps=len(actions),replay_exact=True,
            park_seconds=0.,physical_velocities=44,hardware_realtime_claim=False,
            max_force_norm=float(np.linalg.norm(np.array(actions).astype(float),axis=1).max()))
        write_json(out/'summary.json',dict(passed=True,links=14,qualified=False,task_success=bool(len(actions)==1500 and env.info()['success']),
            controller_identity=identity,result=result,native_steps=total,predictor_native_steps=0,
            inherited_prefix_queries=len(prefix_cache.used),inherited_future_queries=len(future_cache.used),
            all_regenerated_queries_and_actual_sources_byte_exact=True,source_models_rebuilt=False,
            original_reset_episodes=1,new_noisy_samples=0,reserved_cases_consumed=0,
            scope='continuous original-reset retained25 phased witness with every command recomputed; predictor responses transferred from independently audited exact inputs'))
        write_json(out/'completed.json',dict(status='completed'))
        print(read(out/'summary.json'),flush=True)
    except BaseException as exc:
        save('partial_actual_failure.npz')
        write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=env.native_steps+cold.native_steps,accounting_complete=True))
        raise
    finally:
        before.close()
        after.close()
        env.close()
        cold.close()


if __name__=='__main__':
    main()
