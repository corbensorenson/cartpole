"""Complete native admission of a fixed library of successful14 profiles."""
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import write_json
from spatialpole.native_mpc import exact_bytes
from successful_profiles14 import Profile,ranking
from audit_spatial_synthesis_models import load
from eleven_synthesis_protocol import read,sha
from witness_multiplicative_critic14 import AuditedQueryCache


def library(inventory):
    p=read(inventory);run,audit=Path(p['run']),Path(p['audit'])
    assert p['passed'] and p['ready_for_selector_freeze'] and not p['task_success']
    assert sha(run/'summary.json')==p['summary_sha256'] and sha(audit/'summary.json')==p['audit_summary_sha256']
    assert sha(run/'frozen_manifest.json')==p['manifest_sha256'] and sha(run/'profiles.json')==p['profiles_sha256']
    assert read(audit/'summary.json')['all_profile_and_query_proofs_independently_byte_exact']
    m=read(run/'frozen_manifest.json')
    for name,digest in m['helper_sha256'].items():assert sha(run/name)==digest==sha(Path(__file__).with_name(name))
    profiles=read(run/'profiles.json');assert len(profiles)==14
    for profile in profiles:
        for path,digest in profile['input_sha256'].items():assert sha(Path(path))==digest
    return profiles,m


def save_queries(profile,out,prefix=''):
    for engine,name in ((profile.before,'prefix_queries.npz'),(profile.after,'future_queries.npz')):
        if engine is not None and hasattr(engine,'calls') and len(engine.calls):engine.save(out/(prefix+name))


def cut(tape,start,stop):
    return {key:tape[key][start:stop+int(key=='states')] for key in ('states','actions','raw_actions','tilts')}


def forecast(meta,initial,out,deadline):
    out.mkdir()
    if np.array(initial).tobytes()==np.array(meta['initial_state']).tobytes():
        # Full controller parity and source physical success were prerequisites
        # of the independently verified library, including this exact reset.
        source=Path(meta['actual']);assert sha(source)==meta['input_sha256'][str(source.resolve())]
        shutil.copyfile(source,out/'forecast.npz')
        for field,name in (('prefix_trace','prefix_queries.npz'),('future_trace','future_queries.npz')):
            if meta[field] is not None:shutil.copyfile(meta[field],out/name)
        result={key:meta['result'][key] for key in ('max_tilt','best_hold_seconds','plane_margin','physics_warning','success')}
        report=dict(passed=True,profile_seed=meta['seed'],commands=1500,full_native_success=True,result=result,
            predictor_native_steps=0,actual_and_cold_native_steps=0,native_steps=0,cold_replay_byte_exact=True,
            source_cache_used=True,source_actual_sha256=sha(source),source_audit_summary_sha256=sha(Path(meta['audit'])/'summary.json'),
            physical_velocities=44,park_seconds=0.,max_force_norm=meta['result']['max_force_norm'],
            scope='exact original observed-state/profile/controller/physical-success proof transfer; no repeated physics')
        write_json(out/'summary.json',report);write_json(out/'completed.json',dict(status='completed'));return report
    env,cold=SpatialCartpole(Config(n_links=14)),SpatialCartpole(Config(n_links=14))
    policy=Profile(env.config,meta);states=[initial.copy()];actions=[];raws=[];tilts=[]
    def save():
        np.savez_compressed(out/'forecast.npz',states=np.asarray(states).reshape(-1,102),actions=np.asarray(actions,np.float32).reshape(-1,2),
            raw_actions=np.asarray(raws).reshape(-1,2),tilts=np.asarray(tilts).reshape(-1,14))
        save_queries(policy,out)
    try:
        env.set_state(initial)
        for t in range(1500):
            assert time.monotonic()<deadline,'Profile selector actor wall cap'
            raw=policy(states[-1],t);force=env.deliver(raw);state,_,done,truncated,_=env.step_force(force)
            if meta['mode']=='native':
                selected=policy.chosen(t).selected
                exact_bytes(force,selected['force'].astype(np.float32));exact_bytes(state,selected['successor'])
            states.append(state.copy());actions.append(force);raws.append(raw);tilts.append(env.tilts())
            if done or truncated:break
        cold.set_state(initial)
        for t,u in enumerate(actions):
            state,_,_,_,_=cold.step_force(u);exact_bytes(state,states[t+1]);exact_bytes(cold.tilts(),tilts[t])
        assert cold.info()==env.info();save()
        cost=policy.predictor_native_steps+env.native_steps+cold.native_steps;assert cost<=84000
        report=dict(passed=True,profile_seed=meta['seed'],commands=len(actions),full_native_success=bool(len(actions)==1500 and env.info()['success']),
            result=env.info(),predictor_native_steps=policy.predictor_native_steps,actual_and_cold_native_steps=env.native_steps+cold.native_steps,
            native_steps=cost,cold_replay_byte_exact=True,source_cache_used=False,physical_velocities=44,park_seconds=0.,
            max_force_norm=float(np.linalg.norm(np.array(actions).astype(float),axis=1).max()),
            scope='one existing complete feedback profile native forecast on the observed original start; no standalone benchmark claim')
        write_json(out/'summary.json',report);write_json(out/'completed.json',dict(status='completed'));return report
    except BaseException as exc:
        save();write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=policy.predictor_native_steps+env.native_steps+cold.native_steps,accounting_complete=True));raise
    finally:policy.close();env.close();cold.close()


def witness(env,initial,meta,folder,out,deadline):
    tape=load(folder/'forecast.npz');profile=Profile(env.config,meta);caches=[]
    cold=SpatialCartpole(env.config);states=[initial.copy()];actions=[];raws=[];tilts=[]
    try:
        if meta['mode']=='native':
            caches.append(AuditedQueryCache(profile.before.predictor,tape,load(folder/'prefix_queries.npz'),0,meta['switch']))
            if profile.after is not None:
                caches.append(AuditedQueryCache(profile.after.predictor,dict(states=tape['states'][meta['switch']:]),
                    load(folder/'future_queries.npz'),meta['switch'],1500))
        for t in range(1500):
            assert time.monotonic()<deadline,'Profile selector witness wall cap'
            raw=profile(states[-1],t);force=env.deliver(raw);state,_,done,truncated,_=env.step_force(force)
            exact_bytes(raw,tape['raw_actions'][t]);exact_bytes(force,tape['actions'][t]);exact_bytes(state,tape['states'][t+1]);exact_bytes(env.tilts(),tape['tilts'][t])
            states.append(state.copy());actions.append(force);raws.append(raw);tilts.append(env.tilts())
            assert t==1499 or not(done or truncated)
        cold.set_state(initial)
        for t,u in enumerate(actions):
            state,_,_,_,_=cold.step_force(u);exact_bytes(state,states[t+1]);exact_bytes(cold.tilts(),tilts[t])
        assert env.info()==cold.info() and env.info()['success'] and profile.predictor_native_steps==0
        np.savez_compressed(out/'actual.npz',states=np.array(states),actions=np.array(actions,np.float32),raw_actions=np.array(raws),tilts=np.array(tilts))
        save_queries(profile,out,'actual_')
        np.savez_compressed(out/'actual_query_reuse.npz',**{f'source_indices_{j}':np.array(cache.used) for j,cache in enumerate(caches)})
        return dict(passed=True,result=env.info(),native_steps=env.native_steps+cold.native_steps,
            max_force_norm=float(np.linalg.norm(np.array(actions).astype(float),axis=1).max()),physical_velocities=44,park_seconds=0.,
            all_actual_native_states_and_feedback_byte_exact=True,cold_replay_byte_exact=True)
    except BaseException as exc:
        np.savez_compressed(out/'partial_actual.npz',states=np.array(states),actions=np.array(actions,np.float32),raw_actions=np.array(raws),tilts=np.array(tilts))
        save_queries(profile,out,'actual_')
        write_json(out/'actual.failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=env.native_steps+cold.native_steps,accounting_complete=True));raise
    finally:profile.close();cold.close()
