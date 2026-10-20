"""Piecewise execution arrays and adaptive native forecasts on the original clock."""
import json
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import write_json
from spatialpole.native_mpc import exact_bytes
from native_control_bellman14 import NativeControlBellman


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def execution(old_folder,v,F,c,w,phase):
    old_folder=Path(old_folder);ov,om,oc,op=[load(old_folder/name) for name in ('virtual_reference.npz','native_models.npz','coordinates.npz','model_plan.npz')]
    assert len(v['actions'])+phase==1500 and not np.any(w) and not np.any(op['w'])
    joined=dict(states=np.concatenate([ov['states'][:phase],v['states']]),actions=np.concatenate([ov['actions'][:phase],v['actions']]))
    frames=np.concatenate([om['F'][:phase],F]);coordinates={key:np.concatenate([oc[key][:phase],value]) for key,value in c.items()}
    residual=np.concatenate([op['w'][:phase],w])
    assert joined['states'].shape==(1501,102) and frames.shape==(1501,74,88) and residual.shape==(1500,2)
    return joined,dict(F=frames),coordinates,residual


def save_execution(out,bundle):
    v,m,c,w=bundle
    np.savez_compressed(out/'execution_reference.npz',**v);np.savez_compressed(out/'execution_geometry.npz',**m)
    np.savez_compressed(out/'execution_coordinates.npz',**c);np.savez_compressed(out/'execution_plan.npz',w=w)


def read_execution(folder):
    folder=Path(folder)
    return (load(folder/'execution_reference.npz'),load(folder/'execution_geometry.npz'),
        load(folder/'execution_coordinates.npz'),load(folder/'execution_plan.npz')['w'])


def snapshot(env):
    return (env.state().tobytes(),env.data.time,env.steps,env.streak,env.best_streak,env.min_plane_margin,env.warning_seen,env.native_steps,env.data.ctrl.tobytes())


def forecast(config,initial,event,bundle,out):
    assert config==Config(n_links=14)
    env,cold=SpatialCartpole(config),SpatialCartpole(config);v,m,c,w=bundle;policy=NativeControlBellman(config,v,m,c,w)
    phase=event['phase'];H=1500-phase;states=[initial.copy()];actions=[];raw=[];tilts=[]
    def save():
        np.savez_compressed(out/'adaptive_forecast.npz',states=np.asarray(states).reshape(-1,102),actions=np.asarray(actions,np.float32).reshape(-1,2),
            raw_actions=np.asarray(raw).reshape(-1,2),tilts=np.asarray(tilts).reshape(-1,14),config=np.array(json.dumps(config.__dict__)),start_phase=np.array(phase))
        policy.save(out/'adaptive_native_queries.npz')
    try:
        env.set_state(initial);env.steps=phase;env.streak=event['prefix_streak'];env.best_streak=event['prefix_best_streak']
        env.min_plane_margin=event['prefix_plane_margin'];env.warning_seen=event['prefix_warning']
        for t in range(phase,1500):
            protected=snapshot(env);request=policy(states[-1],t);assert snapshot(env)==protected
            force=env.deliver(request);state,_,done,truncated,info=env.step_force(force)
            exact_bytes(force,policy.selected['force'].astype(np.float32));exact_bytes(state,policy.selected['successor'])
            states.append(state.copy());actions.append(force);raw.append(request);tilts.append(env.tilts())
            if (t-phase)%100==0:print(dict(future_commands=t-phase+1,best_hold=info['best_hold_seconds'],predictor_native=policy.predictor.native_steps),flush=True)
            if done or truncated:break
        cold.min_plane_margin=event['prefix_plane_margin'];cold.warning_seen=event['prefix_warning'];state=initial.copy()
        for t,u in enumerate(actions):state=cold.transition(state,u);exact_bytes(state,states[t+1]);exact_bytes(cold.tilts(),tilts[t])
        assert cold.min_plane_margin==env.min_plane_margin and cold.warning_seen==env.warning_seen
        save();total=policy.predictor.native_steps+env.native_steps+cold.native_steps;assert total<=56*H
        return dict(continuation_feasible=bool(len(actions)==H and env.info()['success']),commands=len(actions),start_phase=phase,
            full_physical_horizon=1500,**env.info(),predictor_native_steps=policy.predictor.native_steps,
            rollout_native_steps=env.native_steps,cold_native_steps=cold.native_steps,inclusive_native_steps=total,
            cold_replay_byte_exact=True,selected_next_states_actual_byte_exact=True,benchmark_success=False)
    except BaseException as exc:
        save();write_json(out/'adaptive_forecast.failure.json',dict(type=type(exc).__name__,error=str(exc),predictor_native_steps=policy.predictor.native_steps,
            rollout_native_steps=env.native_steps,cold_native_steps=cold.native_steps,accounting_complete=True));raise
    finally:policy.close();env.close();cold.close()
