"""Owned full-episode rollouts for observed-reset trajectory continuation."""
from dataclasses import asdict
import json
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.native_mpc import exact_bytes


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def initial_at(geometry,nominal,target,fraction):
    if fraction==0.:return nominal.copy()
    if fraction==1.:return target.copy()
    return geometry.integrate(nominal,fraction*geometry.difference(target,nominal))


def forecast(initial,factory,path):
    env=SpatialCartpole(Config(n_links=11));cold=SpatialCartpole(env.config);policy=None
    try:
        env.set_state(initial);policy=factory(env);states=[initial.copy()];actions=[];raws=[];tilts=[]
        for t in range(1500):
            before,clock=env.state().tobytes(),env.data.time
            force=policy(states[-1],t);assert before==env.state().tobytes() and clock==env.data.time
            u=env.deliver(force);state,_,done,truncated,info=env.step_force(u)
            states.append(state);actions.append(u);raws.append(force);tilts.append(env.tilts())
            if done or truncated:break
        state=initial.copy()
        for t,u in enumerate(actions):state=cold.transition(state,u);exact_bytes(state,states[t+1])
        data=dict(states=np.array(states),actions=np.array(actions,dtype=np.float32),raw_actions=np.array(raws),
            tilts=np.array(tilts),config=np.array(json.dumps(asdict(env.config))))
        np.savez_compressed(path,**data)
        feasible=bool(info['success'] and len(actions)==1500 and not info['physics_warning'] and info['plane_margin']>=0)
        return data,dict(continuation_feasible=feasible,commands=len(actions),best_hold_seconds=info['best_hold_seconds'],
            plane_margin=info['plane_margin'],physics_warning=info['physics_warning'],cold_replay_byte_exact=True,
            rollout_native_steps=env.native_steps,cold_native_steps=cold.native_steps,
            inclusive_native_steps=env.native_steps+cold.native_steps,
            benchmark_success=False,scope='internal continuation predictor, not original-noise task evidence')
    finally:
        if policy is not None:policy.close()
        env.close();cold.close()
