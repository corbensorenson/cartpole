"""Input-only native stencils with an independent physical-force spacing."""
import mujoco as mj
import mujoco.rollout
import numpy as np


def input_jacobians(env,states,actions,eps=1/32):
    """Smooth held-force derivatives; actual commands still use F32 delivery.

    This uses owned batch data and never sets env.data. Identification near the
    force disk may probe outside that disk, as in the existing native planner;
    inward stencils respect component actuator caps. No such probe is an actual
    episode or accepted command.
    """
    states,actions=np.asarray(states,float),np.asarray(actions,float)
    count=len(states)
    if (not np.isfinite(eps) or eps<=0 or count<1 or
        states.shape!=(count,env.model.nq+env.model.nv) or actions.shape!=(count,2) or
        not np.isfinite(states).all() or not np.isfinite(actions).all()):
        raise ValueError('finite native states/actions and positive force spacing required')
    centered=np.array([8.,-8.,-1.,1.])/(12*eps)
    initial,controls,weights=[],[],[]
    for state,force in zip(states,actions):
        initial.append(state);controls.append(force);row=[]
        for axis in range(2):
            if force[axis]+2*eps>env.config.force_limit:
                offsets=(-1.,-2.,-3.,-4.);coef=-np.array([48.,-36.,16.,-3.])/(12*eps)
            elif force[axis]-2*eps < -env.config.force_limit:
                offsets=(1.,2.,3.,4.);coef=np.array([48.,-36.,16.,-3.])/(12*eps)
            else:
                offsets=(1.,-1.,2.,-2.);coef=centered
            for offset in offsets:
                u=force.copy();u[axis]+=offset*eps
                initial.append(state);controls.append(u)
            row.append(coef)
        weights.append(row)
    initial=np.c_[np.zeros(len(initial)),np.array(initial)]
    controls=np.repeat(np.asarray(controls)[:,None,:],env.config.frame_skip,axis=1)
    workers=getattr(env,'derivative_workers',1)
    data=[mj.MjData(env.model) for _ in range(workers)]
    with mujoco.rollout.Rollout(nthread=workers if workers>1 else 0) as batch:
        full,_=batch.rollout(env.model,data,initial,controls)
    times=np.diff(np.c_[np.zeros(len(initial)),full[:,:,0]],axis=1)
    env.native_steps+=int(np.count_nonzero(times>0))
    if not np.all(times>0) or not np.isfinite(full).all():
        raise RuntimeError('invalid native input derivative batch')
    results=full[:,-1,1:].reshape(count,9,-1)
    matrices=[]
    for row,coefs in zip(results,weights):
        columns=[]
        for axis in range(2):
            values=np.array([env.difference(row[1+4*axis+k],row[0]) for k in range(4)])
            columns.append(coefs[axis]@values)
        matrices.append(np.array(columns).T)
    return np.array(matrices),results[:,0].copy()
