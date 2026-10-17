"""Sparse simultaneous state/control corrections; proposals are not executions."""
import mujoco as mj
import mujoco.rollout
import numpy as np
from scipy import sparse
from .control import difference_jacobian


def native_successors(env,states,actions):
    """Evaluate independent native control intervals, with exact step accounting."""
    workers = getattr(env,'derivative_workers',1)
    if not hasattr(env,'_fd_rollout'):
        env._fd_rollout = mujoco.rollout.Rollout(nthread=workers if workers>1 else 0)
        env._fd_data = [mj.MjData(env.model) for _ in range(workers)]
    initial = np.c_[np.zeros(len(states)),np.asarray(states)]
    controls = np.repeat(np.asarray(actions)[:,None,:],env.config.frame_skip,axis=1)
    full,_ = env._fd_rollout.rollout(env.model,env._fd_data,initial,controls)
    times = np.diff(np.c_[np.zeros(len(initial)),full[:,:,0]],axis=1)
    env.native_steps += int(np.count_nonzero(times>0))
    if not np.all(times>0) or not np.isfinite(full).all():
        raise RuntimeError('invalid native shooting interval')
    return full[:,-1,1:].copy()


def shooting_defects(env,states,actions,scales):
    predicted = native_successors(env,states[:-1],actions)
    defects = np.array([env.difference(f,x) for f,x in zip(predicted,states[1:])])/scales
    return defects,predicted


def shooting_jacobian(env,states,predicted,models,scales,fix_terminal=True):
    """Exact manifold defect linearization in scaled local increments.

    models are native A/B, BEFORE any virtual-gap transport. The initial state
    is fixed. Optionally fix the terminal node too. Control variables are N/80.
    """
    H,nx = len(models),2*env.model.nv
    nodes = H-1 if fix_terminal else H
    rows,columns,values = [],[],[]

    def block(matrix,row,col):
        i,j = np.nonzero(matrix)
        rows.append(i+row)
        columns.append(j+col)
        values.append(matrix[i,j])

    for t,((A,B),f,x) in enumerate(zip(models,predicted,states[1:])):
        jf = difference_jacobian(env,f,x)
        jn = difference_jacobian(env,x,f)
        if t:
            block((jf@A)*scales[None,:]/scales[:,None],t*nx,(t-1)*nx)
        if t<nodes:
            block(-jn*scales[None,:]/scales[:,None],t*nx,t*nx)
        block((jf@B)*env.config.force_limit/scales[:,None],t*nx,nodes*nx+2*t)
    return sparse.coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(columns))),
                             shape=(H*nx,nodes*nx+2*H)).tocsr()


def apply_shooting_step(env,states,actions,step,scales,alpha=1.,fix_terminal=True):
    """Retract simultaneous proposal nodes and project every proposal force."""
    H,nx = len(actions),2*env.model.nv
    nodes = H-1 if fix_terminal else H
    result = states.copy()
    for t in range(nodes):
        result[t+1] = env.integrate(states[t+1],alpha*step[t*nx:(t+1)*nx]*scales)
    controls = actions+alpha*step[nodes*nx:].reshape(H,2)*env.config.force_limit
    controls *= np.minimum(1.,env.config.force_limit/np.maximum(np.linalg.norm(controls,axis=1),1e-100))[:,None]
    return result,controls
