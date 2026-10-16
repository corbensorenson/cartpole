"""Lift planar route proposals into an unrestricted spatial model."""
import numpy as np
from .control import linearize_many, state_cost
from .physical import world_cost_derivatives


def lift_planar_state(env,qpos,qvel):
    n,nq,nv = env.config.n_links,env.model.nq,env.model.nv
    qpos,qvel = np.asarray(qpos),np.asarray(qvel)
    if qpos.shape != (n+1,) or qvel.shape != (n+1,):
        raise ValueError("planar state has the wrong count")
    out = env.upright_state()
    out[0],out[nq] = qpos[0],qvel[0]
    for i in range(n):
        angle = qpos[i+1]
        out[2+4*i:6+4*i] = [np.cos(angle/2),0,np.sin(angle/2),0]
        out[nq+3+3*i] = qvel[i+1]
    return out


def coordinate_projection(env,cart_position=1.25,angle=0.15,cart_velocity=0.5,joint_velocity=0.75):
    """Tangent projection for the old dimensionless-absolute transform.

    Its ANGLES are cumulative, while its hinge VELOCITIES are relative. Keep
    that historical distinction; treating both as cumulative corrupts feedback.
    """
    n,nv = env.config.n_links,env.model.nv
    transform = np.zeros((2*(n+1),2*nv))
    transform[0,0],transform[n+1,nv] = 1/cart_position,1/cart_velocity
    for i in range(n):
        for j in range(i+1):
            transform[1+i,3+3*j] = 1/angle
        transform[n+2+i,nv+3+3*i] = 1/joint_velocity
    return transform


def spatial_tracking_gains(env,states,actions,design):
    """Finite-horizon feedback on the spatial native model around a proposal."""
    models = linearize_many(env,states[:-1],actions)
    Q,R = state_cost(env),np.eye(2)*0.01
    _,hess = world_cost_derivatives(env,states[-1:],design['P'])
    V = hess[0]
    gains = np.zeros((len(actions),2,2*env.model.nv))
    for t in range(len(actions)-1,-1,-1):
        A,B = models[t]
        K = np.linalg.solve(R+B.T@V@B,B.T@V@A)
        gains[t] = K
        V = Q+K.T@R@K+(A-B@K).T@V@(A-B@K)
        V = (V+V.T)/2
    return gains
