"""Batched spatial kinematics for optimizer residuals; native physics is authority."""
import numpy as np


def world_errors(env, states):
    """World tilt/rate coordinates matching native WorldBalancePolicy.error.

    Ball velocities are child-frame angular velocities. Compose orientations,
    sum world angular velocities, and differentiate the rod's unit direction.
    Axial spin is free, while all transverse physical motion remains visible.
    """
    states = np.atleast_2d(states)
    count, nv, nq = len(states), env.model.nv, env.model.nq
    error = np.zeros((count,2*nv))
    error[:,:2], error[:,nv:nv+2] = states[:,:2], states[:,nq:nq+2]
    rotation = np.broadcast_to(np.eye(3),(count,3,3)).copy()
    omega = np.zeros((count,3))
    previous_tilt, previous_rate = np.zeros((count,2)), np.zeros((count,2))
    for i in range(env.config.n_links):
        w,x,y,z = states[:,2+4*i:6+4*i].T
        local = np.empty((count,3,3))
        local[:,0,0],local[:,0,1],local[:,0,2] = 1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)
        local[:,1,0],local[:,1,1],local[:,1,2] = 2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)
        local[:,2,0],local[:,2,1],local[:,2,2] = 2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)
        rotation = rotation@local
        j = 2+3*i
        omega += np.einsum('bij,bj->bi',rotation,states[:,nq+j:nq+j+3])
        direction = rotation[:,:,2]
        dd = np.cross(omega,direction)
        dx,dy,dz = direction.T
        radius = np.hypot(dx,dy)
        safe = np.maximum(radius,1e-7)
        angle = np.arctan2(radius,dz)
        rd = (dx*dd[:,0]+dy*dd[:,1])/safe
        ad = dz*rd-radius*dd[:,2]
        v = np.c_[-dy,dx]
        tilt = v*(angle/safe)[:,None]
        rate = np.c_[-dd[:,1],dd[:,0]]*(angle/safe)[:,None]+v*(ad/safe-angle*rd/safe**2)[:,None]
        at_pole = radius < 1e-7
        tilt[at_pole] = v[at_pole]
        tilt[at_pole & (dz<0)] = [np.pi,0.]
        rate[at_pole] = np.c_[-dd[at_pole,1],dd[at_pole,0]]
        error[:,j:j+2], error[:,nv+j:nv+j+2] = tilt-previous_tilt,rate-previous_rate
        previous_tilt,previous_rate = tilt,rate
    return error


def world_cost_derivatives(env, states, weight, eps=1e-5):
    """Batched centered tangent secants and Gauss-Newton cost Hessians."""
    states = np.atleast_2d(states)
    nx, width = 2*env.model.nv, 1+4*env.model.nv
    queries = []
    for state in states:
        queries.append(state)
        for j in range(nx):
            delta = np.zeros(nx)
            delta[j] = eps
            queries.extend([env.integrate(state,delta),env.integrate(state,-delta)])
    residual = world_errors(env,np.array(queries)).reshape(len(states),width,nx)
    jac = np.transpose((residual[:,1::2]-residual[:,2::2])/(2*eps),(0,2,1))
    grads = np.einsum('bji,jk,bk->bi',jac,weight,residual[:,0])
    hess = np.einsum('bji,jk,bkl->bil',jac,weight,jac)
    return grads,hess
