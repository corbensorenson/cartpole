"""Axial-frame quotient for planning; the native plant retains every DOF."""
import mujoco as mj
import numpy as np
from .symmetry import reflection_signs, reflect_state
from .affine_value import reflection_tracking_values,affine_tracking_values


def axial_quotient_frame(env,state,planar=False):
    """Differentiate observed polar alignment without forwarding native data.

    D spans constant body axial-frame relabelings. C measures the corresponding
    infinitesimal world-body polar angles. Velocity coordinates are not deleted:
    D includes only their change of local reference basis under frame relabeling.
    """
    nv,nq,n = env.model.nv,env.model.nq,env.config.n_links
    nx = 2*nv
    state = np.asarray(state,float)
    if state.shape!=(nq+nv,) or not np.isfinite(state).all():
        raise ValueError('finite full native state required')
    C,D = np.zeros((n,nx)),np.zeros((nx,n))
    world_body_error = np.zeros((3,nx))
    z = np.array([0.,0.,1.])
    for i in range(n):
        local = np.empty(9)
        mj.mju_quat2Mat(local,state[2+4*i:6+4*i])
        local = local.reshape(3,3)
        j = 2+3*i
        world_body_error = local.T@world_body_error
        world_body_error[:,j:j+3] += np.eye(3)
        C[i] = world_body_error[2]
        D[j:j+3,i] = z
        if i:
            D[j:j+3,i-1] = -local.T@z
        D[nv+j:nv+j+3,i] = -np.cross(z,state[nq+j:nq+j+3])
    G = np.eye(nx)-D@C
    if planar:
        if not np.array_equal(reflect_state(env,state),state):
            raise ValueError('separated quotient requires an exactly reflection-fixed reference')
        even,odd = np.flatnonzero(reflection_signs(env)>0),np.flatnonzero(reflection_signs(env)<0)
        if np.any(C[:,even]) or np.any(D[even]):
            raise ValueError('axial frame generators must lie in the odd planning branch')
        q,_ = np.linalg.qr(C[:,odd].T,mode='complete')
        E = np.zeros((nx,nx-n))
        E[even,np.arange(len(even))] = 1.
        E[np.ix_(odd,np.arange(len(even),nx-n))] = q[:,n:]
        signs = np.r_[np.ones(len(even)),-np.ones(len(odd)-n)]
    else:
        q,_ = np.linalg.qr(C.T,mode='complete')
        E,signs = q[:,n:],None
    F = E.T@G
    return dict(C=C,D=D,G=G,E=E,F=F,signs=signs)


def quotient_tracking_values(models,defects,factors,R,terminal_factor,frames):
    """Finite affine QR in quotient charts, then lift force/value to full errors."""
    H = len(models)
    if len(frames)!=H+1:
        raise ValueError('one quotient frame per reference state required')
    separated=frames[0]['signs'] is not None
    if any((f['signs'] is not None)!=separated for f in frames):
        raise ValueError('cannot mix general and reflection-separated quotient frames')
    reduced = [(frames[t+1]['F']@A@frames[t]['E'],frames[t+1]['F']@B)
        for t,(A,B) in enumerate(models)]
    reduced_defects = np.array([frames[t+1]['F']@d for t,d in enumerate(defects)])
    reduced_factors = [L@frames[t]['E'] for t,L in enumerate(factors)]
    if separated:
        feed,K,S,r = reflection_tracking_values(reduced,reduced_defects,reduced_factors,R,
            terminal_factor@frames[-1]['E'],frames[0]['signs'])
    else:
        feed,K,S,r = affine_tracking_values(reduced,reduced_defects,reduced_factors,R,
            terminal_factor@frames[-1]['E'])
    lifted_K = np.array([k@frames[t]['F'] for t,k in enumerate(K)])
    lifted_S = np.array([s@frames[t]['F'] for t,s in enumerate(S)])
    return dict(feed=feed,gains=lifted_K,roots=lifted_S,residuals=r,
        reduced_gains=K,reduced_roots=S,reduced_models=reduced,
        reduced_defects=reduced_defects,reduced_factors=reduced_factors)
