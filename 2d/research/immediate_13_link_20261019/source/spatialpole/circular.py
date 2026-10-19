"""Offline physical coordinates for native circular-orbit synthesis."""
import mujoco as mj
import numpy as np


def state_from_physical(env, cart_xy, tilt_vectors, cart_velocity, body_omega):
    """Choose axial pose frames; preserve every supplied world angular velocity."""
    n = env.config.n_links
    cart_xy, tilt_vectors, cart_velocity, body_omega = [np.asarray(x, float) for x in
        (cart_xy, tilt_vectors, cart_velocity, body_omega)]
    if (cart_xy.shape != (2,) or tilt_vectors.shape != (n,2) or cart_velocity.shape != (2,)
            or body_omega.shape != (n,3) or not all(np.isfinite(x).all() for x in
                (cart_xy,tilt_vectors,cart_velocity,body_omega))):
        raise ValueError('finite cart, world tilt and full body angular velocities required')
    if np.any(np.linalg.norm(tilt_vectors,axis=1) >= np.pi):
        raise ValueError('physical tilt chart excludes the antipode')
    state = np.zeros(env.model.nq+env.model.nv)
    state[:2] = cart_xy
    state[env.model.nq:env.model.nq+2] = cart_velocity
    parent_quat = np.array([1.,0.,0.,0.]); parent_omega = np.zeros(3)
    for i,(tilt,omega) in enumerate(zip(tilt_vectors,body_omega)):
        radius = np.linalg.norm(tilt)
        factor = .5*np.sinc(radius/(2*np.pi))
        quat = np.array([np.cos(radius/2),-factor*tilt[1],factor*tilt[0],0.])
        inverse_parent = parent_quat.copy(); inverse_parent[1:] *= -1
        mj.mju_mulQuat(state[2+4*i:6+4*i],inverse_parent,quat)
        rotation = np.empty(9); mj.mju_quat2Mat(rotation,quat)
        j = env.model.nq+2+3*i
        state[j:j+3] = rotation.reshape(3,3).T@(omega-parent_omega)
        parent_quat,parent_omega = quat,omega
    return state


def rotate_world_azimuth(env, state, angle):
    """Rotate physical world coordinates; local joint velocities stay intact."""
    if not np.isfinite(angle):
        raise ValueError('finite world azimuth required')
    c,s = np.cos(angle),np.sin(angle)
    rotation = np.array([[c,-s],[s,c]])
    result = np.array(state,copy=True)
    result[:2] = rotation@state[:2]
    nq = env.model.nq
    result[nq:nq+2] = rotation@state[nq:nq+2]
    quat = np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)])
    mj.mju_mulQuat(result[2:6],quat,state[2:6])
    return result
