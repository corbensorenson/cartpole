"""Reflection diagnostics and roundoff cleanup for planar proposal references.

These functions never constrain or change the physical simulator state.
"""
import numpy as np


def reflection_signs(env):
    signs = np.ones(2*env.model.nv)
    for offset in (0,env.model.nv):
        signs[offset+1] = -1
        for i in range(env.config.n_links):
            signs[offset+2+3*i] = signs[offset+4+3*i] = -1
    return signs


def reflect_state(env,state):
    """Reflect Y -> -Y, including axial-vector angular velocity signs."""
    out = np.array(state,float).copy()
    out[1] *= -1
    for i in range(env.config.n_links):
        out[3+4*i] *= -1
        out[5+4*i] *= -1
    out[env.model.nq:] *= reflection_signs(env)[env.model.nv:]
    return out


def reflection_average_feedback(env,states,actions,gains,feed):
    """Average the fitted force law under Y reflection; never edit the plant."""
    states,actions,gains,feed = [np.asarray(x,float) for x in (states,actions,gains,feed)]
    H,nx = len(actions),2*env.model.nv
    if (states.shape!=(H+1,env.model.nq+env.model.nv) or actions.shape!=(H,2)
            or gains.shape!=(H,2,nx) or feed.shape!=(H,2)
            or not all(np.isfinite(x).all() for x in (states,actions,gains,feed))):
        raise ValueError('finite compatible affine reference/feedback arrays required')
    if np.any(actions[:,1]!=0) or any(not np.array_equal(reflect_state(env,x),x) for x in states):
        raise ValueError('reflection averaging requires a reflection-fixed reference')
    parity = np.array([1.,-1.])[:,None]*reflection_signs(env)[None,:]
    new_gains = np.where(parity[None,:,:]>0,gains,0.)
    new_feed = feed.copy()
    new_feed[:,1] = 0.
    return new_gains,new_feed,dict(
        max_gain_change=float(np.max(np.abs(new_gains-gains))),
        max_feed_change=float(np.max(np.abs(new_feed-feed))),
        scope='reflection-averaged affine force law only; all spatial physics and observed states retained')


def clean_planar_reference(env,states,actions,tolerance=1e-8):
    """Remove tiny off-plane proposal components; reject substantive changes.

    tolerance bounds normalized local state changes and force changes/80N.
    It cannot flatten a genuinely spatial reference without raising an error.
    """
    states,actions = np.asarray(states,float),np.asarray(actions,float)
    if states.shape!=(len(actions)+1,env.model.nq+env.model.nv) or actions.shape!=(len(actions),2):
        raise ValueError('invalid reference dimensions')
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError('nonfinite reference')
    xs,us = states.copy(),actions.copy()
    xs[:,1] = 0.
    for i in range(env.config.n_links):
        j = 2+4*i
        norm = np.sqrt(xs[:,j]**2+xs[:,j+2]**2)
        if np.any(norm<1e-12):
            raise ValueError('reference is not nearly planar')
        xs[:,j:j+4] /= norm[:,None]
        xs[:,j+1] = xs[:,j+3] = 0.
    odd = reflection_signs(env)[env.model.nv:]<0
    xs[:,env.model.nq:][:,odd] = 0.
    us[:,1] = 0.
    nv = env.model.nv
    scales = np.r_[[3.,3.],np.full(nv-2,.15),[4.,4.],np.full(nv-2,15.)]
    changes = np.array([env.difference(x,y) for x,y in zip(xs,states)])/scales
    state_change = float(np.max(np.abs(changes)))
    force_change = float(np.max(np.abs(us-actions)))/env.config.force_limit
    if max(state_change,force_change)>tolerance:
        raise ValueError('reference cleanup exceeds declared roundoff tolerance')
    return xs,us,dict(max_normalized_state_change=state_change,
        max_normalized_force_change=force_change,tolerance=tolerance,
        scope='proposal/reference only; actual full spatial states and physics are untouched')
