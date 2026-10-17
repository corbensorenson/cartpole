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


def decode_planar_coordinates(coordinates,source_transform,n_links):
    """Invert the donor's declared physical-relative -> planner map.

    Different historical donors use different coordinates. An identity map
    means physical relative angles/rates; a cumulative/scaled map must be
    inverted as stored, not inferred from the link count or a scale file.
    """
    coordinates,source_transform = np.asarray(coordinates,float),np.asarray(source_transform,float)
    width = 2*(n_links+1)
    if coordinates.ndim!=2 or coordinates.shape[1]!=width or source_transform.shape!=(width,width):
        raise ValueError('source coordinate state/transform dimensions disagree')
    if not np.isfinite(coordinates).all() or not np.isfinite(source_transform).all():
        raise ValueError('nonfinite source coordinate state or transform')
    try:
        return np.linalg.solve(source_transform,coordinates.T).T
    except np.linalg.LinAlgError as exc:
        raise ValueError('singular source coordinate transform') from exc


def declared_coordinate_projection(env,source_transform):
    """Project spatial tangent errors through the explicit donor convention."""
    n,nv = env.config.n_links,env.model.nv
    physical = np.zeros((2*(n+1),2*nv))
    physical[0,0],physical[n+1,nv] = 1.,1.
    for i in range(n):
        physical[1+i,3+3*i] = 1.
        physical[n+2+i,nv+3+3*i] = 1.
    transform = np.asarray(source_transform,float)
    if transform.shape!=(2*(n+1),2*(n+1)) or not np.isfinite(transform).all():
        raise ValueError('invalid source coordinate transform for feedback')
    return transform@physical


def regrid_planar_material_states(physical,source_lengths,target_lengths):
    """Interpolate cumulative angle/rate at material centers, preserve cart.

    Inputs and outputs are physical relative q/v, with unwrapped angle branches
    retained. This is a geometric warm start; target dynamics must be rebuilt.
    """
    physical = np.asarray(physical,float)
    source_lengths,target_lengths = np.asarray(source_lengths,float),np.asarray(target_lengths,float)
    for lengths in (source_lengths,target_lengths):
        if lengths.ndim!=1 or len(lengths)<1 or not np.isfinite(lengths).all() or np.any(lengths<=0):
            raise ValueError('material lengths must be finite and positive')
    ns,nt = len(source_lengths),len(target_lengths)
    if physical.ndim!=2 or physical.shape[1]!=2*(ns+1) or not np.isfinite(physical).all():
        raise ValueError('invalid physical source states')
    source_centers = (np.cumsum(source_lengths)-source_lengths/2)/np.sum(source_lengths)
    target_centers = (np.cumsum(target_lengths)-target_lengths/2)/np.sum(target_lengths)
    angles = np.cumsum(physical[:,1:ns+1],axis=1)
    rates = np.cumsum(physical[:,ns+2:],axis=1)
    new_angles = np.array([np.interp(target_centers,source_centers,row) for row in angles])
    new_rates = np.array([np.interp(target_centers,source_centers,row) for row in rates])
    out = np.zeros((len(physical),2*(nt+1)))
    out[:,0],out[:,nt+1] = physical[:,0],physical[:,ns+1]
    out[:,1:nt+1] = np.diff(np.c_[np.zeros(len(physical)),new_angles],axis=1)
    out[:,nt+2:] = np.diff(np.c_[np.zeros(len(physical)),new_rates],axis=1)
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


def split_reference_states(source_env,target_env,states,split_index):
    """Insert a zero-relative-rotation reference body, never a joint constraint.

    This duplicates one source body's world orientation/rate continuously in
    time. Uniform target lengths/masses are regenerated independently, so the
    resulting proposal generally has native dynamics defects to repair.
    """
    n = source_env.config.n_links
    if target_env.config.n_links != n+1 or not 0 <= split_index < n:
        raise ValueError('split requires one additional target link and a valid source index')
    states = np.asarray(states)
    result = np.repeat(target_env.upright_state()[None],len(states),axis=0)
    result[:,:2] = states[:,:2]
    result[:,target_env.model.nq:target_env.model.nq+2] = states[:,source_env.model.nq:source_env.model.nq+2]
    target = 0
    for i in range(n):
        result[:,2+4*target:6+4*target] = states[:,2+4*i:6+4*i]
        start,old = target_env.model.nq+2+3*target,source_env.model.nq+2+3*i
        result[:,start:start+3] = states[:,old:old+3]
        target += 1
        if i==split_index:
            target += 1  # identity quaternion and zero angular rate already set
    return result
