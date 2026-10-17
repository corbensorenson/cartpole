"""Causal hanging contraction and passive axial-frame alignment at launch."""
import mujoco as mj
import numpy as np
from .control import BalancePolicy,RoutePolicy


def axial_gauge_state(env,state,angles):
    """Change each axisymmetric body's local axial frame, preserving motion."""
    out = state.copy()
    parent = np.array([1.,0.,0.,0.])
    nq,nv = env.model.nq,env.model.nv
    for i,angle in enumerate(angles):
        child = np.array([np.cos(angle/2),0.,0.,np.sin(angle/2)])
        parent_inverse = parent.copy()
        parent_inverse[1:] *= -1
        temp,quat = np.zeros(4),np.zeros(4)
        mj.mju_mulQuat(temp,parent_inverse,state[2+4*i:6+4*i])
        mj.mju_mulQuat(quat,temp,child)
        out[2+4*i:6+4*i] = quat
        rotation = np.zeros(9)
        mj.mju_quat2Mat(rotation,child)
        j = 2+3*i
        out[nq+j:nq+j+3] = rotation.reshape(3,3).T@state[nq+j:nq+j+3]
        parent = child
    return out


def axial_gauge_route(env,states,gains,angles):
    out = np.array([axial_gauge_state(env,x,angles) for x in states])
    updated = gains.copy()
    for i,angle in enumerate(angles):
        c,s = np.cos(angle),np.sin(angle)
        rotation = np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
        j = 2+3*i
        for index in (j,env.model.nv+j):
            updated[:,:,index:index+3] = gains[:,:,index:index+3]@rotation
    return out,updated


def launch_gauge_angles(env,reference,state):
    env.set_state(reference)
    nominal = env.data.xmat[env.link_ids].reshape(-1,3,3).copy()
    env.set_state(state)
    actual = env.data.xmat[env.link_ids].reshape(-1,3,3)
    relative = np.transpose(nominal,(0,2,1))@actual
    return np.arctan2(relative[:,1,0],relative[:,0,0])


class ParkedRoutePolicy:
    def __init__(self,env,states,actions,gains,balance,parking,park_steps,align=True):
        self.env,self.states,self.actions,self.gains,self.balance = env,states,actions,gains,balance
        self.parking = BalancePolicy(env,parking)
        self.park_steps,self.align,self.route = park_steps,align,None
        self.metadata = {'park_steps':park_steps,'park_seconds':park_steps*env.config.dt,
                         'route_steps':len(actions),'route_seconds':len(actions)*env.config.dt,'align_axial_frames':align}

    def __call__(self,state,step):
        if step < self.park_steps:
            return self.parking(state,step)
        if self.route is None:
            angles = launch_gauge_angles(self.env,self.states[0],state) if self.align else np.zeros(self.env.config.n_links)
            states,gains = axial_gauge_route(self.env,self.states,self.gains,angles)
            self.route = RoutePolicy(self.env,states,self.actions,gains,self.balance,True)
            self.env.set_state(state)
            self.metadata.update(launch_state=state.tolist(),gauge_angles=angles.tolist(),
                launch_max_tilt_from_down=float(np.max(np.pi-self.env.tilts())),
                launch_cart_xy=state[:2].tolist())
        return self.route(state,step-self.park_steps)
