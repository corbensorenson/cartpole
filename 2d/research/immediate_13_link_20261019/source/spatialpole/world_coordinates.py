"""Physical rod-direction/body-angular-velocity feedback, with every spin retained."""
import mujoco as mj
import numpy as np
from .env import SpatialCartpole
from .control import BalancePolicy


def native_geometry(env,state):
    """Use an OWNED scratch environment; no integration or live plant mutation."""
    env.set_state(state)
    rotations=env.data.xmat[env.link_ids].reshape(-1,3,3).copy()
    omega=[]
    for body in env.link_ids:
        velocity=np.empty(6)
        mj.mj_objectVelocity(env.model,env.data,mj.mjtObj.mjOBJ_BODY,int(body),velocity,0)
        omega.append(velocity[:3].copy())
    return rotations,np.array(omega)


def world_coordinate_error(env,state,reference,reference_rotations,reference_omega):
    """A local physical chart, not a full nonlinear rewrite of quaternion LQR."""
    rotations,omega=native_geometry(env,state)
    observed,nominal=rotations[:,:,2],reference_rotations[:,:,2]
    local_delta=np.einsum('nji,nj->ni',reference_rotations,observed-nominal)
    cross=np.linalg.norm(np.cross(nominal,observed),axis=1)
    dot=np.sum(nominal*observed,axis=1)
    norms=np.linalg.norm(nominal,axis=1)*np.linalg.norm(observed,axis=1)
    sine=np.clip(cross/norms,0.,1.)
    angle=np.arctan2(cross,dot)
    ratio=np.ones(len(angle))
    moving=sine>1e-8
    ratio[moving]=angle[moving]/sine[moving]
    tilt=np.c_[-local_delta[:,1],local_delta[:,0]]*ratio[:,None]
    # The antipodal sphere log is ambiguous, but it never denotes zero tilt.
    tilt[(~moving)&(dot<0)]=[np.pi,0.]
    angular=np.einsum('nji,nj->ni',reference_rotations,omega-reference_omega)
    nq=env.model.nq
    return np.r_[state[:2]-reference[:2],tilt.ravel(),state[nq:nq+2]-reference[nq:nq+2],angular.ravel()]


def _cross_matrix(v):
    x,y,z=v
    return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])


def world_coordinate_jacobian(env,reference):
    """Analytic derivative of physical features in full native tangent coordinates."""
    n,nv,nq=env.config.n_links,env.model.nv,env.model.nq
    J=np.zeros((4+5*n,2*nv))
    J[:2,:2]=np.eye(2)
    J[2+2*n:4+2*n,nv:nv+2]=np.eye(2)
    rotation_error=np.zeros((3,2*nv))
    angular_error=np.zeros((3,2*nv))
    for i in range(n):
        R=np.empty(9)
        mj.mju_quat2Mat(R,reference[2+4*i:6+4*i])
        R=R.reshape(3,3)
        j=2+3*i
        rotation_error=R.T@rotation_error
        rotation_error[:,j:j+3]+=np.eye(3)
        angular_error=R.T@angular_error-_cross_matrix(reference[nq+j:nq+j+3])@rotation_error
        angular_error[:,nv+j:nv+j+3]+=np.eye(3)
        J[2+2*i:4+2*i]=rotation_error[:2]
        start=4+2*n+3*i
        J[start:start+3]=angular_error
    return J


def physical_tracking_factor(env,reference):
    """Local physical-error cost; full native velocities, including spin, survive."""
    n=env.config.n_links
    weights=np.r_[[4.,4.],np.full(2*n,40.),[.05,.05],np.tile([.05,.05,.005],n)]
    return np.sqrt(weights)[:,None]*world_coordinate_jacobian(env,reference)


class ParkedWorldCoordinatePolicy:
    """Changed nonlinear tracking law; original causal hanging policy is retained."""
    def __init__(self,env,payload):
        if not bool(payload.get('world_coordinate_feedback',False)):
            raise ValueError('explicit physical world-coordinate feedback marker required')
        if any(bool(payload.get(k,False)) for k in ('native_search','online_parking','align_cart_origin','rollout_native_value','aligned_native_value')) or 'launch_reference_profiles' in payload or np.any(payload.get('cart_reference_offset',np.zeros(2))):
            raise ValueError('physical-coordinate law does not combine search, offsets, online parking or admission')
        self.env,self.payload=env,payload
        self.park_steps=int(payload['park_steps'])
        self.scratch=SpatialCartpole(env.config)
        self.parking=BalancePolicy(env,dict(reference=payload['parking_reference'],K=payload['parking_K']))
        self.started=False
        H,n=len(payload['actions']),env.config.n_links
        if not 0<=self.park_steps<env.config.horizon or env.config.horizon-self.park_steps>H:
            self.scratch.close()
            raise ValueError('physical-coordinate policy requires a fully covered execution prefix')
        if (payload['world_gains'].shape!=(H,2,4+5*n) or
            payload['world_reference_rotations'].shape!=(H+1,n,3,3) or
            payload['world_reference_omega'].shape!=(H+1,n,3) or
            any(not np.isfinite(payload[k]).all() for k in ('world_gains','world_reference_rotations','world_reference_omega'))):
            self.scratch.close()
            raise ValueError('physical-coordinate payload dimensions do not match native model')
        # A caller may supply a compressed NpzFile. Read each required array
        # once and keep an immutable owned copy, avoiding per-command decoding.
        self.payload={k:np.array(payload[k],copy=True) for k in
            ('states','actions','feed','world_gains','world_reference_rotations','world_reference_omega')}
        for value in self.payload.values():
            value.setflags(write=False)
        self.metadata=dict(policy='parked spatial physical-coordinate affine tracking',park_steps=self.park_steps,
            park_seconds=self.park_steps*env.config.dt,route_seconds=int(payload['swing_steps'])*env.config.dt,
            design_steps=H,execution_tracking_steps=env.config.horizon-self.park_steps,
            full_native_coordinates=2*env.model.nv,physical_tracking_coordinates=4+5*n,
            all_body_angular_velocities_retained=True,frame_invariant_observations=True,
            online_axial_alignment=False,predictor_native_steps=0,coordinate_forward_queries=0,
            hardware_realtime_claim=False)

    def __call__(self,state,step):
        if step<self.park_steps:
            return self.parking(state,step)
        t=step-self.park_steps
        if not 0<=t<len(self.payload['actions']):
            raise ValueError('physical-coordinate query outside the frozen reference prefix')
        p=self.payload
        if not self.started:
            R,_=native_geometry(self.scratch,state)
            direction=R[:,:,2]
            self.metadata.update(launch_state=state.tolist(),launch_cart_xy=state[:2].tolist(),
                launch_max_tilt_from_down=float(max(np.arctan2(np.linalg.norm(direction[:,:2],axis=1),-direction[:,2]))))
            self.metadata['coordinate_forward_queries']+=1
            self.started=True
        error=world_coordinate_error(self.scratch,state,p['states'][t],p['world_reference_rotations'][t],p['world_reference_omega'][t])
        self.metadata['coordinate_forward_queries']+=1
        return p['actions'][t]+p['feed'][t]-p['world_gains'][t]@error

    def close(self):
        self.scratch.close()


def world_coordinate_policy_from_payload(env,payload):
    return ParkedWorldCoordinatePolicy(env,payload)
