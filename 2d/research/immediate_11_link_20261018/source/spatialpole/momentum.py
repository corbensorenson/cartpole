"""Cart-force yaw-momentum damping during preparation; all native DOFs survive."""
import mujoco as mj
import numpy as np
from .env import SpatialCartpole


def yaw_momentum(env,state):
    """COM-relative generalized momentum, using an OWNED scratch environment."""
    env.set_state(state)
    nv=env.model.nv
    mass=np.empty((nv,nv));mj.mj_fullM(env.model,mass,env.data.qM)
    weights=env.model.body_mass
    com=(weights[:,None]*env.data.xipos).sum(axis=0)/sum(weights)
    lever=np.asarray(state[:2])-com[:2]
    perpendicular=np.array([-lever[1],lever[0]])
    generator=np.zeros(nv);generator[:2]=perpendicular
    generator[2:5]=env.data.xmat[env.link_ids[0]].reshape(3,3)[2]
    return float(generator@mass@state[env.model.nq:]),lever,perpendicular,generator


class MomentumParkingPolicy:
    """Add only a two-force preparation term; delegate the frozen tracking law."""
    def __init__(self,env,payload):
        if not bool(payload.get('momentum_parking',False)) or not bool(payload.get('aligned_native_value',False)):
            raise ValueError('explicit momentum preparation and aligned native tracking required')
        if any(bool(payload.get(k,False)) for k in ('physical_native_value','world_coordinate_feedback',
                'rollout_native_value','controller_bank','online_parking','align_cart_origin')):
            raise ValueError('momentum preparation does not combine other controller families')
        self.rate=float(payload['momentum_rate'])
        self.floor=float(payload['momentum_lever_floor'])
        if not np.isfinite(self.rate) or not 0<self.rate<=20 or not np.isfinite(self.floor) or self.floor<=0:
            raise ValueError('finite positive momentum parameters and rate<=20/s required')
        from .aligned_value import aligned_value_policy_from_payload
        original=dict(payload);original.pop('momentum_parking')
        self.policy=aligned_value_policy_from_payload(env,original)
        self.env=env;self.park_steps=int(payload['park_steps'])
        self.scratch=SpatialCartpole(env.config);self.parking_trace=[]
        self.metadata=self.policy.metadata
        self.metadata.update(momentum_parking=True,momentum_rate=self.rate,momentum_lever_floor=self.floor,
            parking_geometry_queries=0,original_parking_replaced=True,original_prefix_byte_parity_claim=False,
            protected_prefix_scope='tracking coefficients and next-value boundary; preparation changed')

    def __call__(self,state,step):
        baseline=self.policy(state,step)
        if step>=self.park_steps:
            return baseline
        h,lever,perpendicular,_=yaw_momentum(self.scratch,state)
        delta=-self.rate*h*perpendicular/(lever@lever+self.floor**2)
        # Keep the original signed-zero arithmetic when the correction is zero.
        raw=baseline if h==0 else baseline+delta
        self.parking_trace.append(dict(step=step,momentum=h,lever=lever.copy(),delta=delta.copy(),
            baseline=np.asarray(baseline).copy(),delivered=self.env.deliver(raw)))
        self.metadata['parking_geometry_queries']+=1
        assert self.scratch.native_steps==0
        return raw

    @property
    def forecast_tapes(self):
        values=dict(self.policy.forecast_tapes)
        values.update({f'momentum_parking_{k}':np.asarray([r[k] for r in self.parking_trace])
            for k in ('step','momentum','lever','delta','baseline','delivered')})
        return values

    def close(self):
        self.policy.close();self.scratch.close()
