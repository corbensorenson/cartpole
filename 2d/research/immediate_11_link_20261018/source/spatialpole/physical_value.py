"""Native value correction in physical direction/ALL angular-velocity features."""
import numpy as np
from .env import SpatialCartpole
from .control import BalancePolicy
from .world_coordinates import world_coordinate_error
from .aligned_value import relative_value_cost
from .rollout_value import force_proposals


class ParkedPhysicalNativePolicy:
    """Changed nonlinear error chart; the native plant retains every coordinate."""
    def __init__(self,env,payload):
        if not all(bool(payload.get(k,False)) for k in
                ('physical_native_value','world_coordinate_feedback','native_search')):
            raise ValueError('explicit physical native value markers required')
        if any(bool(payload.get(k,False)) for k in ('aligned_native_value','rollout_native_value',
                'controller_bank','protected_native_suffix','online_parking','align_cart_origin','online_gauge')):
            raise ValueError('physical native value is a separately identified policy')
        if 'launch_reference_profiles' in payload or np.any(payload.get('cart_reference_offset',np.zeros(2))):
            raise ValueError('physical native value does not combine other admission/origin policies')
        self.env=env;self.park_steps=int(payload['park_steps'])
        H=len(payload['actions']);n=env.config.n_links;ny=4+5*n
        if not 0<=self.park_steps<env.config.horizon or env.config.horizon-self.park_steps>H:
            raise ValueError('physical native value requires a fully covered execution prefix')
        shapes=dict(states=(H+1,env.model.nq+env.model.nv),actions=(H,2),feed=(H,2),
            world_gains=(H,2,ny),world_value_roots=(H+1,ny,ny),residuals=(H+1,ny),
            tracking_R=(2,2),world_reference_rotations=(H+1,n,3,3),world_reference_omega=(H+1,n,3))
        self.payload={}
        for key,shape in shapes.items():
            value=np.array(payload[key],copy=True)
            if value.shape!=shape or not np.isfinite(value).all():
                raise ValueError('finite compatible physical value arrays required: '+key)
            value.setflags(write=False);self.payload[key]=value
        self.parking=BalancePolicy(env,dict(reference=np.array(payload['parking_reference'],copy=True),
            K=np.array(payload['parking_K'],copy=True)))
        self.predictor=SpatialCartpole(env.config);self.trace=[]
        self.metadata=dict(policy='parked physical-coordinate native affine value',park_steps=self.park_steps,
            park_seconds=self.park_steps*env.config.dt,route_seconds=int(payload['swing_steps'])*env.config.dt,
            design_steps=H,execution_tracking_steps=env.config.horizon-self.park_steps,
            full_native_coordinates=2*env.model.nv,physical_tracking_coordinates=ny,
            all_body_angular_velocities_retained=True,online_axial_alignment=False,
            predictor_native_steps=0,coordinate_forward_queries=0,max_candidates=16,hardware_realtime_claim=False)

    def error(self,state,t):
        p=self.payload
        self.metadata['coordinate_forward_queries']+=1
        return world_coordinate_error(self.predictor,state,p['states'][t],
            p['world_reference_rotations'][t],p['world_reference_omega'][t])

    def __call__(self,state,step):
        if step<self.park_steps:
            return self.parking(state,step)
        t=step-self.park_steps;p=self.payload;oracle=self.predictor
        if not 0<=t<len(p['actions']):
            raise ValueError('physical native query outside declared reference')
        if not self.trace:
            self.metadata['launch_state']=state.tolist()
        raw_baseline=p['actions'][t]+p['feed'][t]-p['world_gains'][t]@self.error(state,t)
        baseline=oracle.deliver(raw_baseline)
        delta0=baseline.astype(float)-p['actions'][t]
        best,best_cost,selected=baseline.copy(),float('inf'),-1
        seen,records=set(),[];error0=None
        for raw in force_proposals(raw_baseline,baseline,p['actions'][t],lambda:best):
            action=oracle.deliver(raw);key=action.tobytes()
            if key in seen or len(records)>=16:
                continue
            seen.add(key);oracle.min_plane_margin,oracle.warning_seen=oracle.config.plane_limit,False
            successor=oracle.transition(state,action)
            margin,warning=oracle.min_plane_margin,bool(oracle.warning_seen)
            finite=bool(np.isfinite(successor).all())
            error=self.error(successor,t+1) if finite else np.full(p['world_gains'].shape[-1],np.nan)
            if not records and finite:
                error0=error
            valid=bool(finite and not warning and margin>=0 and error0 is not None)
            cost=float('inf')
            if valid:
                cost=relative_value_cost(p['world_value_roots'][t+1],p['residuals'][t+1],
                    error0,error,p['tracking_R'],delta0,action.astype(float)-p['actions'][t])
                valid=bool(np.isfinite(cost))
            records.append(dict(raw=np.asarray(raw,float).copy(),action=action.copy(),
                successor=successor.copy(),error=error.copy(),cost=cost,valid=valid,
                plane_margin=margin,physics_warning=warning))
            if valid and cost<best_cost:
                best,best_cost,selected=action.copy(),cost,len(records)-1
        self.trace.append(dict(step=t,candidates=records,selected=best.copy(),selected_index=selected,
            baseline_fallback=selected<0))
        self.metadata.update(predictor_native_steps=oracle.native_steps,native_search_commands=len(self.trace),
            baseline_fallback_commands=sum(r['baseline_fallback'] for r in self.trace))
        return best

    @property
    def forecast_tapes(self):
        records=[c for r in self.trace for c in r['candidates']]
        values={f'physical_search_{k}':np.asarray([r[k] for r in records]) for k in
            ('raw','action','successor','error','cost','valid','plane_margin','physics_warning')}
        values.update({f'physical_search_{k}':np.asarray([r[k] for r in self.trace]) for k in
            ('step','selected','selected_index','baseline_fallback')})
        values['physical_search_offsets']=np.r_[0,np.cumsum([len(r['candidates']) for r in self.trace])]
        return values

    def close(self):
        self.predictor.close()


def physical_value_policy_from_payload(env,payload):
    return ParkedPhysicalNativePolicy(env,payload)
