"""Opt-in axial-aligned native action improvement with retained affine values."""
import math
import numpy as np
from .env import SpatialCartpole
from .control import BalancePolicy
from .affine_value import AffineValuePolicy, axial_aligned_affine_force
from .conditioning import axial_gauge_route, launch_gauge_angles


def relative_value_cost(root, residual, baseline_error, error, R, baseline_delta, delta):
    """Difference of full affine value and input cost in one COMMON chart."""
    w = root @ baseline_error + residual
    dv = root @ (error-baseline_error)
    return math.fsum(((2*w+dv)*dv).tolist()) + float((delta-baseline_delta) @ R @ (delta+baseline_delta))


def aligned_next_value(env, reference, root, observed):
    """Closest axial frame; only the owned scratch environment is forwarded."""
    env.set_state(reference)
    nominal = env.data.xmat[env.link_ids].reshape(-1,3,3).copy()
    env.set_state(observed)
    actual = env.data.xmat[env.link_ids].reshape(-1,3,3)
    relative = np.transpose(nominal,(0,2,1)) @ actual
    angles = np.arctan2(relative[:,1,0]-relative[:,0,1], relative[:,0,0]+relative[:,1,1])
    refs, roots = axial_gauge_route(env, reference[None,:], root[None,:,:], angles)
    return refs[0], roots[0], angles


class AlignedNativeValuePolicy(AffineValuePolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, search=True)
        # All coordinate scratch work, including the baseline, uses this copy.
        self.balance.env = self.predictor
        self.metadata.update(policy='spatial aligned native affine value', online_axial_alignment=True,
                             common_next_value_frame='baseline predicted successor', hardware_realtime_claim=False)

    def __call__(self, state, step):
        if step >= len(self.actions):
            return self.balance(state)
        p = self.predictor
        raw_baseline = axial_aligned_affine_force(p,state,self.states[step],
            self.actions[step],self.feed[step],self.gains[step])
        baseline = p.deliver(raw_baseline)
        seen, records = set(), []
        best, best_cost, selected = baseline.copy(), float('inf'), -1
        next_ref = next_root = anchor_angles = error0 = None
        delta0 = baseline.astype(float)-self.actions[step]

        def query(raw):
            nonlocal best, best_cost, selected, next_ref, next_root, anchor_angles, error0
            action = p.deliver(raw)
            key = action.tobytes()
            if key in seen or len(records)>=16:
                return
            seen.add(key)
            p.min_plane_margin, p.warning_seen = p.config.plane_limit, False
            successor = p.transition(state,action)
            margin, warning = p.min_plane_margin, p.warning_seen
            finite = np.isfinite(successor).all()
            if not records and finite:
                next_ref,next_root,anchor_angles = aligned_next_value(p,self.states[step+1],self.roots[step+1],successor)
                error0 = p.difference(successor,next_ref)
            valid = bool(finite and not warning and margin>=0 and error0 is not None)
            cost = float('inf')
            if valid:
                cost = relative_value_cost(next_root,self.residuals[step+1],error0,
                    p.difference(successor,next_ref),self.R,delta0,action.astype(float)-self.actions[step])
                valid = bool(np.isfinite(cost))
            records.append(dict(raw=np.asarray(raw,float).copy(),action=action.copy(),successor=successor.copy(),
                cost=cost,valid=valid,plane_margin=margin,physics_warning=bool(warning)))
            if valid and cost<best_cost:
                best,best_cost,selected = action.copy(),cost,len(records)-1

        query(raw_baseline)
        query(self.actions[step])
        query((baseline.astype(float)+self.actions[step])/2)
        ulps = np.maximum(abs(np.nextafter(baseline,np.float32(np.inf)).astype(float)-baseline),
                          abs(np.nextafter(baseline,np.float32(-np.inf)).astype(float)-baseline))
        radii = np.maximum.reduce([abs(baseline.astype(float)-self.actions[step]),8*ulps,np.full(2,1e-4)])
        for scale in (1.,.5):
            center = best.astype(float).copy()
            for axis in range(2):
                for sign in (1.,-1.):
                    proposal = center.copy()
                    proposal[axis] += sign*scale*radii[axis]
                    query(proposal)
        center = best.copy()
        for axis in range(2):
            for direction in (np.float32(np.inf),np.float32(-np.inf)):
                proposal = center.copy()
                proposal[axis] = np.nextafter(proposal[axis],direction)
                query(proposal)
        query((best.astype(float)+baseline.astype(float))/2)
        self.trace.append(dict(step=step,candidates=records,selected=best.copy(),selected_index=selected,
            baseline_fallback=selected<0,anchor_angles=anchor_angles if anchor_angles is not None else np.full(p.config.n_links,np.nan)))
        self.metadata['predictor_native_steps'] = p.native_steps
        self.metadata['native_search_commands'] = len(self.trace)
        self.metadata['baseline_fallback_commands'] = sum(r['baseline_fallback'] for r in self.trace)
        return best

    @property
    def forecast_tapes(self):
        records = [c for row in self.trace for c in row['candidates']]
        values = {f'native_search_{name}':np.asarray([c[name] for c in records])
            for name in ('raw','action','successor','cost','valid','plane_margin','physics_warning')}
        values.update({f'native_search_{name}':np.asarray([r[name] for r in self.trace])
            for name in ('step','selected','selected_index','baseline_fallback','anchor_angles')})
        values['native_search_offsets'] = np.r_[0,np.cumsum([len(r['candidates']) for r in self.trace])]
        return values


class ParkedAlignedNativeValuePolicy:
    def __init__(self, env, payload):
        if not all(bool(payload.get(k,False)) for k in ('aligned_native_value','native_search','online_gauge','affine_value','align_gauge')):
            raise ValueError('aligned native value requires explicit affine/search/online/launch alignment markers')
        if bool(payload.get('reflection_averaged_feedback',False)):
            raise ValueError('postfit averaged feedback has no matching native search value roots')
        if any(bool(payload.get(k,False)) for k in ('online_parking','align_cart_origin')) or 'launch_reference_profiles' in payload or np.any(payload.get('cart_reference_offset',np.zeros(2))):
            raise ValueError('aligned native value does not support parking alignment, origins, offsets or admission')
        self.env,self.payload,self.route = env,payload,None
        self.park_steps = int(payload['park_steps'])
        if not 0<=self.park_steps<env.config.horizon or env.config.horizon-self.park_steps>len(payload['actions']):
            raise ValueError('aligned native value requires an entirely covered tracking prefix')
        self.parking = BalancePolicy(env,dict(reference=payload['parking_reference'],K=payload['parking_K']))
        self.metadata = dict(policy='parked spatial aligned native affine value',park_steps=self.park_steps,
            park_seconds=self.park_steps*env.config.dt,route_seconds=int(payload['swing_steps'])*env.config.dt,
            design_steps=len(payload['actions']),execution_tracking_steps=env.config.horizon-self.park_steps,
            online_axial_alignment=True,align_axial_frames=True,predictor_native_steps=0,max_candidates=16,
            hardware_realtime_claim=False)

    def __call__(self,state,step):
        if step<self.park_steps:
            return self.parking(state,step)
        if self.route is None:
            p,e = self.payload,self.env
            scratch = SpatialCartpole(e.config)
            try:
                angles = launch_gauge_angles(scratch,p['states'][0],state)
                states,gains = axial_gauge_route(scratch,p['states'],p['gains'],angles)
                _,roots = axial_gauge_route(scratch,p['states'],p['roots'],angles)
            finally:
                scratch.close()
            design = {k:p[k] for k in ('reference','K','P','Q','R')}
            self.route = AlignedNativeValuePolicy(e,states,p['actions'],p['feed'],gains,roots,
                p['residuals'],p['tracking_R'],design)
            self.metadata.update(launch_state=state.tolist(),gauge_angles=angles.tolist())
        action = self.route(state,step-self.park_steps)
        self.metadata.update({k:self.route.metadata[k] for k in ('predictor_native_steps','native_search_commands','baseline_fallback_commands')})
        return action

    @property
    def forecast_tapes(self):
        return self.route.forecast_tapes if self.route is not None else {}

    def close(self):
        if self.route is not None:
            self.route.close()


def aligned_value_policy_from_payload(env,payload):
    return ParkedAlignedNativeValuePolicy(env,payload)
