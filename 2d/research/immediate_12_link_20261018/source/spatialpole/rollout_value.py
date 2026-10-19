"""Bounded native lookahead with plain future feedback and full affine values."""
import math
import numpy as np
from .aligned_value import AlignedNativeValuePolicy,ParkedAlignedNativeValuePolicy,aligned_next_value
from .affine_value import axial_aligned_affine_force
from .conditioning import launch_gauge_angles,axial_gauge_route
from .env import SpatialCartpole


def quadratic_difference(factor,residual,baseline_error,error):
    w = factor@baseline_error+residual
    dv = factor@(error-baseline_error)
    return math.fsum(((2*w+dv)*dv).tolist())


def force_proposals(raw_baseline,baseline,reference,current_best):
    """Same bounded deterministic first-force proposals as the one-step policy."""
    yield raw_baseline
    yield reference
    yield (baseline.astype(float)+reference)/2
    ulps = np.maximum(abs(np.nextafter(baseline,np.float32(np.inf)).astype(float)-baseline),
                      abs(np.nextafter(baseline,np.float32(-np.inf)).astype(float)-baseline))
    radii = np.maximum.reduce([abs(baseline.astype(float)-reference),8*ulps,np.full(2,1e-4)])
    for scale in (1.,.5):
        center = current_best().astype(float).copy()
        for axis in range(2):
            for sign in (1.,-1.):
                proposal = center.copy()
                proposal[axis] += sign*scale*radii[axis]
                yield proposal
    center = current_best().copy()
    for axis in range(2):
        for direction in (np.float32(np.inf),np.float32(-np.inf)):
            proposal = center.copy()
            proposal[axis] = np.nextafter(proposal[axis],direction)
            yield proposal
    yield (current_best().astype(float)+baseline.astype(float))/2


class NativeRolloutValuePolicy(AlignedNativeValuePolicy):
    def __init__(self,*args,stage_factors,horizon=4,**kwargs):
        super().__init__(*args,**kwargs)
        self.horizon = int(horizon)
        self.stage_factors = np.asarray(stage_factors)
        nx = 2*self.env.model.nv
        if self.horizon!=4 or self.stage_factors.shape!=(len(self.states),nx,nx) or not np.isfinite(self.stage_factors).all():
            raise ValueError('four-command rollout requires full finite stage factors at every reference')
        self.metadata.update(policy='spatial native four-command affine value',native_rollout_horizon=self.horizon,
            future_feedback='plain observed-aligned affine; no nested search',
            common_next_value_frame='one baseline-anchored frame at each forecast poststate')

    def __call__(self,state,step):
        if step>=len(self.actions):
            return self.balance(state)
        p = self.predictor
        H = min(self.horizon,len(self.actions)-step)
        raw_baseline = axial_aligned_affine_force(p,state,self.states[step],self.actions[step],self.feed[step],self.gains[step])
        baseline = p.deliver(raw_baseline)
        best,best_cost,selected = baseline.copy(),float('inf'),-1
        seen,records,anchors = set(),[],None

        def rollout(raw,first):
            native_before = p.native_steps
            states = np.full((self.horizon+1,len(state)),np.nan)
            raw_actions = np.full((self.horizon,2),np.nan)
            actions = np.full((self.horizon,2),np.nan,np.float32)
            margins = np.full(self.horizon,np.nan)
            warnings = np.zeros(self.horizon,bool)
            valid_steps = np.zeros(self.horizon,bool)
            states[0] = state
            complete,valid,used = True,True,0
            for j in range(H):
                try:
                    proposal = raw if j==0 else axial_aligned_affine_force(p,states[j],self.states[step+j],
                        self.actions[step+j],self.feed[step+j],self.gains[step+j])
                    action = p.deliver(proposal)
                except (ValueError,FloatingPointError,OverflowError):
                    complete,valid = False,False
                    break
                raw_actions[j],actions[j] = proposal,action
                p.min_plane_margin,p.warning_seen = p.config.plane_limit,False
                states[j+1] = p.transition(states[j],action)
                used += 1
                margins[j],warnings[j] = p.min_plane_margin,p.warning_seen
                finite = bool(np.isfinite(states[j+1]).all())
                valid_steps[j] = finite and not warnings[j] and margins[j]>=0
                valid = valid and bool(valid_steps[j])
                # The finite baseline may anchor comparisons even if outside
                # bounds. Other invalid candidates stop immediately and cannot win.
                if not finite or (not first and not valid_steps[j]):
                    complete = False
                    break
            return dict(raw=raw_actions,actions=actions,states=states,plane_margins=margins,
                physics_warnings=warnings,valid_steps=valid_steps,used_steps=used,
                complete=complete,valid=valid and used==H,native_steps=p.native_steps-native_before)

        def query(raw):
            nonlocal best,best_cost,selected,anchors
            action = p.deliver(raw)
            if action.tobytes() in seen or len(records)>=16:
                return
            seen.add(action.tobytes())
            first = not records
            record = rollout(raw,first)
            if first and record['complete']:
                anchors=[]
                for j in range(1,H+1):
                    factor = self.roots[step+j] if j==H else self.stage_factors[step+j]
                    ref,root,angle = aligned_next_value(p,self.states[step+j],factor,record['states'][j])
                    residual = self.residuals[step+j] if j==H else np.zeros(len(root))
                    anchors.append((ref,root,residual,angle,p.difference(record['states'][j],ref)))
            state_costs,input_costs = np.full(self.horizon,np.nan),np.full(self.horizon,np.nan)
            cost = float('inf')
            if record['valid'] and anchors is not None:
                try:
                    with np.errstate(over='raise',invalid='raise'):
                        for j,(ref,root,residual,angle,error0) in enumerate(anchors):
                            state_costs[j] = quadratic_difference(root,residual,error0,p.difference(record['states'][j+1],ref))
                            d0 = records[0]['actions'][j].astype(float)-self.actions[step+j] if records else record['actions'][j].astype(float)-self.actions[step+j]
                            d1 = record['actions'][j].astype(float)-self.actions[step+j]
                            input_costs[j] = (d1-d0)@self.R@(d1+d0)
                        cost = math.fsum(state_costs[:H].tolist()+input_costs[:H].tolist())
                    record['valid'] = bool(np.isfinite(cost))
                except (FloatingPointError,OverflowError,ValueError):
                    record['valid'] = False
            else:
                record['valid'] = False
            record.update(cost=cost,state_cost_differences=state_costs,input_cost_differences=input_costs)
            records.append(record)
            if record['valid'] and cost<best_cost:
                best,best_cost,selected = action.copy(),cost,len(records)-1

        for raw in force_proposals(raw_baseline,baseline,self.actions[step],lambda:best):
            query(raw)
        angles = np.full((self.horizon,p.config.n_links),np.nan)
        if anchors is not None:
            angles[:H] = np.array([a[3] for a in anchors])
        self.trace.append(dict(step=step,horizon=H,candidates=records,selected=best.copy(),selected_index=selected,
            baseline_fallback=selected<0,anchor_angles=angles))
        self.metadata.update(predictor_native_steps=p.native_steps,native_rollout_commands=len(self.trace),
            baseline_fallback_commands=sum(r['baseline_fallback'] for r in self.trace))
        return best

    @property
    def forecast_tapes(self):
        records = [c for r in self.trace for c in r['candidates']]
        values = {f'native_rollout_{k}':np.asarray([r[k] for r in records]) for k in
            ('raw','actions','states','plane_margins','physics_warnings','valid_steps','used_steps','complete','valid',
             'native_steps','cost','state_cost_differences','input_cost_differences')}
        values.update({f'native_rollout_{k}':np.asarray([r[k] for r in self.trace]) for k in
            ('step','horizon','selected','selected_index','baseline_fallback','anchor_angles')})
        values['native_rollout_offsets'] = np.r_[0,np.cumsum([len(r['candidates']) for r in self.trace])]
        return values


class ParkedNativeRolloutValuePolicy(ParkedAlignedNativeValuePolicy):
    def __init__(self,env,payload):
        super().__init__(env,payload)
        if not bool(payload.get('rollout_native_value',False)) or int(payload['native_rollout_horizon'])!=4:
            raise ValueError('explicit four-command native rollout marker required')
        self.metadata.update(policy='parked spatial native four-command affine value',native_rollout_horizon=4,
            future_feedback='plain observed-aligned affine; no nested search')

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
                _,factors = axial_gauge_route(scratch,p['states'],p['tracking_stage_factors'],angles)
            finally:
                scratch.close()
            design = {k:p[k] for k in ('reference','K','P','Q','R')}
            self.route = NativeRolloutValuePolicy(e,states,p['actions'],p['feed'],gains,roots,p['residuals'],
                p['tracking_R'],design,stage_factors=factors,horizon=4)
            self.metadata.update(launch_state=state.tolist(),gauge_angles=angles.tolist())
        action = self.route(state,step-self.park_steps)
        self.metadata.update({k:self.route.metadata[k] for k in ('predictor_native_steps','native_rollout_commands','baseline_fallback_commands')})
        return action


def rollout_value_policy_from_payload(env,payload):
    return ParkedNativeRolloutValuePolicy(env,payload)
