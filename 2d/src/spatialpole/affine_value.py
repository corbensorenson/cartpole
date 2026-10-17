"""Finite affine tracking values and bounded two-force native correction.

Spatial adaptation of the ideas reviewed in corben-marc-hybrid.md, not a
reproduction of Marc's scalar B17 algorithm. Values are local surrogates;
only complete actual episodes can admit a controller.
"""
import numpy as np
from .env import SpatialCartpole
from .control import WorldBalancePolicy,BalancePolicy


def axial_aligned_affine_force(env,state,reference,action,feed,gain):
    """Gauge-invariant local feedback; align reference frames, never the plant.

    The closest axial rotation uses the polar angle, which commutes with an
    arbitrary change of the observed body's axial frame. Angular velocities
    retain their measured spin; no physical DOF or rate is removed.
    """
    from .conditioning import axial_gauge_route
    env.set_state(reference)
    nominal = env.data.xmat[env.link_ids].reshape(-1,3,3).copy()
    env.set_state(state)
    observed = env.data.xmat[env.link_ids].reshape(-1,3,3)
    relative = np.transpose(nominal,(0,2,1))@observed
    angles = np.arctan2(relative[:,1,0]-relative[:,0,1],relative[:,0,0]+relative[:,1,1])
    refs,gains = axial_gauge_route(env,np.array([reference]),gain[None,:,:],angles)
    return action+feed-gains[0]@env.difference(state,refs[0])


def affine_tracking_values(models, defects, factors, R, terminal_factor,
                           terminal_residual=None):
    """QR elimination for .5||L e||²+.5 du'R du and e+=A e+B du+d.

    Returns additive offsets and subtractive gains: du=feed-K@e, plus all
    state value roots/residuals, including the terminal design state.
    """
    H, nx = len(models), terminal_factor.shape[1]
    nu = R.shape[0]
    S = np.asarray(terminal_factor).copy()
    s = np.zeros(len(S)) if terminal_residual is None else np.asarray(terminal_residual).copy()
    if len(S)>nx:
        _, qr = np.linalg.qr(np.c_[S,s],mode='reduced')
        S,s = qr[:nx,:nx],qr[:nx,-1]
    roots, residuals = [None]*(H+1), [None]*(H+1)
    roots[H], residuals[H] = S.copy(), s.copy()
    feed, gains = np.zeros((H,nu)), np.zeros((H,nu,nx))
    C = np.linalg.cholesky(R).T
    for t in range(H-1,-1,-1):
        A,B = models[t]
        L = factors[t]
        U = np.vstack((C, S@B, np.zeros((len(L),nu))))
        X = np.vstack((np.zeros((nu,nx)), S@A, L))
        r = np.r_[np.zeros(nu), S@defects[t]+s, np.zeros(len(L))]
        _, qr = np.linalg.qr(np.c_[U,X,r], mode='reduced')
        gains[t] = np.linalg.solve(qr[:nu,:nu], qr[:nu,nu:nu+nx])
        feed[t] = -np.linalg.solve(qr[:nu,:nu], qr[:nu,-1])
        S, s = qr[nu:nu+nx,nu:nu+nx], qr[nu:nu+nx,-1]
        roots[t], residuals[t] = S.copy(), s.copy()
    return feed, gains, roots, residuals


class AffineValuePolicy:
    """Frozen affine law, optionally corrected by <=16 native F32 candidates.

    An owned predictor never advances or resets the physical environment.
    Candidate order/ties are deterministic. Search is a new spatial policy,
    not scalar independent clipping: every query uses the full 80N disk.
    """
    def __init__(self, env, states, actions, feed, gains, roots, residuals,
                 R, balance, search=False):
        self.env, self.states, self.actions = env, states, actions
        self.feed, self.gains = feed, gains
        self.roots, self.residuals, self.R = roots, residuals, R
        self.balance = WorldBalancePolicy(env,balance)
        self.predictor = SpatialCartpole(env.config) if search else None
        self.trace = []
        self.metadata = dict(policy='spatial affine native value' if search else 'spatial affine feedback',
                             design_steps=len(actions), execution_route_steps=min(env.config.horizon,len(actions)),
                             predictor_native_steps=0, max_candidates=16 if search else 0)

    def __call__(self, state, step):
        if step>=len(self.actions):
            return self.balance(state)
        baseline = self.actions[step]+self.feed[step]-self.gains[step]@self.env.difference(state,self.states[step])
        if self.predictor is None:
            return baseline
        p = self.predictor
        seen, records = set(), []
        best, best_cost = self.env.deliver(baseline), float('inf')

        def query(raw):
            nonlocal best,best_cost
            action = p.deliver(raw)
            key = action.tobytes()
            if key in seen or len(seen)>=16:
                return
            seen.add(key)
            p.min_plane_margin, p.warning_seen = p.config.plane_limit, False
            successor = p.transition(state,action)
            valid = not p.warning_seen and p.min_plane_margin>=0 and np.isfinite(successor).all()
            cost = float('inf')
            if valid:
                error = p.difference(successor,self.states[step+1])
                value = self.roots[step+1]@error+self.residuals[step+1]
                delta = action.astype(float)-self.actions[step]
                cost = float(value@value+delta@self.R@delta)
                valid = np.isfinite(cost)
            records.append(dict(action=action.copy(),cost=cost,valid=bool(valid),
                                successor=successor.copy(),plane_margin=p.min_plane_margin))
            if valid and cost<best_cost:
                best,best_cost = action.copy(),cost

        # Keep the delivered baseline and literal reference among candidates.
        query(baseline)
        query(self.actions[step])
        query((self.env.deliver(baseline).astype(float)+self.actions[step])/2)
        for radius in (1.,.1,.01):
            center = best.astype(float).copy()
            for direction in ((1,0),(-1,0),(0,1),(0,-1)):
                query(center+radius*np.array(direction))
        query(np.nextafter(best,np.float32(0)))
        self.trace.append(dict(step=step,candidates=records,selected=best.copy()))
        self.metadata['predictor_native_steps'] = p.native_steps
        return best

    def close(self):
        if self.predictor is not None:
            self.predictor.close()


class ParkedAffineValuePolicy:
    """Causal hanging preparation followed by a prefix of a full value design."""
    def __init__(self,env,payload):
        self.env,self.payload,self.route = env,payload,None
        self.park_steps = int(payload['park_steps'])
        self.online_gauge = bool(payload.get('online_gauge',False))
        self.online_parking = bool(payload.get('online_parking',False))
        self.align_cart_origin = bool(payload.get('align_cart_origin',False))
        self.cart_reference_offset = np.asarray(payload.get('cart_reference_offset',np.zeros(2)),float)
        if self.cart_reference_offset.shape!=(2,) or not np.isfinite(self.cart_reference_offset).all():
            raise ValueError('reference cart offset must contain two finite coordinates')
        if self.online_gauge and bool(payload['native_search']):
            raise ValueError('online axial feedback and native-value search are separate tested families')
        self.parking = BalancePolicy(env,dict(reference=payload['parking_reference'],K=payload['parking_K']))
        self.metadata = dict(park_steps=self.park_steps,park_seconds=self.park_steps*env.config.dt,
            route_seconds=int(payload['swing_steps'])*env.config.dt,
            design_steps=len(payload['actions']),execution_tracking_steps=env.config.horizon-self.park_steps,
            align_axial_frames=bool(payload['align_gauge']),policy='parked spatial affine value')
        self.metadata['online_axial_alignment'] = self.online_gauge
        self.metadata['online_parking_alignment'] = self.online_parking
        self.metadata['align_reference_cart_origin'] = self.align_cart_origin

    def __call__(self,state,step):
        if step<self.park_steps:
            if self.online_parking:
                return axial_aligned_affine_force(self.env,state,self.parking.design['reference'],
                    np.zeros(2),np.zeros(2),self.parking.design['K'])
            return self.parking(state,step)
        if self.route is None:
            from .conditioning import launch_gauge_angles,axial_gauge_route
            p,e = self.payload,self.env
            angles = launch_gauge_angles(e,p['states'][0],state) if bool(p['align_gauge']) else np.zeros(e.config.n_links)
            states,gains = axial_gauge_route(e,p['states'],p['gains'],angles)
            if self.align_cart_origin or np.any(self.cart_reference_offset):
                translation = self.cart_reference_offset.copy()
                if self.align_cart_origin:
                    translation += state[:2]-states[0,:2]
                states = states.copy()
                states[:,:2] += translation
                self.metadata['reference_cart_translation'] = translation.tolist()
            # Values act on the same local error coordinates as feedback gains.
            # Feed offsets/residuals are unchanged by this local axial basis map.
            _,roots = axial_gauge_route(e,p['states'],p['roots'],angles)
            design = {k:p[k] for k in ('reference','K','P','Q','R')}
            self.route = AffineValuePolicy(e,states,p['actions'],p['feed'],gains,roots,
                p['residuals'],p['tracking_R'],design,search=bool(p['native_search']))
            e.set_state(state)
            self.metadata.update(launch_state=state.tolist(),gauge_angles=angles.tolist(),
                launch_max_tilt_from_down=float(np.max(np.pi-e.tilts())),launch_cart_xy=state[:2].tolist())
        t = step-self.park_steps
        if self.online_gauge and t<len(self.route.actions):
            result = axial_aligned_affine_force(self.env,state,self.route.states[t],
                self.route.actions[t],self.route.feed[t],self.route.gains[t])
        else:
            result = self.route(state,t)
        self.metadata['predictor_native_steps'] = self.route.metadata['predictor_native_steps']
        return result

    def close(self):
        if self.route is not None:
            self.route.close()


class LaunchAdmissionPolicy:
    """Choose a frozen reference from complete causal native launch forecasts."""
    def __init__(self,env,payload):
        self.env,self.payload = env,payload
        self.profiles = np.asarray(payload['launch_reference_profiles'],float)
        if self.profiles.shape!=(2,2) or not np.isfinite(self.profiles).all():
            raise ValueError('launch admission requires two finite XY reference profiles')
        if bool(payload['native_search']):
            raise ValueError('launch admission does not combine native action search')
        self.park_steps = int(payload['park_steps'])
        self.policy = ParkedAffineValuePolicy(env,payload)
        self.metadata = self.policy.metadata
        self.forecast_tapes = {}
        self.selected = None

    def __call__(self,state,step):
        if step<self.park_steps:
            return self.policy(state,step)
        observed_time = self.env.data.time
        if self.selected is None:
            if step!=self.park_steps:
                raise ValueError('launch admission must begin at the declared launch command')
            records = []
            for index,offset in enumerate(self.profiles):
                predictor = SpatialCartpole(self.env.config)
                forecast_policy = None
                try:
                    predicted = np.asarray(state).copy()
                    predictor.set_state(predicted)
                    predictor.steps = step
                    predictor.streak = self.env.streak
                    predictor.best_streak = self.env.best_streak
                    predictor.min_plane_margin = self.env.min_plane_margin
                    predictor.warning_seen = self.env.warning_seen
                    candidate = dict(self.payload,cart_reference_offset=offset.copy())
                    forecast_policy = ParkedAffineValuePolicy(predictor,candidate)
                    states,actions,raw_actions,tilts = [predicted],[],[],[]
                    for t in range(step,self.env.config.horizon):
                        raw = forecast_policy(predicted,t)
                        action = predictor.deliver(raw)
                        predicted,_,terminated,truncated,info = predictor.step_force(action)
                        states.append(predicted)
                        actions.append(action)
                        raw_actions.append(raw)
                        tilts.append(predictor.tilts())
                        if terminated or truncated:
                            break
                    for name,values in dict(states=states,actions=actions,
                                            raw_actions=raw_actions,tilts=tilts).items():
                        self.forecast_tapes[f'forecast_{index}_{name}'] = np.asarray(values)
                    records.append(dict(profile=index,reference_cart_offset=offset.tolist(),
                        **info,steps=predictor.steps,native_steps=predictor.native_steps))
                finally:
                    if forecast_policy is not None:
                        forecast_policy.close()
                    predictor.close()
            admissible = [r['profile'] for r in records if r['success'] and
                r['steps']==self.env.config.horizon and not r['physics_warning'] and r['plane_margin']>=0]
            self.selected = admissible[0] if admissible else 0
            self.policy.close()
            self.policy = ParkedAffineValuePolicy(self.env,dict(self.payload,
                cart_reference_offset=self.profiles[self.selected].copy()))
            self.metadata.update(launch_forecasts=records,selected_launch_profile=self.selected,
                launch_admitted=bool(admissible),launch_forecast_start_step=step,
                launch_predictor_native_steps=sum(r['native_steps'] for r in records))
        force = self.policy(state,step)
        # Reference transport forwards scratch states; preserve the caller's
        # clock as well as the already-restored observed physical q/v.
        self.env.data.time = observed_time
        self.metadata.update(self.policy.metadata)
        self.metadata['predictor_native_steps'] = self.metadata['launch_predictor_native_steps']
        return force

    def close(self):
        self.policy.close()


def affine_policy_from_payload(env,payload):
    if 'launch_reference_profiles' in payload:
        return LaunchAdmissionPolicy(env,payload)
    if 'park_steps' in payload:
        return ParkedAffineValuePolicy(env,payload)
    design = {k:payload[k] for k in ('reference','K','P','Q','R')}
    return AffineValuePolicy(env,payload['states'],payload['actions'],payload['feed'],payload['gains'],
        payload['roots'],payload['residuals'],payload['tracking_R'],design,bool(payload['native_search']))
