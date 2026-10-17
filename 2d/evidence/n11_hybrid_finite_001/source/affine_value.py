"""Finite affine tracking values and bounded two-force native correction.

Spatial adaptation of the ideas reviewed in corben-marc-hybrid.md, not a
reproduction of Marc's scalar B17 algorithm. Values are local surrogates;
only complete actual episodes can admit a controller.
"""
import numpy as np
from .env import SpatialCartpole
from .control import WorldBalancePolicy


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
