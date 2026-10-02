"""Bounded one-step nonlinear value minimization using a separate predictor.

The baseline delivered action is always a candidate. This is a component
controller, not a stability certificate or a multi-step reachability solver.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

from .simulation import SimulationError


def select_nonlinear_value_action(transition, state, factor, baseline_action, *,
                                  error_function=None, rail_limit=None,
                                  grid_points=17, max_iterations=40,
                                  control_cost=0., reference_action=0.,
                                  decimal_digits=0, candidate_actions=()):
    state=np.asarray(state,dtype=float);factor=np.asarray(factor,dtype=float)
    if (state.ndim!=1 or factor.ndim!=2 or factor.shape[1]!=state.size
            or not np.all(np.isfinite(state)) or not np.all(np.isfinite(factor))
            or not np.isfinite(baseline_action) or grid_points<3 or max_iterations<1
            or not np.all(np.isfinite([control_cost,reference_action])) or control_cost<0
            or not isinstance(decimal_digits,int) or decimal_digits<0 or 0<decimal_digits<40
            or (rail_limit is not None and (not np.isfinite(rail_limit) or rail_limit<=0))):
        raise ValueError('finite compatible state/value factor, action and positive bounds required')
    candidates=np.asarray(candidate_actions,dtype=float)
    if candidates.ndim!=1 or not np.all(np.isfinite(candidates)):
        raise ValueError('additional action candidates must be a finite vector')
    cache={};exact_values={};invalid=0
    ctx=mp_factor=None
    if decimal_digits:
        import mpmath
        ctx=mpmath.mp.clone();ctx.dps=decimal_digits;mp_factor=ctx.matrix(factor.tolist())
    def objective(action):
        nonlocal invalid
        delivered=float(np.float32(np.clip(action,-1.,1.)))
        if delivered in cache:return cache[delivered]
        try:
            predicted=np.asarray(transition(state,delivered),dtype=float)
            if (predicted.shape!=state.shape or not np.all(np.isfinite(predicted))
                    or (rail_limit is not None and abs(predicted[0])>rail_limit)):
                value=np.inf
            else:
                error=predicted if error_function is None else np.asarray(error_function(predicted),dtype=float)
                if ctx is None:
                    residual=factor@error;value=float(residual@residual)
                    if control_cost:value+=control_cost*(delivered-reference_action)**2
                else:
                    residual=mp_factor*ctx.matrix(error.tolist())
                    exact=(residual.T*residual)[0]+ctx.mpf(control_cost)*(ctx.mpf(delivered)-ctx.mpf(reference_action))**2
                    exact_values[delivered]=exact;value=float(exact)
                if not np.isfinite(value):value=np.inf
        except SimulationError:
            value=np.inf
        invalid+=int(not np.isfinite(value));cache[delivered]=value;return value
    baseline=float(np.float32(np.clip(baseline_action,-1.,1.)))
    baseline_value=objective(baseline)
    actions=np.unique(np.r_[np.linspace(-1.,1.,grid_points).astype(np.float32).astype(float),baseline,
                            np.clip(candidates,-1,1).astype(np.float32).astype(float)])
    values=np.asarray([objective(action) for action in actions])
    ranking=lambda action:(exact_values.get(float(action),cache[float(action)])
                           if np.isfinite(cache[float(action)]) else np.inf)
    best=min(range(len(actions)),key=lambda index:ranking(actions[index]))
    if not np.isfinite(values[best]):
        raise SimulationError('no finite in-rail nonlinear value preview')
    left=max(0,best-1);right=min(len(actions)-1,best+1)
    if actions[left]<actions[right]:
        optimized=minimize_scalar(objective,bounds=(actions[left],actions[right]),method='bounded',
                                  options=dict(maxiter=max_iterations,xatol=1e-12))
        objective(optimized.x)
        if decimal_digits or len(candidates):
            center=np.float32(optimized.x)
            for direction in [np.float32(-np.inf),np.float32(np.inf)]:
                objective(np.nextafter(center,direction))
    action=min(cache,key=ranking);value=cache[action]
    return action,dict(predicted_value=value,baseline_predicted_value=baseline_value,
                      predictor_evaluations=len(cache),invalid_previews=invalid,
                      improves_baseline=bool(ranking(action)<ranking(baseline)),
                      decimal_digits=decimal_digits,control_cost=control_cost,
                      reference_action=reference_action)
