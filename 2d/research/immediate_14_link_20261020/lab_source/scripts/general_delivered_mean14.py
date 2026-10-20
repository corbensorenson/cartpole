"""Original actuator delivery propagated from arbitrary observed model errors."""
import numpy as np
from scipy.linalg import solve_triangular
from spatialpole import Config
from spatialpole.native_mpc import exact_bytes
from delivered_feedback_mean14 import construct as legacy_construct,algebra as legacy_algebra
from eleven_value_sequence import forward,mapped_error,physical_cost


def arguments(v,initial_error,base_white,observed):
    e0=np.zeros(74) if initial_error is None else np.array(initial_error,float,copy=True)
    base=np.zeros((1500,2)) if base_white is None else np.array(base_white,float,copy=True)
    observed=v['states'][0].copy() if observed is None else np.array(observed,float,copy=True)
    assert e0.shape==(74,) and base.shape==(1500,2) and observed.shape==(102,)
    assert all(np.isfinite(x).all() for x in (e0,base,observed))
    return e0,base,observed


def construct(geometry,v,r,c,initial_error=None,base_white=None,observed=None):
    assert geometry.config==Config(n_links=14)
    e0,base,observed=arguments(v,initial_error,base_white,observed)
    if not np.any(e0) and not np.any(base) and observed.tobytes()==v['states'][0].tobytes():
        proposal,data=legacy_construct(geometry,v,r,c)
    else:
        z=[c['S'][0]@e0];raws=[];delivered=[];white=[]
        for t in range(1500):
            raw=v['actions'][t].astype(float)+c['feed'][t]-c['K'][t]@z[-1]
            if np.any(base[t]):raw=raw+c['Cinv'][t]@base[t]
            u=geometry.deliver(raw);w=base[t]+np.linalg.solve(c['Cinv'][t],u.astype(float)-raw)
            following=c['A'][t]@z[-1]+c['B'][t]@w+c['d'][t]
            z.append(following);raws.append(raw);delivered.append(u);white.append(w)
        z,raws,delivered,white=map(np.array,(z,raws,delivered,white))
        errors=np.array([solve_triangular(S,zz) for S,zz in zip(c['S'],z)])
        delta=np.einsum('tij,tj->ti',r['E'],errors)
        states=np.array([geometry.integrate(x,dx) for x,dx in zip(v['states'],delta)])
        states[0]=observed.copy()
        proposal=dict(states=states,actions=delivered)
        data=dict(white_states=z,raw_controls=raws,delivered_controls=delivered,white_controls=white,errors=errors,delta=delta)
    exact_bytes(proposal['states'][0],observed)
    data=dict(data,initial_error=e0,base_white_controls=base,observed_initial=observed)
    return proposal,data


def algebra(geometry,v,r,c,data):
    e0,base,observed=arguments(v,data['initial_error'],data['base_white_controls'],data['observed_initial'])
    z,u,w,e=data['white_states'],data['delivered_controls'],data['white_controls'],data['errors']
    legacy=not np.any(e0) and not np.any(base) and observed.tobytes()==v['states'][0].tobytes()
    if legacy:
        result=legacy_algebra(geometry,v,r,c,data)
        return dict(result,initial_error_norm=0.,base_white_norm=0.,legacy_exact_path=True)
    exact_bytes(z[0],c['S'][0]@e0)
    residuals=[];rounding=[]
    for t in range(1500):
        raw=v['actions'][t].astype(float)+c['feed'][t]-c['K'][t]@z[t]
        if np.any(base[t]):raw=raw+c['Cinv'][t]@base[t]
        exact_bytes(raw,data['raw_controls'][t]);exact_bytes(geometry.deliver(raw),u[t])
        expected=c['A'][t]@z[t]+c['B'][t]@w[t]+c['d'][t];exact_bytes(expected,z[t+1])
        rounding.append(float(np.max(abs(c['Cinv'][t]@(w[t]-base[t])-(u[t].astype(float)-raw)))))
        residuals.append(float(np.max(abs(c['S'][t]@e[t]-z[t]))))
    du=u.astype(float)-v['actions'].astype(float);mapped=mapped_error(r,e,du)
    score=physical_cost(r,e,du,np.eye(2)*.001,terminal_scale=1.)
    _,be,bu=forward(c,e0,np.zeros((1500,2)))
    baseline=physical_cost(r,be,bu,np.eye(2)*.001,terminal_scale=1.)
    completion=abs(score-baseline-.5*float(np.sum(w*w)))/max(1.,abs(score))
    assert mapped<1e-6 and completion<1e-6 and max(rounding)<1e-12 and max(residuals)<1e-6
    return dict(passed=True,mapped_error=mapped,completion_error=completion,model_cost=score,
        continuous_zero_white_cost=baseline,total_white_cost=.5*float(np.sum(w*w)),
        maximum_delivery_identity_error=max(rounding),maximum_white_state_identity_error=max(residuals),
        initial_error_norm=float(np.linalg.norm(e0)),base_white_norm=float(np.linalg.norm(base)),legacy_exact_path=False)
