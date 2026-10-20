"""Propagate original F32 delivery through stabilized model feedback."""
import numpy as np
from scipy.linalg import solve_triangular
from spatialpole.native_mpc import exact_bytes
from eleven_value_sequence import mapped_error,physical_cost,forward


def construct(geometry,v,r,c):
    z=[np.zeros(74)];raws=[];delivered=[];white=[]
    for t in range(1500):
        raw=v['actions'][t].astype(float)+c['feed'][t]-c['K'][t]@z[-1]
        u=geometry.deliver(raw);w=np.linalg.solve(c['Cinv'][t],u.astype(float)-raw)
        following=c['A'][t]@z[-1]+c['B'][t]@w+c['d'][t]
        z.append(following);raws.append(raw);delivered.append(u);white.append(w)
    z,raws,delivered,white=np.array(z),np.array(raws),np.array(delivered),np.array(white)
    errors=np.array([solve_triangular(S,zz) for S,zz in zip(c['S'],z)])
    delta=np.einsum('tij,tj->ti',r['E'],errors)
    states=np.array([geometry.integrate(x,dx) for x,dx in zip(v['states'],delta)])
    states[0]=v['states'][0].copy();exact_bytes(states[0],v['states'][0])
    return dict(states=states,actions=delivered),dict(white_states=z,raw_controls=raws,delivered_controls=delivered,white_controls=white,errors=errors,delta=delta)


def algebra(geometry,v,r,c,data):
    z,u,w,e=data['white_states'],data['delivered_controls'],data['white_controls'],data['errors']
    residuals=[];rounding=[]
    for t in range(1500):
        raw=v['actions'][t].astype(float)+c['feed'][t]-c['K'][t]@z[t]
        exact_bytes(raw,data['raw_controls'][t]);exact_bytes(geometry.deliver(raw),u[t])
        expected=c['A'][t]@z[t]+c['B'][t]@w[t]+c['d'][t];exact_bytes(expected,z[t+1])
        rounding.append(float(np.max(abs(c['Cinv'][t]@w[t]-(u[t].astype(float)-raw)))))
        residuals.append(float(np.max(abs(c['S'][t]@e[t]-z[t]))))
    du=u.astype(float)-v['actions'].astype(float);mapped=mapped_error(r,e,du)
    score=physical_cost(r,e,du,np.eye(2)*.001,terminal_scale=1.)
    _,be,bu=forward(c,np.zeros(74),np.zeros((1500,2)))
    baseline=physical_cost(r,be,bu,np.eye(2)*.001,terminal_scale=1.)
    completion=abs(score-baseline-.5*float(np.sum(w*w)))/max(1.,abs(score))
    assert mapped<1e-6 and completion<1e-6 and max(rounding)<1e-12 and max(residuals)<1e-6
    return dict(passed=True,mapped_error=mapped,completion_error=completion,model_cost=score,
        continuous_zero_white_cost=baseline,white_rounding_cost=.5*float(np.sum(w*w)),
        maximum_delivery_identity_error=max(rounding),maximum_white_state_identity_error=max(residuals))
