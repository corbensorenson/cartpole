"""Full-horizon affine correction of a continuation secant predictor."""
import numpy as np
from spatialpole import SpatialCartpole
from spatialpole.control import linearize_many
from spatialpole.quotient import axial_quotient_frame
from spatialpole.world_coordinates import physical_tracking_factor
from spatialpole.aligned_value import aligned_next_value
from probe_eleven_native_chart import error,derivative
from eleven_initial_continuation import initial_at


def predictor(geometry,previous,current,target,fa,fb,fraction):
    weight=(fraction-fb)/(fb-fa)
    xs=np.array([geometry.integrate(xb,weight*geometry.difference(xb,xa)) for xa,xb in zip(previous['states'],current['states'])])
    initial=initial_at(geometry,target['nominal'],target['target'],fraction)
    assert np.max(abs(geometry.difference(xs[0],initial)))<1e-10;xs[0]=initial
    raw=current['actions'].astype(float)+weight*(current['actions'].astype(float)-previous['actions'].astype(float))
    us=np.array([geometry.deliver(u) for u in raw])
    return dict(states=xs,actions=us)


def corrector_model(native,geometry,v,out):
    native.derivative_workers=4
    raw,predicted=linearize_many(native,v['states'][:-1],v['actions'].astype(float),eps=.001,order=4,return_next=True)
    frames=[axial_quotient_frame(geometry,x) for x in v['states']]
    F=np.array([f['F'] for f in frames]);E=np.array([f['E'] for f in frames]);cache=dict(F=F)
    factors=np.array([physical_tracking_factor(geometry,x) for x in v['states']])
    assert all(np.linalg.matrix_rank(L[:,35:])==35 for L in factors)
    aa=[];bb=[];dd=[];jacobians=[]
    for t,((A,B),state) in enumerate(zip(raw,predicted)):
        J=derivative(geometry,v,cache,state,t+1,1e-5)
        assert np.linalg.matrix_rank(J[:,35:])==35
        aa.append(J@A@E[t]);bb.append(J@B);dd.append(error(geometry,v,cache,state,t+1));jacobians.append(J)
    r=dict(A=np.array(aa),B=np.array(bb),d=np.array(dd),factors=factors@E,cart=E[:,:2])
    np.savez_compressed(out/'native_models.npz',reference_states=v['states'],reference_actions=v['actions'],
        raw_A=np.array([A for A,B in raw]),raw_B=np.array([B for A,B in raw]),predicted=predicted,
        E=E,F=F,successor_error_jacobian=np.array(jacobians),**r)
    return r,F,E,predicted,raw


class CorrectorPolicy:
    def __init__(self,env,v,F,c,w):
        self.geometry=SpatialCartpole(env.config);self.v=v;self.F=F;self.c=c;self.w=w
    def __call__(self,state,t):
        if np.array_equal(state,self.v['states'][t]):e=np.zeros(59)
        else:
            ref,F,_=aligned_next_value(self.geometry,self.v['states'][t],self.F[t],state)
            e=F@self.geometry.difference(state,ref)
        return self.v['actions'][t].astype(float)+self.c['feed'][t]-self.c['K'][t]@(self.c['S'][t]@e)+self.c['Cinv'][t]@self.w[t]
    def close(self):
        assert self.geometry.native_steps==0;self.geometry.close()
