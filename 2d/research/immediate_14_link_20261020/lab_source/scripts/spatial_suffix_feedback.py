"""Variable-length planning components in the unchanged full physical model."""
import numpy as np
from spatialpole import SpatialCartpole
from spatialpole.control import linearize_many
from spatialpole.quotient import axial_quotient_frame
from spatialpole.world_coordinates import physical_tracking_factor
from spatial_synthesis_geometry import Dimensions,derivative,error


def build_model(native,geometry,reference,out):
    d=Dimensions.from_config(native.config);H=len(reference['actions'])
    assert native.config==geometry.config and reference['states'].shape==(H+1,d.state)
    assert 0<H<=native.config.horizon
    native.derivative_workers=4;before=native.native_steps
    raw,predicted=linearize_many(native,reference['states'][:-1],
        reference['actions'].astype(float),eps=.001,order=4,return_next=True)
    assert native.native_steps-before==H*(1+4*(d.tangent+2))*native.config.frame_skip
    charts=[axial_quotient_frame(geometry,x) for x in reference['states']]
    F=np.array([chart['F'] for chart in charts]);E=np.array([chart['E'] for chart in charts])
    factors=np.array([physical_tracking_factor(geometry,x) for x in reference['states']])
    assert all(np.linalg.matrix_rank(L[:,d.velocities:])==d.velocities for L in factors)
    aa=[];bb=[];dd=[];jacobians=[]
    for t,((A,B),state) in enumerate(zip(raw,predicted)):
        J=derivative(geometry,reference,F,state,t+1)
        assert np.linalg.matrix_rank(J[:,d.velocities:])==d.velocities
        aa.append(J@A@E[t]);bb.append(J@B)
        dd.append(error(geometry,reference,F,state,t+1));jacobians.append(J)
    r=dict(A=np.array(aa),B=np.array(bb),d=np.array(dd),factors=factors@E,cart=E[:,:2])
    np.savez_compressed(out/'native_models.npz',reference_states=reference['states'],
        reference_actions=reference['actions'],raw_A=np.array([A for A,B in raw]),
        raw_B=np.array([B for A,B in raw]),predicted=predicted,E=E,F=F,
        successor_error_jacobian=np.array(jacobians),**r)
    return r,F,E,predicted


class SegmentFeedback:
    def __init__(self,config,reference,frames,c,w):
        d=Dimensions.from_config(config);H=len(reference['actions'])
        assert reference['states'].shape==(H+1,d.state)
        assert frames.shape==(H+1,d.quotient,d.tangent) and w.shape==(H,2)
        self.geometry=SpatialCartpole(config)
        self.reference,self.frames,self.c,self.w=reference,frames,c,w
        self.dimension=d.quotient

    def __call__(self,state,phase):
        assert 0<=phase<len(self.reference['actions'])
        e=(np.zeros(self.dimension) if np.array_equal(state,self.reference['states'][phase])
            else error(self.geometry,self.reference,self.frames,state,phase))
        c=self.c
        return (self.reference['actions'][phase].astype(float)+c['feed'][phase]
            -c['K'][phase]@(c['S'][phase]@e)+c['Cinv'][phase]@self.w[phase])

    def close(self):
        assert self.geometry.native_steps==0
        self.geometry.close()


class JoinedFeedback:
    def __init__(self,env,prefix,reference,frames,c,w):
        self.phase=len(prefix['actions'])
        assert self.phase+len(reference['actions'])==env.config.horizon
        self.prefix=SegmentFeedback(env.config,prefix,prefix['F'],prefix,prefix['w'])
        self.suffix=SegmentFeedback(env.config,reference,frames,c,w)

    def __call__(self,state,phase):
        return self.prefix(state,phase) if phase<self.phase else self.suffix(state,phase-self.phase)

    def close(self):
        self.prefix.close();self.suffix.close()
