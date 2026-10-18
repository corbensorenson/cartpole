"""Owned-scratch instantaneous link loads; no integration or plant mutation."""
import mujoco as mj
import numpy as np
from scipy.optimize import minimize
from spatialpole import Config,SpatialCartpole


class Loads:
    def __init__(self,n_links=11):
        self.env=SpatialCartpole(Config(n_links=n_links));self.forward_calls=0;self.post_calls=0
        masses=self.env.model.body_mass[self.env.link_ids]
        self.weight=np.cumsum(masses[::-1])[::-1]*9.81

    def at(self,state,force):
        e=self.env;e.set_state(state);self.forward_calls+=1
        e.data.ctrl[:]=force;mj.mj_forward(e.model,e.data);self.forward_calls+=1
        mj.mj_rnePostConstraint(e.model,e.data);self.post_calls+=1
        directions=e.data.xmat[e.link_ids].reshape(-1,3,3)[:,:,2]
        tension=-np.sum(directions*e.data.cfrc_int[e.link_ids,3:],axis=1)
        assert e.native_steps==0 and np.isfinite(tension).all()
        return tension

    def affine(self,state):
        offset=self.at(state,np.zeros(2))
        matrix=np.column_stack([self.at(state,np.eye(2)[j])-offset for j in range(2)])
        return offset,matrix

    def close(self):
        assert self.env.native_steps==0;self.env.close()


def maximin(offset,matrix,weight):
    """Maximize minimum weight-normalized axial tension under the 80 N disk.

    A simplex dual gives a separate upper bound: c'lambda+80||G'lambda||.
    The returned force is a diagnostic, not a stabilizing-controller claim.
    """
    c=offset/weight;g=matrix/weight[:,None]
    primal=minimize(lambda x:-x[2],np.r_[np.zeros(2),c.min()],
        jac=lambda x:np.array([0.,0.,-1.]),method='SLSQP',
        constraints=[dict(type='ineq',fun=lambda x:c+g@x[:2]-x[2],jac=lambda x:np.c_[g,-np.ones(len(c))]),
                     dict(type='ineq',fun=lambda x:6400.-x[:2]@x[:2],jac=lambda x:np.r_[-2*x[:2],0.])],
        options=dict(ftol=1e-11,maxiter=400))
    u=primal.x[:2];u*=min(1.,80./max(np.linalg.norm(u),1e-30))
    lower=float(np.min(c+g@u))
    def dual(lam):return float(c@lam+80.*np.linalg.norm(g.T@lam))
    dual_result=minimize(dual,np.full(len(c),1./len(c)),method='SLSQP',bounds=[(0.,1.)]*len(c),
        constraints=[dict(type='eq',fun=lambda x:x.sum()-1.)],options=dict(ftol=1e-12,maxiter=600))
    lam=np.maximum(dual_result.x,0.);lam/=lam.sum();upper=dual(lam)
    assert np.isfinite(lower) and np.isfinite(upper) and upper>=lower-1e-8
    return dict(force=u,lower=lower,upper=upper,gap=upper-lower,
                primal_success=bool(primal.success),dual_success=bool(dual_result.success))


def project_halfplanes(desired,matrix,rhs,radius=80.):
    """Nearest force in a disk intersected with eleven closed halfplanes.

    Enumerate the possible boundary optima in two dimensions. Return None for
    an empty intersection. The caller explicitly records any relaxed floor.
    """
    desired=np.asarray(desired,float);G=np.asarray(matrix,float);b=np.asarray(rhs,float)
    tolerance=1e-9;candidates=[]
    def add(u):
        if np.isfinite(u).all() and np.linalg.norm(u)<=radius+1e-8 and np.min(G@u-b)>=-tolerance:
            candidates.append(u.copy())
    add(desired*min(1.,radius/max(np.linalg.norm(desired),1e-30)))
    if candidates:return candidates[0]
    for i,(g,r) in enumerate(zip(G,b)):
        norm2=float(g@g)
        if norm2<1e-28:continue
        point=g*(r/norm2);add(desired+g*((r-g@desired)/norm2))
        height=radius*radius-point@point
        if height>=-1e-9:
            normal=np.array([-g[1],g[0]])/np.sqrt(norm2)
            for sign in (-1.,1.):add(point+sign*np.sqrt(max(0.,height))*normal)
        for j in range(i):
            other=G[j];det=g[0]*other[1]-g[1]*other[0]
            if abs(det)<1e-13*np.sqrt(norm2*float(other@other)):continue
            add(np.linalg.solve(np.vstack([g,other]),np.array([r,b[j]])))
    if not candidates:return None
    return min(candidates,key=lambda u:float(np.sum((u-desired)**2)))
