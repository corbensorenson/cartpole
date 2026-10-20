"""The same constrained tracking problem in stabilized Riccati coordinates."""
import numpy as np
from scipy import sparse
from scipy.linalg import solve_triangular
from spatialpole.affine_value import affine_tracking_values
from donor_conic_step import solver_module


def right(matrix,root):
    return solve_triangular(root.T,matrix.T,lower=True).T


def coordinates(r,R,terminal_scale=10.):
    _,terminal=np.linalg.qr(terminal_scale*r['factors'][-1],mode='reduced')
    f,k,S,s=affine_tracking_values(list(zip(r['A'],r['B'])),r['d'],r['factors'][:-1],R,terminal)
    S,s=np.array(S),np.array(s);H,nx=len(f),S.shape[1]
    aa=[];bb=[];dd=[];kk=[];cc=[];cart=[]
    for t in range(H):
        SB=S[t+1]@r['B'][t]
        _,C=np.linalg.qr(np.vstack([np.linalg.cholesky(R).T,SB]),mode='reduced')
        Ci=solve_triangular(C,np.eye(2));cc.append(Ci)
        aa.append(right(S[t+1]@r['A'][t]-SB@k[t],S[t]))
        bb.append(SB@Ci);dd.append(S[t+1]@(r['d'][t]+r['B'][t]@f[t]))
        kk.append(right(k[t],S[t]))
        cart.append(right(r['cart'][t],S[t]))
    cart.append(right(r['cart'][-1],S[-1]))
    return dict(A=np.array(aa),B=np.array(bb),d=np.array(dd),K=np.array(kk),
        Cinv=np.array(cc),cart=np.array(cart),S=S,feed=f,value_residual=s)


def forward(c,initial,w):
    z=[c['S'][0]@initial];du=[]
    for aa,bb,dd,K,Ci,f,ww in zip(c['A'],c['B'],c['d'],c['K'],c['Cinv'],c['feed'],w):
        du.append(f-K@z[-1]+Ci@ww);z.append(aa@z[-1]+bb@ww+dd)
    z=np.array(z)
    errors=np.array([solve_triangular(S,zz) for S,zz in zip(c['S'],z)])
    return z,errors,np.array(du)


def physical_cost(r,errors,du,R,terminal_scale=10.):
    physical=np.einsum('tij,tj->ti',r['factors'][1:],errors[1:]);physical[-1]*=terminal_scale
    return .5*float(np.sum(physical*physical)+np.einsum('ti,ij,tj->',du,R,du))


def mapped_error(r,errors,du):
    residual=errors[1:]-np.einsum('tij,tj->ti',r['A'],errors[:-1])-np.einsum('tij,tj->ti',r['B'],du)-r['d']
    return float(np.max(abs(residual)))


def solve_value_sequence(c,initial,reference,actions,runtime,path=None,time_limit=30.):
    H,nx=len(actions),len(initial);n=H*(nx+2)
    # Zero residual has the global minimum of this nonnegative objective.
    # It needs no iterative solve if every future physical constraint is met.
    zero=np.zeros((H,2));zz,ee,du0=forward(c,initial,zero)
    raw0=actions+du0;cart0=reference[:,:2]+np.einsum('tij,tj->ti',c['cart'],zz)
    if np.linalg.norm(raw0,axis=1).max()<=80. and np.max(abs(cart0[1:]))<=3.:
        residual=zz[1:]-np.einsum('tij,tj->ti',c['A'],zz[:-1])-c['d']
        primal=float(np.max(abs(residual)));assert primal<1e-6
        if path is not None:
            np.savez_compressed(path,kind='zero_residual',white_states=zz,initial=initial,H=H,nx=nx)
        return raw0,ee,zero,dict(passed=True,status='CertifiedZeroResidual',seconds=0.,iterations=0,
            primal=primal,stationarity=0.,cone_error=0.,relative_gap=0.,complementarity=0.,objective=0.)
    P=sparse.diags(np.r_[np.zeros(H*nx),np.ones(2*H)],format='csc')
    eq=sparse.lil_matrix((H*nx,n));rhs=c['d'].copy();z0=c['S'][0]@initial
    for t in range(H):
        rows=slice(t*nx,(t+1)*nx);eq[rows,t*nx:(t+1)*nx]=np.eye(nx)
        if t:eq[rows,(t-1)*nx:t*nx]=-c['A'][t]
        else:rhs[t]+=c['A'][t]@z0
        eq[rows,H*nx+2*t:H*nx+2*t+2]=-c['B'][t]
    pos=sparse.block_diag([sparse.csc_matrix(C) for C in c['cart'][1:]],format='csc')
    pos=sparse.hstack([pos,sparse.csc_matrix((2*H,2*H))],format='csc')
    rail=sparse.vstack([pos,-pos],format='csc')
    rb=np.r_[(3.-reference[1:,:2]).ravel(),(3.+reference[1:,:2]).ravel()]
    force=sparse.lil_matrix((3*H,n));fb=np.ones((H,3));fb[:,1:]=(actions+c['feed'])/80.
    for t in range(H):
        if t:force[3*t+1:3*t+3,(t-1)*nx:t*nx]=c['K'][t]/80.
        else:fb[t,1:]-=c['K'][t]@z0/80.
        force[3*t+1:3*t+3,H*nx+2*t:H*nx+2*t+2]=-c['Cinv'][t]/80.
    A=sparse.vstack([eq.tocsc(),rail,force.tocsc()],format='csc');b=np.r_[rhs.ravel(),rb,fb.ravel()]
    clarabel=solver_module(runtime);settings=clarabel.DefaultSettings();settings.verbose=False
    settings.direct_solve_method='qdldl';settings.max_iter=100;settings.time_limit=time_limit
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-9
    cones=[clarabel.ZeroConeT(H*nx),clarabel.NonnegativeConeT(4*H)]+[clarabel.SecondOrderConeT(3) for _ in range(H)]
    solver=clarabel.DefaultSolver(P,np.zeros(n),A,b,cones,settings);result=solver.solve()
    x,y,s=np.array(result.x),np.array(result.z),np.array(result.s)
    # The independent checker also covers primal/dual cone membership and gap.
    from verify_eleven_sequence_refinement import check
    proof=check(P,A,b,x,y,s,H,nx);proof.update(status=str(result.status),seconds=result.solve_time,iterations=result.iterations)
    if path is not None:
        sparse.save_npz(path.with_name(path.stem+'_P.npz'),P);sparse.save_npz(path.with_name(path.stem+'_A.npz'),A)
        np.savez_compressed(path,decision=x,dual=y,slack=s,b=b,H=H,nx=nx,initial=initial)
    assert proof['passed'] and str(result.status) in ('Solved','AlmostSolved'),proof
    z=np.vstack([z0,x[:H*nx].reshape(H,nx)]);w=x[H*nx:].reshape(H,2)
    errors=np.array([solve_triangular(S,zz) for S,zz in zip(c['S'],z)])
    du=c['feed']-np.einsum('tij,tj->ti',c['K'],z[:-1])+np.einsum('tij,tj->ti',c['Cinv'],w)
    controls=actions+du
    assert np.linalg.norm(controls,axis=1).max()<=80.+1e-5
    cart=reference[:,:2]+np.einsum('tij,tj->ti',c['cart'],z)
    assert np.max(abs(cart[1:]))<=3.+1e-6
    return controls,errors,w,proof
