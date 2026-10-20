"""The existing full-horizon tracking QP in physical error coordinates."""
import numpy as np
from scipy import sparse
from donor_conic_step import solver_module
from verify_eleven_sequence_refinement import check


def problem(r,initial,reference,actions):
    """No horizon, cost, force, cart or dynamics changes; terminal factor is one."""
    H,nx=len(actions),len(initial);n=H*(nx+2)
    assert r['A'].shape==(H,nx,nx) and r['B'].shape==(H,nx,2)
    blocks=[sparse.csc_matrix(L.T@L) for L in r['factors'][1:]]
    P=sparse.block_diag(blocks+[sparse.csc_matrix(80.*80.*.001*np.eye(2))]*H,format='csc')
    eq=sparse.lil_matrix((H*nx,n));rhs=r['d'].copy()
    for t in range(H):
        rows=slice(t*nx,(t+1)*nx);eq[rows,t*nx:(t+1)*nx]=np.eye(nx)
        if t:eq[rows,(t-1)*nx:t*nx]=-r['A'][t]
        else:rhs[t]+=r['A'][t]@initial
        eq[rows,H*nx+2*t:H*nx+2*t+2]=-80.*r['B'][t]
    pos=sparse.block_diag([sparse.csc_matrix(C) for C in r['cart'][1:]],format='csc')
    pos=sparse.hstack([pos,sparse.csc_matrix((2*H,2*H))],format='csc')
    rail=sparse.vstack([pos,-pos],format='csc')
    rb=np.r_[(3.-reference[1:,:2]).ravel(),(3.+reference[1:,:2]).ravel()]
    indices=np.arange(2*H);rows=np.array([3*t+j for t in range(H) for j in (1,2)])
    force=sparse.coo_matrix((-np.ones(2*H),(rows,H*nx+indices)),shape=(3*H,n)).tocsc()
    fb=np.c_[np.ones(H),actions.astype(float)/80.].ravel()
    A=sparse.vstack([eq.tocsc(),rail,force],format='csc');b=np.r_[rhs.ravel(),rb,fb]
    return P,A,b


def algebra_check(P,A,b,r,initial,reference,actions):
    """Check matrix assembly against the original dynamics and physical cost."""
    H,nx=len(actions),len(initial);rng=np.random.default_rng(141601)
    e=np.vstack([initial,rng.normal(size=(H,nx))*1e-4]);du=rng.normal(size=(H,2))*1e-3
    x=np.r_[e[1:].ravel(),(du/80.).ravel()]
    residual=e[1:]-np.einsum('tij,tj->ti',r['A'],e[:-1])-np.einsum('tij,tj->ti',r['B'],du)-r['d']
    np.testing.assert_allclose((A@x-b)[:H*nx].reshape(H,nx),residual,atol=1e-11,rtol=1e-11)
    cart=reference[1:,:2]+np.einsum('tij,tj->ti',r['cart'][1:],e[1:])
    slacks=b-A@x;np.testing.assert_allclose(slacks[H*nx:H*nx+4*H],np.r_[(3.-cart).ravel(),(3.+cart).ravel()],atol=1e-12,rtol=1e-13)
    force=slacks[H*nx+4*H:].reshape(H,3)
    np.testing.assert_allclose(force,np.c_[np.ones(H),(actions+du)/80.],atol=1e-13,rtol=1e-13)
    f=np.einsum('tij,tj->ti',r['factors'][1:],e[1:]);cost=.5*float(np.sum(f*f)+.001*np.sum(du*du))
    assert abs(.5*float(x@(P@x))-cost)<1e-10*max(1.,cost)
    return dict(passed=True,dynamics_cart_force_and_original_physical_objective=True,terminal_scale=1.,R=.001)


def solve_physical(r,initial,reference,actions,runtime,out):
    P,A,b=problem(r,initial,reference,actions);H,nx=len(actions),len(initial)
    algebra=algebra_check(P,A,b,r,initial,reference,actions)
    solver=solver_module(runtime);settings=solver.DefaultSettings();settings.verbose=False
    settings.direct_solve_method='qdldl';settings.max_iter=100;settings.time_limit=120.
    settings.tol_gap_abs=settings.tol_gap_rel=settings.tol_feas=1e-9
    cones=[solver.ZeroConeT(H*nx),solver.NonnegativeConeT(4*H)]+[solver.SecondOrderConeT(3) for _ in range(H)]
    sol=solver.DefaultSolver(sparse.triu(P,format='csc'),np.zeros(P.shape[0]),A,b,cones,settings).solve()
    x,y,s=np.array(sol.x),np.array(sol.z),np.array(sol.s)
    sparse.save_npz(out/'physical_P.npz',P);sparse.save_npz(out/'physical_A.npz',A)
    np.savez_compressed(out/'physical_certificate.npz',decision=x,dual=y,slack=s,b=b,initial=initial,H=H,nx=nx)
    proof=check(P,A,b,x,y,s,H,nx);proof.update(status=str(sol.status),iterations=sol.iterations,seconds=sol.solve_time)
    admitted=bool(proof['passed'] and str(sol.status) in ('Solved','AlmostSolved'))
    if not admitted:return None,dict(proof=proof,algebra=algebra,admitted=False)
    errors=np.vstack([initial,x[:H*nx].reshape(H,nx)]);du=80.*x[H*nx:].reshape(H,2);controls=actions.astype(float)+du
    assert np.linalg.norm(controls,axis=1).max()<=80.+1e-5
    cart=reference[:,:2]+np.einsum('tij,tj->ti',r['cart'],errors);assert np.max(abs(cart[1:]))<=3.+1e-6
    return (controls,errors),dict(proof=proof,algebra=algebra,admitted=True)
