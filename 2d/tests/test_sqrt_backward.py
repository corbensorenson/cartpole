"""Independent control optima, including a value beyond normal-matrix precision."""
import numpy as np
import pytest
from spatialpole.fddp import affine_backward,least_squares_backward
from spatialpole import Config,SpatialCartpole
from spatialpole.tracking_repair import repair_tracking


@pytest.mark.parametrize('backend',['normal','qr'])
def test_backward_includes_active_circle_value_curvature(backend):
    # Last-stage min .5||x+u+g||²+.5||u||², ||u||<=2, at x=0:
    # V(x)=.5||x+g||²-2||x+g||+4. Its tangential Hessian is .6.
    # Center the preceding control's optimum at zero. Its gain then depends
    # on this EXACT constrained value Hessian, including the circle curvature.
    g,limit = np.array([3.,4.]),2.
    I,Z = np.eye(2),np.zeros((2,2))
    reference = np.array([.6*g,np.zeros(2)])
    if backend=='normal':
        k,K,active = affine_backward([(I,I)]*2,np.zeros((2,2)),np.zeros((2,2)),
            np.zeros((2,2)),np.zeros((2,2,2)),g,I,I,0.,limit,action_reference=reference)
    else:
        k,K,active = least_squares_backward([(I,I)]*2,np.zeros((2,2)),np.zeros((2,2)),
            [Z]*2,[np.zeros(2)]*2,I,g,I,limit,action_reference=reference)
    value_hessian = I-limit/np.linalg.norm(g)*(I-np.outer(g,g)/(g@g))
    expected = np.linalg.solve(I+value_hessian,value_hessian)
    np.testing.assert_allclose(k[0],0.,atol=1e-11)
    np.testing.assert_allclose(K[0],expected,atol=1e-11)
    assert active==1


def test_residual_qr_matches_independently_condensed_affine_tracking_with_trust():
    rng = np.random.default_rng(9341)
    H,nx,nu = 8,3,2
    A = np.array([[1.,.1,0.],[0.,.95,.1],[0.,0.,.8]])
    B = np.array([[.01,0.],[.1,.01],[0.,.15]])
    Q,R,P = np.diag([1.,2.,.3]),.2*np.eye(nu),5*np.eye(nx)
    S,T = np.diag([.3,2.,.7]),.15
    xs,us,reference = rng.normal(size=(H+1,nx)),rng.normal(size=(H,nu)),rng.normal(size=(H,nu))
    initial = xs[0].copy()
    offsets = np.array([A@xs[t]+B@us[t]-xs[t+1] for t in range(H)])
    L,F = np.sqrt(Q),np.sqrt(S)
    k,K,_ = least_squares_backward([(A,B)]*H,offsets,us,[L]*H,
        [L@x for x in xs[:-1]],np.sqrt(P),np.sqrt(P)@xs[-1],R,1000.,F,T,reference)
    actual,x = [],initial.copy()
    for t in range(H):
        u = us[t]+k[t]-K[t]@(x-xs[t])
        actual.append(u)
        x = A@x+B@u
    base,mapping = initial.copy(),np.zeros((nx,H*nu))
    hessian = np.kron(np.eye(H),R+T*np.eye(nu))
    gradient = -(reference@R).ravel()-T*us.ravel()
    for t in range(H+1):
        weight = P if t==H else Q
        hessian += mapping.T@(weight+S)@mapping
        gradient += mapping.T@(weight@base+S@(base-xs[t]))
        if t<H:
            base,mapping = A@base,A@mapping
            mapping[:,t*nu:(t+1)*nu] += B
    expected = -np.linalg.solve(hessian,gradient).reshape(H,nu)
    np.testing.assert_allclose(actual,expected,atol=1e-10)


def test_qr_retains_small_control_cost_under_huge_terminal_value():
    # min u0²+u1²+1e18*(.5+u0+u1)² has equal controls -1e18*.5/(1+2e18).
    # Forming/subtracting normal value matrices loses the unit control cost.
    A,B = np.eye(2),np.array([[1.],[0.]])
    S = np.array([[1e9,1e9],[0.,1.]])
    initial = np.array([1.,-.5])
    us = np.zeros((2,1))
    k,K,_ = least_squares_backward([(A,B)]*2,np.zeros((2,2)),us,
        [np.zeros((2,2))]*2,[np.zeros(2)]*2,S,S@initial,np.eye(1),100.)
    x,actual = initial.copy(),[]
    for t in range(2):
        u = k[t]-K[t]@(x-initial)
        actual.append(u[0])
        x = A@x+B@u
    np.testing.assert_allclose(actual,[-.25,-.25],atol=1e-10)


def test_qr_circle_bound_and_tangent_feedback_with_large_residual_scale():
    v,limit = np.array([3.,4.]),2.
    S = 1e9*np.eye(2)
    k,K,active = least_squares_backward([(np.eye(2),np.eye(2))],
        np.zeros((1,2)),np.zeros((1,2)),[np.zeros((2,2))],[np.zeros(2)],
        S,S@v,np.eye(2)*.001,limit)
    np.testing.assert_allclose(k[0],-limit*v/np.linalg.norm(v),atol=1e-11)
    expected = limit/np.linalg.norm(v)*(np.eye(2)-np.outer(v,v)/(v@v))
    np.testing.assert_allclose(K[0],expected,atol=1e-11)
    assert active==1 and np.linalg.norm(k[0])<=limit


def test_square_root_repair_closes_actual_native_manifold_defects():
    e = SpatialCartpole(Config(n_links=1))
    initial = e.upright_state()
    controls = np.c_[np.sin(np.arange(20))*.2,np.zeros(20)]
    target = [initial]
    for u in controls:
        target.append(e.transition(target[-1],u,quantize=False))
    target = np.array(target)
    delta = np.zeros(2*e.model.nv)
    delta[3] = .01
    target[10] = e.integrate(target[10],delta)
    xs,us,_,_,history,_ = repair_tracking(e,initial,target,controls,iterations=6,
                                          penalty=100000.,sqrt_backward=True)
    for t,u in enumerate(us):
        np.testing.assert_array_equal(e.transition(xs[t],u,quantize=False),xs[t+1])
    assert min(r['max_scaled_defect'] for r in history)<1e-8
    assert np.max(np.abs(xs[:,:2]))<.01
    e.close()
