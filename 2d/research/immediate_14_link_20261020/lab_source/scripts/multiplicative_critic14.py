"""Finite affine QR control with a declared empirical multiplicative remainder."""
import numpy as np


class DesignRangeError(RuntimeError):
    def __init__(self, phase, reason, partial):
        super().__init__(reason)
        self.phase, self.partial = phase, partial


def design(m, directions, checkpoint=None):
    """E[w w'] = Z Z' * (||L x||² + .001 ||du||²), Z=L_next^-1 C.

    The physical simulator has no added w. This covariance is only a design
    surrogate fitted to already recorded native/model discrepancies.
    """
    H, nx = len(m['A']), m['A'].shape[1]
    _, S = np.linalg.qr(m['factors'][-1], mode='reduced')
    s = np.zeros(nx)
    roots, residuals = np.zeros((H+1,nx,nx)), np.zeros((H+1,nx))
    gains, feed, h = np.zeros((H,2,nx)), np.zeros((H,2)), np.zeros(H)
    roots[-1] = S
    computed = np.zeros(H,bool)
    result = dict(S=roots,value_residual=residuals,gain=gains,feed=feed,uncertainty_weight=h,computed=computed)
    for t in range(H-1,-1,-1):
        Z = np.linalg.solve(m['factors'][t+1], directions)
        SZ = S@Z
        if not np.isfinite(SZ).all() or np.max(abs(SZ),initial=0.) > 1e50:
            raise DesignRangeError(t,'uncertainty weight exceeds declared finite range',result)
        h[t] = float(np.sum(SZ*SZ))
        if h[t] > 1e100:
            raise DesignRangeError(t,'uncertainty weight exceeds declared finite range',result)
        scale = np.sqrt(1.+h[t])
        A,B,L = m['A'][t],m['B'][t],m['factors'][t]
        U = np.vstack((np.eye(2)*np.sqrt(.001)*scale, S@B, np.zeros((len(L),2))))
        X = np.vstack((np.zeros((2,nx)), S@A, scale*L))
        rr = np.r_[np.zeros(2), S@m['d'][t]+s, np.zeros(len(L))]
        _, qr = np.linalg.qr(np.c_[U,X,rr], mode='reduced')
        gains[t] = np.linalg.solve(qr[:2,:2],qr[:2,2:2+nx])
        feed[t] = -np.linalg.solve(qr[:2,:2],qr[:2,-1])
        S,s = qr[2:2+nx,2:2+nx],qr[2:2+nx,-1]
        if not np.isfinite(S).all() or not np.isfinite(s).all() or np.max(abs(S)) > 1e100:
            raise DesignRangeError(t,'value root exceeds declared finite range',result)
        roots[t], residuals[t] = S,s
        computed[t] = True
        if checkpoint and t%100 == 0:
            checkpoint(t,result)
    return result


def fit(m, x, actions):
    N = 11
    e = x['native_errors']
    r = e[1:N+1]-x['predicted_native_errors'][:N]
    physical = np.einsum('tij,tj->ti',m['factors'][:N],e[:N])
    du = actions[:N].astype(float)-m['reference_actions'][:N]
    den = np.sum(physical*physical,axis=1)+.001*np.sum(du*du,axis=1)
    assert np.all(den>0.)
    measured = np.einsum('tij,tj->ti',m['factors'][1:N+1],r)
    return (measured/np.sqrt(N*den[:,None])).T, dict(physical_residual=measured,denominator=den)


def dense_toy_check():
    """Independent Hessian Schur elimination versus square-root recursion."""
    rng = np.random.default_rng(290370)
    H,nx = 4,3
    m = dict(A=np.array([.6*np.eye(nx)+rng.normal(size=(nx,nx))*.03 for _ in range(H)]),
        B=rng.normal(size=(H,nx,2))*.1,d=rng.normal(size=(H,nx))*.001,
        factors=np.array([np.eye(nx)+rng.normal(size=(nx,nx))*.02 for _ in range(H+1)]))
    C = rng.normal(size=(nx,2))*.02
    qr = design(m,C)
    P = m['factors'][-1].T@m['factors'][-1]
    p = np.zeros(nx)
    maximum = 0.
    for t in range(H-1,-1,-1):
        A,B,d,L = m['A'][t],m['B'][t],m['d'][t],m['factors'][t]
        Z = np.linalg.solve(m['factors'][t+1],C)
        h = np.trace(Z.T@P@Z)
        U = np.eye(2)*.001*(1+h)+B.T@P@B
        cross = B.T@P@A
        g = B.T@(P@d+p)
        G = np.linalg.solve(U,cross)
        f = -np.linalg.solve(U,g)
        pn = A.T@(P@d+p)-cross.T@np.linalg.solve(U,g)
        Pn = (1+h)*(L.T@L)+A.T@P@A-cross.T@np.linalg.solve(U,cross)
        tests = ((G,qr['gain'][t]),(f,qr['feed'][t]),(Pn,qr['S'][t].T@qr['S'][t]),(pn,qr['S'][t].T@qr['value_residual'][t]))
        for actual,expected in tests:
            err = float(np.max(abs(actual-expected)))
            maximum = max(maximum,err)
            assert err<1e-11
        P,p = Pn,pn
    return dict(passed=True,max_absolute_error=maximum,horizon=H,state_dimension=nx)
