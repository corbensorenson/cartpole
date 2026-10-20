"""Sparse finite-horizon tracking with a disk constraint on every future input."""
import numpy as np
from scipy import sparse
from donor_conic_step import solver_module


def reduced_cache(cache, end=600):
    E, F = cache['E'], cache['F']
    A = F[1:end+1] @ cache['A'][:end] @ E[:end]
    B = F[1:end+1] @ cache['B'][:end]
    d = np.einsum('tij,tj->ti', F[1:end+1], cache['defects'][:end])
    factors = cache['factors'][:end+1] @ E[:end+1]
    assert E.shape[1:] == (70, 59) and np.linalg.matrix_rank(E[0,35:]) == 35
    return dict(A=A, B=B, d=d, factors=factors, cart=E[:end+1,:2])


def solve_sequence(A, B, defects, factors, cart, reference_states, actions,
                   initial, R, runtime, certificate_path=None):
    """State errors are model predictions only, never physical state updates."""
    clarabel = solver_module(runtime)
    H, nx = len(A), len(initial)
    assert len(factors) == H+1 and actions.shape == (H,2)
    count = H*(nx+2)
    blocks = [sparse.csc_matrix(L.T @ L) for L in factors[1:]]
    blocks[-1] *= 100.
    P = sparse.block_diag(blocks + [80.*80.*sparse.csc_matrix(R)]*H, format='csc')
    eq = sparse.lil_matrix((H*nx, count))
    rhs = defects.copy()
    for t in range(H):
        rows = slice(t*nx, (t+1)*nx)
        eq[rows, t*nx:(t+1)*nx] = np.eye(nx)
        if t:
            eq[rows, (t-1)*nx:t*nx] = -A[t]
        else:
            rhs[t] += A[t] @ initial
        eq[rows, H*nx+2*t:H*nx+2*t+2] = -80.*B[t]
    position = sparse.block_diag([sparse.csc_matrix(C) for C in cart[1:]], format='csc')
    position = sparse.hstack([position, sparse.csc_matrix((2*H,2*H))], format='csc')
    rail = sparse.vstack([position, -position], format='csc')
    rail_rhs = np.r_[(3.-reference_states[1:,:2]).ravel(), (3.+reference_states[1:,:2]).ravel()]
    rows, cols, values = [], [], []
    for t in range(H):
        rows.extend([3*t+1, 3*t+2])
        cols.extend([H*nx+2*t, H*nx+2*t+1])
        values.extend([-1., -1.])
    force = sparse.coo_matrix((values,(rows,cols)),shape=(3*H,count)).tocsc()
    force_rhs = np.c_[np.ones(H), actions.astype(float)/80.].ravel()
    matrix = sparse.vstack([eq.tocsc(), rail, force], format='csc')
    b = np.r_[rhs.ravel(),rail_rhs,force_rhs]
    cones = [clarabel.ZeroConeT(H*nx), clarabel.NonnegativeConeT(4*H)]
    cones += [clarabel.SecondOrderConeT(3) for _ in range(H)]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.direct_solve_method = 'qdldl'
    settings.max_iter = 100
    settings.time_limit = 30.
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = 1e-9
    solver = clarabel.DefaultSolver(sparse.triu(P,format='csc'), np.zeros(count), matrix, b, cones, settings)
    result = solver.solve()
    x, dual, slack = np.array(result.x), np.array(result.z), np.array(result.s)
    assert np.isfinite(x).all() and np.isfinite(dual).all() and np.isfinite(slack).all()
    primal = float(np.max(abs(matrix@x+slack-b)))
    px, aty = P@x, matrix.T@dual
    stationarity = float(np.max(abs(px+aty))/max(1.,float(np.max(abs(px))),float(np.max(abs(aty)))))
    cone_error = max(float(np.max(abs(slack[:H*nx]))),
        float(-slack[H*nx:H*nx+4*H].min()), float(-dual[H*nx:H*nx+4*H].min()), 0.)
    for t in range(H):
        start = H*nx+4*H+3*t
        for vector in (slack[start:start+3], dual[start:start+3]):
            cone_error = max(cone_error,float(np.linalg.norm(vector[1:])-vector[0]))
    objective = .5*float(x@px)
    gap = abs(float(x@px+b@dual))/max(1.,2*objective)
    complementarity = abs(float(slack@dual))/max(1.,2*objective)
    report = dict(status=str(result.status),iterations=result.iterations,solve_seconds=result.solve_time,
        primal_error=primal, normalized_stationarity=stationarity, cone_error=cone_error,
        normalized_gap=gap, normalized_complementarity=complementarity, objective=objective,
        horizon=H, variables=count, model_only=True)
    if certificate_path is not None:
        path = certificate_path
        sparse.save_npz(path.with_name(path.stem+'_P.npz'),P)
        sparse.save_npz(path.with_name(path.stem+'_A.npz'),matrix)
        np.savez_compressed(path, decision=x, dual=dual, slack=slack, b=b,
            H=H, nx=nx, initial=initial, actions=actions, report=str(report))
    assert str(result.status) in ('Solved','AlmostSolved'), report
    assert max(primal,stationarity,cone_error,gap,complementarity) < 1e-6, report
    errors = np.vstack([initial,x[:H*nx].reshape(H,nx)])
    controls = actions.astype(float) + 80.*x[H*nx:].reshape(H,2)
    assert np.linalg.norm(controls,axis=1).max() <= 80.+1e-5
    planned_cart = reference_states[:,:2]+np.einsum('tij,tj->ti',cart,errors)
    assert np.max(abs(planned_cart[1:])) <= 3.+1e-6
    report.update(maximum_raw_force=float(np.linalg.norm(controls,axis=1).max()),
        model_cart_margin=3.-float(np.max(abs(planned_cart[1:]))))
    return controls, errors, report
