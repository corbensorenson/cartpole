"""Sparse bounded QP adapter; native convergence is reported without override."""
from types import SimpleNamespace

import numpy as np
from scipy import sparse


def solve_clarabel_bounded_qp(hessian, gradient, constraints, lower, upper, *, tolerance,
                             max_iterations):
    import clarabel
    lower, upper = np.asarray(lower), np.asarray(upper)
    equal = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    high, low = ~equal & np.isfinite(upper), ~equal & np.isfinite(lower)
    matrix = sparse.vstack([constraints[equal], constraints[high], -constraints[low]], format='csc')
    rhs = np.r_[upper[equal], upper[high], -lower[low]]
    cones = [clarabel.ZeroConeT(int(equal.sum())),
             clarabel.NonnegativeConeT(int(high.sum()+low.sum()))]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.max_iter = max_iterations
    settings.tol_gap_abs = settings.tol_gap_rel = settings.tol_feas = tolerance
    # Reduced convergence must obey the requested accuracy as well.
    settings.reduced_tol_gap_abs = settings.reduced_tol_gap_rel = settings.reduced_tol_feas = tolerance
    solver = clarabel.DefaultSolver(sparse.triu(hessian).tocsc(), gradient, matrix, rhs, cones, settings)
    solution = solver.solve()
    native_status = str(solution.status)
    status_val = {'Solved': 1, 'AlmostSolved': 2}.get(native_status, -1)
    # Other statuses remain unusable; the OSQP inexact-candidate rule is never
    # applied to a different solver's residual definition or termination code.
    return SimpleNamespace(x=np.asarray(solution.x),
        info=SimpleNamespace(status=native_status, status_val=status_val,
            prim_res=solution.r_prim, dual_res=solution.r_dual, iter=solution.iterations))
