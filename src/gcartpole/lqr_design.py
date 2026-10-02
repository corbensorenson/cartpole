"""Validate numerical discrete LQR designs before they become controllers."""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are


def checked_discrete_lqr(
    a: np.ndarray,
    b: np.ndarray,
    q: np.ndarray,
    r: np.ndarray,
    *,
    residual_tolerance: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Use the existing DARE design, rejecting invalid returned solutions.

    These are numerical checks on the unsaturated linear model, not a
    nonlinear capture or robustness certificate. The residual tolerance
    accommodates the ill-conditioned but stable released ten-link design.
    """
    p = solve_discrete_are(a, b, q, r)
    if not np.all(np.isfinite(p)):
        raise ValueError("Riccati solution contains non-finite values")
    gain = np.linalg.solve(r + b.T @ p @ b, b.T @ p @ a)
    if not np.all(np.isfinite(gain)):
        raise ValueError("LQR gain contains non-finite values")
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(a - b @ gain))))
    minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(0.5 * (p + p.T))))
    ap = a.T @ p @ a
    correction = a.T @ p @ b @ gain
    residual = ap - p - correction + q
    scale = max(np.linalg.norm(p), np.linalg.norm(ap), np.linalg.norm(correction), np.linalg.norm(q), 1.0)
    relative_residual = float(np.linalg.norm(residual) / scale)
    diagnostics = {
        "spectral_radius": spectral_radius,
        "minimum_riccati_eigenvalue": minimum_eigenvalue,
        "relative_riccati_residual": relative_residual,
        "residual_tolerance": float(residual_tolerance),
    }
    if (
        not all(np.isfinite(value) for value in diagnostics.values())
        or spectral_radius >= 1.0
        or minimum_eigenvalue <= 0.0
        or relative_residual > residual_tolerance
    ):
        raise ValueError(
            f"Invalid discrete LQR: spectral radius={spectral_radius:.9g}, "
            f"minimum Riccati eigenvalue={minimum_eigenvalue:.9g}, "
            f"relative Riccati residual={relative_residual:.9g}"
        )
    return gain, p, diagnostics


def _precision_input_matrices(values, ctx, preserve_input_precision):
    if preserve_input_precision:
        matrices = [ctx.matrix(value.tolist() if hasattr(value, "tolist") else value)
                    for value in values]
        if not all(ctx.isfinite(entry) for matrix in matrices for entry in matrix):
            raise ValueError("finite precision inputs required")
        arrays = [np.asarray(matrix.tolist(), dtype=np.float64) for matrix in matrices]
    else:
        arrays = [np.asarray(value, dtype=np.float64) for value in values]
        matrices = [ctx.matrix(value.tolist()) for value in arrays]
    return matrices, arrays


def _precision_eigenvalues(matrix, ctx):
    # mpmath's scalar special case returns a tuple even with both eigenvector
    # flags disabled. Its sole eigenvalue is the scalar itself.
    if matrix.rows == matrix.cols == 1:
        return [matrix[0, 0]]
    return ctx.eig(matrix, left=False, right=False)


def high_precision_discrete_lqr(a, b, q, r, *, decimal_digits=80, max_iterations=50,
                              preserve_input_precision=False):
    """Diagnostic SDA solve; binary64 inputs are promoted without recovering lost digits.

    Implements the doubling recurrence in Poloni (2020), equation 33:
    https://arxiv.org/html/2005.08903. This does not certify the nonlinear plant.
    The returned gain is rounded for the existing binary64 runtime. The
    opt-in precision-input path accepts MP matrices or decimal strings
    without first rounding the plant to binary64.
    """
    import mpmath

    if decimal_digits < 30 or max_iterations < 1:
        raise ValueError("high-precision design needs at least 30 digits and a positive iteration budget")
    ctx = mpmath.mp.clone()
    ctx.dps = decimal_digits
    matrices, arrays = _precision_input_matrices((a, b, q, r), ctx, preserve_input_precision)
    a, b, q, r = arrays
    n, nu = a.shape[0], b.shape[1]
    if (a.shape != (n, n) or b.shape != (n, nu) or q.shape != (n, n)
            or r.shape != (nu, nu) or not all(np.all(np.isfinite(x)) for x in arrays)):
        raise ValueError("LQR matrices must have compatible dimensions and finite entries")
    am, bm, qm, rm = matrices
    ctx.cholesky(qm)
    ctx.cholesky(rm)
    current_a = am.copy()
    g = bm * ctx.inverse(rm) * bm.T
    h = qm.copy()
    identity = ctx.eye(n)
    tolerance = ctx.mpf(10) ** (-decimal_digits // 2)
    converged = False
    for iteration in range(1, max_iterations + 1):
        inverse = ctx.inverse(identity + g * h)
        next_a = current_a * inverse * current_a
        next_g = g + current_a * inverse * g * current_a.T
        next_h = h + current_a.T * h * inverse * current_a
        next_g = (next_g + next_g.T) / 2
        next_h = (next_h + next_h.T) / 2
        relative_change = ctx.norm(next_h - h) / max(ctx.norm(next_h), ctx.mpf(1))
        current_a, g, h = next_a, next_g, next_h
        if relative_change <= tolerance and ctx.norm(current_a) <= ctx.sqrt(tolerance):
            converged = True
            break
    if not converged:
        raise ValueError("high-precision doubling did not converge within its iteration budget")
    ctx.cholesky(h)
    km = ctx.inverse(rm + bm.T * h * bm) * bm.T * h * am
    gain = np.array(km.tolist(), dtype=np.float64)
    p = np.array(h.tolist(), dtype=np.float64)
    rounded_km = ctx.matrix(gain.tolist())
    exact_closed_loop = am - bm * km
    rounded_closed_loop = am - bm * rounded_km
    eigenvalues = _precision_eigenvalues(exact_closed_loop, ctx)
    rounded_eigenvalues = _precision_eigenvalues(rounded_closed_loop, ctx)
    residual = am.T * h * am - h - am.T * h * bm * km + qm
    diagnostics = dict(
        method="arbitrary_precision_structure_preserving_doubling",
        input_precision=("precision inputs retained; model scalar errors and runtime gain rounding remain"
                         if preserve_input_precision
                         else "binary64 matrices promoted exactly; input errors remain"),
        preserved_input_precision=bool(preserve_input_precision),
        decimal_digits=decimal_digits, iterations=iteration,
        relative_riccati_residual=str(ctx.norm(residual) / max(ctx.norm(h), ctx.mpf(1))),
        high_precision_spectral_radius=str(max(abs(value) for value in eigenvalues)),
        rounded_gain_high_precision_spectral_radius=str(max(abs(value) for value in rounded_eigenvalues)),
        rounded_gain_numpy_spectral_radius=float(np.max(np.abs(np.linalg.eigvals(a - b @ gain)))),
        gain_norm=float(np.linalg.norm(gain)),
        nonlinear_certified=False,
    )
    if not (np.all(np.isfinite(gain)) and np.all(np.isfinite(p))):
        raise ValueError("high-precision design cannot be represented in binary64")
    if max(abs(value) for value in eigenvalues) >= 1:
        raise ValueError("high-precision design is not stabilizing for its input matrices")
    if max(abs(value) for value in rounded_eigenvalues) >= 1:
        raise ValueError("rounded high-precision gain is not stabilizing for its input matrices")
    return gain, p, diagnostics


def high_precision_lyapunov_factor(a, b, gain, transform, *, feedback_scale=1., decimal_digits=100,
                                  preserve_input_precision=False):
    """Factor the unit-cost Lyapunov value for the actually rounded gain.

    Binary64 inputs are promoted exactly. The factor improves value and
    gradient evaluation; it does not improve identification or certify the
    bounded nonlinear plant. Return P as the Gram matrix of the saved factor.
    """
    import mpmath
    if decimal_digits < 30 or not np.isfinite(feedback_scale) or feedback_scale <= 0:
        raise ValueError("Lyapunov design requires >=30 digits and a positive finite feedback scale")
    ctx = mpmath.mp.clone(); ctx.dps = decimal_digits
    if preserve_input_precision:
        gain_input = gain if hasattr(gain, "rows") else np.atleast_2d(gain)
        matrices, arrays = _precision_input_matrices((a, b, gain_input, transform), ctx, True)
    else:
        arrays = [np.atleast_2d(np.asarray(value, dtype=float)) for value in (a, b, gain, transform)]
        matrices = [ctx.matrix(v.tolist()) for v in arrays]
    am, bm, km, tm = arrays
    nx = am.shape[0]
    if (am.shape != (nx, nx) or bm.shape != (nx, 1)
            or km.shape != (1, nx) or tm.shape != (nx, nx)
            or not all(np.all(np.isfinite(v)) for v in arrays)):
        raise ValueError("finite compatible plant, single-input gain and transform required")
    am, bm, km, tm = matrices
    closed = tm*(am-bm*(ctx.mpf(float(feedback_scale))*km))*(tm**-1)
    radius = max(abs(value) for value in _precision_eigenvalues(closed, ctx))
    if radius >= 1:
        raise ValueError(f"rounded-gain promoted-input closed loop is unstable: {radius}")
    power, value = closed.copy(), ctx.eye(nx)
    tolerance = ctx.mpf(10)**(-decimal_digits//2)
    for iteration in range(40):
        value = value+power.T*value*power
        power = power*power
        if max(abs(v) for v in power) < tolerance:
            break
    else:
        raise ValueError("Lyapunov doubling did not converge")
    residual = value-closed.T*value*closed-ctx.eye(nx)
    relative_residual = ctx.norm(residual)/max(ctx.norm(value), ctx.mpf(1))
    if relative_residual > ctx.sqrt(tolerance):
        raise ValueError(f"high-precision Lyapunov residual too large: {relative_residual}")
    root = np.asarray(ctx.cholesky(value).T.tolist(), dtype=float)
    if not np.all(np.isfinite(root)):
        raise ValueError("Lyapunov factor cannot be represented in binary64")
    matrix = root.T@root
    return matrix, root, dict(method="promoted_rounded_gain_lyapunov_doubling_cholesky",
        input_precision=("precision plant inputs retained; model scalar and gain rounding errors remain"
                         if preserve_input_precision else
                         "binary64 inputs promoted exactly; identification and gain rounding errors remain"),
        preserved_input_precision=bool(preserve_input_precision),
        decimal_digits=decimal_digits, doubling_iterations=iteration+1,
        spectral_radius=str(radius), relative_equation_residual=str(relative_residual),
        absolute_equation_residual=str(max(abs(v) for v in residual)), nonlinear_certified=False)
