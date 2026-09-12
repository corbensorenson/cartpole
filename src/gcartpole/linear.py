from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.linalg import eigvals, matrix_rank, cond, svd

from .morphology import exp_gradient


@dataclass
class LinearMetrics:
    n_links: int
    alpha_length: float
    alpha_mass: float
    lambda_max_pos: float
    lambda_min_pos: float
    unstable_count: int
    controllability_rank: int
    controllability_cond: float
    weakest_unstable_coupling: float

    def to_dict(self) -> dict[str, float | int]:
        return self.__dict__.copy()


def build_linear_state_space(
    lengths: np.ndarray,
    masses: np.ndarray,
    *,
    cart_mass: float = 1.0,
    damping: np.ndarray | None = None,
    cart_damping: float = 0.0,
    g: float = 9.81,
):
    """Simplified linearized upright n-link point-mass model.

    This is for ranking candidate morphologies, not for exact MuJoCo matching.
    q = [x, theta_1, ..., theta_n]
    M qddot + D qdot - K q = b u around the upright equilibrium.
    """
    lengths = np.asarray(lengths, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    n = len(lengths)
    d = n + 1
    M = np.zeros((d, d), dtype=np.float64)
    K = np.zeros((d, d), dtype=np.float64)
    D = np.zeros((d, d), dtype=np.float64)

    M[0, 0] = cart_mass + masses.sum()
    cumulative = np.array([masses[i:].sum() for i in range(n)], dtype=np.float64)
    h = lengths * cumulative
    M[0, 1:] = h
    M[1:, 0] = h
    for i in range(n):
        for j in range(n):
            M[1 + i, 1 + j] = lengths[i] * lengths[j] * masses[max(i, j) :].sum()
    for i in range(n):
        K[1 + i, 1 + i] = g * lengths[i] * cumulative[i]
    D[0, 0] = cart_damping
    if damping is not None:
        damping = np.asarray(damping, dtype=np.float64)
        for i in range(n):
            D[1 + i, 1 + i] = damping[i]

    Minv = np.linalg.inv(M)
    Z = np.zeros((d, d), dtype=np.float64)
    I = np.eye(d, dtype=np.float64)
    A = np.block([[Z, I], [Minv @ K, -Minv @ D]])
    b = np.zeros((d, 1), dtype=np.float64)
    b[0, 0] = 1.0
    B = np.vstack([np.zeros((d, 1), dtype=np.float64), Minv @ b])
    return A, B


def controllability_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    parts = [B]
    cur = B
    for _ in range(1, A.shape[0]):
        cur = A @ cur
        parts.append(cur)
    return np.concatenate(parts, axis=1)


def modal_input_coupling(A: np.ndarray, B: np.ndarray) -> float:
    vals, vecs = np.linalg.eig(A)
    unstable = np.where(np.real(vals) > 1e-7)[0]
    if unstable.size == 0:
        return 0.0
    couplings = []
    for idx in unstable:
        v = vecs[:, idx]
        denom = np.linalg.norm(v) * np.linalg.norm(B)
        c = abs(np.vdot(v, B.reshape(-1))) / max(denom, 1e-12)
        couplings.append(float(c))
    return float(np.min(couplings))


def analyze_morphology(
    n_links: int,
    total_length: float,
    total_mass: float,
    cart_mass: float,
    alpha_length: float,
    alpha_mass: float,
    total_damping: float = 0.0,
) -> LinearMetrics:
    lengths = exp_gradient(total_length, n_links, alpha_length, min_value=0.02)
    masses = exp_gradient(total_mass, n_links, alpha_mass, min_value=1e-4)
    damping = exp_gradient(total_damping, n_links, 0.0, min_value=0.0) if total_damping > 0 else np.zeros(n_links)
    A, B = build_linear_state_space(lengths, masses, cart_mass=cart_mass, damping=damping)
    vals = eigvals(A)
    positive = np.sort(np.real(vals[np.real(vals) > 1e-7]))
    C = controllability_matrix(A, B)
    _, svals, _ = svd(C, full_matrices=False)
    ccond = float(svals[0] / max(svals[-1], 1e-18))
    return LinearMetrics(
        n_links=n_links,
        alpha_length=float(alpha_length),
        alpha_mass=float(alpha_mass),
        lambda_max_pos=float(positive[-1]) if positive.size else 0.0,
        lambda_min_pos=float(positive[0]) if positive.size else 0.0,
        unstable_count=int(positive.size),
        controllability_rank=int(matrix_rank(C, tol=1e-9)),
        controllability_cond=ccond,
        weakest_unstable_coupling=modal_input_coupling(A, B),
    )
