"""Offline antiperiodic orbit construction for an odd physical step map.

An orbit is a component initializer, never hanging-start evidence. Simultaneous
nodes here belong only to the offline solver; the live plant must be executed
continuously to establish tracking. No Gram matrix or unstable long-horizon
state-transition product is used in the boundary-value solve.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def antiperiodic_dynamics_matrix(jacobians):
    """Jacobian of F(x_i,u_i)-x_{i+1}, with x_M=-x_0."""
    jacobians = np.asarray(jacobians, dtype=float)
    if jacobians.ndim != 3 or jacobians.shape[1] != jacobians.shape[2] or not len(jacobians):
        raise ValueError('a nonempty sequence of square dynamics Jacobians is required')
    steps, nx, _ = jacobians.shape
    matrix = sparse.lil_matrix((steps*nx, steps*nx))
    for step, a in enumerate(jacobians):
        rows = slice(step*nx, (step+1)*nx)
        matrix[rows, rows] = a
        column = ((step+1) % steps)*nx
        matrix[rows, column:column+nx] += np.eye(nx) if step == steps-1 else -np.eye(nx)
    return matrix.tocsc()


def linear_antiperiodic_nodes(a, b, half_controls):
    """Solve the linear boundary problem, including actual delivered controls."""
    controls = np.asarray(half_controls, dtype=float)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float).ravel()
    matrix = antiperiodic_dynamics_matrix(np.repeat(a[None], len(controls), axis=0))
    nodes = spsolve(matrix, -(controls[:, None]*b).ravel()).reshape(len(controls), len(b))
    if not np.all(np.isfinite(nodes)):
        raise np.linalg.LinAlgError('nonfinite linear antiperiodic solution')
    return nodes


def refine_antiperiodic_orbit(transition, nodes, half_controls, *,
                             max_iterations=12, tolerance=1e-13,
                             state_epsilon=1e-5, action_epsilon=1e-4):
    """Newton-refine fixed-action half-cycle nodes with an actual-map merit."""
    nodes = np.asarray(nodes, dtype=float).copy()
    controls = np.asarray(half_controls, dtype=np.float32).astype(float)
    if (nodes.ndim != 2 or len(nodes) != len(controls) or not len(nodes)
            or not np.all(np.isfinite(nodes)) or not np.all(np.isfinite(controls))
            or np.any(np.abs(controls) > 1) or max_iterations < 1
            or not np.isfinite(tolerance) or tolerance <= 0):
        raise ValueError('finite half-cycle nodes, bounded controls and a positive budget/tolerance required')

    def defects(values):
        successors = np.vstack([values[1:], -values[:1]])
        return np.asarray([transition(x, float(u)) for x, u in zip(values, controls)])-successors

    gap = defects(nodes)
    history = []
    for iteration in range(max_iterations):
        largest = float(np.max(np.abs(gap)))
        if largest <= tolerance:
            break
        jacobians = [transition.linearize(x, float(u), state_epsilon=state_epsilon,
                                         action_epsilon=action_epsilon)[0]
                     for x, u in zip(nodes, controls)]
        delta = spsolve(antiperiodic_dynamics_matrix(jacobians), -gap.ravel()).reshape(nodes.shape)
        if not np.all(np.isfinite(delta)):
            raise np.linalg.LinAlgError('nonfinite antiperiodic Newton step')
        accepted, alpha_used = False, None
        old_merit = np.linalg.norm(gap)
        for alpha in (1., .5, .25, .1, .01, .001):
            candidate = nodes+alpha*delta
            trial = defects(candidate)
            if np.linalg.norm(trial) < old_merit:
                nodes, gap, accepted, alpha_used = candidate, trial, True, alpha
                break
        history.append(dict(iteration=iteration+1, accepted=accepted, alpha=alpha_used,
                            maximum_physical_gap=float(np.max(np.abs(gap))),
                            gap_l2=float(np.linalg.norm(gap))))
        if not accepted:
            break
    # Verify the mirrored second half explicitly; odd symmetry is not assumed
    # to be exact in the numerical simulator.
    full_nodes = np.vstack([nodes, -nodes, nodes[:1]])
    full_controls = np.r_[controls, -controls]
    full_gaps = np.asarray([transition(full_nodes[i], float(u))-full_nodes[i+1]
                            for i, u in enumerate(full_controls)])
    maximum_gap = float(np.max(np.abs(full_gaps)))
    return full_nodes, full_controls, dict(
        history=history, iterations=len(history), tolerance=tolerance,
        maximum_physical_gap=maximum_gap,
        converged=maximum_gap <= tolerance, exactly_feasible=maximum_gap == 0.,
        offline_nodes_only=True, hanging_start_evidence=False)
