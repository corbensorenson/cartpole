"""Experimental residual-form sequential convexification with virtual nodes.

State and control increments are simultaneous optimizer variables. Dynamics
slacks are explicitly penalized and measured, never applied to the live plant.
Neither native QP convergence nor small local gaps is a physical certificate.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse

from .ilqr import ILQRResult, resolve_running_costs
from .simulation import SimulationError
from .sqrt_ilqr import square_root_backward_pass


def _root(cost, terminal=False):
    factor = cost.terminal_factor if terminal else cost.stage_factor
    matrix = cost.terminal_state if terminal else cost.stage_state
    result = np.linalg.cholesky(matrix).T if factor is None else np.asarray(factor, dtype=float)
    if result.ndim != 2 or result.shape[1] != matrix.shape[0] or not np.all(np.isfinite(result)):
        raise ValueError('cost factors must be finite and match the state dimension')
    return result


class FactoredShootingProblem:
    def __init__(self, transition, initial_state, states, controls, cost, *,
                 running_costs=None, defect_factor=None, state_epsilon=1e-5, action_epsilon=1e-4):
        self.transition = transition
        self.initial_state = np.asarray(initial_state, dtype=float).copy()
        self.steps, self.nx = len(controls), len(initial_state)
        self.cost = cost
        self.costs = resolve_running_costs(cost, self.steps, running_costs)
        if self.steps < 1 or any(c.wrap_angles for c in (*self.costs, cost)):
            raise ValueError('sparse shooting requires a nonempty continuous-coordinate horizon')
        if any(c.control <= 0 for c in self.costs):
            raise ValueError('positive running control costs required')
        if min(state_epsilon, action_epsilon) <= 0 or not np.all(np.isfinite([state_epsilon, action_epsilon])):
            raise ValueError('finite positive derivative increments required')
        self.state_epsilon, self.action_epsilon = state_epsilon, action_epsilon
        self.defect_factor = np.eye(self.nx) if defect_factor is None else np.asarray(defect_factor, dtype=float)
        if self.defect_factor.shape != (self.nx, self.nx) or not np.all(np.isfinite(self.defect_factor)):
            raise ValueError('defect factor must be a finite square state matrix')
        self.state_values = self.steps*self.nx
        self.variables = self.state_values+self.steps
        self.roots = tuple(_root(c) for c in self.costs)+(_root(cost, True),)
        self.initial = self.pack(states, controls)

    def pack(self, states, controls):
        states, controls = np.asarray(states, dtype=float), np.asarray(controls, dtype=float)
        if states.shape != (self.steps+1, self.nx) or controls.shape != (self.steps,):
            raise ValueError('state/control horizon dimensions do not match')
        if not np.array_equal(states[0], self.initial_state):
            raise ValueError('initial node must equal the fixed physical initial state')
        if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)) or np.any(np.abs(controls) > 1):
            raise ValueError('finite states and bounded controls required')
        return np.r_[states[1:].ravel(), controls.astype(np.float32).astype(float)]

    def unpack(self, values):
        return (np.vstack([self.initial_state, values[:self.state_values].reshape(self.steps, self.nx)]),
                values[self.state_values:])

    def evaluate(self, values, derivatives=False):
        states, controls = self.unpack(values)
        gaps = np.array([self.transition(states[i], float(u))-states[i+1] for i, u in enumerate(controls)])
        scaled_gaps = (gaps @ self.defect_factor.T).ravel()
        residuals, objective_blocks = [], []
        for node, (state, c, factor) in enumerate(zip(states, (*self.costs, self.cost), self.roots)):
            target = c.terminal_target if node == self.steps else c.stage_target
            residuals.append(factor @ (state if target is None else state-target))
            if derivatives:
                block = sparse.lil_matrix((factor.shape[0], self.variables))
                if node > 0:
                    block[:, (node-1)*self.nx:node*self.nx] = factor
                objective_blocks.append(block.tocsr())
            # Exact residual representation of the quartic soft rail cost.
            width = max(1e-9, c.rail_limit-c.rail_soft_limit)
            ratio = max(0., abs(float(state[0]))-c.rail_soft_limit)/width
            residuals.append(np.array([np.sqrt(2*c.rail_weight)*ratio**2]))
            if derivatives:
                block = sparse.lil_matrix((1, self.variables))
                if node > 0 and ratio > 0:
                    block[0, (node-1)*self.nx] = 2*np.sqrt(2*c.rail_weight)*ratio*np.sign(state[0])/width
                objective_blocks.append(block.tocsr())
        control_root = np.sqrt([c.control for c in self.costs])
        residuals.append(control_root*controls)
        if derivatives:
            objective_blocks.append(sparse.hstack([sparse.csr_matrix((self.steps, self.state_values)),
                                                   sparse.diags(control_root)], format='csr'))
        residual = np.concatenate(residuals)
        if not derivatives:
            return gaps, scaled_gaps, residual
        dynamics = sparse.lil_matrix((self.state_values, self.variables))
        for step, u in enumerate(controls):
            a, b = self.transition.linearize(states[step], float(u), state_epsilon=self.state_epsilon,
                                             action_epsilon=self.action_epsilon)
            rows = slice(step*self.nx, (step+1)*self.nx)
            if step:
                dynamics[rows, (step-1)*self.nx:step*self.nx] = self.defect_factor @ a
            dynamics[rows, step*self.nx:(step+1)*self.nx] = -self.defect_factor
            dynamics[rows, self.state_values+step] = self.defect_factor @ np.asarray(b).reshape(self.nx, 1)
        return gaps, scaled_gaps, residual, dynamics.tocsc(), sparse.vstack(objective_blocks, format='csc')


def optimize_constrained_shooting(
    transition, initial_state, initial_controls, initial_states, cost, *,
    running_costs=None, defect_factor=None, max_iterations=20, defect_penalty=1e5,
    initial_regularization=1., state_trust=.05, control_trust=.1,
    state_abs_limit=150., state_epsilon=1e-5, action_epsilon=1e-4,
    qp_max_iterations=10000, qp_tolerance=1e-8, qp_initial_tolerance=None,
    qp_inexact_dual_tolerance=0., callback=None,
    node_constraint_matrix=None, node_lower=None, node_upper=None,
):
    import osqp
    if (min(max_iterations, defect_penalty, state_trust, control_trust, state_abs_limit,
            qp_max_iterations, qp_tolerance) <= 0 or initial_regularization < 0
            or not np.all(np.isfinite([defect_penalty, state_trust, control_trust,
                                       state_abs_limit, initial_regularization, qp_tolerance]))):
        raise ValueError('finite positive budgets, trust regions and tolerances required')
    initial_qp_tolerance = qp_tolerance if qp_initial_tolerance is None else qp_initial_tolerance
    if not np.isfinite(initial_qp_tolerance) or initial_qp_tolerance < qp_tolerance:
        raise ValueError('initial QP tolerance must be finite and no smaller than its refinement floor')
    if not np.isfinite(qp_inexact_dual_tolerance) or qp_inexact_dual_tolerance < 0:
        raise ValueError('inexact dual tolerance must be finite and nonnegative')
    p = FactoredShootingProblem(transition, initial_state, initial_states, initial_controls, cost,
                               running_costs=running_costs, defect_factor=defect_factor,
                               state_epsilon=state_epsilon, action_epsilon=action_epsilon)
    n, m = p.variables, p.state_values
    values = p.initial.copy()
    limits = np.array([c.rail_limit for c in (*p.costs[1:], cost)])
    if np.any(~np.isfinite(limits)) or np.any(limits <= 0) or abs(initial_state[0]) > p.costs[0].rail_limit:
        raise ValueError('finite positive rail limits and an in-rail initial state required')
    lower, upper = np.full(n, -state_abs_limit), np.full(n, state_abs_limit)
    lower[m:], upper[m:] = -1., 1.
    carts = np.arange(p.steps)*p.nx
    lower[carts], upper[carts] = -limits, limits
    linear = sparse.csc_matrix((0, n))
    linear_lower, linear_upper = np.array([]), np.array([])
    if node_constraint_matrix is not None:
        matrix = np.asarray(node_constraint_matrix, dtype=float)
        lo, hi = np.asarray(node_lower, dtype=float), np.asarray(node_upper, dtype=float)
        if (matrix.ndim != 2 or matrix.shape[1] != p.nx or not np.all(np.isfinite(matrix))
                or lo.shape != (p.steps, matrix.shape[0]) or hi.shape != lo.shape
                or np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(lo > hi)):
            raise ValueError('linear node constraints have invalid dimensions or bounds')
        linear = sparse.hstack([sparse.kron(sparse.eye(p.steps), matrix),
                                sparse.csc_matrix((lo.size, p.steps))], format='csc')
        mask = np.isfinite(lo.ravel()) | np.isfinite(hi.ravel())
        linear, linear_lower, linear_upper = linear[mask].tocsc(), lo.ravel()[mask], hi.ravel()[mask]
    elif node_lower is not None or node_upper is not None:
        raise ValueError('node bounds require a constraint matrix')

    def admissible(candidate):
        mapped = linear @ candidate
        return bool(np.all(np.isfinite(candidate)) and np.all(candidate >= lower) and np.all(candidate <= upper)
                    and np.all(mapped >= linear_lower) and np.all(mapped <= linear_upper))

    if not admissible(values):
        raise ValueError('initial virtual nodes violate hard bounds')
    trust = np.r_[np.full(m, state_trust), np.full(p.steps, control_trust)]
    initial_trust = trust.copy()
    regularization = max(1e-9, initial_regularization)
    gaps, scaled, residual = p.evaluate(values)
    merit = lambda g, r: float(defect_penalty*np.sum(np.abs(g))+.5*(r @ r))
    current_merit = merit(scaled, residual)
    history, consecutive_native_failures = [], 0
    cached = None
    for iteration in range(max_iterations):
        if cached is None:
            gaps, scaled, residual, dynamics, objective = p.evaluate(values, derivatives=True)
            cached = dynamics, objective
        else:
            dynamics, objective = cached
        residual_rows = len(residual)
        # Residual variables retain the cost factors without J.T@J. A uniform
        # objective scale preserves the minimizer; the residual is scaled by
        # its square root so its QP diagonal stays one.
        scale = max(1., defect_penalty)
        root_scale = np.sqrt(scale)
        eye_n, eye_m = sparse.eye(n, format='csc'), sparse.eye(m, format='csc')
        eye_r = sparse.eye(residual_rows, format='csc')
        zero_nm, zero_nr = sparse.csc_matrix((n, m)), sparse.csc_matrix((n, residual_rows))
        zero_mm, zero_mr = sparse.csc_matrix((m, m)), sparse.csc_matrix((m, residual_rows))
        zero_rm = sparse.csc_matrix((residual_rows, m))
        zero_lm = sparse.csc_matrix((linear.shape[0], m))
        zero_lr = sparse.csc_matrix((linear.shape[0], residual_rows))
        hessian = sparse.block_diag([regularization*eye_n/scale, zero_mm, zero_mm, eye_r], format='csc')
        gradient = np.r_[np.zeros(n), np.full(2*m, defect_penalty/scale), np.zeros(residual_rows)]
        constraints = sparse.bmat([
            [dynamics, -eye_m, eye_m, zero_mr],
            [objective/root_scale, zero_rm, zero_rm, -eye_r],
            [eye_n, zero_nm, zero_nm, zero_nr],
            [linear, zero_lm, zero_lm, zero_lr],
            [sparse.csc_matrix((m, n)), eye_m, zero_mm, zero_mr],
            [sparse.csc_matrix((m, n)), zero_mm, eye_m, zero_mr]], format='csc')
        step_lower, step_upper = np.maximum(lower-values, -trust), np.minimum(upper-values, trust)
        mapped = linear @ values
        bound_lower = np.r_[-scaled, -residual/root_scale, step_lower, linear_lower-mapped, np.zeros(2*m)]
        bound_upper = np.r_[-scaled, -residual/root_scale, step_upper, linear_upper-mapped, np.full(2*m, np.inf)]
        qp = osqp.OSQP()
        used_qp_tolerance = max(qp_tolerance, min(initial_qp_tolerance, .1*float(np.max(np.abs(scaled)))))
        qp.setup(P=hessian, q=gradient, A=constraints, l=bound_lower, u=bound_upper,
                 verbose=False, eps_abs=used_qp_tolerance, eps_rel=used_qp_tolerance,
                 max_iter=qp_max_iterations, polishing=True)
        qp.warm_start(x=np.r_[np.zeros(n), np.maximum(scaled, 0.), np.maximum(-scaled, 0.), residual/root_scale])
        solved = qp.solve(raise_error=False)
        native_ok = solved.info.status_val in (1, 2)
        inexact_primal_tolerance = min(qp_inexact_dual_tolerance, .1*float(np.max(np.abs(scaled))))
        inexact_candidate = bool(solved.info.status_val == 7 and qp_inexact_dual_tolerance > 0
                                 and solved.info.prim_res <= inexact_primal_tolerance
                                 and solved.info.dual_res <= qp_inexact_dual_tolerance)
        usable_step = bool((native_ok or inexact_candidate) and solved.x is not None
                           and np.all(np.isfinite(solved.x)))
        accepted, alpha_used, invalid_trials, bound_rejections = False, None, 0, 0
        previous_merit = current_merit
        step = None
        if usable_step:
            step = solved.x[:n]
            for alpha in (1., .5, .25, .1, .05, .01, .001, .0001):
                candidate = values+alpha*step
                candidate[m:] = candidate[m:].astype(np.float32).astype(float)
                if not admissible(candidate):
                    bound_rejections += 1
                    continue
                try:
                    trial_gaps, trial_scaled, trial_residual = p.evaluate(candidate)
                    trial_merit = merit(trial_scaled, trial_residual)
                except SimulationError:
                    invalid_trials += 1
                    continue
                if np.isfinite(trial_merit) and trial_merit < current_merit:
                    values, gaps, scaled, residual = candidate, trial_gaps, trial_scaled, trial_residual
                    current_merit, accepted, alpha_used = trial_merit, True, alpha
                    break
        if accepted:
            cached = None
            consecutive_native_failures = 0
            regularization = max(1e-9, regularization/2)
            if alpha_used == 1.:
                trust = np.minimum(trust*1.3, initial_trust*10)
        elif usable_step:
            consecutive_native_failures = 0
            trust *= .5
            regularization = min(1e12, regularization*10)
        else:
            # Native failure says nothing about nonlinear model agreement.
            consecutive_native_failures += 1
            regularization = min(1e12, regularization*10)
        improvement = (previous_merit-current_merit)/max(1., abs(previous_merit))
        row = dict(iteration=iteration+1, accepted=accepted, alpha=alpha_used,
                   merit=current_merit, cost=float(.5*(residual @ residual)),
                   maximum_dynamics_defect=float(np.max(np.abs(gaps))),
                   l1_scaled_dynamics_defect=float(np.sum(np.abs(scaled))),
                   relative_improvement=improvement, regularization=regularization,
                   maximum_state_trust=float(np.max(trust[:m])), maximum_control_trust=float(np.max(trust[m:])),
                   qp_status=solved.info.status, qp_iterations=int(solved.info.iter),
                   qp_primal_residual=float(solved.info.prim_res), qp_dual_residual=float(solved.info.dual_res),
                   qp_native_accepted=bool(native_ok), objective_scale=scale,
                   qp_step_usable=usable_step, inexact_qp_candidate=inexact_candidate,
                   inexact_primal_tolerance=inexact_primal_tolerance,
                   qp_requested_tolerance=used_qp_tolerance,
                   qp_variables=int(hessian.shape[0]), qp_constraint_rows=int(constraints.shape[0]),
                   bounded_linear_node_rows=int(linear.shape[0]),
                   invalid_transition_trials=invalid_trials, hard_bound_rejected_trials=bound_rejections)
        history.append(row)
        if callback:
            callback(row.copy())
        if consecutive_native_failures >= 3 or np.max(trust) < 1e-8:
            break
        if accepted and improvement < 1e-8 and np.max(np.abs(gaps)) <= 1e-10:
            break
    states, controls = p.unpack(values)
    _, gains, active = square_root_backward_pass(transition, states, controls, cost, gaps=gaps,
                                                running_costs=p.costs, regularization=regularization,
                                                state_epsilon=state_epsilon, action_epsilon=action_epsilon)
    converged = bool(history[-1]['accepted'] and history[-1]['relative_improvement'] < 1e-8
                     and np.max(np.abs(gaps)) <= 1e-10)
    return ILQRResult(controls, states, gains, float(.5*(residual @ residual)), len(history),
                      converged, int(np.count_nonzero(active)), history)
