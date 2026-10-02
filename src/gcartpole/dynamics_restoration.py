"""Sparse bounded restoration of continuous-angle MuJoCo trajectories."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares


@dataclass
class RestorationProblem:
    transition: object
    initial_state: np.ndarray
    reference_states: np.ndarray
    reference_controls: np.ndarray
    terminal_target: np.ndarray
    terminal_factor: np.ndarray
    defect_weight: float
    terminal_weight: float
    reference_weight: float
    control_weight: float
    state_epsilon: float = 1e-5
    action_epsilon: float = 1e-4
    defect_factor: np.ndarray | None = None
    reference_factor: np.ndarray | None = None

    def __post_init__(self):
        self.steps = len(self.reference_controls)
        self.nx = len(self.initial_state)
        for name in ("defect_factor", "reference_factor"):
            factor = getattr(self, name)
            factor = np.eye(self.nx) if factor is None else np.asarray(factor, dtype=float)
            if factor.shape != (self.nx, self.nx) or not np.all(np.isfinite(factor)):
                raise ValueError(f"{name} must be a finite square state matrix")
            setattr(self, name, factor)
        if self.reference_states.shape != (self.steps + 1, self.nx):
            raise ValueError("reference state/control dimensions do not match")
        if self.terminal_target.shape != (self.nx,) or self.terminal_factor.shape[1] != self.nx:
            raise ValueError("terminal target/factor dimensions do not match")
        if min(self.defect_weight, self.terminal_weight, self.state_epsilon, self.action_epsilon) <= 0:
            raise ValueError("dynamics/terminal weights and derivative steps must be positive")
        if min(self.reference_weight, self.control_weight) < 0:
            raise ValueError("reference/control weights must be nonnegative")
        self.state_values = self.steps * self.nx
        self.defect_rows = self.state_values
        self.terminal_rows = self.terminal_factor.shape[0]
        self.reference_start = self.defect_rows + self.terminal_rows
        self.control_start = self.reference_start + self.state_values
        self.residual_size = self.control_start + self.steps
        self.history = []

    def unpack(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (self.state_values + self.steps,):
            raise ValueError("restoration decision has inconsistent dimensions")
        states = np.vstack([self.initial_state, values[:self.state_values].reshape(self.steps, self.nx)])
        return states, values[self.state_values:]

    def defects(self, states, controls):
        return np.asarray([
            self.transition(states[index], float(action)) - states[index + 1]
            for index, action in enumerate(controls)
        ])

    def residual(self, values):
        states, controls = self.unpack(values)
        defects = self.defects(states, controls)
        terminal = self.terminal_factor @ (states[-1] - self.terminal_target)
        residual = np.r_[
            np.sqrt(self.defect_weight) * (defects @ self.defect_factor.T).ravel(),
            np.sqrt(self.terminal_weight) * terminal,
            np.sqrt(self.reference_weight) * ((states[1:] - self.reference_states[1:]) @ self.reference_factor.T).ravel(),
            np.sqrt(self.control_weight) * (controls - self.reference_controls),
        ]
        self.history.append(dict(evaluation=len(self.history) + 1,
                                 cost=float(0.5 * residual @ residual),
                                 maximum_defect=float(np.max(np.abs(defects))),
                                 rms_defect=float(np.sqrt(np.mean(defects**2))),
                                 terminal_error=float(np.linalg.norm(terminal))))
        callback = getattr(self, "residual_callback", None)
        if callback is not None:
            callback(self.history[-1].copy())
        return residual

    def jacobian(self, values):
        states, controls = self.unpack(values)
        jacobian = sparse.lil_matrix((self.residual_size, len(values)))
        sqrt_defect = np.sqrt(self.defect_weight)
        identity = self.defect_factor
        for step, action in enumerate(controls):
            a, b = self.transition.linearize(
                states[step], float(action), state_epsilon=self.state_epsilon,
                action_epsilon=self.action_epsilon,
            )
            rows = slice(step * self.nx, (step + 1) * self.nx)
            if step > 0:
                jacobian[rows, (step - 1) * self.nx:step * self.nx] = sqrt_defect * self.defect_factor @ a
            jacobian[rows, step * self.nx:(step + 1) * self.nx] = -sqrt_defect * identity
            jacobian[rows, self.state_values + step] = sqrt_defect * self.defect_factor @ b.reshape(self.nx, 1)
        jacobian[self.defect_rows:self.reference_start,
                 self.state_values - self.nx:self.state_values] = np.sqrt(self.terminal_weight) * self.terminal_factor
        jacobian[self.reference_start:self.control_start, :self.state_values] = (
            np.sqrt(self.reference_weight) * sparse.kron(sparse.eye(self.steps), self.reference_factor)
        )
        jacobian[self.control_start:, self.state_values:] = np.sqrt(self.control_weight) * sparse.eye(self.steps)
        return jacobian.tocsr()


def restore_trajectory(
    problem: RestorationProblem,
    *,
    rail_limit: float,
    max_evaluations: int,
    state_abs_limit: float = 150.0,
    lsmr_max_iterations: int = 300,
    solver: str = "lsmr",
    progress_callback=None,
    inexact_qp_tolerance: float = 0.0,
    monotone_defects: bool = False,
    state_lower_bounds=None,
    state_upper_bounds=None,
):
    if min(rail_limit, max_evaluations, state_abs_limit, lsmr_max_iterations) <= 0:
        raise ValueError("restoration bounds and budgets must be positive")
    if not np.isfinite(inexact_qp_tolerance) or inexact_qp_tolerance < 0:
        raise ValueError("inexact QP tolerance must be finite and nonnegative")
    initial = np.r_[problem.reference_states[1:].ravel(), problem.reference_controls]
    lower = np.full(initial.size, -state_abs_limit)
    upper = np.full(initial.size, state_abs_limit)
    lower[problem.state_values:] = -1.0
    upper[problem.state_values:] = 1.0
    cart_columns = np.arange(problem.steps) * problem.nx
    lower[cart_columns] = -rail_limit
    upper[cart_columns] = rail_limit
    for requested, bound, combine in ((state_lower_bounds, lower, np.maximum),
                                       (state_upper_bounds, upper, np.minimum)):
        if requested is not None:
            requested = np.asarray(requested, dtype=np.float64)
            if requested.shape != (problem.steps, problem.nx) or np.any(np.isnan(requested)):
                raise ValueError("additional state bounds must match all decision nodes and contain no NaNs")
            bound[:problem.state_values] = combine(bound[:problem.state_values], requested.ravel())
    if np.any(lower >= upper):
        raise ValueError("restoration bounds leave no interior")
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("reference trajectory lies outside restoration bounds")
    if solver == "l1-osqp":
        return restore_l1_qp(problem, initial, lower, upper, max_iterations=max_evaluations,
                             progress_callback=progress_callback,
                             inexact_qp_tolerance=inexact_qp_tolerance,
                             monotone_defects=monotone_defects)
    if solver != "lsmr":
        raise ValueError(f"unknown restoration solver: {solver}")
    problem.residual_callback = progress_callback
    result = least_squares(
        problem.residual, initial, jac=problem.jacobian,
        bounds=(lower, upper), x_scale="jac", tr_solver="lsmr",
        tr_options={"maxiter": lsmr_max_iterations, "atol": 1e-8, "btol": 1e-8},
        max_nfev=max_evaluations, ftol=1e-10, xtol=1e-10, gtol=1e-10,
    )
    return result, *problem.unpack(result.x)


def restore_l1_qp(problem, initial, lower, upper, *, max_iterations, progress_callback=None,
                  inexact_qp_tolerance=0.0, monotone_defects=False):
    """Experimental convexified dynamics repair with penalized virtual gaps.

    Slack variables exist only in the optimizer. They must vanish in the
    independently measured exact dynamics before a trajectory is feasible.
    This is an implementation inspired by SCvx, without a convergence claim.
    """
    import osqp

    values = initial.copy()
    n = values.size
    m = problem.defect_rows
    identity = sparse.eye(n, format="csc")
    gap_identity = sparse.eye(m, format="csc")
    zero_nm = sparse.csc_matrix((n, m))
    zero_mn = sparse.csc_matrix((m, n))
    zero_mm = sparse.csc_matrix((m, m))
    trust = np.r_[np.full(problem.state_values, 0.5), np.full(problem.steps, 0.1)]
    regularization = 1e-3
    qp_history = []
    residual = problem.residual(values)

    def merit(res):
        return float(
            np.sqrt(problem.defect_weight) * np.sum(np.abs(res[:m]))
            + 0.5 * res[m:] @ res[m:]
        )

    current_merit = merit(residual)
    def maximum_gap(res):
        return float(np.max(np.abs(res[:m])) / np.sqrt(problem.defect_weight))

    jac = None
    consecutive_native_failures = 0
    for iteration in range(max_iterations):
        if jac is None:
            jac = problem.jacobian(values)
        dynamics_jac = jac[:m] / np.sqrt(problem.defect_weight)
        dynamics_gap = residual[:m] / np.sqrt(problem.defect_weight)
        objective_jac = jac[m:]
        p_delta = objective_jac.T @ objective_jac + regularization * identity
        p = sparse.block_diag([p_delta, zero_mm, zero_mm], format="csc")
        q = np.r_[np.asarray(objective_jac.T @ residual[m:]).ravel(),
                  np.full(2 * m, problem.defect_weight)]
        # A positive scalar preserves the QP minimizer while avoiding huge
        # slack gradients and correspondingly loose relative dual tolerances.
        objective_scale = max(problem.defect_weight, float(np.max(np.abs(p.diagonal()))), 1.0)
        p = p / objective_scale
        q = q / objective_scale
        constraints = sparse.bmat([
            [dynamics_jac, -gap_identity, gap_identity],
            [identity, zero_nm, zero_nm],
            [zero_mn, gap_identity, zero_mm],
            [zero_mn, zero_mm, gap_identity],
        ], format="csc")
        step_lower = np.maximum(lower - values, -trust)
        step_upper = np.minimum(upper - values, trust)
        l = np.r_[-dynamics_gap, step_lower, np.zeros(2 * m)]
        u = np.r_[-dynamics_gap, step_upper, np.full(2 * m, np.inf)]
        qp = osqp.OSQP()
        qp.setup(P=sparse.triu(p, format="csc"), q=q, A=constraints, l=l, u=u,
                 verbose=False, eps_abs=1e-9, eps_rel=1e-9, max_iter=10000,
                 polishing=True)
        solved = qp.solve(raise_error=False)
        accepted = False
        alpha = 1.0
        native_optimal = solved.info.status_val in (1, 2)
        candidate_tolerance = min(inexact_qp_tolerance, 0.1 * maximum_gap(residual))
        inexact_candidate = (solved.info.status_val == 7 and candidate_tolerance > 0
                            and max(solved.info.prim_res, solved.info.dual_res) <= candidate_tolerance)
        usable_step = (solved.x is not None and np.all(np.isfinite(solved.x))
                       and (native_optimal or inexact_candidate))
        if usable_step:
            step = np.clip(solved.x[:n], step_lower, step_upper)
            while alpha >= 1 / 128:
                candidate = np.clip(values + alpha * step, lower, upper)
                trial = problem.residual(candidate)
                trial_merit = merit(trial)
                gap_allowed = not monotone_defects or maximum_gap(trial) <= maximum_gap(residual) + 1e-12
                if trial_merit < current_merit and gap_allowed:
                    values = candidate
                    residual = trial
                    current_merit = trial_merit
                    accepted = True
                    break
                alpha *= 0.5
        qp_history.append(dict(iteration=iteration + 1, accepted=accepted,
                               qp_status=solved.info.status, qp_iterations=int(solved.info.iter),
                               qp_primal_residual=float(solved.info.prim_res),
                               qp_dual_residual=float(solved.info.dual_res),
                               objective_scale=objective_scale,
                               inexact_candidate=bool(inexact_candidate),
                               candidate_tolerance=candidate_tolerance,
                               alpha=alpha, merit=current_merit,
                               maximum_defect=maximum_gap(residual)))
        qp_history[-1]["gap_norm"] = "linf"
        if progress_callback is not None:
            progress_callback(qp_history[-1].copy())
        if accepted:
            jac = None
            consecutive_native_failures = 0
            regularization = max(1e-8, regularization * 0.5)
            if alpha == 1.0:
                trust = np.minimum(trust * 1.3, np.r_[np.full(problem.state_values, 2.0), np.full(problem.steps, 0.5)])
        elif usable_step:
            consecutive_native_failures = 0
            trust *= 0.5
            regularization *= 10.0
        else:
            # Native nonconvergence does not measure nonlinear model accuracy.
            # Keep the trust region and reuse derivatives at unchanged values.
            consecutive_native_failures += 1
            regularization *= 10.0
            if consecutive_native_failures >= 3:
                break
        if np.max(trust) < 1e-6:
            break
    final_defect = maximum_gap(residual)
    result = SimpleNamespace(x=values, cost=current_merit, nfev=len(problem.history),
                             status=1 if final_defect <= 1e-8 else 0,
                             success=final_defect <= 1e-8,
                             message="Experimental L1 dynamics restoration; nonlinear feasibility is checked independently",
                             qp_history=qp_history)
    return result, *problem.unpack(values)
