"""Feasible iLQR with QR elimination of cost residuals.

The value Hessian stays in factored form throughout the backward pass.
This avoids forming and subtracting ill-conditioned Riccati matrices.
"""
from __future__ import annotations

import numpy as np

from .ilqr import (
    ILQRResult, QuadraticTrajectoryCost, _rail_cost_derivatives,
    rollout, stage_cost, terminal_cost, wrapped_state, resolve_running_costs,
)
from .simulation import SimulationError


def _compact(factor, residual):
    nx = factor.shape[1]
    _, upper = np.linalg.qr(np.column_stack([factor, residual]), mode="reduced")
    return upper[:nx, :nx], upper[:nx, nx]


def _inside_rail(state, cost):
    """Use the same inclusive cart bound as the physical episode gate."""
    return bool(np.isfinite(state[0]) and abs(state[0]) <= cost.rail_limit)


def _validate_initial_rail(states, cost, costs):
    # A rejection filter preserves an admissible initializer; it cannot repair
    # an inadmissible one. FDDP nodes can still have explicit dynamics defects.
    node_costs = (*costs, cost)
    if any(not np.isfinite(c.rail_limit) or c.rail_limit <= 0 for c in node_costs):
        raise ValueError("hard rail enforcement requires finite positive limits")
    for step, (state, node_cost) in enumerate(zip(states, node_costs)):
        if not _inside_rail(state, node_cost):
            raise ValueError(f"initial trajectory violates hard rail at node {step}")


def _state_residual(state, cost, n_links, *, terminal):
    state_x = wrapped_state(state, n_links) if cost.wrap_angles else state
    target = cost.terminal_target if terminal else cost.stage_target
    error = state_x if target is None else state_x - target
    if terminal and cost.terminal_factor is not None:
        factor = cost.terminal_factor.copy()
    elif not terminal and cost.stage_factor is not None:
        factor = cost.stage_factor.copy()
    else:
        matrix = cost.terminal_state if terminal else cost.stage_state
        factor = np.linalg.cholesky(matrix).T
    residual = factor @ error
    _, gradient, curvature = _rail_cost_derivatives(float(state_x[0]), cost)
    if curvature > 0:
        row = np.zeros(state.size)
        row[0] = np.sqrt(curvature)
        factor = np.vstack([factor, row])
        residual = np.r_[residual, gradient / row[0]]
    return factor, residual


def square_root_backward_pass(
    transition, states, controls, cost: QuadraticTrajectoryCost, *,
    regularization=1e-4, state_epsilon=1e-5, action_epsilon=1e-4,
    gaps=None, running_costs=None,
):
    costs = resolve_running_costs(cost, len(controls), running_costs)
    if regularization < 0 or any(item.control <= 0 for item in costs):
        raise ValueError("control cost must be positive and regularization nonnegative")
    nx = states.shape[1]
    factor, residual = _state_residual(states[-1], cost, transition.env.n, terminal=True)
    root, offset = _compact(factor, residual)
    feedforward = np.zeros(controls.size)
    feedback = np.zeros((controls.size, nx))
    active = np.zeros(controls.size, dtype=bool)
    for step in range(controls.size - 1, -1, -1):
        a, b = transition.linearize(
            states[step], float(controls[step]),
            state_epsilon=state_epsilon, action_epsilon=action_epsilon,
        )
        stage_root, stage_offset = _state_residual(
            states[step], costs[step], transition.env.n, terminal=False,
        )
        control_root = np.sqrt(costs[step].control)
        jx = np.vstack([stage_root, np.zeros((2, nx)), root @ a])
        ju = np.r_[np.zeros(stage_root.shape[0]), control_root,
                   np.sqrt(regularization), (root @ b).reshape(-1)]
        next_offset = offset if gaps is None else offset + root @ gaps[step]
        r = np.r_[stage_offset, control_root * controls[step], 0., next_offset]
        # Eliminate delta-u from the residual system before propagating the
        # state value. Orthogonal QR replaces the subtractive Schur update.
        _, upper = np.linalg.qr(np.column_stack([ju, jx, r]), mode="reduced")
        pivot = upper[0, 0]
        if not np.isfinite(pivot) or pivot == 0:
            raise np.linalg.LinAlgError(f"invalid control factor at step {step}")
        unconstrained = -upper[0, -1] / pivot
        delta = float(np.clip(unconstrained, -1 - controls[step], 1 - controls[step]))
        active[step] = delta != unconstrained
        feedforward[step] = delta
        if active[step]:
            # At a scalar bound the local feedback derivative is zero.
            root, offset = _compact(jx, r + ju * delta)
        else:
            feedback[step] = -upper[0, 1:nx + 1] / pivot
            root = upper[1:nx + 1, 1:nx + 1]
            offset = upper[1:nx + 1, -1]
        if not np.all(np.isfinite(root)) or not np.all(np.isfinite(offset)):
            raise np.linalg.LinAlgError(f"nonfinite value factor at step {step}")
    return feedforward, feedback, active


def square_root_tracking_gains(
    transition, states, controls, cost: QuadraticTrajectoryCost, *,
    regularization=0., state_epsilon=1e-5, action_epsilon=1e-4,
    running_costs=None,
):
    """Design pure perturbation feedback about the supplied nominal route.

    Unlike the optimization backward pass, this recurrence has no affine
    descent step whose clipping can disable feedback. The delivered action
    is still clipped by the physical controller; these gains do not certify
    tracking or actuator feasibility. Targets affect the nominal objective,
    not the perturbation Hessian used here.
    """
    costs = resolve_running_costs(cost, len(controls), running_costs)
    if not np.isfinite(regularization) or regularization < 0 or any(c.control <= 0 for c in costs):
        raise ValueError('tracking regularization must be finite and nonnegative; control costs positive')
    nx = states.shape[1]
    factor, _ = _state_residual(states[-1], cost, transition.env.n, terminal=True)
    _, root = np.linalg.qr(factor, mode='reduced')
    gains = np.zeros((len(controls), nx))
    for step in range(len(controls)-1, -1, -1):
        a, b = transition.linearize(states[step], float(controls[step]),
                                    state_epsilon=state_epsilon, action_epsilon=action_epsilon)
        stage_root, _ = _state_residual(states[step], costs[step], transition.env.n, terminal=False)
        jx = np.vstack([stage_root, np.zeros((2, nx)), root @ a])
        ju = np.r_[np.zeros(stage_root.shape[0]), np.sqrt(costs[step].control),
                   np.sqrt(regularization), (root @ b).ravel()]
        _, upper = np.linalg.qr(np.column_stack([ju, jx]), mode='reduced')
        pivot = upper[0, 0]
        if not np.isfinite(pivot) or pivot == 0:
            raise np.linalg.LinAlgError(f'invalid tracking control factor at step {step}')
        gains[step] = -upper[0, 1:nx+1]/pivot
        root = upper[1:nx+1, 1:nx+1]
        if not np.all(np.isfinite(root)) or not np.all(np.isfinite(gains[step])):
            raise np.linalg.LinAlgError(f'nonfinite tracking factor at step {step}')
    return gains


def optimize_sqrt_ilqr(
    transition, initial_state, initial_controls, cost: QuadraticTrajectoryCost, *,
    max_iterations=40, initial_regularization=1e-4,
    state_epsilon=1e-5, action_epsilon=1e-4, tolerance=1e-8, callback=None,
    running_costs=None, enforce_rail=False,
):
    controls = np.clip(np.asarray(initial_controls, dtype=float), -1., 1.)
    costs = resolve_running_costs(cost, len(controls), running_costs)
    states, current_cost = rollout(transition, initial_state, controls, cost, transition.env.n,
                                  running_costs=costs)
    if enforce_rail:
        _validate_initial_rail(states, cost, costs)
    regularization = max(1e-9, float(initial_regularization))
    history = []
    converged = False
    feedback = np.zeros((controls.size, initial_state.size))
    active = np.zeros(controls.size, dtype=bool)
    for iteration in range(max_iterations):
        previous_cost = current_cost
        accepted = False
        alpha_used = None
        invalid_trials = 0
        rail_rejections = 0
        try:
            delta, gains, candidate_active = square_root_backward_pass(
                transition, states, controls, cost, regularization=regularization,
                state_epsilon=state_epsilon, action_epsilon=action_epsilon,
                running_costs=costs,
            )
        except np.linalg.LinAlgError as error:
            history.append(dict(iteration=iteration + 1, cost=current_cost,
                                accepted=False, regularization=regularization, error=str(error)))
            if callback is not None:
                callback(history[-1])
            regularization = min(1e12, regularization * 10)
            continue
        for alpha in (1., .5, .25, .1, .05, .01, .001, .0001):
            candidate_controls = np.empty_like(controls)
            candidate_states = np.empty_like(states)
            candidate_states[0] = initial_state
            candidate_cost = 0.
            try:
                for step in range(controls.size):
                    error = transition.difference(candidate_states[step], states[step])
                    action = float(np.clip(controls[step] + alpha * delta[step] + gains[step] @ error, -1., 1.))
                    candidate_controls[step] = action
                    candidate_cost += stage_cost(candidate_states[step], action, costs[step], transition.env.n)
                    candidate_states[step + 1] = transition(candidate_states[step], action)
                    next_cost = cost if step + 1 == controls.size else costs[step + 1]
                    if enforce_rail and not _inside_rail(candidate_states[step + 1], next_cost):
                        rail_rejections += 1
                        candidate_cost = np.inf
                        break
                if np.isfinite(candidate_cost):
                    candidate_cost += terminal_cost(candidate_states[-1], cost, transition.env.n)
            except SimulationError:
                invalid_trials += 1
                candidate_cost = np.inf
            if np.isfinite(candidate_cost) and candidate_cost < current_cost:
                controls, states = candidate_controls, candidate_states
                current_cost = float(candidate_cost)
                feedback, active = gains, candidate_active
                regularization = max(1e-9, regularization / 3)
                accepted = True
                alpha_used = alpha
                break
        if not accepted:
            regularization = min(1e12, regularization * 10)
        improvement = (previous_cost - current_cost) / max(1., abs(previous_cost))
        history.append(dict(iteration=iteration + 1, cost=current_cost, accepted=accepted,
                            alpha=alpha_used,
                            relative_improvement=improvement, regularization=regularization,
                            invalid_transition_trials=invalid_trials,
                            rail_rejected_trials=rail_rejections,
                            active_control_steps=int(np.count_nonzero(candidate_active))))
        if callback is not None:
            callback(history[-1])
        if accepted and improvement < tolerance:
            converged = True
            break
    # Feedback must describe the final saved trajectory, including when a
    # bounded run ends before accepting a step.
    _, feedback, active = square_root_backward_pass(
        transition, states, controls, cost, regularization=regularization,
        state_epsilon=state_epsilon, action_epsilon=action_epsilon,
        running_costs=costs,
    )
    return ILQRResult(controls, states, feedback, float(current_cost), len(history),
                      converged, int(np.count_nonzero(active)), history)
