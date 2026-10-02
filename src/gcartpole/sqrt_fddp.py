"""Factored local quadratic refinement with explicit dynamics defects.

Virtual nodes are optimizer variables only. A unit step is a serial physical
rollout; smaller steps retain declared defects and never constitute release
evidence without an independent full physical replay.
"""
from __future__ import annotations

import numpy as np

from .ilqr import ILQRResult, stage_cost, terminal_cost, resolve_running_costs
from .simulation import SimulationError
from .sqrt_ilqr import square_root_backward_pass, _inside_rail, _validate_initial_rail


def trajectory_gaps(transition, states, controls):
    return np.asarray([transition(states[i], float(u)) - states[i + 1]
                       for i, u in enumerate(controls)])


def trajectory_cost(states, controls, cost, n, running_costs=None):
    costs = resolve_running_costs(cost, len(controls), running_costs)
    return float(sum(stage_cost(x, float(u), stage, n) for x, u, stage in zip(states[:-1], controls, costs))
                 + terminal_cost(states[-1], cost, n))


def optimize_sqrt_fddp(
    transition, initial_state, initial_controls, initial_states, cost, *,
    max_iterations=40, initial_regularization=10., defect_penalty=1e8,
    state_epsilon=1e-5, action_epsilon=1e-4, tolerance=1e-8, callback=None,
    running_costs=None, enforce_rail=False,
):
    if defect_penalty <= 0 or not np.isfinite(defect_penalty):
        raise ValueError("finite positive dynamics defect penalty required")
    controls = np.clip(np.asarray(initial_controls, dtype=float), -1., 1.)
    costs = resolve_running_costs(cost, len(controls), running_costs)
    states = np.asarray(initial_states, dtype=float).copy()
    if states.shape != (controls.size + 1, initial_state.size):
        raise ValueError("initial state nodes do not match the control horizon")
    states[0] = initial_state
    if enforce_rail:
        _validate_initial_rail(states, cost, costs)
    gaps = trajectory_gaps(transition, states, controls)
    current_cost = trajectory_cost(states, controls, cost, transition.env.n, costs)
    merit = current_cost + defect_penalty * float(np.sum(np.abs(gaps)))
    regularization = max(1e-9, float(initial_regularization))
    history = []
    converged = False
    for iteration in range(max_iterations):
        previous_merit = merit
        accepted = False
        invalid_trials = 0
        rail_rejections = 0
        delta, gains, active = square_root_backward_pass(
            transition, states, controls, cost, gaps=gaps,
            regularization=regularization,
            state_epsilon=state_epsilon, action_epsilon=action_epsilon,
            running_costs=costs,
        )
        alpha_used = None
        for alpha in (1., .5, .25, .1, .05, .01, .001, .0001):
            trial_states = np.empty_like(states)
            trial_controls = np.empty_like(controls)
            trial_states[0] = initial_state
            rail_violated = False
            try:
                for step in range(controls.size):
                    error = trial_states[step] - states[step]
                    action = float(np.clip(controls[step] + alpha * delta[step] + gains[step] @ error, -1., 1.))
                    trial_controls[step] = action
                    # The retained defect is explicit, not hidden inside the
                    # physical transition or the eventual controller replay.
                    trial_states[step + 1] = transition(trial_states[step], action) - (1 - alpha) * gaps[step]
                    next_cost = cost if step + 1 == controls.size else costs[step + 1]
                    if enforce_rail and not _inside_rail(trial_states[step + 1], next_cost):
                        rail_rejections += 1
                        rail_violated = True
                        break
                if rail_violated:
                    continue
                trial_gaps = trajectory_gaps(transition, trial_states, trial_controls)
                trial_cost = trajectory_cost(trial_states, trial_controls, cost, transition.env.n, costs)
                trial_merit = trial_cost + defect_penalty * float(np.sum(np.abs(trial_gaps)))
            except SimulationError:
                invalid_trials += 1
                continue
            if np.isfinite(trial_merit) and trial_merit < merit:
                controls, states, gaps = trial_controls, trial_states, trial_gaps
                current_cost, merit = trial_cost, trial_merit
                regularization = max(1e-9, regularization / 3)
                accepted = True
                alpha_used = alpha
                break
        if not accepted:
            regularization = min(1e12, regularization * 10)
        improvement = (previous_merit - merit) / max(1., abs(previous_merit))
        history.append(dict(iteration=iteration + 1, cost=current_cost, merit=merit,
                            accepted=accepted, alpha=alpha_used,
                            relative_improvement=improvement, regularization=regularization,
                            maximum_dynamics_defect=float(np.max(np.abs(gaps))),
                            l1_dynamics_defect=float(np.sum(np.abs(gaps))),
                            invalid_transition_trials=invalid_trials,
                            rail_rejected_trials=rail_rejections,
                            active_control_steps=int(np.count_nonzero(active))))
        if callback is not None:
            callback(history[-1])
        if accepted and improvement < tolerance and np.max(np.abs(gaps)) <= 1e-8:
            converged = True
            break
    _, gains, active = square_root_backward_pass(
        transition, states, controls, cost, gaps=gaps, regularization=regularization,
        state_epsilon=state_epsilon, action_epsilon=action_epsilon,
        running_costs=costs,
    )
    return ILQRResult(controls, states, gains, current_cost, len(history), converged,
                      int(np.count_nonzero(active)), history)
