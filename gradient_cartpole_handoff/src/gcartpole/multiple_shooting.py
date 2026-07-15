from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import BFGS, Bounds, NonlinearConstraint, least_squares, minimize

from gcartpole.ilqr import Array, MujocoTransition


@dataclass(frozen=True)
class MultipleShootingResult:
    controls: Array
    node_states: Array
    exact_states: Array
    segment_defects: Array
    cost: float
    optimality: float
    evaluations: int
    status: int
    message: str
    success: bool


def segment_count(horizon_steps: int, segment_steps: int) -> int:
    if horizon_steps < 1 or segment_steps < 1 or horizon_steps % segment_steps != 0:
        raise ValueError("horizon_steps must be a positive multiple of segment_steps")
    return horizon_steps // segment_steps


def pack_decision(node_states: Array, controls: Array) -> Array:
    nodes = np.asarray(node_states, dtype=np.float64)
    actions = np.asarray(controls, dtype=np.float64)
    if nodes.ndim != 2 or actions.ndim != 1:
        raise ValueError("node states must be a matrix and controls must be a vector")
    return np.r_[nodes.reshape(-1), actions]


def unpack_decision(decision: Array, *, nx: int, horizon_steps: int, segment_steps: int) -> tuple[Array, Array]:
    segments = segment_count(horizon_steps, segment_steps)
    decision = np.asarray(decision, dtype=np.float64)
    state_values = segments * nx
    if decision.shape != (state_values + horizon_steps,):
        raise ValueError("multiple-shooting decision has the wrong dimension")
    return decision[:state_values].reshape(segments, nx), decision[state_values:]


def exact_rollout(transition: MujocoTransition, initial_state: Array, controls: Array) -> Array:
    controls = np.asarray(controls, dtype=np.float64)
    states = np.empty((len(controls) + 1, len(initial_state)), dtype=np.float64)
    states[0] = initial_state
    for step, action in enumerate(controls):
        states[step + 1] = transition(states[step], float(action))
    return states


def shooting_sparsity(*, nx: int, horizon_steps: int, segment_steps: int) -> sparse.csr_matrix:
    segments = segment_count(horizon_steps, segment_steps)
    state_values = segments * nx
    residual_count = segments * nx + nx + horizon_steps + horizon_steps
    variable_count = state_values + horizon_steps
    pattern = sparse.lil_matrix((residual_count, variable_count), dtype=np.int8)

    for segment in range(segments):
        rows = slice(segment * nx, (segment + 1) * nx)
        end_columns = slice(segment * nx, (segment + 1) * nx)
        pattern[rows, end_columns] = 1
        if segment > 0:
            start_columns = slice((segment - 1) * nx, segment * nx)
            pattern[rows, start_columns] = 1
        action_start = state_values + segment * segment_steps
        action_end = action_start + segment_steps
        pattern[rows, action_start:action_end] = 1

    terminal_start = segments * nx
    pattern[terminal_start : terminal_start + nx, (segments - 1) * nx : segments * nx] = 1
    control_start = terminal_start + nx
    for step in range(horizon_steps):
        pattern[control_start + step, state_values + step] = 1

    rail_start = control_start + horizon_steps
    for step in range(horizon_steps):
        segment = step // segment_steps
        if segment > 0:
            pattern[rail_start + step, (segment - 1) * nx : segment * nx] = 1
        segment_action_start = state_values + segment * segment_steps
        pattern[rail_start + step, segment_action_start : state_values + step + 1] = 1
    return pattern.tocsr()


def defect_sparsity(*, nx: int, horizon_steps: int, segment_steps: int) -> sparse.csr_matrix:
    segments = segment_count(horizon_steps, segment_steps)
    return shooting_sparsity(
        nx=nx,
        horizon_steps=horizon_steps,
        segment_steps=segment_steps,
    )[: segments * nx]


def _decision_bounds(
    *,
    segments: int,
    nx: int,
    horizon_steps: int,
    rail_limit: float,
    state_abs_limit: float = 100.0,
) -> tuple[Array, Array]:
    state_values = segments * nx
    lower = np.full(state_values + horizon_steps, -state_abs_limit, dtype=np.float64)
    upper = np.full(state_values + horizon_steps, state_abs_limit, dtype=np.float64)
    lower[state_values:] = -1.0
    upper[state_values:] = 1.0
    for segment in range(segments):
        lower[segment * nx] = -rail_limit
        upper[segment * nx] = rail_limit
    return lower, upper


def optimize_multiple_shooting(
    transition: MujocoTransition,
    initial_state: Array,
    initial_controls: Array,
    *,
    segment_steps: int,
    defect_weight: float,
    terminal_weight: float,
    control_weight: float,
    rail_weight: float,
    rail_soft_limit: float,
    rail_limit: float,
    max_evaluations: int,
    initial_node_states: Array | None = None,
) -> MultipleShootingResult:
    controls = np.clip(np.asarray(initial_controls, dtype=np.float64), -1.0, 1.0)
    horizon_steps = len(controls)
    segments = segment_count(horizon_steps, segment_steps)
    nx = len(initial_state)
    initial_exact_states = exact_rollout(transition, initial_state, controls)
    node_states = (
        initial_exact_states[segment_steps::segment_steps].copy()
        if initial_node_states is None
        else np.asarray(initial_node_states, dtype=np.float64).copy()
    )
    if node_states.shape != (segments, nx):
        raise ValueError("initial node states do not match the shooting grid")
    decision = pack_decision(node_states, controls)
    lower, upper = _decision_bounds(
        segments=segments,
        nx=nx,
        horizon_steps=horizon_steps,
        rail_limit=rail_limit,
    )

    sqrt_defect = float(np.sqrt(defect_weight))
    sqrt_terminal = float(np.sqrt(terminal_weight))
    sqrt_control = float(np.sqrt(control_weight))
    sqrt_rail = float(np.sqrt(rail_weight))
    rail_width = max(1e-9, rail_limit - rail_soft_limit)

    def residuals(values: Array) -> Array:
        nodes, actions = unpack_decision(
            values,
            nx=nx,
            horizon_steps=horizon_steps,
            segment_steps=segment_steps,
        )
        defects = np.empty((segments, nx), dtype=np.float64)
        rail = np.empty(horizon_steps, dtype=np.float64)
        for segment in range(segments):
            state = initial_state if segment == 0 else nodes[segment - 1]
            for local_step in range(segment_steps):
                step = segment * segment_steps + local_step
                state = transition(state, float(actions[step]))
                rail[step] = max(0.0, abs(float(state[0])) - rail_soft_limit) / rail_width
            defects[segment] = transition.difference(state, nodes[segment])
        return np.r_[
            sqrt_defect * defects.reshape(-1),
            sqrt_terminal * nodes[-1],
            sqrt_control * actions,
            sqrt_rail * rail,
        ]

    result = least_squares(
        residuals,
        decision,
        jac="2-point",
        jac_sparsity=shooting_sparsity(
            nx=nx,
            horizon_steps=horizon_steps,
            segment_steps=segment_steps,
        ),
        bounds=(lower, upper),
        x_scale="jac",
        tr_solver="lsmr",
        max_nfev=max_evaluations,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
        verbose=0,
    )
    nodes, controls = unpack_decision(
        result.x,
        nx=nx,
        horizon_steps=horizon_steps,
        segment_steps=segment_steps,
    )
    exact_states = exact_rollout(transition, initial_state, controls)
    defects = np.empty((segments, nx), dtype=np.float64)
    for segment in range(segments):
        start = initial_state if segment == 0 else nodes[segment - 1]
        predicted = start
        for step in range(segment * segment_steps, (segment + 1) * segment_steps):
            predicted = transition(predicted, float(controls[step]))
        defects[segment] = transition.difference(predicted, nodes[segment])
    return MultipleShootingResult(
        controls=controls,
        node_states=nodes,
        exact_states=exact_states,
        segment_defects=defects,
        cost=float(result.cost),
        optimality=float(result.optimality),
        evaluations=int(result.nfev),
        status=int(result.status),
        message=str(result.message),
        success=bool(result.success),
    )


def optimize_direct_collocation(
    transition: MujocoTransition,
    initial_state: Array,
    initial_controls: Array,
    *,
    segment_steps: int,
    terminal_weight: float,
    control_weight: float,
    rail_limit: float,
    max_iterations: int,
    initial_node_states: Array | None = None,
    solver: str = "slsqp",
) -> MultipleShootingResult:
    controls = np.clip(np.asarray(initial_controls, dtype=np.float64), -1.0, 1.0)
    horizon_steps = len(controls)
    segments = segment_count(horizon_steps, segment_steps)
    nx = len(initial_state)
    initial_exact_states = exact_rollout(transition, initial_state, controls)
    node_states = (
        initial_exact_states[segment_steps::segment_steps].copy()
        if initial_node_states is None
        else np.asarray(initial_node_states, dtype=np.float64).copy()
    )
    if node_states.shape != (segments, nx):
        raise ValueError("initial node states do not match the collocation grid")
    decision = pack_decision(node_states, controls)
    state_values = segments * nx
    lower, upper = _decision_bounds(
        segments=segments,
        nx=nx,
        horizon_steps=horizon_steps,
        rail_limit=rail_limit,
    )

    def defects(values: Array) -> Array:
        nodes, actions = unpack_decision(
            values,
            nx=nx,
            horizon_steps=horizon_steps,
            segment_steps=segment_steps,
        )
        result = np.empty((segments, nx), dtype=np.float64)
        for segment in range(segments):
            state = initial_state if segment == 0 else nodes[segment - 1]
            for step in range(segment * segment_steps, (segment + 1) * segment_steps):
                state = transition(state, float(actions[step]))
            result[segment] = transition.difference(state, nodes[segment])
        return result.reshape(-1)

    def defect_jacobian(values: Array) -> sparse.csr_matrix:
        nodes, actions = unpack_decision(
            values,
            nx=nx,
            horizon_steps=horizon_steps,
            segment_steps=segment_steps,
        )
        jacobian = sparse.lil_matrix((segments * nx, len(values)), dtype=np.float64)
        for segment in range(segments):
            state = initial_state if segment == 0 else nodes[segment - 1]
            state_jacobian = np.eye(nx, dtype=np.float64)
            control_jacobians: list[Array] = []
            for step in range(segment * segment_steps, (segment + 1) * segment_steps):
                a, b = transition.linearize(
                    state,
                    float(actions[step]),
                    state_epsilon=1e-5,
                    action_epsilon=1e-4,
                )
                state_jacobian = a @ state_jacobian
                control_jacobians = [a @ column for column in control_jacobians]
                control_jacobians.append(b)
                state = transition(state, float(actions[step]))
            rows = slice(segment * nx, (segment + 1) * nx)
            if segment > 0:
                jacobian[rows, (segment - 1) * nx : segment * nx] = state_jacobian
            jacobian[rows, segment * nx : (segment + 1) * nx] = -np.eye(nx)
            for local_step, column in enumerate(control_jacobians):
                action_column = state_values + segment * segment_steps + local_step
                jacobian[rows, action_column] = column.reshape(-1, 1)
        return jacobian.tocsr()

    def objective(values: Array) -> float:
        nodes, actions = unpack_decision(
            values,
            nx=nx,
            horizon_steps=horizon_steps,
            segment_steps=segment_steps,
        )
        return float(
            0.5 * terminal_weight * nodes[-1] @ nodes[-1]
            + 0.5 * control_weight * actions @ actions
        )

    def objective_gradient(values: Array) -> Array:
        nodes, actions = unpack_decision(
            values,
            nx=nx,
            horizon_steps=horizon_steps,
            segment_steps=segment_steps,
        )
        gradient = np.zeros_like(values)
        gradient[(segments - 1) * nx : segments * nx] = terminal_weight * nodes[-1]
        gradient[state_values:] = control_weight * actions
        return gradient

    defect_tolerance = 1e-8
    best_decision = decision.copy()
    best_objective = objective(best_decision)
    best_defect = float(np.max(np.abs(defects(best_decision)), initial=0.0))

    def retain_feasible(values: Array, *_: object) -> bool:
        nonlocal best_decision, best_objective, best_defect
        candidate_defect = float(np.max(np.abs(defects(values)), initial=0.0))
        candidate_objective = objective(values)
        if (
            np.isfinite(candidate_defect)
            and np.isfinite(candidate_objective)
            and candidate_defect <= defect_tolerance
            and candidate_objective < best_objective
        ):
            best_decision = np.asarray(values, dtype=np.float64).copy()
            best_objective = float(candidate_objective)
            best_defect = float(candidate_defect)
        return False

    if solver == "slsqp":
        result = minimize(
            objective,
            decision,
            method="SLSQP",
            jac=objective_gradient,
            constraints=[
                {
                    "type": "eq",
                    "fun": defects,
                    "jac": lambda values: defect_jacobian(values).toarray(),
                }
            ],
            bounds=Bounds(lower, upper),
            callback=retain_feasible,
            options={
                "maxiter": int(max_iterations),
                "ftol": 1e-9,
                "disp": False,
            },
        )
    elif solver == "trust-constr":
        constraint = NonlinearConstraint(
            defects,
            np.zeros(segments * nx, dtype=np.float64),
            np.zeros(segments * nx, dtype=np.float64),
            jac=defect_jacobian,
            hess=BFGS(),
        )
        result = minimize(
            objective,
            decision,
            method="trust-constr",
            jac=objective_gradient,
            hess=BFGS(),
            constraints=[constraint],
            bounds=Bounds(lower, upper),
            callback=retain_feasible,
            options={
                "maxiter": int(max_iterations),
                "gtol": 1e-6,
                "xtol": 1e-8,
                "barrier_tol": 1e-8,
                "sparse_jacobian": True,
                "verbose": 0,
            },
        )
    else:
        raise ValueError(f"unknown direct-collocation solver: {solver}")
    retain_feasible(result.x)
    nodes, controls = unpack_decision(
        best_decision,
        nx=nx,
        horizon_steps=horizon_steps,
        segment_steps=segment_steps,
    )
    exact_states = exact_rollout(transition, initial_state, controls)
    final_defects = defects(best_decision).reshape(segments, nx)
    if hasattr(result, "optimality"):
        optimality = float(result.optimality)
    elif sparse.issparse(result.jac):
        optimality = float(np.max(np.abs(result.jac.data), initial=0.0))
    else:
        optimality = float(np.linalg.norm(result.jac, ord=np.inf))
    return MultipleShootingResult(
        controls=controls,
        node_states=nodes,
        exact_states=exact_states,
        segment_defects=final_defects,
        cost=float(best_objective),
        optimality=optimality,
        evaluations=int(result.nfev),
        status=int(result.status),
        message=(
            f"{result.message}; retained best iterate with max defect {best_defect:.3e}"
        ),
        success=bool(result.success and best_defect <= defect_tolerance),
    )
