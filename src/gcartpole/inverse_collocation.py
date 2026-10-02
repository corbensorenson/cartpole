"""Continuous inverse-dynamics spline discovery; exact replay is still mandatory."""
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy import sparse
from scipy.interpolate import BSpline
from scipy.optimize import least_squares


@dataclass
class SplineInverseProblem:
    model: object
    initial_position: np.ndarray
    terminal_position: np.ndarray
    seconds: float
    control_points: int
    sample_times: np.ndarray
    force_limit: float
    dynamics_weight: float = 1.0
    force_weight: float = 100.0
    effort_weight: float = 1e-6
    reference_weight: float = 1e-5
    progress_callback: object = None
    dynamics_residual: str = "torque"
    knots: object = None
    initial_control_points: object = None

    def __post_init__(self):
        self.d = self.model.nv
        self.initial_position = np.asarray(self.initial_position, dtype=float)
        self.terminal_position = np.asarray(self.terminal_position, dtype=float)
        self.sample_times = np.asarray(self.sample_times, dtype=float)
        if self.model.nq != self.d or self.control_points < 8 or self.seconds <= 0:
            raise ValueError("scalar-joint model, at least eight points, and positive duration required")
        if (self.initial_position.shape != (self.d,) or self.terminal_position.shape != (self.d,)
                or self.sample_times.ndim != 1 or not len(self.sample_times)
                or not np.all(np.isfinite(np.r_[self.initial_position, self.terminal_position, self.sample_times]))
                or np.any(self.sample_times < 0) or np.any(self.sample_times > self.seconds)):
            raise ValueError("finite scalar-joint endpoints and sample times within the interval required")
        if min(self.force_limit, self.dynamics_weight, self.force_weight) <= 0 or min(self.effort_weight, self.reference_weight) < 0:
            raise ValueError("invalid inverse-collocation weights")
        if self.dynamics_residual not in ("torque", "acceleration"):
            raise ValueError("unknown dynamics residual")
        degree = 5
        if self.knots is None:
            interior = np.linspace(0, self.seconds, self.control_points - degree + 1)[1:-1]
            self.knots = np.r_[np.zeros(degree + 1), interior, np.full(degree + 1, self.seconds)]
        self.knots = np.asarray(self.knots, dtype=float)
        if (self.knots.shape != (self.control_points + degree + 1,)
                or not np.all(np.isfinite(self.knots)) or np.any(np.diff(self.knots) < 0)
                or not np.all(self.knots[:degree + 1] == 0)
                or not np.all(self.knots[-degree - 1:] == self.seconds)):
            raise ValueError("finite clamped quintic knots required")
        spline = BSpline(self.knots, np.eye(self.control_points), degree)
        self.basis = [spline.derivative(order)(self.sample_times) for order in range(3)]
        greville = np.array([np.mean(self.knots[i + 1:i + degree + 1]) for i in range(self.control_points)]) / self.seconds
        blend = 6 * greville**5 - 15 * greville**4 + 10 * greville**3
        self.reference = self.initial_position + blend[:, None] * (self.terminal_position - self.initial_position)
        self.reference[:3] = self.initial_position
        self.reference[-3:] = self.terminal_position
        if self.initial_control_points is not None:
            points = np.asarray(self.initial_control_points, dtype=float)
            if (points.shape != self.reference.shape or not np.all(np.isfinite(points))
                    or not np.allclose(points[:3], self.initial_position, atol=1e-12, rtol=0)
                    or not np.allclose(points[-3:], self.terminal_position, atol=1e-12, rtol=0)):
                raise ValueError("initial spline must preserve all endpoint conditions")
            self.reference = points.copy()
        self.free_points = self.control_points - 6
        self.nvars = self.free_points * self.d
        self.initial_values = self.reference[3:-3].ravel().copy()
        self.data = mujoco.MjData(self.model)
        self.inverse_calls = 0
        self.derivative_calls = 0
        horizontal = np.zeros(self.d)
        horizontal[1] = np.pi / 2
        self.torque_scales = np.maximum(np.abs(self.inverse(horizontal, np.zeros(self.d), np.zeros(self.d))[1:]), 1e-4)
        self.node_count = len(self.sample_times)
        self.residual_dimension = self.d if self.dynamics_residual == "acceleration" else self.d - 1
        self.joint_rows = self.node_count * self.residual_dimension
        self.force_start = self.joint_rows
        self.effort_start = self.force_start + self.node_count
        self.reference_start = self.effort_start + self.node_count
        self.history = []

    def unpack(self, values):
        points = self.reference.copy()
        points[3:-3] = np.asarray(values).reshape(self.free_points, self.d)
        return points

    def trajectory(self, values):
        points = self.unpack(values)
        return [basis @ points for basis in self.basis]

    def inverse(self, q, v, a):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = q
        self.data.qvel[:] = v
        self.data.qacc[:] = a
        mujoco.mj_inverse(self.model, self.data)
        self.inverse_calls += 1
        force = self.data.qfrc_inverse.copy()
        if not np.all(np.isfinite(force)):
            raise ValueError("inverse dynamics returned non-finite forces")
        return force

    def forces(self, values):
        positions, velocities, accelerations = self.trajectory(values)
        forces = np.array([self.inverse(q, v, a) for q, v, a in zip(positions, velocities, accelerations)])
        return forces

    def acceleration_residual(self, q, v, a):
        force = self.inverse(q, v, a)
        mass = np.zeros((self.d, self.d))
        mujoco.mj_fullM(self.model, mass, self.data.qM)
        missing_force = force.copy()
        missing_force[0] = 0.
        return np.linalg.solve(mass, missing_force)

    def acceleration_derivatives(self, q, v, a):
        self.inverse(q, v, a)
        mass = np.zeros((self.d, self.d))
        mujoco.mj_fullM(self.model, mass, self.data.qM)
        projected_mass = mass.copy()
        projected_mass[0] = 0.
        da = np.linalg.solve(mass, projected_mass)
        dq, dv = np.zeros_like(da), np.zeros_like(da)
        for column in range(self.d):
            for state, destination, position in ((q, dq, True), (v, dv, False)):
                step = 1e-6 * max(1., abs(state[column]))
                plus, minus = state.copy(), state.copy()
                plus[column] += step
                minus[column] -= step
                if position:
                    rplus, rminus = self.acceleration_residual(plus, v, a), self.acceleration_residual(minus, v, a)
                else:
                    rplus, rminus = self.acceleration_residual(q, plus, a), self.acceleration_residual(q, minus, a)
                destination[:, column] = (rplus - rminus) / (2 * step)
        return dq, dv, da

    def inverse_derivatives(self, q, v, a):
        """Differentiate continuous inverse dynamics without changing RK4.

        MuJoCo's mjd_inverseFD rejects RK4 models. Central differences of
        mj_inverse itself work with the benchmark model; acceleration enters
        affinely through the mass matrix, including joint armature.
        """
        self.inverse(q, v, a)
        da = np.zeros((self.d, self.d))
        mujoco.mj_fullM(self.model, da, self.data.qM)
        dq, dv = np.zeros_like(da), np.zeros_like(da)
        for column in range(self.d):
            for state, destination, position in ((q, dq, True), (v, dv, False)):
                step = 1e-6 * max(1.0, abs(state[column]))
                plus, minus = state.copy(), state.copy()
                plus[column] += step
                minus[column] -= step
                if position:
                    fplus, fminus = self.inverse(plus, v, a), self.inverse(minus, v, a)
                else:
                    fplus, fminus = self.inverse(q, plus, a), self.inverse(q, minus, a)
                destination[:, column] = (fplus - fminus) / (2 * step)
        self.derivative_calls += 1
        return dq, dv, da

    def residual(self, values):
        forces = self.forces(values)
        normalized_force = forces[:, 0] / self.force_limit
        excess = np.maximum(np.abs(normalized_force) - 1, 0)
        if self.dynamics_residual == "acceleration":
            q, v, a = self.trajectory(values)
            dynamics = np.array([self.acceleration_residual(qi, vi, ai) for qi, vi, ai in zip(q, v, a)])
        else:
            dynamics = forces[:, 1:] / self.torque_scales
        residual = np.r_[
            (np.sqrt(self.dynamics_weight) * dynamics).ravel(),
            np.sqrt(self.force_weight) * excess,
            np.sqrt(self.effort_weight) * normalized_force,
            np.sqrt(self.reference_weight) * (values - self.initial_values),
        ]
        self.history.append(dict(evaluation=len(self.history) + 1, cost=float(0.5 * residual @ residual),
                                 max_joint_torque=float(np.max(np.abs(forces[:, 1:]))),
                                 max_normalized_joint_residual=float(np.max(np.abs(forces[:, 1:] / self.torque_scales))),
                                 max_force=float(np.max(np.abs(forces[:, 0]))),
                                 max_dynamics_residual=float(np.max(np.abs(dynamics))),
                                 max_force_excess=float(np.max(excess))))
        if self.progress_callback is not None:
            self.progress_callback(self.history[-1])
        return residual

    def jacobian(self, values):
        positions, velocities, accelerations = self.trajectory(values)
        rows, columns, entries = [], [], []
        for node, (q, v, a) in enumerate(zip(positions, velocities, accelerations)):
            basis = [matrix[node, 3:-3] for matrix in self.basis]
            support = np.flatnonzero(np.any(np.array(basis) != 0, axis=0))
            if not len(support):
                continue
            force = self.inverse(q, v, a)
            dq, dv, da = self.inverse_derivatives(q, v, a)
            variable_columns = (support[:, None] * self.d + np.arange(self.d)).ravel()
            local = np.hstack([dq * basis[0][k] + dv * basis[1][k] + da * basis[2][k] for k in support])
            if self.dynamics_residual == "acceleration":
                aq, av, aa = self.acceleration_derivatives(q, v, a)
                block = np.sqrt(self.dynamics_weight) * np.hstack([
                    aq * basis[0][k] + av * basis[1][k] + aa * basis[2][k] for k in support])
            else:
                block = np.sqrt(self.dynamics_weight) * local[1:] / self.torque_scales[:, None]
            rows.append(np.repeat(node * self.residual_dimension + np.arange(self.residual_dimension), len(variable_columns)))
            columns.append(np.tile(variable_columns, self.residual_dimension)); entries.append(block.ravel())
            force_gradient = local[0] / self.force_limit
            if abs(force[0]) > self.force_limit:
                rows.append(np.full(len(variable_columns), self.force_start + node))
                columns.append(variable_columns); entries.append(np.sqrt(self.force_weight) * np.sign(force[0]) * force_gradient)
            rows.append(np.full(len(variable_columns), self.effort_start + node))
            columns.append(variable_columns); entries.append(np.sqrt(self.effort_weight) * force_gradient)
        rows.append(self.reference_start + np.arange(self.nvars))
        columns.append(np.arange(self.nvars)); entries.append(np.full(self.nvars, np.sqrt(self.reference_weight)))
        return sparse.coo_matrix((np.concatenate(entries), (np.concatenate(rows), np.concatenate(columns))),
                                 shape=(self.reference_start + self.nvars, self.nvars)).tocsr()


def solve_inverse_spline(problem, *, rail_limit, max_evaluations):
    lower = np.full((problem.free_points, problem.d), -4 * np.pi)
    upper = -lower
    lower[:, 0], upper[:, 0] = -rail_limit, rail_limit
    return least_squares(problem.residual, problem.initial_values, jac=problem.jacobian,
                         bounds=(lower.ravel(), upper.ravel()), x_scale="jac", tr_solver="lsmr",
                         tr_options={"maxiter": 300, "atol": 1e-8, "btol": 1e-8},
                         max_nfev=max_evaluations, ftol=1e-10, xtol=1e-10, gtol=1e-10)
