"""Morphology-derived normal modes for deterministic n-link control.

The generalized swing-up controller needs coordinates that retain the phase of
every internal degree of freedom.  A total-energy scalar cannot distinguish two
states with identical energy but opposite distal-link phases.  This module
derives a mass-normalized modal basis directly from the exact MuJoCo plant,
without fitting a policy or introducing link-count-specific parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.linalg import eigh

from gcartpole.env import NLinkCartPoleEnv, wrap_angle
from gcartpole.generalized_solver import setup_from_config


@dataclass(frozen=True)
class ChainNormalModes:
    """Mass-normalized small-oscillation modes about a chain equilibrium."""

    equilibrium: str
    relative_equilibrium: np.ndarray
    joint_mass_matrix: np.ndarray
    stiffness_matrix: np.ndarray
    damping_matrix: np.ndarray
    squared_frequencies: np.ndarray
    angular_frequencies: np.ndarray
    dimensionless_frequencies: np.ndarray
    relative_shapes: np.ndarray
    absolute_shapes: np.ndarray
    cart_acceleration_coupling: np.ndarray
    normalized_coupling: np.ndarray

    @property
    def n_links(self) -> int:
        return int(self.relative_equilibrium.size)

    def coordinates(
        self,
        relative_angles: np.ndarray,
        hinge_rates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project a physical state into mass-normalized modal coordinates."""

        angles = np.asarray(relative_angles, dtype=np.float64)
        rates = np.asarray(hinge_rates, dtype=np.float64)
        expected = (self.n_links,)
        if angles.shape != expected or rates.shape != expected:
            raise ValueError("angle and rate shapes do not match the modal basis")
        displacement = wrap_angle(angles - self.relative_equilibrium)
        projector = self.relative_shapes.T @ self.joint_mass_matrix
        return projector @ displacement, projector @ rates

    def phase_features(
        self,
        relative_angles: np.ndarray,
        hinge_rates: np.ndarray,
        *,
        energy_scale: float,
    ) -> dict[str, np.ndarray]:
        """Return per-mode phase, amplitude, and normalized energy features."""

        if not np.isfinite(energy_scale) or energy_scale <= 0.0:
            raise ValueError("energy_scale must be finite and positive")
        positions, velocities = self.coordinates(relative_angles, hinge_rates)
        frequency_positions = self.angular_frequencies * positions
        amplitudes = np.hypot(frequency_positions, velocities)
        phases = np.arctan2(frequency_positions, velocities)
        energies = 0.5 * (frequency_positions**2 + velocities**2)
        return {
            "position": positions,
            "velocity": velocities,
            "phase": phases,
            "amplitude": amplitudes / np.sqrt(energy_scale),
            "energy_fraction": energies / energy_scale,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "equilibrium": self.equilibrium,
            "squared_frequencies": self.squared_frequencies.astype(float).tolist(),
            "angular_frequencies": self.angular_frequencies.astype(float).tolist(),
            "dimensionless_frequencies": self.dimensionless_frequencies.astype(
                float
            ).tolist(),
            "relative_shapes": self.relative_shapes.astype(float).tolist(),
            "absolute_shapes": self.absolute_shapes.astype(float).tolist(),
            "cart_acceleration_coupling": self.cart_acceleration_coupling.astype(
                float
            ).tolist(),
            "normalized_coupling": self.normalized_coupling.astype(float).tolist(),
            "joint_mass_matrix": self.joint_mass_matrix.astype(float).tolist(),
            "stiffness_matrix": self.stiffness_matrix.astype(float).tolist(),
            "damping_matrix": self.damping_matrix.astype(float).tolist(),
        }


def _generalized_bias(env: NLinkCartPoleEnv) -> np.ndarray:
    """Return the passive/bias term in ``M qdd + b = actuator``."""

    return np.asarray(
        env.data.qfrc_bias - env.data.qfrc_passive,
        dtype=np.float64,
    ).copy()


def chain_normal_modes(
    env: NLinkCartPoleEnv,
    *,
    equilibrium: str = "hanging",
    epsilon: float = 1.0e-6,
) -> ChainNormalModes:
    """Derive exact morphology-specific modes from MuJoCo finite differences.

    The cart acceleration is treated as the base input.  For joint coordinates
    ``q`` the linearized dynamics are

    ``M_jj qdd + D qdot + K q = -M_jx xdd``.

    The returned eigenvectors satisfy ``V.T @ M_jj @ V = I`` and are ordered by
    increasing natural frequency.  Their signs are canonicalized so serialized
    analyses are reproducible across runs.
    """

    if equilibrium not in {"hanging", "upright"}:
        raise ValueError("equilibrium must be 'hanging' or 'upright'")
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be finite and positive")

    n = int(env.n)
    d = n + 1
    saved_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    saved_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
    saved_ctrl = np.asarray(env.data.ctrl, dtype=np.float64).copy()
    saved_time = float(env.data.time)

    relative_equilibrium = np.zeros(n, dtype=np.float64)
    if equilibrium == "hanging":
        relative_equilibrium[0] = np.pi
    base_qpos = np.r_[0.0, relative_equilibrium]
    zero_qvel = np.zeros(d, dtype=np.float64)

    def evaluate(qpos: np.ndarray, qvel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        env.data.qpos[:] = qpos
        env.data.qvel[:] = qvel
        env.data.ctrl[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        full_mass = np.zeros((d, d), dtype=np.float64)
        mujoco.mj_fullM(env.model, full_mass, env.data.qM)
        return full_mass, _generalized_bias(env)

    full_mass, _ = evaluate(base_qpos, zero_qvel)
    stiffness = np.empty((n, n), dtype=np.float64)
    damping = np.empty((n, n), dtype=np.float64)
    for column in range(n):
        q_delta = np.zeros(d, dtype=np.float64)
        q_delta[1 + column] = epsilon
        _, plus = evaluate(base_qpos + q_delta, zero_qvel)
        _, minus = evaluate(base_qpos - q_delta, zero_qvel)
        stiffness[:, column] = (plus[1:] - minus[1:]) / (2.0 * epsilon)

        v_delta = np.zeros(d, dtype=np.float64)
        v_delta[1 + column] = epsilon
        _, plus = evaluate(base_qpos, zero_qvel + v_delta)
        _, minus = evaluate(base_qpos, zero_qvel - v_delta)
        damping[:, column] = (plus[1:] - minus[1:]) / (2.0 * epsilon)

    # Roundoff and MuJoCo finite differencing can introduce tiny asymmetric
    # components even though these physical matrices are symmetric.
    joint_mass = 0.5 * (full_mass[1:, 1:] + full_mass[1:, 1:].T)
    stiffness = 0.5 * (stiffness + stiffness.T)
    damping = 0.5 * (damping + damping.T)
    squared_frequencies, shapes = eigh(stiffness, joint_mass)
    # Hanging eigenvalues are positive while upright eigenvalues are negative.
    # Order both equilibria by oscillation/growth-rate magnitude so mode zero is
    # always the collective, lowest-frequency shape.
    order = np.argsort(np.sqrt(np.abs(squared_frequencies)))
    squared_frequencies = squared_frequencies[order]
    shapes = shapes[:, order]

    cumulative = np.tril(np.ones((n, n), dtype=np.float64))
    absolute_shapes = cumulative @ shapes
    coupling = -(shapes.T @ full_mass[1:, 0])
    for mode in range(n):
        pivot = int(np.argmax(np.abs(absolute_shapes[:, mode])))
        if absolute_shapes[pivot, mode] < 0.0:
            shapes[:, mode] *= -1.0
            absolute_shapes[:, mode] *= -1.0
            coupling[mode] *= -1.0

    coupling_norm = float(np.linalg.norm(coupling))
    normalized_coupling = (
        coupling / coupling_norm if coupling_norm > 1.0e-15 else np.zeros_like(coupling)
    )
    frequencies = np.sqrt(np.abs(squared_frequencies))
    setup = setup_from_config(env.cfg, progress=env.plant_progress)

    env.data.qpos[:] = saved_qpos
    env.data.qvel[:] = saved_qvel
    env.data.ctrl[:] = saved_ctrl
    env.data.time = saved_time
    mujoco.mj_forward(env.model, env.data)

    return ChainNormalModes(
        equilibrium=equilibrium,
        relative_equilibrium=relative_equilibrium,
        joint_mass_matrix=joint_mass,
        stiffness_matrix=stiffness,
        damping_matrix=damping,
        squared_frequencies=squared_frequencies,
        angular_frequencies=frequencies,
        dimensionless_frequencies=frequencies * setup.natural_time,
        relative_shapes=shapes,
        absolute_shapes=absolute_shapes,
        cart_acceleration_coupling=coupling,
        normalized_coupling=normalized_coupling,
    )


def modal_energy_acceleration_ratio(
    modes: ChainNormalModes,
    relative_angles: np.ndarray,
    hinge_rates: np.ndarray,
    *,
    total_energy_error: float,
    energy_scale: float,
    collective_gain: float,
    internal_damping_gain: float,
) -> tuple[float, dict[str, object]]:
    """Compute a count-agnostic energy-pump/internal-damping command.

    Mode zero is the lowest-frequency collective swing.  Remaining modes are
    internal deformation modes.  The law pumps the collective mode toward the
    upright energy while removing net input power from the internal modes.  It
    uses two scalar gains regardless of link count; all per-mode coefficients
    come from the measured plant.
    """

    if min(collective_gain, internal_damping_gain) < 0.0:
        raise ValueError("modal gains must be nonnegative")
    features = modes.phase_features(
        relative_angles,
        hinge_rates,
        energy_scale=energy_scale,
    )
    velocity = np.asarray(features["velocity"], dtype=np.float64) / np.sqrt(
        energy_scale
    )
    coupling = modes.normalized_coupling
    collective_power_direction = float(coupling[0] * velocity[0])
    internal_power_direction = float(coupling[1:] @ velocity[1:])
    ratio = (
        float(collective_gain) * float(total_energy_error) * collective_power_direction
        - float(internal_damping_gain) * internal_power_direction
    )
    diagnostics: dict[str, object] = {
        "modal_phase": np.asarray(features["phase"]).astype(float).tolist(),
        "modal_amplitude": np.asarray(features["amplitude"]).astype(float).tolist(),
        "modal_energy_fraction": np.asarray(features["energy_fraction"])
        .astype(float)
        .tolist(),
        "collective_power_direction": collective_power_direction,
        "internal_power_direction": internal_power_direction,
        "modal_acceleration_ratio": float(ratio),
    }
    return float(ratio), diagnostics
