"""Deterministic energy-shaping and Riccati control for arbitrary link chains."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import mujoco
import numpy as np
from scipy.linalg import solve_discrete_are

from gcartpole.env import NLinkCartPoleEnv, serial_absolute_angles, wrap_angle
from gcartpole.generalized_modes import (
    chain_normal_modes,
    modal_energy_acceleration_ratio,
)
from gcartpole.generalized_solver import dimensionless_setup, setup_from_config
from gcartpole.ilqr import MujocoTransition, data_state
from gcartpole.lqr_design import checked_discrete_lqr


def absolute_state_cost(n_links: int) -> np.ndarray:
    """Repository-standard upright cost in relative MuJoCo coordinates."""

    n = int(n_links)
    d = n + 1
    transform = np.zeros((2 * d, 2 * d), dtype=np.float64)
    transform[0, 0] = 1.0
    transform[d, d] = 1.0
    for link in range(n):
        transform[1 + link, 1 : 2 + link] = 1.0
        transform[d + 1 + link, d + 1 : d + 2 + link] = 1.0
    weights = np.diag([0.1] + [100.0] * n + [0.1] + [1.0] * n)
    cost = transform.T @ weights @ transform
    cost += np.diag([0.0] + [1.0] * n + [0.0] + [0.01] * n)
    return cost


def upright_lqr_gain(
    env: NLinkCartPoleEnv, *, control_cost: float = 10.0
) -> np.ndarray:
    """Linearize the exact target plant and compute normalized-action LQR."""

    transition = MujocoTransition(env)
    equilibrium = np.zeros(2 * (env.n + 1), dtype=np.float64)
    state_matrix, input_matrix = transition.linearize(
        equilibrium, 0.0, state_epsilon=1e-7, action_epsilon=1e-7
    )
    state_cost = absolute_state_cost(env.n)
    input_cost = np.array([[float(control_cost)]], dtype=np.float64)
    gain, _, _ = checked_discrete_lqr(state_matrix, input_matrix, state_cost, input_cost)
    return gain.reshape(-1)


def hanging_lqr_gain(
    env: NLinkCartPoleEnv, *, control_cost: float = 1000.0,
    cart_position_cost: float = 0.1, cart_velocity_cost: float = 0.1,
) -> np.ndarray:
    """Compute an exact local regulator without wrapping across the pi branch."""

    n = env.n
    if any(not np.isfinite(value) or value <= 0.0 for value in
           (control_cost, cart_position_cost, cart_velocity_cost)):
        raise ValueError("hanging regulator costs must be finite and positive")
    d = n + 1
    state_size = 2 * d
    equilibrium = np.zeros(state_size, dtype=np.float64)
    equilibrium[1] = np.pi
    saved_qpos = np.asarray(env.data.qpos, dtype=np.float64).copy()
    saved_qvel = np.asarray(env.data.qvel, dtype=np.float64).copy()
    saved_ctrl = np.asarray(env.data.ctrl, dtype=np.float64).copy()

    def step_map(state: np.ndarray, action: float) -> np.ndarray:
        env.data.qpos[:] = state[:d]
        env.data.qvel[:] = state[d:]
        env.data.ctrl[0] = float(np.clip(action, -1.0, 1.0)) * env.force_limit
        mujoco.mj_forward(env.model, env.data)
        for _ in range(env.frame_skip):
            mujoco.mj_step(env.model, env.data)
        return data_state(env.data)

    epsilon = 1e-7
    state_matrix = np.empty((state_size, state_size), dtype=np.float64)
    for column in range(state_size):
        delta = np.zeros(state_size, dtype=np.float64)
        delta[column] = epsilon
        state_matrix[:, column] = (
            step_map(equilibrium + delta, 0.0) - step_map(equilibrium - delta, 0.0)
        ) / (2.0 * epsilon)
    input_matrix = (step_map(equilibrium, epsilon) - step_map(equilibrium, -epsilon))[
        :, None
    ] / (2.0 * epsilon)
    state_cost = absolute_state_cost(n)
    state_cost[0, 0] = float(cart_position_cost)
    state_cost[d, d] = float(cart_velocity_cost)
    input_cost = np.array([[float(control_cost)]], dtype=np.float64)
    riccati = solve_discrete_are(state_matrix, input_matrix, state_cost, input_cost)
    gain = np.linalg.solve(
        input_matrix.T @ riccati @ input_matrix + input_cost,
        input_matrix.T @ riccati @ state_matrix,
    ).reshape(-1)
    env.data.qpos[:] = saved_qpos
    env.data.qvel[:] = saved_qvel
    env.data.ctrl[:] = saved_ctrl
    mujoco.mj_forward(env.model, env.data)
    return gain


def hanging_lqr_action(env: NLinkCartPoleEnv, gain: np.ndarray) -> float:
    state = data_state(env.data)
    state[0] = env.data.qpos[0]
    state[1] = wrap_angle(env.data.qpos[1] - np.pi)
    if env.n > 1:
        state[2 : env.n + 1] = wrap_angle(env.data.qpos[2 : env.n + 1])
    return float(np.clip(-np.asarray(gain, dtype=np.float64) @ state, -1.0, 1.0))


@dataclass(frozen=True)
class EnergySwingParameters:
    """Dimensionless gains shared across dynamically similar plants."""

    energy_gain: float = 2.2
    cart_position_gain: float = 0.31
    cart_velocity_gain: float = 0.28
    kick_acceleration_ratio: float = 0.102
    kick_frequency_ratio: float = 0.166
    kick_duration_ratio: float = 3.62
    kick_phase: float = 0.0
    collective_modal_gain: float = 0.0
    internal_modal_damping_gain: float = 0.0
    modal_acceleration_limit_ratio: float = 2.0
    coherence_position_gain: float = 0.0
    coherence_velocity_gain: float = 0.0
    coherence_correction_weight: float = 0.0
    coherence_acceleration_limit_ratio: float = 2.0
    coherence_gate_energy_error: float = 0.0
    vcl_position_gain: float = 0.0
    vcl_velocity_gain: float = 0.0
    vcl_correction_weight: float = 0.0
    vcl_acceleration_limit_ratio: float = 2.0
    vcl_gate_energy_error: float = 0.0
    vcl_pump_gain: float = 0.0
    vcl_phase_gain: float = 0.0
    vcl_rate_gain: float = 0.0
    enter_angle: float = 0.20
    enter_absolute_rate_ratio: float = 0.67
    enter_cart_velocity_ratio: float = 0.14
    lqr_control_cost: float = 10.0
    lqr_scale: float = 1.0

    def to_dict(self, *, include_extensions: bool = False) -> dict[str, float]:
        values = asdict(self)
        if not include_extensions:
            values = {
                key: values[key]
                for key in (
                    "energy_gain",
                    "cart_position_gain",
                    "cart_velocity_gain",
                    "kick_acceleration_ratio",
                    "kick_frequency_ratio",
                    "kick_duration_ratio",
                    "kick_phase",
                    "collective_modal_gain",
                    "internal_modal_damping_gain",
                    "modal_acceleration_limit_ratio",
                    "enter_angle",
                    "enter_absolute_rate_ratio",
                    "enter_cart_velocity_ratio",
                    "lqr_control_cost",
                    "lqr_scale",
                )
            }
        return {key: float(value) for key, value in values.items()}


def chain_energy_features(env: NLinkCartPoleEnv) -> dict[str, float]:
    """Exact passive-chain energy and base-acceleration coupling features."""

    mass_matrix = np.zeros((env.model.nv, env.model.nv), dtype=np.float64)
    mujoco.mj_fullM(env.model, mass_matrix, env.data.qM)
    joint_rates = np.asarray(env.data.qvel[1:], dtype=np.float64)
    joint_kinetic = 0.5 * float(joint_rates @ mass_matrix[1:, 1:] @ joint_rates)
    mujoco.mj_energyPos(env.model, env.data)
    potential = float(env.data.energy[0])
    passive_energy = potential + joint_kinetic
    gap = float(env._energy_gap)
    energy_error = (passive_energy - float(env._upright_potential_energy)) / gap
    setup = setup_from_config(env.cfg, progress=env.plant_progress)
    velocity_scale = np.sqrt(setup.gravity * setup.chain_length)
    relative_horizontal_momentum = float(mass_matrix[0, 1:] @ joint_rates)
    momentum_ratio = relative_horizontal_momentum / (setup.link_mass * velocity_scale)
    absolute_angles = serial_absolute_angles(
        np.asarray(env.data.qpos[1:], dtype=np.float64)
    )
    absolute_rates = np.cumsum(joint_rates)
    return {
        "energy_error": float(energy_error),
        "relative_horizontal_momentum_ratio": float(momentum_ratio),
        "cart_position_ratio": float(env.data.qpos[0] / setup.chain_length),
        "cart_velocity_ratio": float(env.data.qvel[0] / velocity_scale),
        "max_abs_angle": float(np.max(np.abs(absolute_angles))),
        "absolute_rate_rms_ratio": float(
            np.sqrt(np.mean(absolute_rates**2)) * setup.natural_time
        ),
    }


def force_for_desired_cart_acceleration(
    env: NLinkCartPoleEnv, desired_acceleration: float
) -> float:
    """Exact partial feedback linearization from cart acceleration to force."""

    mass_matrix = np.zeros((env.model.nv, env.model.nv), dtype=np.float64)
    mujoco.mj_fullM(env.model, mass_matrix, env.data.qM)
    # MuJoCo forward dynamics: M*qacc + bias = passive + actuator.
    generalized_bias = np.asarray(
        env.data.qfrc_bias - env.data.qfrc_passive, dtype=np.float64
    )
    joint_mass = mass_matrix[1:, 1:]
    joint_acceleration = np.linalg.solve(
        joint_mass,
        -generalized_bias[1:] - mass_matrix[1:, 0] * float(desired_acceleration),
    )
    return float(
        mass_matrix[0, 0] * float(desired_acceleration)
        + mass_matrix[0, 1:] @ joint_acceleration
        + generalized_bias[0]
    )


def _vcl_state(
    absolute_angles: np.ndarray,
    absolute_rates: np.ndarray,
    lengths: np.ndarray,
    masses: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tail virtual-composite-link angles, rates, and weights.

    VCL ``i`` is the center-of-mass vector of links ``i..n`` measured from
    joint ``i``.  The construction is deliberately kinematic: it gives the
    controller a recursive shape coordinate for every distal sub-chain while
    the live MuJoCo mass matrix supplies the exact acceleration response.
    """

    angles = np.asarray(absolute_angles, dtype=np.float64)
    rates = np.asarray(absolute_rates, dtype=np.float64)
    lengths = np.asarray(lengths, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    n_links = int(angles.size)
    joint_positions = np.zeros((n_links + 1, 2), dtype=np.float64)
    joint_velocities = np.zeros((n_links + 1, 2), dtype=np.float64)
    com_positions = np.zeros((n_links, 2), dtype=np.float64)
    com_velocities = np.zeros((n_links, 2), dtype=np.float64)
    for index in range(n_links):
        direction = np.asarray(
            [np.sin(angles[index]), np.cos(angles[index])], dtype=np.float64
        )
        direction_rate = np.asarray(
            [
                np.cos(angles[index]) * rates[index],
                -np.sin(angles[index]) * rates[index],
            ],
            dtype=np.float64,
        )
        com_positions[index] = joint_positions[index] + 0.5 * lengths[index] * direction
        com_velocities[index] = (
            joint_velocities[index] + 0.5 * lengths[index] * direction_rate
        )
        joint_positions[index + 1] = joint_positions[index] + lengths[index] * direction
        joint_velocities[index + 1] = (
            joint_velocities[index] + lengths[index] * direction_rate
        )

    vcl_angles = np.zeros(n_links, dtype=np.float64)
    vcl_rates = np.zeros(n_links, dtype=np.float64)
    weights = np.zeros(n_links, dtype=np.float64)
    for index in range(n_links):
        tail_mass = max(float(np.sum(masses[index:])), 1e-12)
        vector = np.sum(
            masses[index:, None]
            * (com_positions[index:] - joint_positions[index]),
            axis=0,
        ) / tail_mass
        velocity = np.sum(
            masses[index:, None]
            * (com_velocities[index:] - joint_velocities[index]),
            axis=0,
        ) / tail_mass
        radius_squared = max(float(vector @ vector), 1e-12)
        vcl_angles[index] = float(np.arctan2(vector[0], vector[1]))
        vcl_rates[index] = float(
            (vector[1] * velocity[0] - vector[0] * velocity[1]) / radius_squared
        )
        weights[index] = tail_mass * np.sqrt(radius_squared)
    weights /= max(float(np.sum(weights)), 1e-12)
    return vcl_angles, vcl_rates, weights


def vcl_coherence_acceleration(
    env: NLinkCartPoleEnv,
    base_acceleration: float,
    *,
    position_gain: float,
    velocity_gain: float,
    correction_weight: float = 1.0,
    max_correction_ratio: float = 2.0,
) -> tuple[float, dict[str, float]]:
    """Project tail-composite-link damping through the cart input channel.

    This is a cart-pole adaptation of the virtual-composite-link idea: the
    target is not each raw hinge angle, but the orientation and rate of every
    distal composite mass.  The one available cart acceleration is chosen by a
    weighted least-squares projection, then bounded before it is converted to
    force.  It is a discovery primitive and must be validated by exact replay.
    """

    if min(position_gain, velocity_gain, correction_weight, max_correction_ratio) < 0.0:
        raise ValueError("VCL gains and bounds must be nonnegative")
    setup = setup_from_config(env.cfg, progress=env.plant_progress)
    relative_angles = wrap_angle(np.asarray(env.data.qpos[1:], dtype=np.float64))
    absolute_angles = serial_absolute_angles(relative_angles)
    absolute_rates = np.cumsum(np.asarray(env.data.qvel[1:], dtype=np.float64))
    lengths = np.asarray(env.morphology.lengths, dtype=np.float64)
    masses = np.asarray(env.morphology.masses, dtype=np.float64)
    vcl_angles, vcl_rates, weights = _vcl_state(
        absolute_angles, absolute_rates, lengths, masses
    )
    angle_error = wrap_angle(vcl_angles)

    mass_matrix = np.zeros((env.model.nv, env.model.nv), dtype=np.float64)
    mujoco.mj_fullM(env.model, mass_matrix, env.data.qM)
    generalized_bias = np.asarray(
        env.data.qfrc_bias - env.data.qfrc_passive, dtype=np.float64
    )
    joint_mass = mass_matrix[1:, 1:]

    def absolute_acceleration(cart_acceleration: float) -> np.ndarray:
        joint = np.linalg.solve(
            joint_mass,
            -generalized_bias[1:] - mass_matrix[1:, 0] * float(cart_acceleration),
        )
        return np.cumsum(joint)

    zero_acceleration = absolute_acceleration(0.0)
    response_acceleration = absolute_acceleration(1.0) - zero_acceleration
    response_rate = _vcl_state(
        absolute_angles,
        response_acceleration,
        lengths,
        masses,
    )[1]
    zero_rate = _vcl_state(
        absolute_angles,
        zero_acceleration,
        lengths,
        masses,
    )[1]
    # The position-dependent part of VCL rate acceleration is the directional
    # derivative of its kinematic rate along the current angular velocity.
    epsilon = 1.0e-6
    rate_plus = _vcl_state(
        absolute_angles + epsilon * absolute_rates,
        absolute_rates,
        lengths,
        masses,
    )[1]
    rate_minus = _vcl_state(
        absolute_angles - epsilon * absolute_rates,
        absolute_rates,
        lengths,
        masses,
    )[1]
    drift = (rate_plus - rate_minus) / (2.0 * epsilon)
    predicted = drift + zero_rate + response_rate * float(base_acceleration)
    desired = (
        -float(position_gain) * angle_error / (setup.natural_time**2)
        - float(velocity_gain) * vcl_rates / setup.natural_time
    )
    residual = desired - predicted
    denominator = float(weights @ (response_rate**2))
    raw_correction = 0.0
    if denominator > 1.0e-12:
        raw_correction = float(weights @ (response_rate * residual) / denominator)
    limit = float(max_correction_ratio * setup.gravity)
    correction = float(np.clip(correction_weight * raw_correction, -limit, limit))
    return float(base_acceleration + correction), {
        "vcl_angle_rms": float(np.sqrt(weights @ (angle_error**2))),
        "vcl_rate_rms_ratio": float(
            np.sqrt(weights @ (vcl_rates**2)) * setup.natural_time
        ),
        "vcl_acceleration_correction_ratio": float(correction / setup.gravity),
    }


def vcl_phase_features(
    env: NLinkCartPoleEnv,
) -> tuple[float, float]:
    """Return weighted VCL phase and natural-time-scaled phase rate."""

    setup = setup_from_config(env.cfg, progress=env.plant_progress)
    relative_angles = wrap_angle(np.asarray(env.data.qpos[1:], dtype=np.float64))
    absolute_angles = serial_absolute_angles(relative_angles)
    absolute_rates = np.cumsum(np.asarray(env.data.qvel[1:], dtype=np.float64))
    lengths = np.asarray(env.morphology.lengths, dtype=np.float64)
    masses = np.asarray(env.morphology.masses, dtype=np.float64)
    vcl_angles, vcl_rates, weights = _vcl_state(
        absolute_angles, absolute_rates, lengths, masses
    )
    phase = float(weights @ np.sin(vcl_angles))
    phase_rate = float(
        weights @ (np.cos(vcl_angles) * vcl_rates) * setup.natural_time
    )
    return phase, phase_rate


def modal_coherence_acceleration(
    env: NLinkCartPoleEnv,
    base_acceleration: float,
    *,
    position_gain: float,
    velocity_gain: float,
    correction_weight: float = 1.0,
    max_correction_ratio: float = 2.0,
) -> tuple[float, dict[str, float]]:
    """Project internal-mode synchronization onto the one available input.

    Absolute link angles are decomposed into a mass/length-weighted rigid-chain
    phase and internal deviations.  Exact MuJoCo mass-matrix equations provide
    the absolute-angular-acceleration response to prescribed cart acceleration.
    A scalar least-squares projection then selects the acceleration correction
    that best damps the internal deviations.  For one link the internal space
    is empty and this function is exactly the identity.
    """

    if min(position_gain, velocity_gain, correction_weight, max_correction_ratio) < 0.0:
        raise ValueError("modal-coherence gains and bounds must be nonnegative")
    if env.n == 1 or correction_weight == 0.0:
        return float(base_acceleration), {
            "modal_angle_rms": 0.0,
            "modal_rate_rms_ratio": 0.0,
            "modal_acceleration_correction_ratio": 0.0,
        }

    setup = setup_from_config(env.cfg, progress=env.plant_progress)
    relative_angles = wrap_angle(np.asarray(env.data.qpos[1:], dtype=np.float64))
    absolute_angles = serial_absolute_angles(relative_angles)
    absolute_rates = np.cumsum(np.asarray(env.data.qvel[1:], dtype=np.float64))
    weights = np.asarray(
        env.morphology.masses * env.morphology.lengths, dtype=np.float64
    )
    weights /= float(np.sum(weights))
    mean_angle = float(
        np.arctan2(weights @ np.sin(absolute_angles), weights @ np.cos(absolute_angles))
    )
    angle_error = wrap_angle(absolute_angles - mean_angle)
    mean_rate = float(weights @ absolute_rates)
    rate_error = absolute_rates - mean_rate

    mass_matrix = np.zeros((env.model.nv, env.model.nv), dtype=np.float64)
    mujoco.mj_fullM(env.model, mass_matrix, env.data.qM)
    generalized_bias = np.asarray(
        env.data.qfrc_bias - env.data.qfrc_passive, dtype=np.float64
    )
    joint_mass = mass_matrix[1:, 1:]

    def absolute_acceleration(cart_acceleration: float) -> np.ndarray:
        joint = np.linalg.solve(
            joint_mass,
            -generalized_bias[1:] - mass_matrix[1:, 0] * float(cart_acceleration),
        )
        return np.cumsum(joint)

    zero = absolute_acceleration(0.0)
    response = absolute_acceleration(1.0) - zero

    def internal(values: np.ndarray) -> np.ndarray:
        return values - float(weights @ values)

    predicted = internal(zero + response * float(base_acceleration))
    response_internal = internal(response)
    desired = (
        -float(position_gain) * angle_error / (setup.natural_time**2)
        - float(velocity_gain) * rate_error / setup.natural_time
    )
    denominator = float(weights @ (response_internal**2))
    correction = 0.0
    if denominator > 1e-12:
        correction = float(
            weights @ (response_internal * (desired - predicted)) / denominator
        )
    limit = float(max_correction_ratio * setup.gravity)
    correction = float(np.clip(correction_weight * correction, -limit, limit))
    return float(base_acceleration + correction), {
        "modal_angle_rms": float(np.sqrt(weights @ (angle_error**2))),
        "modal_rate_rms_ratio": float(
            np.sqrt(weights @ (rate_error**2)) * setup.natural_time
        ),
        "modal_acceleration_correction_ratio": float(correction / setup.gravity),
    }


class GeneralizedEnergyController:
    """Energy pump with a state-gated exact upright LQR handoff."""

    def __init__(
        self, env: NLinkCartPoleEnv, parameters: EnergySwingParameters
    ) -> None:
        self.parameters = parameters
        self.setup = setup_from_config(env.cfg, progress=env.plant_progress)
        self.pi = dimensionless_setup(self.setup)
        self.lqr_gain = upright_lqr_gain(env, control_cost=parameters.lqr_control_cost)
        self.hanging_modes = chain_normal_modes(env, equilibrium="hanging")
        self.mode = "energy"
        self.switch_time: float | None = None

    def reset(self) -> None:
        self.mode = "energy"
        self.switch_time = None

    def _should_capture(
        self, env: NLinkCartPoleEnv, features: dict[str, float]
    ) -> bool:
        p = self.parameters
        return bool(
            features["max_abs_angle"] <= p.enter_angle
            and features["absolute_rate_rms_ratio"] <= p.enter_absolute_rate_ratio
            and abs(features["cart_velocity_ratio"]) <= p.enter_cart_velocity_ratio
        )

    def action(
        self, env: NLinkCartPoleEnv, time_seconds: float
    ) -> tuple[float, dict[str, float | str]]:
        features = chain_energy_features(env)
        if self.mode == "energy" and self._should_capture(env, features):
            self.mode = "lqr"
            self.switch_time = float(time_seconds)
        if self.mode == "lqr":
            state = data_state(env.data)
            state[1 : env.n + 1] = wrap_angle(state[1 : env.n + 1])
            action = float(
                np.clip(-self.parameters.lqr_scale * self.lqr_gain @ state, -1.0, 1.0)
            )
            return action, {**features, "mode": self.mode}

        tau = float(time_seconds / self.setup.natural_time)
        kick = 0.0
        if tau < self.parameters.kick_duration_ratio:
            kick = self.parameters.kick_acceleration_ratio * np.sin(
                2.0 * np.pi * self.parameters.kick_frequency_ratio * tau
                + self.parameters.kick_phase
            )
        acceleration_ratio = (
            self.parameters.energy_gain
            * features["energy_error"]
            * features["relative_horizontal_momentum_ratio"]
            - self.parameters.cart_position_gain * features["cart_position_ratio"]
            - self.parameters.cart_velocity_gain * features["cart_velocity_ratio"]
            + kick
        )
        modal_diagnostics: dict[str, object] = {}
        if (
            self.parameters.collective_modal_gain > 0.0
            or self.parameters.internal_modal_damping_gain > 0.0
        ):
            modal_ratio, modal_diagnostics = modal_energy_acceleration_ratio(
                self.hanging_modes,
                np.asarray(env.data.qpos[1:], dtype=np.float64),
                np.asarray(env.data.qvel[1:], dtype=np.float64),
                total_energy_error=features["energy_error"],
                energy_scale=float(env._energy_gap),
                collective_gain=self.parameters.collective_modal_gain,
                internal_damping_gain=self.parameters.internal_modal_damping_gain,
            )
            modal_ratio = float(
                np.clip(
                    modal_ratio,
                    -self.parameters.modal_acceleration_limit_ratio,
                    self.parameters.modal_acceleration_limit_ratio,
                )
            )
            acceleration_ratio += modal_ratio
            modal_diagnostics["limited_modal_acceleration_ratio"] = modal_ratio
        desired_acceleration = self.setup.gravity * float(acceleration_ratio)
        vcl_phase_diagnostics: dict[str, object] = {}
        if (
            self.parameters.vcl_pump_gain != 0.0
            or self.parameters.vcl_phase_gain != 0.0
            or self.parameters.vcl_rate_gain != 0.0
        ):
            vcl_phase, vcl_phase_rate = vcl_phase_features(env)
            vcl_acceleration_ratio = (
                self.parameters.vcl_pump_gain
                * features["energy_error"]
                * vcl_phase_rate
                + self.parameters.vcl_phase_gain * vcl_phase
                + self.parameters.vcl_rate_gain * vcl_phase_rate
            )
            desired_acceleration += self.setup.gravity * float(vcl_acceleration_ratio)
            vcl_phase_diagnostics = {
                "vcl_phase": float(vcl_phase),
                "vcl_phase_rate_ratio": float(vcl_phase_rate),
                "vcl_pump_acceleration_ratio": float(vcl_acceleration_ratio),
            }
        coherence_diagnostics: dict[str, object] = {}
        if (
            self.parameters.coherence_correction_weight > 0.0
            and (
                self.parameters.coherence_position_gain > 0.0
                or self.parameters.coherence_velocity_gain > 0.0
            )
        ):
            coherence_gate = 1.0
            if self.parameters.coherence_gate_energy_error > 0.0:
                coherence_gate = float(
                    np.clip(
                        (
                            self.parameters.coherence_gate_energy_error
                            - abs(float(features["energy_error"]))
                        )
                        / self.parameters.coherence_gate_energy_error,
                        0.0,
                        1.0,
                    )
                )
            desired_acceleration, coherence_diagnostics = modal_coherence_acceleration(
                env,
                desired_acceleration,
                position_gain=self.parameters.coherence_position_gain,
                velocity_gain=self.parameters.coherence_velocity_gain,
                correction_weight=(
                    self.parameters.coherence_correction_weight * coherence_gate
                ),
                max_correction_ratio=self.parameters.coherence_acceleration_limit_ratio,
            )
            coherence_diagnostics["coherence_gate"] = coherence_gate
        vcl_diagnostics: dict[str, object] = {}
        if (
            self.parameters.vcl_correction_weight > 0.0
            and (
                self.parameters.vcl_position_gain > 0.0
                or self.parameters.vcl_velocity_gain > 0.0
            )
        ):
            vcl_gate = 1.0
            if self.parameters.vcl_gate_energy_error > 0.0:
                vcl_gate = float(
                    np.clip(
                        (
                            self.parameters.vcl_gate_energy_error
                            - abs(float(features["energy_error"]))
                        )
                        / self.parameters.vcl_gate_energy_error,
                        0.0,
                        1.0,
                    )
                )
            desired_acceleration, vcl_diagnostics = vcl_coherence_acceleration(
                env,
                desired_acceleration,
                position_gain=self.parameters.vcl_position_gain,
                velocity_gain=self.parameters.vcl_velocity_gain,
                correction_weight=self.parameters.vcl_correction_weight * vcl_gate,
                max_correction_ratio=self.parameters.vcl_acceleration_limit_ratio,
            )
            vcl_diagnostics["vcl_gate"] = vcl_gate
        force = force_for_desired_cart_acceleration(env, desired_acceleration)
        action = float(np.clip(force / env.force_limit, -1.0, 1.0))
        return action, {
            **features,
            "mode": self.mode,
            "desired_acceleration_ratio": float(acceleration_ratio),
            "unclipped_action": float(force / env.force_limit),
            **vcl_phase_diagnostics,
            **modal_diagnostics,
            **coherence_diagnostics,
            **vcl_diagnostics,
        }
