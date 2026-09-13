"""Morphology-normalized building blocks for a generalized cart-pole solver.

This module deliberately contains no learned parameters and no link-count-specific
constants.  It turns a physical setup into dimensionless groups, transfers state
and feedback trajectories between arbitrary serial-chain discretizations, and
provides a conservative online calibration layer for actuator mismatch.

The transfer operation is a *seed generator*, not a claim that one force trace is
optimal for every morphology.  Exact dynamics optimization is still responsible
for closing the remaining gap after a transfer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

GRAVITY = 9.81


def _positive_vector(name: str, values: np.ndarray | list[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional vector")
    if not np.all(np.isfinite(vector)) or np.any(vector <= 0.0):
        raise ValueError(f"{name} must contain finite positive values")
    return vector


@dataclass(frozen=True)
class PhysicalSetup:
    """Physical quantities needed to instantiate the count-agnostic solver."""

    lengths: np.ndarray
    masses: np.ndarray
    cart_mass: float
    rail_half_length: float
    force_limit: float
    joint_damping: np.ndarray
    cart_damping: float = 0.0
    joint_armature: float = 0.0
    cart_half_length: float = 0.18
    link_radius: float = 0.025
    timestep: float = 0.005
    frame_skip: int = 1
    gravity: float = GRAVITY

    def __post_init__(self) -> None:
        lengths = _positive_vector("lengths", self.lengths)
        masses = _positive_vector("masses", self.masses)
        if lengths.shape != masses.shape:
            raise ValueError("lengths and masses must have the same shape")
        damping = np.asarray(self.joint_damping, dtype=np.float64)
        if damping.shape != lengths.shape:
            raise ValueError("joint_damping must have one value per link")
        if not np.all(np.isfinite(damping)) or np.any(damping < 0.0):
            raise ValueError("joint_damping must contain finite nonnegative values")
        positive_scalars = {
            "cart_mass": self.cart_mass,
            "rail_half_length": self.rail_half_length,
            "force_limit": self.force_limit,
            "cart_half_length": self.cart_half_length,
            "link_radius": self.link_radius,
            "timestep": self.timestep,
            "gravity": self.gravity,
        }
        for name, value in positive_scalars.items():
            if not np.isfinite(value) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.cart_damping) or self.cart_damping < 0.0:
            raise ValueError("cart_damping must be finite and nonnegative")
        if not np.isfinite(self.joint_armature) or self.joint_armature < 0.0:
            raise ValueError("joint_armature must be finite and nonnegative")
        if int(self.frame_skip) != self.frame_skip or int(self.frame_skip) < 1:
            raise ValueError("frame_skip must be a positive integer")
        object.__setattr__(self, "lengths", lengths.copy())
        object.__setattr__(self, "masses", masses.copy())
        object.__setattr__(self, "joint_damping", damping.copy())
        object.__setattr__(self, "frame_skip", int(self.frame_skip))

    @property
    def n_links(self) -> int:
        return int(self.lengths.size)

    @property
    def chain_length(self) -> float:
        return float(np.sum(self.lengths))

    @property
    def link_mass(self) -> float:
        return float(np.sum(self.masses))

    @property
    def system_mass(self) -> float:
        return float(self.cart_mass + self.link_mass)

    @property
    def natural_time(self) -> float:
        return float(np.sqrt(self.chain_length / self.gravity))

    @property
    def policy_dt(self) -> float:
        return float(self.timestep * self.frame_skip)


@dataclass(frozen=True)
class DimensionlessSetup:
    """Buckingham-pi description of a cart plus serial pendulum chain."""

    n_links: int
    length_fractions: np.ndarray
    mass_fractions: np.ndarray
    cart_to_link_mass: float
    force_authority: float
    rail_ratio: float
    usable_rail_ratio: float
    cart_half_length_ratio: float
    link_radius_ratio: float
    policy_dt_ratio: float
    cart_damping_ratio: float
    joint_damping_ratios: np.ndarray
    joint_armature_ratio: float
    chain_length: float
    system_mass: float
    natural_time: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for key, value in tuple(result.items()):
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
        return result


def dimensionless_setup(setup: PhysicalSetup) -> DimensionlessSetup:
    """Return the complete scale-free setup used by the generalized recipe.

    Translational force and damping are normalized by ``M g`` and ``M/t0``.
    Joint damping and armature are normalized by ``M L^2/t0`` and ``M L^2``.
    ``usable_rail_ratio`` accounts for the cart body's half length.
    """

    length = setup.chain_length
    mass = setup.system_mass
    time = setup.natural_time
    rotational_damping_scale = mass * length * length / time
    return DimensionlessSetup(
        n_links=setup.n_links,
        length_fractions=setup.lengths / length,
        mass_fractions=setup.masses / setup.link_mass,
        cart_to_link_mass=float(setup.cart_mass / setup.link_mass),
        force_authority=float(setup.force_limit / (mass * setup.gravity)),
        rail_ratio=float(setup.rail_half_length / length),
        usable_rail_ratio=float((setup.rail_half_length - setup.cart_half_length) / length),
        cart_half_length_ratio=float(setup.cart_half_length / length),
        link_radius_ratio=float(setup.link_radius / length),
        policy_dt_ratio=float(setup.policy_dt / time),
        cart_damping_ratio=float(setup.cart_damping * time / mass),
        joint_damping_ratios=setup.joint_damping / rotational_damping_scale,
        joint_armature_ratio=float(setup.joint_armature / (mass * length * length)),
        chain_length=length,
        system_mass=mass,
        natural_time=time,
    )


def setup_from_config(cfg: dict[str, Any], *, progress: float = 1.0) -> PhysicalSetup:
    """Construct a physical setup using the repository's morphology builder."""

    from gcartpole.morphology import build_morphology

    env = cfg["env"]
    morphology = build_morphology(env, cfg["morphology"], progress=progress)
    return PhysicalSetup(
        lengths=morphology.lengths,
        masses=morphology.masses,
        cart_mass=float(env["cart_mass"]),
        rail_half_length=float(env["rail_limit"]),
        force_limit=float(env["force_limit"]),
        joint_damping=morphology.damping,
        cart_damping=float(env.get("cart_damping", 0.0)),
        joint_armature=float(env.get("joint_armature", 0.0)),
        cart_half_length=float(env.get("cart_half_length", 0.18)),
        link_radius=float(env.get("link_radius", 0.025)),
        timestep=float(env["timestep"]),
        frame_skip=int(env.get("frame_skip", 1)),
    )


def normalized_link_centers(lengths: np.ndarray | list[float]) -> np.ndarray:
    """Arc-length coordinates of link centers on the normalized chain [0, 1]."""

    lengths = _positive_vector("lengths", lengths)
    edges = np.r_[0.0, np.cumsum(lengths)] / float(np.sum(lengths))
    return 0.5 * (edges[:-1] + edges[1:])


def interpolation_matrix(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    """Piecewise-linear interpolation with constant boundary extrapolation."""

    source = np.asarray(source_points, dtype=np.float64)
    target = np.asarray(target_points, dtype=np.float64)
    if source.ndim != 1 or target.ndim != 1 or source.size == 0 or target.size == 0:
        raise ValueError("source_points and target_points must be nonempty vectors")
    if np.any(np.diff(source) <= 0.0) or not np.all(np.isfinite(source)):
        raise ValueError("source_points must be finite and strictly increasing")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_points must be finite")
    matrix = np.zeros((target.size, source.size), dtype=np.float64)
    for row, point in enumerate(target):
        if point <= source[0]:
            matrix[row, 0] = 1.0
        elif point >= source[-1]:
            matrix[row, -1] = 1.0
        else:
            right = int(np.searchsorted(source, point, side="right"))
            left = right - 1
            fraction = float((point - source[left]) / (source[right] - source[left]))
            matrix[row, left] = 1.0 - fraction
            matrix[row, right] = fraction
    return matrix


def _relative_to_absolute_matrix(n_links: int) -> np.ndarray:
    return np.tril(np.ones((n_links, n_links), dtype=np.float64))


def _absolute_to_relative_matrix(n_links: int) -> np.ndarray:
    result = np.eye(n_links, dtype=np.float64)
    if n_links > 1:
        result[1:, :-1] -= np.eye(n_links - 1, dtype=np.float64)
    return result


def state_transfer_matrix(source: PhysicalSetup, target: PhysicalSetup) -> np.ndarray:
    """Map a scale-normalized physical state from one morphology to another.

    MuJoCo state order is ``[x, relative_angles, xdot, hinge_rates]``.  Angular
    positions and rates are first converted into an absolute orientation field,
    interpolated by normalized arc length, and converted back to relative joints.
    """

    ns = source.n_links
    nt = target.n_links
    source_centers = normalized_link_centers(source.lengths)
    target_centers = normalized_link_centers(target.lengths)
    interpolate = interpolation_matrix(source_centers, target_centers)
    angular = (
        _absolute_to_relative_matrix(nt)
        @ interpolate
        @ _relative_to_absolute_matrix(ns)
    )
    result = np.zeros((2 * (nt + 1), 2 * (ns + 1)), dtype=np.float64)
    result[0, 0] = target.chain_length / source.chain_length
    result[1 : nt + 1, 1 : ns + 1] = angular
    source_velocity = ns + 1
    target_velocity = nt + 1
    result[target_velocity, source_velocity] = np.sqrt(
        target.chain_length / source.chain_length
    )
    result[target_velocity + 1 :, source_velocity + 1 :] = (
        np.sqrt(source.chain_length / target.chain_length) * angular
    )
    return result


def transfer_state(state: np.ndarray, source: PhysicalSetup, target: PhysicalSetup) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    expected = 2 * (source.n_links + 1)
    if state.shape != (expected,):
        raise ValueError(f"source state must have shape ({expected},)")
    return state_transfer_matrix(source, target) @ state


def force_action_scale(source: PhysicalSetup, target: PhysicalSetup) -> float:
    """Scale normalized actions to preserve force/(system weight)."""

    source_authority = dimensionless_setup(source).force_authority
    target_authority = dimensionless_setup(target).force_authority
    return float(source_authority / target_authority)


def transfer_feedback_gains(
    gains: np.ndarray, source: PhysicalSetup, target: PhysicalSetup
) -> np.ndarray:
    """Transfer source feedback to a target's state coordinates.

    The target state is projected back onto the source chain before the source
    gain is applied.  Gain magnitude is then adjusted for force authority.
    """

    gains = np.asarray(gains, dtype=np.float64)
    was_vector = gains.ndim == 1
    if was_vector:
        gains = gains[None, :]
    expected = 2 * (source.n_links + 1)
    if gains.ndim != 2 or gains.shape[1] != expected:
        raise ValueError(f"source gains must have {expected} columns")
    project_target_to_source = state_transfer_matrix(target, source)
    transferred = force_action_scale(source, target) * gains @ project_target_to_source
    return transferred.reshape(-1) if was_vector else transferred


def resample_controls(
    controls: np.ndarray,
    source: PhysicalSetup,
    target: PhysicalSetup,
    *,
    target_steps: int | None = None,
) -> np.ndarray:
    """Resample a force sequence on the natural-time clock of the target."""

    controls = np.asarray(controls, dtype=np.float64)
    if controls.ndim != 1 or controls.size == 0 or not np.all(np.isfinite(controls)):
        raise ValueError("controls must be a nonempty finite vector")
    dimensionless_duration = controls.size * source.policy_dt / source.natural_time
    if target_steps is None:
        target_steps = max(
            1,
            round(dimensionless_duration * target.natural_time / target.policy_dt),
        )
    if target_steps < 1:
        raise ValueError("target_steps must be positive")
    source_tau = (np.arange(controls.size, dtype=np.float64) + 0.5) * (
        source.policy_dt / source.natural_time
    )
    target_tau = (np.arange(target_steps, dtype=np.float64) + 0.5) * (
        target.policy_dt / target.natural_time
    )
    result = np.interp(target_tau, source_tau, controls, left=controls[0], right=controls[-1])
    return np.clip(force_action_scale(source, target) * result, -1.0, 1.0)


def rail_requirement(
    cart_positions: np.ndarray,
    setup: PhysicalSetup,
    *,
    clearance: float = 0.0,
) -> dict[str, float]:
    """Measure trajectory rail demand in physical and chain-normalized units."""

    positions = np.asarray(cart_positions, dtype=np.float64)
    if positions.size == 0 or not np.all(np.isfinite(positions)):
        raise ValueError("cart_positions must contain finite values")
    if not np.isfinite(clearance) or clearance < 0.0:
        raise ValueError("clearance must be finite and nonnegative")
    center_excursion = float(np.max(np.abs(positions)))
    required_half_length = center_excursion + setup.cart_half_length + float(clearance)
    return {
        "max_cart_center_excursion": center_excursion,
        "required_rail_half_length": required_half_length,
        "required_rail_ratio": required_half_length / setup.chain_length,
        "configured_rail_ratio": setup.rail_half_length / setup.chain_length,
        "margin": setup.rail_half_length - required_half_length,
    }


def mirror_feedback_route(
    controls: np.ndarray,
    coordinate_states: np.ndarray,
    feedback_gains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reflect a planar feedback route through the cart origin.

    The dynamics are invariant under ``(q, qdot, u) -> (-q, -qdot, -u)``.
    The gain itself is unchanged because both the live and nominal state errors
    reverse sign.
    """

    actions = np.asarray(controls, dtype=np.float64)
    states = np.asarray(coordinate_states, dtype=np.float64)
    gains = np.asarray(feedback_gains, dtype=np.float64)
    if actions.ndim != 1:
        raise ValueError("controls must be one-dimensional")
    if states.ndim != 2 or states.shape[0] != actions.size + 1:
        raise ValueError("coordinate_states must contain one state per boundary")
    if gains.shape != (actions.size, states.shape[1]):
        raise ValueError("feedback_gains do not match the route dimensions")
    return -actions.copy(), -states.copy(), gains.copy()


@dataclass
class BoundedForceAdapter:
    """Projected RLS calibration for force gain and bias mismatch.

    The exact nominal model supplies its local action Jacobian ``B``.  After an
    observed transition, the model error is projected onto ``B`` to estimate the
    equivalent action actually delivered.  A two-parameter recursive least
    squares fit tracks ``u_equivalent = gain*u_command + bias``.  Projection and
    bounds keep this layer conservative; structural mismatch remains the job of
    receding-horizon replanning.
    """

    forgetting: float = 0.995
    covariance: float = 10.0
    gain_bounds: tuple[float, float] = (0.5, 1.5)
    bias_bound: float = 0.25
    correction_bound: float = 0.35
    projection_floor: float = 1e-10
    command_update_limit: float = 0.85
    structural_fraction_limit: float = 0.50
    equivalent_delta_limit: float = 0.75

    def __post_init__(self) -> None:
        if not 0.0 < self.forgetting <= 1.0:
            raise ValueError("forgetting must lie in (0, 1]")
        if self.covariance <= 0.0:
            raise ValueError("covariance must be positive")
        if self.gain_bounds[0] <= 0.0 or self.gain_bounds[0] > self.gain_bounds[1]:
            raise ValueError("gain_bounds must be ordered and positive")
        if self.bias_bound < 0.0 or self.correction_bound < 0.0:
            raise ValueError("adapter bounds must be nonnegative")
        if not 0.0 < self.command_update_limit <= 1.0:
            raise ValueError("command_update_limit must lie in (0, 1]")
        if not 0.0 <= self.structural_fraction_limit <= 1.0:
            raise ValueError("structural_fraction_limit must lie in [0, 1]")
        if self.equivalent_delta_limit <= 0.0:
            raise ValueError("equivalent_delta_limit must be positive")
        self.theta = np.array([1.0, 0.0], dtype=np.float64)
        self.P = float(self.covariance) * np.eye(2, dtype=np.float64)
        self.updates = 0
        self.rejections = 0
        self.last_observation: dict[str, float | bool | str] | None = None

    @property
    def gain(self) -> float:
        return float(self.theta[0])

    @property
    def bias(self) -> float:
        return float(self.theta[1])

    def command(self, nominal_action: float) -> float:
        desired = float(np.clip(nominal_action, -1.0, 1.0))
        compensated = (desired - self.bias) / self.gain
        correction = float(np.clip(compensated - desired, -self.correction_bound, self.correction_bound))
        return float(np.clip(desired + correction, -1.0, 1.0))

    def observe(
        self,
        commanded_action: float,
        predicted_next_state: np.ndarray,
        observed_next_state: np.ndarray,
        action_jacobian: np.ndarray,
    ) -> bool:
        return bool(
            self.observe_diagnostic(
                commanded_action,
                predicted_next_state,
                observed_next_state,
                action_jacobian,
            )["updated"]
        )

    def observe_diagnostic(
        self,
        commanded_action: float,
        predicted_next_state: np.ndarray,
        observed_next_state: np.ndarray,
        action_jacobian: np.ndarray,
    ) -> dict[str, float | bool | str]:
        """Project one-step error onto actuation and conditionally update RLS.

        State vectors and the action Jacobian should use the same dimensionless
        coordinates.  Updates are rejected when the command is too close to
        saturation, when the inferred action jump is implausibly large, or when
        too much model error lies orthogonal to the action direction.  The last
        condition is the explicit boundary between this narrow calibration
        layer and structural mismatch that requires model correction/replanning.
        """

        predicted = np.asarray(predicted_next_state, dtype=np.float64)
        observed = np.asarray(observed_next_state, dtype=np.float64)
        jacobian = np.asarray(action_jacobian, dtype=np.float64).reshape(-1)
        if predicted.shape != observed.shape or predicted.shape != jacobian.shape:
            raise ValueError("predicted, observed, and action_jacobian shapes must match")
        denominator = float(jacobian @ jacobian)
        if denominator <= self.projection_floor or not np.isfinite(denominator):
            diagnostic: dict[str, float | bool | str] = {
                "updated": False,
                "reason": "unobservable_action_direction",
                "commanded_action": float(commanded_action),
                "equivalent_action_delta": 0.0,
                "structural_fraction": 1.0,
                "innovation_norm": float(np.linalg.norm(observed - predicted)),
            }
            self.rejections += 1
            self.last_observation = diagnostic
            return diagnostic
        innovation = observed - predicted
        equivalent_delta = float(jacobian @ innovation / denominator)
        projected = equivalent_delta * jacobian
        innovation_norm = float(np.linalg.norm(innovation))
        orthogonal_norm = float(np.linalg.norm(innovation - projected))
        structural_fraction = (
            orthogonal_norm / innovation_norm
            if innovation_norm > self.projection_floor
            else 0.0
        )
        equivalent_action = float(commanded_action) + equivalent_delta
        reason = "accepted"
        if abs(float(commanded_action)) > self.command_update_limit:
            reason = "command_near_saturation"
        elif abs(equivalent_delta) > self.equivalent_delta_limit:
            reason = "action_innovation_out_of_bounds"
        elif structural_fraction > self.structural_fraction_limit:
            reason = "structural_mismatch"
        diagnostic = {
            "updated": reason == "accepted",
            "reason": reason,
            "commanded_action": float(commanded_action),
            "equivalent_action": equivalent_action,
            "equivalent_action_delta": equivalent_delta,
            "structural_fraction": structural_fraction,
            "innovation_norm": innovation_norm,
            "orthogonal_innovation_norm": orthogonal_norm,
        }
        if reason != "accepted":
            self.rejections += 1
            self.last_observation = diagnostic
            return diagnostic
        feature = np.array([float(commanded_action), 1.0], dtype=np.float64)
        p_feature = self.P @ feature
        denominator_rls = self.forgetting + float(feature @ p_feature)
        kalman = p_feature / denominator_rls
        self.theta += kalman * (equivalent_action - float(feature @ self.theta))
        self.P = (self.P - np.outer(kalman, feature) @ self.P) / self.forgetting
        self.theta[0] = np.clip(self.theta[0], *self.gain_bounds)
        self.theta[1] = np.clip(self.theta[1], -self.bias_bound, self.bias_bound)
        self.updates += 1
        self.last_observation = diagnostic
        return diagnostic

    def to_dict(self) -> dict[str, float | int]:
        return {
            "gain": self.gain,
            "bias": self.bias,
            "updates": self.updates,
            "rejections": self.rejections,
            "covariance_trace": float(np.trace(self.P)),
        }


def embed_morphology(
    lengths: np.ndarray | list[float],
    masses: np.ndarray | list[float],
    target_count: int,
    *,
    ghost_length: float = 0.02,
    ghost_mass: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Embed a chain in a larger coordinate space using distal ghost links.

    Total length and mass are conserved by taking each ghost's small positive
    amount from the last real link.  Adjacent link-count continuation can then
    remain in one fixed-dimensional MuJoCo model until the final projection.
    """

    source_lengths = _positive_vector("lengths", lengths)
    source_masses = _positive_vector("masses", masses)
    if source_lengths.shape != source_masses.shape:
        raise ValueError("lengths and masses must have the same shape")
    if target_count < source_lengths.size:
        raise ValueError("target_count cannot be smaller than the source count")
    extra = int(target_count - source_lengths.size)
    if extra == 0:
        return source_lengths.copy(), source_masses.copy()
    if min(ghost_length, ghost_mass) <= 0.0:
        raise ValueError("ghost dimensions must be positive")
    length_taken = extra * float(ghost_length)
    mass_taken = extra * float(ghost_mass)
    if source_lengths[-1] <= length_taken or source_masses[-1] <= mass_taken:
        raise ValueError("last source link is too small to create requested ghosts")
    embedded_lengths = np.r_[
        source_lengths[:-1], source_lengths[-1] - length_taken, np.full(extra, ghost_length)
    ]
    embedded_masses = np.r_[
        source_masses[:-1], source_masses[-1] - mass_taken, np.full(extra, ghost_mass)
    ]
    return embedded_lengths, embedded_masses


def common_count_morphologies(
    source_lengths: np.ndarray | list[float],
    source_masses: np.ndarray | list[float],
    target_lengths: np.ndarray | list[float],
    target_masses: np.ndarray | list[float],
    *,
    ghost_length: float = 0.02,
    ghost_mass: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Embed two morphologies into a common count for smooth homotopy."""

    source_lengths = _positive_vector("source_lengths", source_lengths)
    source_masses = _positive_vector("source_masses", source_masses)
    target_lengths = _positive_vector("target_lengths", target_lengths)
    target_masses = _positive_vector("target_masses", target_masses)
    count = max(source_lengths.size, target_lengths.size)
    source_l, source_m = embed_morphology(
        source_lengths, source_masses, count,
        ghost_length=ghost_length, ghost_mass=ghost_mass,
    )
    target_l, target_m = embed_morphology(
        target_lengths, target_masses, count,
        ghost_length=ghost_length, ghost_mass=ghost_mass,
    )
    if not np.isclose(np.sum(source_l), np.sum(target_l)):
        raise ValueError("source and target total lengths must match")
    if not np.isclose(np.sum(source_m), np.sum(target_m)):
        raise ValueError("source and target total link masses must match")
    return source_l, source_m, target_l, target_m


def homotopy_morphology(
    source_lengths: np.ndarray | list[float],
    source_masses: np.ndarray | list[float],
    target_lengths: np.ndarray | list[float],
    target_masses: np.ndarray | list[float],
    progress: float,
    *,
    ghost_length: float = 0.02,
    ghost_mass: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate arbitrary source/target chains in a common link space."""

    if not 0.0 <= progress <= 1.0:
        raise ValueError("progress must lie in [0, 1]")
    source_l, source_m, target_l, target_m = common_count_morphologies(
        source_lengths,
        source_masses,
        target_lengths,
        target_masses,
        ghost_length=ghost_length,
        ghost_mass=ghost_mass,
    )
    return (
        (1.0 - progress) * source_l + progress * target_l,
        (1.0 - progress) * source_m + progress * target_m,
    )


@dataclass
class AdaptiveHomotopy:
    """Deterministic accept/grow, reject/bisect continuation schedule."""

    progress: float = 0.0
    step: float = 0.001
    minimum_step: float = 1e-5
    maximum_step: float = 0.10
    growth: float = 1.6

    def __post_init__(self) -> None:
        if not 0.0 <= self.progress <= 1.0:
            raise ValueError("progress must lie in [0, 1]")
        if not 0.0 < self.minimum_step <= self.step <= self.maximum_step:
            raise ValueError("step bounds must be positive and ordered")
        if self.growth <= 1.0:
            raise ValueError("growth must exceed one")

    def proposal(self) -> float:
        return float(min(1.0, self.progress + self.step))

    def accept(self, proposed: float) -> None:
        if proposed <= self.progress or proposed > 1.0:
            raise ValueError("accepted progress must advance within [0, 1]")
        self.progress = float(proposed)
        self.step = float(min(self.maximum_step, self.step * self.growth))

    def reject(self) -> None:
        next_step = self.step / 2.0
        if next_step < self.minimum_step:
            raise RuntimeError("homotopy step fell below minimum_step")
        self.step = float(next_step)
