from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eig, schur


@dataclass(frozen=True)
class StateScales:
    cart_position: float
    absolute_angle: float
    cart_velocity: float
    hinge_velocity: float

    def __post_init__(self) -> None:
        values = (
            self.cart_position,
            self.absolute_angle,
            self.cart_velocity,
            self.hinge_velocity,
        )
        if not all(np.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("all state scales must be finite and positive")


@dataclass(frozen=True)
class ModalDecomposition:
    eigenvalues: np.ndarray
    right_eigenvectors: np.ndarray
    inverse_right_eigenvectors: np.ndarray
    input_coupling: np.ndarray
    eigenvector_condition: float

    def coordinates(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.right_eigenvectors.shape[0],):
            raise ValueError("state shape does not match modal decomposition")
        return self.inverse_right_eigenvectors @ state

    def reconstruct(self, coordinates: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=np.complex128)
        if coordinates.shape != (self.right_eigenvectors.shape[1],):
            raise ValueError("coordinate shape does not match modal decomposition")
        return np.real_if_close(self.right_eigenvectors @ coordinates, tol=1000).real


@dataclass(frozen=True)
class RealSchurDecomposition:
    schur_matrix: np.ndarray
    orthogonal_basis: np.ndarray
    groups: tuple[tuple[int, ...], ...]
    group_eigenvalues: tuple[np.ndarray, ...]
    group_input_coupling: np.ndarray

    def coordinates(self, state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (self.orthogonal_basis.shape[0],):
            raise ValueError("state shape does not match Schur decomposition")
        return self.orthogonal_basis.T @ state

    def reconstruct(self, coordinates: np.ndarray) -> np.ndarray:
        coordinates = np.asarray(coordinates, dtype=np.float64)
        if coordinates.shape != (self.orthogonal_basis.shape[1],):
            raise ValueError("coordinate shape does not match Schur decomposition")
        return self.orthogonal_basis @ coordinates

    def grouped_amplitudes(self, state: np.ndarray) -> np.ndarray:
        coordinates = self.coordinates(state)
        return np.asarray(
            [np.linalg.norm(coordinates[list(group)]) for group in self.groups],
            dtype=np.float64,
        )


def dimensionless_absolute_transform(n_links: int, scales: StateScales) -> np.ndarray:
    """Map [x, relative angles, xdot, hinge velocities] to scaled physical coordinates."""
    if n_links < 1:
        raise ValueError("n_links must be positive")
    d = n_links + 1
    transform = np.zeros((2 * d, 2 * d), dtype=np.float64)
    transform[0, 0] = 1.0 / scales.cart_position
    for link in range(n_links):
        transform[1 + link, 1 : 2 + link] = 1.0 / scales.absolute_angle
    transform[d, d] = 1.0 / scales.cart_velocity
    transform[d + 1 :, d + 1 :] = np.eye(n_links, dtype=np.float64) / scales.hinge_velocity
    return transform


def transform_dynamics(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    transform: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    state_matrix = np.asarray(state_matrix, dtype=np.float64)
    input_matrix = np.asarray(input_matrix, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if state_matrix.ndim != 2 or state_matrix.shape[0] != state_matrix.shape[1]:
        raise ValueError("state matrix must be square")
    if transform.shape != state_matrix.shape:
        raise ValueError("transform shape must match state matrix")
    if input_matrix.ndim != 2 or input_matrix.shape[0] != state_matrix.shape[0]:
        raise ValueError("input matrix row count must match state matrix")
    inverse = np.linalg.inv(transform)
    return transform @ state_matrix @ inverse, transform @ input_matrix


def transform_feedback_gain(gain: np.ndarray, transform: np.ndarray) -> np.ndarray:
    gain = np.asarray(gain, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    vector_input = gain.ndim == 1
    if vector_input:
        gain = gain[None, :]
    if gain.ndim != 2 or gain.shape[1] != transform.shape[0] or transform.shape[0] != transform.shape[1]:
        raise ValueError("gain and transform shapes do not match")
    transformed = gain @ np.linalg.inv(transform)
    return transformed.reshape(-1) if vector_input else transformed


def modal_decomposition(state_matrix: np.ndarray, input_matrix: np.ndarray) -> ModalDecomposition:
    state_matrix = np.asarray(state_matrix, dtype=np.float64)
    input_matrix = np.asarray(input_matrix, dtype=np.float64)
    if state_matrix.ndim != 2 or state_matrix.shape[0] != state_matrix.shape[1]:
        raise ValueError("state matrix must be square")
    if input_matrix.ndim != 2 or input_matrix.shape[0] != state_matrix.shape[0]:
        raise ValueError("input matrix row count must match state matrix")

    eigenvalues, left, right = eig(state_matrix, left=True, right=True)
    order = np.asarray(
        sorted(
            range(eigenvalues.size),
            key=lambda index: (
                -abs(complex(eigenvalues[index])),
                -float(np.real(eigenvalues[index])),
                -float(np.imag(eigenvalues[index])),
            ),
        ),
        dtype=np.int64,
    )
    eigenvalues = eigenvalues[order]
    left = left[:, order]
    right = right[:, order]
    inverse = np.linalg.inv(right)
    coupling = np.asarray(
        [
            np.linalg.norm(left[:, index].conj() @ input_matrix)
            / max(np.linalg.norm(left[:, index]), 1e-15)
            for index in range(eigenvalues.size)
        ],
        dtype=np.float64,
    )
    return ModalDecomposition(
        eigenvalues=eigenvalues,
        right_eigenvectors=right,
        inverse_right_eigenvectors=inverse,
        input_coupling=coupling,
        eigenvector_condition=float(np.linalg.cond(right)),
    )


def real_schur_decomposition(
    state_matrix: np.ndarray,
    input_matrix: np.ndarray,
    tolerance: float = 1e-10,
) -> RealSchurDecomposition:
    state_matrix = np.asarray(state_matrix, dtype=np.float64)
    input_matrix = np.asarray(input_matrix, dtype=np.float64)
    if state_matrix.ndim != 2 or state_matrix.shape[0] != state_matrix.shape[1]:
        raise ValueError("state matrix must be square")
    if input_matrix.ndim != 2 or input_matrix.shape[0] != state_matrix.shape[0]:
        raise ValueError("input matrix row count must match state matrix")

    schur_matrix, basis = schur(state_matrix, output="real")
    groups: list[tuple[int, ...]] = []
    index = 0
    while index < schur_matrix.shape[0]:
        if index + 1 < schur_matrix.shape[0] and abs(float(schur_matrix[index + 1, index])) > tolerance:
            groups.append((index, index + 1))
            index += 2
        else:
            groups.append((index,))
            index += 1
    eigenvalues = tuple(
        np.linalg.eigvals(schur_matrix[np.ix_(group, group)]).astype(np.complex128)
        for group in groups
    )
    coupling = np.asarray(
        [np.linalg.norm(basis[:, list(group)].T @ input_matrix) for group in groups],
        dtype=np.float64,
    )
    return RealSchurDecomposition(
        schur_matrix=schur_matrix,
        orthogonal_basis=basis,
        groups=tuple(groups),
        group_eigenvalues=eigenvalues,
        group_input_coupling=coupling,
    )


def scale_feedback_by_schur_group(
    dimensionless_gain: np.ndarray,
    decomposition: RealSchurDecomposition,
    group_multipliers: dict[int, float],
) -> np.ndarray:
    dimensionless_gain = np.asarray(dimensionless_gain, dtype=np.float64)
    vector_input = dimensionless_gain.ndim == 1
    if vector_input:
        dimensionless_gain = dimensionless_gain[None, :]
    if dimensionless_gain.ndim != 2 or dimensionless_gain.shape[1] != decomposition.orthogonal_basis.shape[0]:
        raise ValueError("gain shape does not match Schur decomposition")
    schur_gain = dimensionless_gain @ decomposition.orthogonal_basis
    for group_index, multiplier in group_multipliers.items():
        if group_index < 0 or group_index >= len(decomposition.groups):
            raise IndexError(f"Schur group index {group_index} is out of range")
        if not np.isfinite(multiplier) or multiplier < 0.0:
            raise ValueError("Schur group multipliers must be finite and nonnegative")
        schur_gain[:, list(decomposition.groups[group_index])] *= float(multiplier)
    scaled = schur_gain @ decomposition.orthogonal_basis.T
    return scaled.reshape(-1) if vector_input else scaled


def conjugate_mode_groups(eigenvalues: np.ndarray, tolerance: float = 1e-8) -> list[tuple[int, ...]]:
    eigenvalues = np.asarray(eigenvalues, dtype=np.complex128)
    unused = set(range(eigenvalues.size))
    groups: list[tuple[int, ...]] = []
    while unused:
        index = min(unused)
        unused.remove(index)
        value = eigenvalues[index]
        if abs(float(np.imag(value))) <= tolerance:
            groups.append((index,))
            continue
        matches = sorted(
            unused,
            key=lambda candidate: abs(eigenvalues[candidate] - np.conj(value)),
        )
        if not matches or abs(eigenvalues[matches[0]] - np.conj(value)) > tolerance:
            groups.append((index,))
            continue
        partner = matches[0]
        unused.remove(partner)
        groups.append(tuple(sorted((index, partner))))
    return groups


def grouped_modal_amplitudes(coordinates: np.ndarray, groups: list[tuple[int, ...]]) -> np.ndarray:
    coordinates = np.asarray(coordinates, dtype=np.complex128)
    return np.asarray(
        [np.sqrt(np.sum(np.abs(coordinates[list(group)]) ** 2)) for group in groups],
        dtype=np.float64,
    )
