"""Exact replay of a saved time-varying state-feedback route."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .generalized_solver import periodic_coordinate_error
from .modal import StateScales, dimensionless_absolute_transform, dimensionless_wrapped_state


_ROUTE_CACHE: dict[Path, dict[str, Any]] = {}


def _load_payload(path: Path) -> dict[str, Any]:
    cached = _ROUTE_CACHE.get(path)
    if cached is not None:
        return cached
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("feedback route must be a JSON object")
    _ROUTE_CACHE[path] = payload
    return payload


class TimeVaryingFeedbackRoute:
    """Route controller with phase adaptation over a saved nominal trajectory."""

    def __init__(
        self,
        path: str | Path,
        n_links: int,
        *,
        tracking_gain_scale: float = 1.0,
        phase_window: int = 12,
    ) -> None:
        route_path = Path(path).expanduser().resolve()
        payload = _load_payload(route_path)
        controller = payload.get("controller")
        search = payload.get("search")
        if not isinstance(controller, dict) or not isinstance(search, dict):
            raise ValueError("feedback route must contain controller and search objects")
        controls = np.asarray(controller.get("controls", []), dtype=np.float64)
        gains = np.asarray(controller.get("feedback_gains", []), dtype=np.float64)
        nominal = np.asarray(search.get("nominal_coordinate_states", []), dtype=np.float64)
        state_dim = 2 * (int(n_links) + 1)
        if controls.ndim != 1 or controls.size < 2:
            raise ValueError("feedback route controls must contain at least two values")
        if gains.shape != (controls.size, state_dim):
            raise ValueError("feedback route gains do not match the target state dimension")
        if nominal.shape != (controls.size + 1, state_dim):
            raise ValueError("feedback route nominal states do not match the controls")
        if tracking_gain_scale < 0.0 or phase_window < 0:
            raise ValueError("tracking_gain_scale and phase_window must be nonnegative")
        self.path = route_path
        self.controls = np.clip(controls, -1.0, 1.0)
        self.feedback_gains = gains
        self.nominal_coordinate_states = nominal
        self.transform = dimensionless_absolute_transform(
            int(n_links),
            StateScales(
                cart_position=1.25,
                absolute_angle=0.15,
                cart_velocity=0.50,
                hinge_velocity=0.75,
            ),
        )
        self.periodic_coordinate_errors = bool(controller.get("periodic_coordinate_errors", False))
        self.tracking_gain_scale = float(tracking_gain_scale)
        self.phase_window = int(phase_window)
        self.phase_cursor = 0
        self.last_route_index: int | None = None
        self.last_base_action = 0.0

    @property
    def route_length(self) -> int:
        return int(self.controls.size)

    @property
    def phase_fraction(self) -> float:
        return float(np.clip(self.phase_cursor / max(1, self.route_length), 0.0, 1.0))

    def reset(self) -> None:
        self.phase_cursor = 0
        self.last_route_index = None
        self.last_base_action = 0.0

    def action(self, qpos: np.ndarray, qvel: np.ndarray) -> tuple[float, int | None]:
        if self.phase_cursor >= self.route_length:
            self.last_route_index = None
            self.last_base_action = 0.0
            return 0.0, None
        coordinate_state = dimensionless_wrapped_state(qpos, qvel, self.transform)
        nominal = self.nominal_coordinate_states
        upper = min(
            self.route_length,
            self.phase_cursor + self.phase_window + 1,
        )
        candidates = np.arange(self.phase_cursor, upper, dtype=np.int64)
        errors = np.asarray(
            [self._error(coordinate_state, nominal[index]) for index in candidates],
            dtype=np.float64,
        )
        route_index = int(
            candidates[int(np.argmin(np.einsum("ij,ij->i", errors, errors)))]
        )
        self.phase_cursor = route_index + 1
        action = self.controls[route_index] + self.tracking_gain_scale * float(
            self.feedback_gains[route_index] @ self._error(
                coordinate_state, nominal[route_index]
            )
        )
        action = float(np.clip(action, -1.0, 1.0))
        self.last_route_index = route_index
        self.last_base_action = action
        return action, route_index

    def _error(self, current: np.ndarray, reference: np.ndarray) -> np.ndarray:
        if self.periodic_coordinate_errors:
            return periodic_coordinate_error(current, reference, self.transform)
        return np.asarray(current, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
