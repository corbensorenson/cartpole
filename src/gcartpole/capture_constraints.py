from __future__ import annotations

import numpy as np

from gcartpole.modal import StateScales


def terminal_bounds(
    n_links: int,
    scales: StateScales,
    *,
    cart_abs: float,
    angle_abs: float,
    cart_velocity_abs: float,
    hinge_velocity_abs: float,
) -> tuple[np.ndarray, np.ndarray]:
    bounds = np.r_[
        cart_abs / scales.cart_position,
        np.full(n_links, angle_abs / scales.absolute_angle),
        cart_velocity_abs / scales.cart_velocity,
        np.full(n_links, hinge_velocity_abs / scales.hinge_velocity),
    ]
    return -bounds, bounds


def exact_constraint_margins(
    states: np.ndarray,
    lyapunov: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    rail_limit: float,
    handoff_lyapunov: float,
) -> dict[str, float]:
    terminal = states[-1]
    rail_margin = float(rail_limit - np.max(np.abs(states[:, 0])))
    terminal_margin = float(min(np.min(terminal - lower), np.min(upper - terminal)))
    terminal_value = float(max(0.0, terminal @ lyapunov @ terminal))
    return {
        "rail": rail_margin,
        "terminal_box": terminal_margin,
        "lyapunov": float(handoff_lyapunov - terminal_value),
        "minimum_hard": min(rail_margin, terminal_margin),
    }


def normalized_constraint_score(
    margins: dict[str, float], *, handoff_lyapunov: float, rail_limit: float
) -> float:
    return max(
        0.0,
        -margins["terminal_box"],
        -margins["rail"] / rail_limit,
        -margins["lyapunov"] / handoff_lyapunov,
    )
