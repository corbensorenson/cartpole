from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Morphology:
    n_links: int
    lengths: np.ndarray
    masses: np.ndarray
    damping: np.ndarray
    total_length: float
    total_mass: float
    total_damping: float
    alpha_length: float
    alpha_mass: float
    alpha_damping: float

    def fingerprint(self) -> np.ndarray:
        """Normalized morphology vector for optional policy conditioning."""
        l = self.lengths / (self.total_length / self.n_links)
        m = self.masses / (self.total_mass / self.n_links)
        if self.total_damping > 0:
            d = self.damping / (self.total_damping / self.n_links)
        else:
            d = np.zeros_like(self.damping)
        return np.concatenate([l, m, d]).astype(np.float32)


def exp_gradient(total: float, n: int, alpha: float, min_value: float = 1e-9) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be positive")
    if n == 1:
        return np.array([float(total)], dtype=np.float64)
    s = np.linspace(0.0, 1.0, n, dtype=np.float64)
    weights = np.exp(-float(alpha) * s)
    values = float(total) * weights / np.sum(weights)
    if np.any(values < min_value):
        values = np.maximum(values, min_value)
        values *= float(total) / np.sum(values)
    return values.astype(np.float64)


def interpolate(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(np.clip(t, 0.0, 1.0))


def _stage_value(start: float, end: float, progress: float, begin: float, finish: float) -> float:
    if progress <= begin:
        return float(start)
    if progress >= finish:
        return float(end)
    local = (progress - begin) / (finish - begin)
    # smoothstep avoids sudden morphology jumps at stage boundaries
    local = local * local * (3 - 2 * local)
    return interpolate(start, end, local)


def scheduled_params(schedule_cfg: dict[str, Any], progress: float) -> dict[str, float]:
    start = schedule_cfg.get("start", {})
    end = schedule_cfg.get("end", {})
    mode = schedule_cfg.get("schedule_mode", "mass_last")
    t = float(np.clip(progress, 0.0, 1.0))

    if mode == "all_linear":
        return {
            "alpha_length": interpolate(start.get("alpha_length", 0.0), end.get("alpha_length", 0.0), t),
            "alpha_mass": interpolate(start.get("alpha_mass", 0.0), end.get("alpha_mass", 0.0), t),
            "alpha_damping": interpolate(start.get("alpha_damping", 0.0), end.get("alpha_damping", 0.0), t),
            "total_damping": interpolate(start.get("total_damping", 0.0), end.get("total_damping", 0.0), t),
        }

    if mode == "mass_last":
        # Remove the easiest/passive crutches first, keep mass gradient until late.
        return {
            "alpha_damping": _stage_value(start.get("alpha_damping", 0.0), end.get("alpha_damping", 0.0), t, 0.00, 0.25),
            "total_damping": _stage_value(start.get("total_damping", 0.0), end.get("total_damping", 0.0), t, 0.00, 0.35),
            "alpha_length": _stage_value(start.get("alpha_length", 0.0), end.get("alpha_length", 0.0), t, 0.20, 0.55),
            "alpha_mass": _stage_value(start.get("alpha_mass", 0.0), end.get("alpha_mass", 0.0), t, 0.45, 1.00),
        }

    if mode == "swingup_slow":
        # Swing-up is more sensitive than near-upright stabilization. Keep the
        # damping/length/mass training wheels longer while the hanging-start
        # angle is ramping up, then remove everything before final progress.
        return {
            "alpha_damping": _stage_value(start.get("alpha_damping", 0.0), end.get("alpha_damping", 0.0), t, 0.10, 0.60),
            "total_damping": _stage_value(start.get("total_damping", 0.0), end.get("total_damping", 0.0), t, 0.10, 0.65),
            "alpha_length": _stage_value(start.get("alpha_length", 0.0), end.get("alpha_length", 0.0), t, 0.35, 0.80),
            "alpha_mass": _stage_value(start.get("alpha_mass", 0.0), end.get("alpha_mass", 0.0), t, 0.65, 1.00),
        }

    raise ValueError(f"Unknown morphology.schedule_mode: {mode}")


def build_morphology(env_cfg: dict[str, Any], morph_cfg: dict[str, Any], progress: float = 0.0) -> Morphology:
    n = int(env_cfg["n_links"])
    total_length = float(env_cfg["total_length"])
    total_mass = float(env_cfg["total_mass"])

    params = scheduled_params(morph_cfg, progress)
    total_damping = float(params["total_damping"])
    lengths = exp_gradient(total_length, n, params["alpha_length"], min_value=0.02)
    masses = exp_gradient(total_mass, n, params["alpha_mass"], min_value=1e-4)
    damping = exp_gradient(total_damping, n, params["alpha_damping"], min_value=0.0) if total_damping > 0 else np.zeros(n)

    return Morphology(
        n_links=n,
        lengths=lengths,
        masses=masses,
        damping=damping,
        total_length=total_length,
        total_mass=total_mass,
        total_damping=total_damping,
        alpha_length=float(params["alpha_length"]),
        alpha_mass=float(params["alpha_mass"]),
        alpha_damping=float(params["alpha_damping"]),
    )
