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
    frictionloss: np.ndarray
    joint_stiffness: np.ndarray
    joint_lock: np.ndarray
    total_length: float
    total_mass: float
    total_damping: float
    total_frictionloss: float
    alpha_length: float
    alpha_mass: float
    alpha_damping: float
    alpha_frictionloss: float

    def fingerprint(self) -> np.ndarray:
        """Normalized morphology vector for optional policy conditioning."""
        l = self.lengths / (self.total_length / self.n_links)
        m = self.masses / (self.total_mass / self.n_links)
        if self.total_damping > 0:
            d = self.damping / (self.total_damping / self.n_links)
        else:
            d = np.zeros_like(self.damping)
        return np.concatenate([l, m, d]).astype(np.float32)

    def frictionloss_fingerprint(self) -> np.ndarray:
        if self.total_frictionloss > 0:
            return (self.frictionloss / (self.total_frictionloss / self.n_links)).astype(np.float32)
        return np.zeros_like(self.frictionloss, dtype=np.float32)


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
            "alpha_frictionloss": interpolate(start.get("alpha_frictionloss", 0.0), end.get("alpha_frictionloss", 0.0), t),
            "total_frictionloss": interpolate(start.get("total_frictionloss", 0.0), end.get("total_frictionloss", 0.0), t),
        }

    if mode == "mass_last":
        # Remove the easiest/passive crutches first, keep mass gradient until late.
        return {
            "alpha_damping": _stage_value(start.get("alpha_damping", 0.0), end.get("alpha_damping", 0.0), t, 0.00, 0.25),
            "total_damping": _stage_value(start.get("total_damping", 0.0), end.get("total_damping", 0.0), t, 0.00, 0.35),
            "alpha_frictionloss": _stage_value(start.get("alpha_frictionloss", 0.0), end.get("alpha_frictionloss", 0.0), t, 0.00, 0.35),
            "total_frictionloss": _stage_value(start.get("total_frictionloss", 0.0), end.get("total_frictionloss", 0.0), t, 0.00, 0.45),
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
            "alpha_frictionloss": _stage_value(start.get("alpha_frictionloss", 0.0), end.get("alpha_frictionloss", 0.0), t, 0.10, 0.65),
            "total_frictionloss": _stage_value(start.get("total_frictionloss", 0.0), end.get("total_frictionloss", 0.0), t, 0.10, 0.70),
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
    total_frictionloss = float(params["total_frictionloss"])
    def scheduled_profile(name: str, total: float, minimum: float) -> np.ndarray | None:
        start = morph_cfg.get(f"{name}_start")
        end = morph_cfg.get(f"{name}_end")
        if start is None and end is None:
            return None
        if not isinstance(start, list) or not isinstance(end, list):
            raise ValueError(f"{name}_start and {name}_end must both be lists")
        if len(start) != n or len(end) != n:
            raise ValueError(f"{name}_start and {name}_end must have {n} entries")
        values = np.asarray(start, dtype=np.float64) + (
            np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        ) * float(np.clip(progress, 0.0, 1.0))
        if np.any(values < minimum) or not np.all(np.isfinite(values)):
            raise ValueError(f"{name} profile contains invalid values")
        if not np.isclose(float(np.sum(values)), total, rtol=1e-8, atol=1e-8):
            raise ValueError(f"{name} profile must sum to {total}")
        return values

    lengths = scheduled_profile("lengths", total_length, 0.02)
    if lengths is None:
        lengths = exp_gradient(total_length, n, params["alpha_length"], min_value=0.02)
    masses = scheduled_profile("masses", total_mass, 1e-4)
    if masses is None:
        masses = exp_gradient(total_mass, n, params["alpha_mass"], min_value=1e-4)
    damping = scheduled_profile("damping", total_damping, 0.0)
    if damping is None:
        damping = (
            exp_gradient(total_damping, n, params["alpha_damping"], min_value=0.0)
            if total_damping > 0
            else np.zeros(n)
        )
    frictionloss = scheduled_profile("frictionloss", total_frictionloss, 0.0)
    if frictionloss is None:
        frictionloss = (
            exp_gradient(total_frictionloss, n, params["alpha_frictionloss"], min_value=0.0)
            if total_frictionloss > 0
            else np.zeros(n)
        )

    stiffness_start = morph_cfg.get("joint_stiffness_start")
    stiffness_end = morph_cfg.get("joint_stiffness_end")
    if stiffness_start is None and stiffness_end is None:
        joint_stiffness = np.full(
            n, float(morph_cfg.get("joint_stiffness", 0.0)), dtype=np.float64
        )
    else:
        if not isinstance(stiffness_start, list) or not isinstance(stiffness_end, list):
            raise ValueError(
                "joint_stiffness_start and joint_stiffness_end must both be lists"
            )
        if len(stiffness_start) != n or len(stiffness_end) != n:
            raise ValueError(
                f"joint_stiffness_start and joint_stiffness_end must have {n} entries"
            )
        joint_stiffness = np.asarray(stiffness_start, dtype=np.float64) + (
            np.asarray(stiffness_end, dtype=np.float64)
            - np.asarray(stiffness_start, dtype=np.float64)
        ) * float(np.clip(progress, 0.0, 1.0))
        if np.any(joint_stiffness < 0.0) or not np.all(np.isfinite(joint_stiffness)):
            raise ValueError("joint stiffness profile contains invalid values")

    lock_start = morph_cfg.get("joint_lock_start")
    lock_end = morph_cfg.get("joint_lock_end")
    if lock_start is None and lock_end is None:
        joint_lock = np.zeros(n, dtype=np.float64)
    else:
        if not isinstance(lock_start, list) or not isinstance(lock_end, list):
            raise ValueError("joint_lock_start and joint_lock_end must both be lists")
        if len(lock_start) != n or len(lock_end) != n:
            raise ValueError(f"joint_lock_start and joint_lock_end must have {n} entries")
        joint_lock = np.asarray(lock_start, dtype=np.float64) + (
            np.asarray(lock_end, dtype=np.float64) - np.asarray(lock_start, dtype=np.float64)
        ) * float(np.clip(progress, 0.0, 1.0))
        if np.any(joint_lock < 0.0) or np.any(joint_lock > 1.0) or not np.all(np.isfinite(joint_lock)):
            raise ValueError("joint lock profile must be finite and lie in [0, 1]")

    return Morphology(
        n_links=n,
        lengths=lengths,
        masses=masses,
        damping=damping,
        frictionloss=frictionloss,
        joint_stiffness=joint_stiffness,
        joint_lock=joint_lock,
        total_length=total_length,
        total_mass=total_mass,
        total_damping=total_damping,
        total_frictionloss=total_frictionloss,
        alpha_length=float(params["alpha_length"]),
        alpha_mass=float(params["alpha_mass"]),
        alpha_damping=float(params["alpha_damping"]),
        alpha_frictionloss=float(params["alpha_frictionloss"]),
    )
