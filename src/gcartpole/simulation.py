"""Reject invalid MuJoCo integration instead of accepting automatic resets."""
from __future__ import annotations

import mujoco
import math
import numpy as np


class SimulationError(ValueError):
    pass


def advance_checked(model: mujoco.MjModel, data: mujoco.MjData, steps: int) -> None:
    warnings_before = tuple(int(item.number) for item in data.warning)
    timestep = float(model.opt.timestep)
    for index in range(steps):
        previous_time = float(data.time)
        mujoco.mj_step(model, data)
        warnings_after = tuple(int(item.number) for item in data.warning)
        if warnings_after != warnings_before:
            raise SimulationError(f"MuJoCo warning during physics step {index + 1}")
        current_time = float(data.time)
        if not math.isfinite(current_time) or abs(current_time - previous_time - timestep) > 1e-10:
            raise SimulationError(f"MuJoCo time discontinuity during physics step {index + 1}")
        if not all(np.isfinite(values).all() for values in (data.qpos, data.qvel, data.qacc)):
            raise SimulationError(f"non-finite MuJoCo state during physics step {index + 1}")
