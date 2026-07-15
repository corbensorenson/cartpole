from __future__ import annotations

import numpy as np


def trajectory_conditioned_features(
    observations: np.ndarray,
    initial_observations: np.ndarray | None,
    steps: np.ndarray,
    *,
    maximum_steps: int,
    include_initial_observation: bool = True,
) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float32)
    steps = np.asarray(steps, dtype=np.float32)
    if observations.ndim != 2:
        raise ValueError("observations must be a matrix")
    if steps.shape != (len(observations),):
        raise ValueError("steps must contain one value per observation")
    if maximum_steps <= 0:
        raise ValueError("maximum_steps must be positive")
    phase = np.clip(steps / float(maximum_steps), 0.0, 1.0)[:, None]
    if not include_initial_observation:
        return np.concatenate((observations, phase), axis=1)
    initial_observations = np.asarray(initial_observations, dtype=np.float32)
    if initial_observations.shape != observations.shape:
        raise ValueError("initial observations must match observations")
    return np.concatenate((observations, initial_observations, phase), axis=1)
