"""Higher-order secants of the checked MuJoCo transition map."""
from __future__ import annotations

import numpy as np

from .ilqr import MujocoTransition


class FourthOrderMujocoTransition(MujocoTransition):
    def linearize(self, state, action, *, state_epsilon, action_epsilon):
        if min(state_epsilon, action_epsilon) <= 0:
            raise ValueError("derivative increments must be positive")
        state = np.asarray(state, dtype=np.float64)
        a = np.empty((self.nx, self.nx))
        for column in range(self.nx):
            offset = np.zeros(self.nx)
            offset[column] = state_epsilon
            first = self.difference(self(state + offset, action), self(state - offset, action))
            second = self.difference(self(state + 2 * offset, action), self(state - 2 * offset, action))
            # Subtract paired values before scaling; do not combine large
            # unwrapped angle offsets directly with the stencil weights.
            a[:, column] = (8 * first - second) / (12 * state_epsilon)
        center_action = float(np.float32(np.clip(action, -1., 1.)))
        if center_action + 2 * action_epsilon > 1:
            shifts = np.arange(-4, 1)
        elif center_action - 2 * action_epsilon < -1:
            shifts = np.arange(5)
        else:
            shifts = np.arange(-2, 3)
        knots = np.asarray([float(np.float32(np.clip(center_action + k * action_epsilon, -1., 1.)))
                            for k in shifts])
        if np.unique(knots).size != 5:
            raise ValueError("fourth-order action knots coincide at applied action precision")
        offsets = (knots - center_action) / action_epsilon
        weights = np.linalg.solve(np.vstack([offsets ** k for k in range(5)]),
                                  np.array([0., 1., 0., 0., 0.])) / action_epsilon
        center = self(state, center_action)
        differences = np.asarray([self.difference(self(state, float(k)), center) for k in knots])
        b = (weights @ differences)[:, None]
        return a, b
