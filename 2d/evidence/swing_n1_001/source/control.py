"""Native finite differences, discrete LQR, and manifold trajectory optimization.

All runtime forces pass through float32 delivery. Smooth optimization derivatives
deliberately exclude that quantizer; only actual delivered rollouts qualify.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_discrete_are
from .env import SpatialCartpole


def linearize(env: SpatialCartpole, state, force, eps=1e-5):
    nx, nu = 2 * env.model.nv, 2
    base = env.transition(state, force, quantize=False)
    A, B = np.empty((nx, nx)), np.empty((nx, nu))
    for j in range(nx):
        d = np.zeros(nx)
        d[j] = eps
        hi = env.transition(env.integrate(state, d), force, quantize=False)
        lo = env.transition(env.integrate(state, -d), force, quantize=False)
        A[:, j] = (env.difference(hi, base) - env.difference(lo, base)) / (2 * eps)
    for j in range(nu):
        d = np.zeros(nu)
        d[j] = eps
        hi = env.transition(state, force + d, quantize=False)
        lo = env.transition(state, force - d, quantize=False)
        B[:, j] = (env.difference(hi, base) - env.difference(lo, base)) / (2 * eps)
    return A, B


def state_cost(env, angle_weight=20., cart_weight=1., velocity_weight=0.15):
    nv = env.model.nv
    # Cost cumulative WORLD tilt in the upright linearization, not just each
    # relative joint angle. Passive axial spin is neither actuated nor penalized.
    C = np.zeros((2 * env.config.n_links, nv))
    for i in range(env.config.n_links):
        for j in range(i + 1):
            C[2 * i, 2 + 3 * j] = 1
            C[2 * i + 1, 3 + 3 * j] = 1
    Q = np.zeros((2 * nv, 2 * nv))
    Q[:nv, :nv] = angle_weight * C.T @ C
    Q[:2, :2] = cart_weight * np.eye(2)
    Q[nv:, nv:] = velocity_weight * (np.eye(nv) + C.T @ C)
    for j in range(env.config.n_links):
        Q[nv + 4 + 3 * j, nv + 4 + 3 * j] = 0
    return Q


def balance_design(env, r=0.01, angle_weight=20., cart_weight=1.):
    ref = env.upright_state()
    A, B = linearize(env, ref, np.zeros(2))
    Q = state_cost(env, angle_weight, cart_weight)
    R = r * np.eye(2)
    # Axial spin positions are unobservable cyclic coordinates at upright.
    # Exclude them (and their decoupled rates) from DARE rather than add fictitious
    # yaw motors or numerical state penalties to claim full-state stabilization.
    ids = [j for j in range(env.model.nv) if j < 2 or (j - 2) % 3 != 2]
    ids += [j + env.model.nv for j in ids.copy()]
    ar, br, qr = A[np.ix_(ids, ids)], B[ids], Q[np.ix_(ids, ids)]
    p = solve_discrete_are(ar, br, qr, R)
    k = np.linalg.solve(R + br.T @ p @ br, br.T @ p @ ar)
    K, P = np.zeros((2, len(A))), np.zeros_like(A)
    K[:, ids], P[np.ix_(ids, ids)] = k, p
    return {"reference": ref, "K": K, "P": P, "Q": Q, "R": R,
            "closed_loop_radius": float(np.max(np.abs(np.linalg.eigvals(ar - br @ k))))}


class BalancePolicy:
    def __init__(self, env, design):
        self.env, self.design = env, design

    def __call__(self, state, step=0):
        return -self.design["K"] @ self.env.difference(state, self.design["reference"])


class RoutePolicy:
    def __init__(self, env, states, actions, gains, balance):
        self.env, self.states, self.actions, self.gains = env, states, actions, gains
        self.balance = BalancePolicy(env, balance)

    def __call__(self, state, step):
        if step >= len(self.actions):
            return self.balance(state)
        return self.actions[step] - self.gains[step] @ self.env.difference(state, self.states[step])


def optimize_route(env, initial, actions, iterations=80, callback=None, terminal_scale=1.):
    """iLQR with native RK4 rollout, tangent-space feedback, and line search.

    Starts feasible. The selected route is later rebuilt/qualified with actual
    float32 delivery. Solver merit is not the task's success test.
    """
    design = balance_design(env)
    ref, nv = design["reference"], env.model.nv
    H, nx = len(actions), 2 * nv
    Q = state_cost(env, angle_weight=0.08, cart_weight=0.3, velocity_weight=0.015)
    R = np.eye(2) * 0.0006
    P = design["P"] * terminal_scale
    reg = 1e-3
    history = []

    def cost(x, u=None, final=False):
        e = env.difference(x, ref)
        val = 0.5 * e @ (P if final else Q) @ e
        if not final:
            val += 0.5 * u @ R @ u
            # Smooth penalty outside a conservative optimization boundary.
            val += 1000 * np.sum(np.maximum(np.abs(x[:2]) - 2.7, 0.) ** 2)
        return float(val)

    def rollout(us, xs=None, ks=None, Ks=None, alpha=1.):
        out, controls = [initial.copy()], []
        total = 0.
        for t in range(H):
            u = us[t].copy()
            if ks is not None:
                u += alpha * ks[t] - Ks[t] @ env.difference(out[-1], xs[t])
            u *= min(1., env.config.force_limit / max(np.linalg.norm(u), 1e-30))
            total += cost(out[-1], u)
            xn = env.transition(out[-1], u, quantize=False)
            if not np.isfinite(xn).all() or np.max(np.abs(xn)) > 1e5:
                return None, None, float("inf")
            controls.append(u)
            out.append(xn)
        total += cost(out[-1], final=True)
        return np.array(out), np.array(controls), total

    states, actions, merit = rollout(np.asarray(actions, float))
    final_gains = np.zeros((H, 2, nx))
    for iteration in range(iterations):
        models = [linearize(env, states[t], actions[t]) for t in range(H)]
        Vx = P @ env.difference(states[-1], ref)
        Vxx = P.copy()
        gains, feed = np.zeros((H, 2, nx)), np.zeros((H, 2))
        try:
            for t in range(H - 1, -1, -1):
                A, B = models[t]
                e = env.difference(states[t], ref)
                lx, lxx = Q @ e, Q.copy()
                for j in range(2):
                    over = max(abs(states[t, j]) - 2.7, 0.)
                    if over:
                        lx[j] += 2000 * over * np.sign(states[t, j])
                        lxx[j, j] += 2000
                qx, qu = lx + A.T @ Vx, R @ actions[t] + B.T @ Vx
                qxx, quu, qux = lxx + A.T @ Vxx @ A, R + B.T @ Vxx @ B, B.T @ Vxx @ A
                h = quu + reg * np.eye(2)
                np.linalg.cholesky(h)
                k, K = -np.linalg.solve(h, qu), np.linalg.solve(h, qux)
                # The feedback sign convention is u_new = u + k - K dx.
                Vx = qx - K.T @ qu + qux.T @ k - K.T @ quu @ k
                Vxx = qxx - K.T @ qux - qux.T @ K + K.T @ quu @ K
                Vxx = (Vxx + Vxx.T) * 0.5
                feed[t], gains[t] = k, K
        except np.linalg.LinAlgError:
            reg *= 10
            continue
        accepted = False
        for alpha in (1., 0.5, 0.25, 0.1, 0.05, 0.01):
            trial_x, trial_u, trial_merit = rollout(actions, states, feed, gains, alpha)
            if trial_merit < merit:
                improvement = merit - trial_merit
                states, actions, merit = trial_x, trial_u, trial_merit
                final_gains = gains
                reg = max(reg / 3, 1e-8)
                accepted = True
                break
        if not accepted:
            reg *= 10
            improvement = 0.
        row = {"iteration": iteration, "merit": merit, "reg": reg, "accepted": accepted,
               "terminal_error": float(np.linalg.norm(env.difference(states[-1], ref))),
               "native_steps": env.native_steps}
        history.append(row)
        if callback:
            callback(row)
        if reg > 1e9 or (accepted and improvement < 1e-7):
            break
    # Recompute feedback at the FINAL trajectory, not the preceding nominal.
    V = P.copy()
    for t in range(H - 1, -1, -1):
        A, B = linearize(env, states[t], actions[t])
        K = np.linalg.solve(R + B.T @ V @ B, B.T @ V @ A)
        final_gains[t] = K
        V = Q + K.T @ R @ K + (A - B @ K).T @ V @ (A - B @ K)
        V = (V + V.T) / 2
    return states, actions, final_gains, design, history
