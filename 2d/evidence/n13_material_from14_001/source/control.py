"""Native finite differences, discrete LQR, and manifold trajectory optimization.

All runtime forces pass through float32 delivery. Smooth optimization derivatives
deliberately exclude that quantizer; only actual delivered rollouts qualify.
"""
from __future__ import annotations

import numpy as np
import mujoco.rollout
import mujoco as mj
from scipy.linalg import solve_discrete_are
from .env import SpatialCartpole
from .physical import world_errors, world_cost_derivatives,direction_residuals,direction_cost_derivatives


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
        if force[j]+eps > env.config.force_limit:
            B[:,j] = -env.difference(lo,base)/eps
        elif force[j]-eps < -env.config.force_limit:
            B[:,j] = env.difference(hi,base)/eps
        else:
            B[:, j] = (env.difference(hi, base) - env.difference(lo, base)) / (2 * eps)
    return A, B


def linearize_many(env, states, actions, eps=1e-5, return_next=False,order=2):
    """Same centered stencil in a native C rollout batch (RK4, four substeps)."""
    if order==4:
        return linearize_many_fourth(env,states,actions,eps,return_next)
    if order!=2:
        raise ValueError('derivative order must be two or four')
    nx, count = 2*env.model.nv, len(actions)
    width = 1+2*nx+4
    initial, control = [], []
    for state, force in zip(states, actions):
        initial.append(state)
        control.append(force)
        for j in range(nx):
            d = np.zeros(nx)
            d[j] = eps
            initial.extend([env.integrate(state,d), env.integrate(state,-d)])
            control.extend([force, force])
        for j in range(2):
            d = np.zeros(2)
            d[j] = eps
            initial.extend([state,state])
            control.extend([force+d,force-d])
    initial = np.c_[np.zeros(len(initial)), np.array(initial)]
    controls = np.repeat(np.asarray(control)[:,None,:], env.config.frame_skip, axis=1)
    workers = getattr(env,'derivative_workers',1)
    if not hasattr(env,'_fd_rollout'):
        env._fd_rollout = mujoco.rollout.Rollout(nthread=workers if workers>1 else 0)
        env._fd_data = [mj.MjData(env.model) for _ in range(workers)]
    full, _ = env._fd_rollout.rollout(env.model, env._fd_data, initial, controls)
    # A stalled native rollout would repeat timestamps. Do not call it valid.
    times = np.diff(np.c_[np.zeros(len(initial)),full[:,:,0]],axis=1)
    env.native_steps += int(np.count_nonzero(times > 0))
    if not np.all(times > 0) or not np.isfinite(full).all():
        raise RuntimeError("invalid native derivative batch")
    results = full[:,-1,1:].reshape(count,width,-1)
    models = []
    for row,force in zip(results,actions):
        columns = [(env.difference(row[1+2*j],row[0])-env.difference(row[2+2*j],row[0]))/(2*eps) for j in range(nx+2)]
        matrix = np.array(columns).T
        for j in range(2):
            column = nx+j
            if force[j]+eps > env.config.force_limit:
                matrix[:,column] = -env.difference(row[2+2*column],row[0])/eps
            elif force[j]-eps < -env.config.force_limit:
                matrix[:,column] = env.difference(row[1+2*column],row[0])/eps
        models.append((matrix[:,:nx],matrix[:,nx:]))
    return (models,results[:,0].copy()) if return_next else models


def linearize_many_fourth(env,states,actions,eps=1e-3,return_next=False):
    """Fourth-order native stencils with inward control probes at component caps.

    Like the original batch, these identify the smooth, unquantized native
    force map. Force constraints are imposed separately on optimization
    rollouts and every actual command. Identification probes near a circular
    boundary may lie infinitesimally outside its feasible set.
    """
    if not np.isfinite(eps) or eps<=0:
        raise ValueError('derivative epsilon must be finite and positive')
    nx,count = 2*env.model.nv,len(actions)
    width = 1+4*(nx+2)
    initial,control,weights = [],[],[]
    centered = np.array([8.,-8.,-1.,1.])/(12*eps)
    for state,force in zip(states,actions):
        initial.append(state)
        control.append(force)
        for j in range(nx):
            for scale in (1.,-1.,2.,-2.):
                d = np.zeros(nx)
                d[j] = scale*eps
                initial.append(env.integrate(state,d))
                control.append(force)
        row_weights = []
        for j in range(2):
            if force[j]+2*eps>env.config.force_limit:
                offsets,coefficients = (-1.,-2.,-3.,-4.),-np.array([48.,-36.,16.,-3.])/(12*eps)
            elif force[j]-2*eps < -env.config.force_limit:
                offsets,coefficients = (1.,2.,3.,4.),np.array([48.,-36.,16.,-3.])/(12*eps)
            else:
                offsets,coefficients = (1.,-1.,2.,-2.),centered
            for scale in offsets:
                d = np.zeros(2)
                d[j] = scale*eps
                initial.append(state)
                control.append(force+d)
            row_weights.append(coefficients)
        weights.append(row_weights)
    initial = np.c_[np.zeros(len(initial)),np.array(initial)]
    controls = np.repeat(np.asarray(control)[:,None,:],env.config.frame_skip,axis=1)
    workers = getattr(env,'derivative_workers',1)
    if not hasattr(env,'_fd_rollout'):
        env._fd_rollout = mujoco.rollout.Rollout(nthread=workers if workers>1 else 0)
        env._fd_data = [mj.MjData(env.model) for _ in range(workers)]
    full,_ = env._fd_rollout.rollout(env.model,env._fd_data,initial,controls)
    times = np.diff(np.c_[np.zeros(len(initial)),full[:,:,0]],axis=1)
    env.native_steps += int(np.count_nonzero(times>0))
    if not np.all(times>0) or not np.isfinite(full).all():
        raise RuntimeError('invalid fourth-order native derivative batch')
    results = full[:,-1,1:].reshape(count,width,-1)
    models = []
    for row,control_weights in zip(results,weights):
        columns = []
        for j in range(nx+2):
            values = np.array([env.difference(row[1+4*j+k],row[0]) for k in range(4)])
            coefficients = centered if j<nx else control_weights[j-nx]
            columns.append(coefficients@values)
        matrix = np.array(columns).T
        models.append((matrix[:,:nx],matrix[:,nx:]))
    return (models,results[:,0].copy()) if return_next else models


def bounded_action_step(u,gradient,hessian,cross,limit,return_multiplier=False):
    """Exact two-dimensional SPD quadratic step inside a circular force bound.

    The scalar KKT multiplier is solved in the Hessian eigenbasis. The feedback
    derivative obeys the active circle's tangent constraint.
    """
    k = -np.linalg.solve(hessian,gradient)
    if np.linalg.norm(u+k) <= limit:
        result = k,np.linalg.solve(hessian,cross),False
        return (*result,0.) if return_multiplier else result
    values,vectors = np.linalg.eigh(hessian)
    if values[0] <= 0:
        raise np.linalg.LinAlgError('action Hessian is not positive definite')
    rhs = hessian@u-gradient
    coeff = vectors.T@rhs
    low,high = 0.,max(np.linalg.norm(rhs)/limit-values[0],1e-12)
    for _ in range(45):
        middle = (low+high)/2
        if np.linalg.norm(coeff/(values+middle)) > limit:
            low = middle
        else:
            high = middle
    v = vectors@(coeff/(values+high))
    H = hessian+high*np.eye(2)
    K = np.linalg.solve(H,cross)
    normal = v/limit
    col = np.linalg.solve(H,normal)
    K -= np.outer(col,normal@K)/(normal@col)
    result = v-u,K,True
    return (*result,high) if return_multiplier else result


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


def cost_derivatives(env, state, reference, weight):
    """Gradient and Gauss-Newton Hessian of quaternion-log quadratic cost.

    Tangent increments are LOCAL rotations; the derivative of a finite rotation
    log is not identity. Its right-Jacobian inverse matters during spatial swing.
    """
    error = env.difference(state,reference)
    jac = difference_jacobian(env,state,reference)
    return jac.T@weight@error, jac.T@weight@jac


def difference_jacobian(env,state,reference):
    """Derivative of reference-relative coordinates under a local increment."""
    error = env.difference(state,reference)
    jac = np.eye(2*env.model.nv)
    for i in range(env.config.n_links):
        j = 2+3*i
        v = error[j:j+3]
        angle = np.linalg.norm(v)
        x,y,z = v
        hat = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        coef = 1/12+angle**2/720 if angle < 1e-4 else (1-0.5*angle/np.tan(0.5*angle))/angle**2
        jac[j:j+3,j:j+3] = np.eye(3)+0.5*hat+coef*(hat@hat)
    return jac


def balance_design(env, r=0.01, angle_weight=20., cart_weight=1., reference=None,axis_separated=False,
                   value_backend='dare',finite_steps=1500):
    if value_backend not in ('dare','finite_qr'):
        raise ValueError('unknown upright value backend')
    if value_backend=='finite_qr' and (not axis_separated or reference is not None or finite_steps<1):
        raise ValueError('finite QR value requires upright axis separation and a positive horizon')
    ref = env.upright_state() if reference is None else np.asarray(reference).copy()
    A, B = linearize(env, ref, np.zeros(2))
    Q = state_cost(env, angle_weight, cart_weight)
    R = r * np.eye(2)
    # Axial spin positions are unobservable cyclic coordinates at upright.
    # Exclude them (and their decoupled rates) from DARE rather than add fictitious
    # yaw motors or numerical state penalties to claim full-state stabilization.
    ids = [j for j in range(env.model.nv) if j < 2 or (j - 2) % 3 != 2]
    ids += [j + env.model.nv for j in ids.copy()]
    ar, br, qr = A[np.ix_(ids, ids)], B[ids], Q[np.ix_(ids, ids)]
    K, P = np.zeros((2, len(A))), np.zeros_like(A)
    if axis_separated:
        if reference is not None:
            raise ValueError('axis-separated design currently requires the upright reference')
        # Upright rotational symmetry gives two identical independent planar
        # linear systems. Solving their degenerate combined pencil at high
        # counts can introduce large numerical cross-axis terms. Average the
        # native X/Y identifications, solve one SISO DARE and rotate its result.
        n,nv = env.config.n_links,env.model.nv
        ix = np.array([0]+[3+3*j for j in range(n)])
        iy = np.array([1]+[2+3*j for j in range(n)])
        ix,iy = np.r_[ix,ix+nv],np.r_[iy,iy+nv]
        signs = np.tile(np.r_[1.,-np.ones(n)],2)
        ax = (A[np.ix_(ix,ix)]+signs[:,None]*A[np.ix_(iy,iy)]*signs[None,:])/2
        bx = (B[ix,0:1]+signs[:,None]*B[iy,1:2])/2
        qx = Q[np.ix_(ix,ix)]
        if value_backend=='finite_qr':
            from .affine_value import affine_tracking_values
            _,gains,roots,_ = affine_tracking_values([(ax,bx)]*finite_steps,
                np.zeros((finite_steps,len(ax))),[np.linalg.cholesky(qx).T]*finite_steps,
                R[:1,:1],np.zeros_like(qx))
            kx,sx = gains[0,0],roots[0]
            px = sx.T@sx
            # Preserve this factor: forming a highly conditioned P in F64 can
            # introduce negative rounded eigenvalues despite a positive cost.
            value_factor = np.zeros_like(A)
            value_factor[np.ix_(ix,ix)] = sx
            value_factor[np.ix_(iy,iy)] = sx*signs[None,:]
        else:
            px = solve_discrete_are(ax,bx,qx,R[:1,:1])
            kx = np.linalg.solve(R[:1,:1]+bx.T@px@bx,bx.T@px@ax)[0]
        K[0,ix],K[1,iy] = kx,signs*kx
        P[np.ix_(ix,ix)] = px
        P[np.ix_(iy,iy)] = signs[:,None]*px*signs[None,:]
        k = K[:,ids]
    else:
        p = solve_discrete_are(ar, br, qr, R)
        k = np.linalg.solve(R + br.T @ p @ br, br.T @ p @ ar)
        K[:, ids], P[np.ix_(ids, ids)] = k, p
    result = {"reference": ref, "K": K, "P": P, "Q": Q, "R": R,
            "closed_loop_radius": float(np.max(np.abs(np.linalg.eigvals(ar - br @ k)))),
            "axis_separated":bool(axis_separated)}
    if value_backend=='finite_qr':
        result.update(value_factor=value_factor,value_backend=value_backend,finite_value_steps=finite_steps)
    return result


class BalancePolicy:
    def __init__(self, env, design):
        self.env, self.design = env, design

    def __call__(self, state, step=0):
        return -self.design["K"] @ self.env.difference(state, self.design["reference"])


class WorldBalancePolicy(BalancePolicy):
    """Yaw-invariant physical tilt/rate coordinates for the same upright gain.

    This changes the nonlinear feedback law, not the plant or LQR design. It
    removes a coordinate artifact when passive ball-joint twist accumulates.
    """
    def error(self, state):
        e, nv = self.env, self.env.model.nv
        e.set_state(state)
        error = np.zeros(2*nv)
        error[:2], error[nv:nv+2] = state[:2], state[e.model.nq:e.model.nq+2]
        last_tilt, last_rate = np.zeros(2), np.zeros(2)
        for i, body in enumerate(e.link_ids):
            direction = e.data.xmat[body].reshape(3,3)[:,2]
            vel = np.zeros(6)
            mj.mj_objectVelocity(e.model,e.data,mj.mjtObj.mjOBJ_BODY,int(body),vel,0)
            dd = np.cross(vel[:3],direction)
            x,y,z = direction
            radius = np.hypot(x,y)
            if radius < 1e-7:
                # At the south pole the shortest tilt axis is ambiguous, but
                # its magnitude is pi, never zero. Choose a fixed finite axis.
                tilt = np.array([-y,x]) if z >= 0 else np.array([np.pi,0.])
                rate = np.array([-dd[1],dd[0]])
            else:
                angle = np.arctan2(radius,z)
                rd = (x*dd[0]+y*dd[1])/radius
                ad = z*rd-radius*dd[2]
                v = np.array([-y,x])
                tilt = v*angle/radius
                rate = np.array([-dd[1],dd[0]])*angle/radius + v*(ad/radius-angle*rd/radius**2)
            j = 2+3*i
            error[j:j+2], error[nv+j:nv+j+2] = tilt-last_tilt, rate-last_rate
            last_tilt, last_rate = tilt, rate
        return error

    def __call__(self, state, step=0):
        return -self.design["K"] @ self.error(state)


class RoutePolicy:
    def __init__(self, env, states, actions, gains, balance, world_balance=False):
        self.env, self.states, self.actions, self.gains = env, states, actions, gains
        self.balance = (WorldBalancePolicy if world_balance else BalancePolicy)(env, balance)

    def __call__(self, state, step):
        if step >= len(self.actions):
            return self.balance(state)
        return self.actions[step] - self.gains[step] @ self.env.difference(state, self.states[step])


def optimize_route(env, initial, actions, iterations=80, callback=None, terminal_scale=1., world_terminal=False, checkpoint=None, hold_start=None,bounded_backward=False,direction_cost=False):
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
    world_policy = WorldBalancePolicy(env,design)
    hold_q = state_cost(env,angle_weight=20.,cart_weight=1.,velocity_weight=0.3)
    direction_q = np.diag(np.r_[np.full(nv,0.08),np.full(nv,0.015)])
    direction_q[:2,:2] = np.eye(2)*0.3
    direction_hold_q = np.diag(np.r_[np.full(nv,20.),np.full(nv,0.3)])
    direction_hold_q[:2,:2] = np.eye(2)

    def direction_derivatives(xs):
        split = hold_start if hold_start is not None else len(xs)
        g,h = direction_cost_derivatives(env,xs[:split],direction_q)
        if split < len(xs):
            tail_g,tail_h = direction_cost_derivatives(env,xs[split:],direction_hold_q)
            g,h = np.concatenate([g,tail_g]),np.concatenate([h,tail_h])
        return g,h

    def terminal_derivatives(x):
        if not world_terminal:
            return cost_derivatives(env,x,ref,P)
        grads,hess = world_cost_derivatives(env,np.array([x]),P)
        return grads[0],hess[0]

    def cost(x, u=None, final=False, t=0):
        hold = hold_start is not None and t >= hold_start
        if direction_cost and not final:
            e = direction_residuals(env,np.array([x]))[0]
            weight = direction_hold_q if hold else direction_q
        else:
            physical = (final and world_terminal) or (not final and hold)
            e = world_errors(env,np.array([x]))[0] if physical else env.difference(x, ref)
            weight = P if final else hold_q if hold else Q
        val = 0.5 * e @ weight @ e
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
            total += cost(out[-1], u,t=t)
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
        models = linearize_many(env, states[:-1], actions)
        Vx, Vxx = terminal_derivatives(states[-1])
        gains, feed = np.zeros((H, 2, nx)), np.zeros((H, 2))
        bounded_steps = 0
        hold_derivatives = world_cost_derivatives(env,states[hold_start:-1],hold_q) if hold_start is not None else None
        direction_stage = direction_derivatives(states[:-1]) if direction_cost else None
        try:
            for t in range(H - 1, -1, -1):
                A, B = models[t]
                if direction_cost:
                    lx,lxx = direction_stage[0][t].copy(),direction_stage[1][t].copy()
                elif hold_start is not None and t >= hold_start:
                    lx,lxx = hold_derivatives[0][t-hold_start].copy(),hold_derivatives[1][t-hold_start].copy()
                else:
                    lx, lxx = cost_derivatives(env,states[t],ref,Q)
                for j in range(2):
                    over = max(abs(states[t, j]) - 2.7, 0.)
                    if over:
                        lx[j] += 2000 * over * np.sign(states[t, j])
                        lxx[j, j] += 2000
                qx, qu = lx + A.T @ Vx, R @ actions[t] + B.T @ Vx
                qxx, quu, qux = lxx + A.T @ Vxx @ A, R + B.T @ Vxx @ B, B.T @ Vxx @ A
                h = quu + reg * np.eye(2)
                np.linalg.cholesky(h)
                if bounded_backward:
                    k,K,active,multiplier = bounded_action_step(actions[t],qu,h,qux,env.config.force_limit,return_multiplier=True)
                    bounded_steps += active
                else:
                    k, K = -np.linalg.solve(h, qu), np.linalg.solve(h, qux)
                    multiplier = 0.
                # The feedback sign convention is u_new = u + k - K dx.
                Vx = qx - K.T @ qu + qux.T @ k - K.T @ quu @ k
                Vxx = qxx - K.T @ qux - qux.T @ K + K.T @ quu @ K + multiplier*(K.T@K)
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
               "native_steps": env.native_steps,"bounded_steps":bounded_steps}
        history.append(row)
        if callback:
            callback(row)
        if checkpoint is not None and iteration % 10 == 0:
            checkpoint(iteration,states,actions)
        if reg > 1e9 or (accepted and improvement < 1e-7):
            break
    # Recompute feedback at the FINAL trajectory, not the preceding nominal.
    _, V = terminal_derivatives(states[-1])
    models = linearize_many(env, states[:-1], actions)
    hold_derivatives = world_cost_derivatives(env,states[hold_start:-1],hold_q) if hold_start is not None else None
    direction_stage = direction_derivatives(states[:-1]) if direction_cost else None
    for t in range(H - 1, -1, -1):
        A, B = models[t]
        K = np.linalg.solve(R + B.T @ V @ B, B.T @ V @ A)
        final_gains[t] = K
        local_q = direction_stage[1][t] if direction_cost else hold_derivatives[1][t-hold_start] if hold_start is not None and t >= hold_start else cost_derivatives(env,states[t],ref,Q)[1]
        V = local_q + K.T @ R @ K + (A - B @ K).T @ V @ (A - B @ K)
        V = (V + V.T) / 2
    return states, actions, final_gains, design, history
