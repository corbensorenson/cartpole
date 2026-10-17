"""Repair a foreign motion by tracking its geometry with the new native plant."""
import numpy as np
from .control import balance_design, linearize_many, cost_derivatives,difference_jacobian
from .fddp import affine_backward,least_squares_backward,reference_gaps,gap_transport,gap_rollout


def positive_value_factor(matrix):
    """Factor a rounded Riccati value, projecting only numerical negative modes.

    A very ill-conditioned stored binary64 value can acquire small negative
    eigenvalues. Reject material indefiniteness, report the projection, and
    evaluate the new optional optimizer's terminal cost through this factor.
    This never changes the native plant or the saved stationary feedback gain.
    """
    values,vectors = np.linalg.eigh((matrix+matrix.T)/2)
    if values[0] < -1e-12*max(float(values[-1]),1.):
        raise ValueError('terminal value is materially indefinite')
    return np.sqrt(np.maximum(values,0.))[:,None]*vectors.T,float(values[0])


def repair_tracking(env, initial, target, target_actions, iterations=40,
                    callback=None, checkpoint=None, physics_budget=15000000, penalty=1000.,
                    sqrt_backward=False,axis_design=False,minimum_alpha=.005,
                    small_step_trust_factor=3.):
    target, target_actions = np.array(target, float), np.array(target_actions, float)
    H, nv = len(target_actions), env.model.nv
    nx = 2*nv
    if target.shape != (H+1, env.model.nq+nv) or H > env.config.horizon:
        raise ValueError('invalid target trajectory')
    if not np.isfinite(penalty) or penalty<=0:
        raise ValueError('penalty must be finite and positive')
    if not np.isfinite(minimum_alpha) or not 0<minimum_alpha<=.005:
        raise ValueError('minimum alpha must be positive and at most .005')
    if not np.isfinite(small_step_trust_factor) or small_step_trust_factor<1:
        raise ValueError('small-step trust factor must be finite and at least one')
    alphas = [1.,.5,.25,.1,.05,.01,.005]
    while alphas[-1]>minimum_alpha:
        alphas.append(max(alphas[-1]/2,minimum_alpha))
    states, actions = target.copy(), target_actions.copy()
    states[0] = initial
    design = balance_design(env,axis_separated=axis_design)
    Q = np.diag(np.r_[[4.,4.], np.full(nv-2,40.), np.full(nv,.05)])
    for j in range(env.config.n_links):
        Q[4+3*j,4+3*j] = .1
        Q[nv+4+3*j,nv+4+3*j] = .005
    P, R = 10*design['P']+Q, .001*np.eye(2)
    Qroot = np.diag(np.sqrt(np.diag(Q)))
    if sqrt_backward:
        value_root,projected_eigenvalue = positive_value_factor(10*design['P'])
        terminal_root = np.vstack((value_root,Qroot))
    scales = np.r_[[3.,3.],np.full(nv-2,.15),[4.,4.],np.full(nv-2,15.)]
    rho, trust = float(penalty), 1.
    history, gains = [], np.zeros((H,2,nx))

    def objective(xs, us):
        errors = np.array([env.difference(x,y) for x,y in zip(xs,target)])
        du = us-target_actions
        terminal = .5*np.sum((terminal_root@errors[-1])**2) if sqrt_backward else .5*errors[-1]@P@errors[-1]
        return float(.5*np.einsum('bi,ij,bj->',errors[:-1],Q,errors[:-1])
                     +.5*np.einsum('bi,ij,bj->',du,R,du)
                     +terminal
                     +1000*np.sum(np.maximum(np.abs(xs[:,:2])-2.7,0.)**2))

    def residual_model(state,reference,root):
        error = env.difference(state,reference)
        factor = root@difference_jacobian(env,state,reference)
        residual = root@error
        # Exact residual/Jacobian for the same soft cart-boundary penalty.
        over = np.maximum(np.abs(state[:2])-2.7,0.)
        boundary = np.zeros((2,nx))
        boundary[0,0],boundary[1,1] = np.sqrt(2000.)*(over>0)*np.sign(state[:2])
        return np.vstack((factor,boundary)),np.r_[residual,np.sqrt(2000.)*over]

    def defect(xs, predicted):
        errors = np.array([env.difference(f,x) for f,x in zip(predicted,xs[1:])])/scales
        return .5*float(np.sum(errors**2)), float(np.max(np.abs(errors)))

    for iteration in range(iterations):
        if env.native_steps+(2+4*nv+4)*H*env.config.frame_skip > physics_budget:
            break
        models,predicted = linearize_many(env,states[:-1],actions,return_next=True)
        offsets = np.array([env.difference(f,x) for f,x in zip(predicted,states[1:])])
        gaps = reference_gaps(env,states,predicted)
        models = [(gap_transport(env,c)@A,gap_transport(env,c)@B)
                  for (A,B),c in zip(models,offsets)]
        if sqrt_backward:
            factors,residuals = zip(*(residual_model(x,y,Qroot) for x,y in zip(states[:-1],target[:-1])))
            terminal_factor,terminal_residual = residual_model(states[-1],target[-1],terminal_root)
        else:
            derivatives = [cost_derivatives(env,x,y,Q) for x,y in zip(states[:-1],target[:-1])]
            gradients,hessians = map(np.array,zip(*derivatives))
            for j in range(2):
                over = np.maximum(np.abs(states[:-1,j])-2.7,0.)
                gradients[:,j] += 2000*over*np.sign(states[:-1,j])
                hessians[:,j,j] += 2000*(over>0)
            tg,th = cost_derivatives(env,states[-1],target[-1],P)
            over = np.maximum(np.abs(states[-1,:2])-2.7,0.)
            tg[:2] += 2000*over*np.sign(states[-1,:2])
            th[:2,:2] += np.diag(2000*(over>0))
        value, maximum = defect(states,predicted)
        cost = objective(states,actions)
        merit = cost+rho*value
        accepted,alpha_used = False,0.
        trials = []
        try:
            if sqrt_backward:
                feed,K,active = least_squares_backward(models,offsets,actions,factors,residuals,
                    terminal_factor,terminal_residual,R,env.config.force_limit,
                    np.sqrt(trust)*Qroot,trust*.001,action_reference=target_actions)
            else:
                feed,K,active = affine_backward(models,offsets,actions,gradients,hessians,tg,th,R,0.,
                    env.config.force_limit,trust*Q,trust*.001,action_reference=target_actions)
        except np.linalg.LinAlgError:
            trust *= 10
            active = 0
        else:
            for alpha in alphas:
                if env.native_steps+2*H*env.config.frame_skip > physics_budget:
                    break
                xs,us,fx = gap_rollout(env,initial,states,actions,gaps,feed,K,alpha)
                if xs is None:
                    continue
                candidate_cost = objective(xs,us)
                candidate_defect,candidate_max = defect(xs,fx)
                trials.append(dict(alpha=alpha,cost=candidate_cost,defect_value=candidate_defect,
                    max_scaled_defect=candidate_max,max_cart=float(np.max(np.abs(xs[:,:2])))))
                if candidate_cost+rho*candidate_defect < merit-1e-10:
                    states,actions,gains = xs,us,K
                    cost,value,maximum = candidate_cost,candidate_defect,candidate_max
                    accepted,alpha_used = True,alpha
                    if alpha>=.5:
                        trust = max(trust/3,1e-8)
                    elif alpha<.1:
                        trust *= small_step_trust_factor
                    break
            if not accepted:
                trust *= 10
        row = dict(iteration=iteration,cost=cost,defect_value=value,max_scaled_defect=maximum,
                   penalty=rho,trust=trust,alpha=alpha_used,accepted=accepted,
                   bounded_steps=int(active),native_steps=env.native_steps,
                   reference_max_cart=float(np.max(np.abs(states[:,:2]))),trials=trials)
        if sqrt_backward:
            row.update(backward='residual QR',terminal_projected_min_eigenvalue=projected_eigenvalue)
        history.append(row)
        if callback:
            callback(row)
        if checkpoint and iteration%10==0:
            checkpoint(iteration,states,actions,gains)
        if (iteration+1)%10==0 and maximum>1e-8:
            rho = min(rho*10,1e12)
        if trust>1e10:
            break
    proposal = dict(states=states.copy(),actions=actions.copy(),gains=gains.copy())
    rebuilt,controls,_ = gap_rollout(env,initial,states,actions,np.zeros((H,nx)),
                                     np.zeros_like(actions),gains,1.)
    if rebuilt is None:
        raise RuntimeError('gap-free tracking rebuild diverged')
    return rebuilt,controls,gains,design,history,proposal
