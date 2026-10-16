"""Repair a foreign motion by tracking its geometry with the new native plant."""
import numpy as np
from .control import balance_design, linearize_many, cost_derivatives
from .fddp import affine_backward, reference_gaps, gap_transport, gap_rollout


def repair_tracking(env, initial, target, target_actions, iterations=40,
                    callback=None, checkpoint=None, physics_budget=15000000):
    target, target_actions = np.array(target, float), np.array(target_actions, float)
    H, nv = len(target_actions), env.model.nv
    nx = 2*nv
    if target.shape != (H+1, env.model.nq+nv) or H > env.config.horizon:
        raise ValueError('invalid target trajectory')
    states, actions = target.copy(), target_actions.copy()
    states[0] = initial
    design = balance_design(env)
    Q = np.diag(np.r_[[4.,4.], np.full(nv-2,40.), np.full(nv,.05)])
    for j in range(env.config.n_links):
        Q[4+3*j,4+3*j] = .1
        Q[nv+4+3*j,nv+4+3*j] = .005
    P, R = 10*design['P']+Q, .001*np.eye(2)
    scales = np.r_[[3.,3.],np.full(nv-2,.15),[4.,4.],np.full(nv-2,15.)]
    rho, trust = 1000., 1.
    history, gains = [], np.zeros((H,2,nx))

    def objective(xs, us):
        errors = np.array([env.difference(x,y) for x,y in zip(xs,target)])
        du = us-target_actions
        return float(.5*np.einsum('bi,ij,bj->',errors[:-1],Q,errors[:-1])
                     +.5*np.einsum('bi,ij,bj->',du,R,du)
                     +.5*errors[-1]@P@errors[-1]
                     +1000*np.sum(np.maximum(np.abs(xs[:,:2])-2.7,0.)**2))

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
        try:
            feed,K,active = affine_backward(models,offsets,actions,gradients,hessians,tg,th,R,0.,
                env.config.force_limit,trust*Q,trust*.001,action_reference=target_actions)
        except np.linalg.LinAlgError:
            trust *= 10
            active = 0
        else:
            for alpha in (1.,.5,.25,.1,.05,.01,.005):
                if env.native_steps+2*H*env.config.frame_skip > physics_budget:
                    break
                xs,us,fx = gap_rollout(env,initial,states,actions,gaps,feed,K,alpha)
                if xs is None:
                    continue
                candidate_cost = objective(xs,us)
                candidate_defect,candidate_max = defect(xs,fx)
                if candidate_cost+rho*candidate_defect < merit-1e-10:
                    states,actions,gains = xs,us,K
                    cost,value,maximum = candidate_cost,candidate_defect,candidate_max
                    accepted,alpha_used = True,alpha
                    if alpha>=.5:
                        trust = max(trust/3,1e-8)
                    elif alpha<.1:
                        trust *= 3
                    break
            if not accepted:
                trust *= 10
        row = dict(iteration=iteration,cost=cost,defect_value=value,max_scaled_defect=maximum,
                   penalty=rho,trust=trust,alpha=alpha_used,accepted=accepted,
                   bounded_steps=int(active),native_steps=env.native_steps,
                   reference_max_cart=float(np.max(np.abs(states[:,:2]))))
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
