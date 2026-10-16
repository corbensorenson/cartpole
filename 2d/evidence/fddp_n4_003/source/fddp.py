"""Defect-aware manifold iLQR with FDDP-style gap contraction.

Inspired by the feasibility-driven forward pass described by Mastalli et al.
(Crocoddyl, arXiv:1909.04947). This implementation uses native finite differences,
a circular action QP, Gauss-Newton costs and an explicit merit penalty. It is
not a binding to, or an exact reproduction of, Crocoddyl's solver.

Intermediate shooting states may be inconsistent. They never alter the actual
simulator. Every returned execution candidate is rebuilt in a gap-free rollout.
"""
import numpy as np
from .control import balance_design,linearize_many,bounded_action_step
from .physical import direction_residuals,direction_cost_derivatives,world_errors,world_cost_derivatives


def affine_backward(models,offsets,actions,gradients,hessians,terminal_gradient,terminal_hessian,R,regularization,limit):
    H,nx = len(actions),len(terminal_gradient)
    feed,gains = np.zeros_like(actions),np.zeros((H,actions.shape[1],nx))
    Vx,Vxx = terminal_gradient.copy(),terminal_hessian.copy()
    active_count = 0
    for t in range(H-1,-1,-1):
        A,B = models[t]
        shifted = Vx+Vxx@offsets[t]
        qx,qu = gradients[t]+A.T@shifted,R@actions[t]+B.T@shifted
        qxx,quu,qux = hessians[t]+A.T@Vxx@A,R+B.T@Vxx@B,B.T@Vxx@A
        h = quu+regularization*np.eye(actions.shape[1])
        np.linalg.cholesky(h)
        k,K,active = bounded_action_step(actions[t],qu,h,qux,limit)
        active_count += active
        Vx = qx-K.T@qu+qux.T@k-K.T@quu@k
        Vxx = qxx-K.T@qux-qux.T@K+K.T@quu@K
        Vxx = (Vxx+Vxx.T)/2
        feed[t],gains[t] = k,K
    return feed,gains,active_count


def reference_gaps(env,states,predicted):
    """Right-tangent correction taking each native successor to its reference."""
    return np.array([env.difference(x,y) for x,y in zip(states[1:],predicted)])


def gap_transport(env,offset):
    """Eliminate BOTH manifold defect Jacobians, not just the successor one.

    For c=log(q_ref^-1 q_F), the implicit defect equation gives
    delta_next = c + Ad(exp(c)) delta_F. This is also the derivative of
    F_perturbed * exp(-c), the gap-preserving forward map at alpha=0.
    Raw joint velocities and cart coordinates use additive differences.
    """
    transport = np.eye(2*env.model.nv)
    for i in range(env.config.n_links):
        j = 2+3*i
        v = offset[j:j+3]
        angle = np.linalg.norm(v)
        x,y,z = v
        hat = np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
        s = 1-angle**2/6 if angle<1e-5 else np.sin(angle)/angle
        c = .5-angle**2/24 if angle<1e-5 else (1-np.cos(angle))/angle**2
        transport[j:j+3,j:j+3] = np.eye(3)+s*hat+c*hat@hat
    return transport


def gap_rollout(env,initial,states,actions,gaps,feed,gains,alpha):
    output,controls,predicted = [initial.copy()],[],[]
    for t in range(len(actions)):
        u = actions[t]+alpha*feed[t]-gains[t]@env.difference(output[-1],states[t])
        u *= min(1.,env.config.force_limit/max(np.linalg.norm(u),1e-30))
        xn = env.transition(output[-1],u,quantize=False)
        if not np.isfinite(xn).all() or np.max(np.abs(xn)) > 1e5:
            return None,None,None
        predicted.append(xn)
        # Even a zero mj_integratePos increment can renormalize quaternions by
        # a few ulps. A full step must retain the actual native state bytes.
        output.append(xn if alpha==1. else env.integrate(xn,(1-alpha)*gaps[t]))
        controls.append(u)
    return np.array(output),np.array(controls),np.array(predicted)


def repair_reference(env,initial,states,actions,iterations=100,terminal_scale=10.,penalty=1000.,callback=None,checkpoint=None,physics_budget=25000000):
    states,actions = np.array(states,float).copy(),np.array(actions,float).copy()
    H,nv,nx = len(actions),env.model.nv,2*env.model.nv
    if states.shape != (H+1,env.model.nq+env.model.nv):
        raise ValueError('reference states have the wrong spatial dimension')
    states[0] = initial
    actions *= np.minimum(1.,env.config.force_limit/np.maximum(np.linalg.norm(actions,axis=1),1e-30))[:,None]
    design = balance_design(env)
    P,R = terminal_scale*design['P'],np.eye(2)*0.0006
    Q = np.diag(np.r_[np.full(nv,0.08),np.full(nv,0.015)])
    Q[:2,:2] = np.eye(2)*0.3
    scales = np.r_[[3.,3.],np.full(nv-2,0.15),[4.,4.],np.full(nv-2,15.)]
    regularization,rho = 1e-3,float(penalty)
    history = []
    last_gains = np.zeros((H,2,nx))

    def objective(xs,us):
        residual = direction_residuals(env,xs[:-1])
        total = 0.5*np.einsum('bi,ij,bj->',residual,Q,residual)+0.5*np.einsum('bi,ij,bj->',us,R,us)
        total += 1000*np.sum(np.maximum(np.abs(xs[:-1,:2])-2.7,0.)**2)
        end = world_errors(env,xs[-1:])[0]
        return float(total+0.5*end@P@end)

    def defect_value(xs,pred):
        error = np.array([env.difference(f,x) for f,x in zip(pred,xs[1:])])/scales
        return 0.5*float(np.sum(error*error)),float(np.max(np.abs(error)))

    for iteration in range(iterations):
        if env.native_steps+(2+4*nv+4)*H*env.config.frame_skip > physics_budget:
            break
        models,predicted = linearize_many(env,states[:-1],actions,return_next=True)
        gaps = reference_gaps(env,states,predicted)
        offsets = np.array([env.difference(f,x) for f,x in zip(predicted,states[1:])])
        transported = []
        for t,(A,B) in enumerate(models):
            jac = gap_transport(env,offsets[t])
            transported.append((jac@A,jac@B))
        gradients,hessians = direction_cost_derivatives(env,states[:-1],Q)
        for j in range(2):
            over = np.maximum(np.abs(states[:-1,j])-2.7,0.)
            gradients[:,j] += 2000*over*np.sign(states[:-1,j])
            hessians[:,j,j] += 2000*(over>0)
        tg,th = world_cost_derivatives(env,states[-1:],P)
        raw_cost = objective(states,actions)
        defect,maximum = defect_value(states,predicted)
        merit = raw_cost+rho*defect
        accepted = False
        alpha_used = 0.
        try:
            feed,gains,active = affine_backward(transported,offsets,actions,gradients,hessians,tg[0],th[0],R,regularization,env.config.force_limit)
        except np.linalg.LinAlgError:
            regularization *= 10
            active = 0
            feed = np.zeros_like(actions)
            gains = last_gains
        else:
            for alpha in (1.,0.5,0.25,0.1,0.05,0.01,0.005):
                if env.native_steps+2*H*env.config.frame_skip > physics_budget:
                    break
                trial_x,trial_u,trial_f = gap_rollout(env,initial,states,actions,gaps,feed,gains,alpha)
                if trial_x is None:
                    continue
                trial_cost = objective(trial_x,trial_u)
                trial_defect,trial_max = defect_value(trial_x,trial_f)
                trial_merit = trial_cost+rho*trial_defect
                if trial_merit < merit-1e-10:
                    states,actions,predicted = trial_x,trial_u,trial_f
                    raw_cost,defect,maximum,merit = trial_cost,trial_defect,trial_max,trial_merit
                    last_gains = gains
                    regularization = max(regularization/3,1e-8)
                    accepted,alpha_used = True,alpha
                    break
            if not accepted:
                regularization *= 10
        row = {'iteration':iteration,'cost':raw_cost,'merit':merit,'penalty':rho,
               'defect_value':defect,'max_scaled_defect':maximum,'alpha':alpha_used,
               'accepted':accepted,'regularization':regularization,'bounded_steps':int(active),
               'native_steps':env.native_steps,'reference_max_cart':float(np.max(np.abs(states[:,:2])))}
        history.append(row)
        if callback:
            callback(row)
        if checkpoint and iteration%10==0:
            checkpoint(iteration,states,actions,last_gains,reference_gaps(env,states,predicted))
        if (iteration+1)%10==0 and maximum>1e-8:
            rho = min(rho*10,1e12)
            regularization = min(regularization,1e-2)
        if regularization > 1e10:
            if maximum>1e-8 and rho<1e12:
                rho *= 10
                regularization = 1e-2
            else:
                break
    # Preserve the last shooting proposal and close every gap in a SEPARATE
    # native feedback rollout. The latter is the only returned execution route.
    proposal = {'states':states.copy(),'actions':actions.copy(),'gains':last_gains.copy()}
    if checkpoint:
        checkpoint(len(history),states,actions,last_gains,reference_gaps(env,states,predicted))
    no_gap = np.zeros((H,nx))
    rebuilt,rebuilt_actions,_ = gap_rollout(env,initial,states,actions,no_gap,np.zeros_like(actions),last_gains,1.)
    if rebuilt is None:
        raise RuntimeError('gap-free execution rebuild diverged; proposal is not an execution controller')
    # Keep the useful feedback centered on its proposal; record a second actual
    # reference with the SAME gains and native-applied SMOOTH forces. The later
    # actual episode must still pass float32 delivery and independent replay.
    return rebuilt,rebuilt_actions,last_gains,design,history,proposal
