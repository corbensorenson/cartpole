"""Bounded native trajectory replanning; plans are proposals, never plant resets."""
import numpy as np
from .env import SpatialCartpole
from .world_coordinates import native_geometry,world_coordinate_jacobian
from .control import linearize_many
from .fddp import least_squares_backward
from .physical import world_errors


class NativePlanningBudgetError(RuntimeError):
    pass


def linear_control_rollout(env,models,actions,feed,gains,alpha):
    """Freeze controls using only the linear delta model, with actual delivery.

    This does not integrate or set any environment state. Every tangent row,
    including every velocity, participates in the propagated delta state.
    """
    actions,feed,gains=[np.asarray(x) for x in (actions,feed,gains)]
    H,nx=len(actions),2*env.model.nv
    if (len(models)!=H or actions.shape!=(H,2) or feed.shape!=(H,2) or gains.shape!=(H,2,nx)
            or not np.isfinite(alpha) or not all(np.isfinite(x).all() for x in (actions,feed,gains))):
        raise ValueError('finite full-native linear control-increment arrays required')
    delta=np.zeros(nx);errors=[delta];raws=[];forces=[]
    for t,(A,B) in enumerate(models):
        if A.shape!=(nx,nx) or B.shape!=(nx,2):
            raise ValueError('linear model must retain all native tangent coordinates')
        raw=actions[t].astype(float)+alpha*feed[t]-gains[t]@delta
        force=env.deliver(raw)
        delta=A@delta+B@(force.astype(float)-actions[t].astype(float))
        errors.append(delta);raws.append(raw);forces.append(force)
    return dict(linear_errors=np.array(errors),raw_actions=np.array(raws),actions=np.array(forces))


def physical_residual_factor(env,state,reference,reference_rotation,reference_omega):
    """Physical tracking residual and analytic full-native tangent derivative.

    Pass an owned geometry environment. Direction chords avoid pole-log cuts;
    all body angular velocities, including spin, contribute to the cost.
    """
    n,nv=env.config.n_links,env.model.nv
    R,w=native_geometry(env,state)
    J=world_coordinate_jacobian(env,state)
    residual=np.zeros(6*n+4);factor=np.zeros((6*n+4,2*nv))
    residual[:2]=2*(state[:2]-reference[:2]);factor[:2,:2]=2*np.eye(2)
    start=2+3*n
    residual[start:start+2]=np.sqrt(.05)*(state[env.model.nq:env.model.nq+2]-reference[env.model.nq:env.model.nq+2])
    factor[start:start+2,nv:nv+2]=np.sqrt(.05)*np.eye(2)
    for i in range(n):
        row=2+3*i;tiltrow=2+2*i;angularrow=4+2*n+3*i
        residual[row:row+3]=np.sqrt(40)*(R[i,:,2]-reference_rotation[i,:,2])
        factor[row:row+3]=np.sqrt(40)*np.c_[-R[i,:,1],R[i,:,0]]@J[tiltrow:tiltrow+2]
        row=4+3*n+3*i;weight=np.sqrt([.05,.05,.005])
        residual[row:row+3]=weight*(reference_rotation[i].T@(w[i]-reference_omega[i]))
        factor[row:row+3]=weight[:,None]*(reference_rotation[i].T@R[i]@J[angularrow:angularrow+3])
    return residual,factor


def upright_value_residual_factor(env,state,root,eps=1e-5):
    """Factored destination value in the existing world upright feedback chart.

    Kinematic tangent secants only: no integration, live state or velocity reset.
    This chart's axial null directions supplement, never replace, the complete
    physical angular-velocity residuals used by NativeReplanner.
    """
    nx=2*env.model.nv
    root=np.asarray(root)
    if root.ndim!=2 or root.shape[1]!=nx or not np.isfinite(root).all() or eps<=0:
        raise ValueError('finite full-native upright value factor required')
    queries=[]
    for j in range(nx):
        delta=np.zeros(nx);delta[j]=eps
        queries.extend((env.integrate(state,delta),env.integrate(state,-delta)))
    errors=world_errors(env,np.array(queries))
    J=((errors[::2]-errors[1::2])/(2*eps)).T
    return root@world_errors(env,state)[0],root@J


class NativeReplanner:
    """One fixed-window native planning operation, with charged owned predictors."""
    def __init__(self,config,states,actions,horizon=64,native_budget=1450000,forward_mode='feedback',terminal_value_root=None):
        if horizon!=64 or len(states)!=len(actions)+1 or len(actions)<horizon:
            raise ValueError('registered replanner requires a covered64-command horizon')
        if forward_mode not in ('feedback','linear_controls'):
            raise ValueError('explicit feedback or linear-control forward mode required')
        self.forward_mode=forward_mode
        self.env=SpatialCartpole(config);self.env.derivative_workers=4
        self.terminal_value_root=None if terminal_value_root is None else np.array(terminal_value_root,copy=True)
        if self.terminal_value_root is not None and (self.terminal_value_root.ndim!=2 or
                self.terminal_value_root.shape[1]!=2*self.env.model.nv or not np.isfinite(self.terminal_value_root).all()):
            self.env.close();raise ValueError('full-native finite terminal factor required')
        self.states=np.array(states,copy=True);self.actions=np.array(actions,copy=True)
        self.horizon=horizon;self.native_budget=native_budget;self.R=np.eye(2)*.001
        geometry=[native_geometry(self.env,x) for x in self.states]
        self.rotations=np.array([r for r,w in geometry]);self.omega=np.array([w for r,w in geometry])
        self.geometry_queries=len(geometry);self.history=[];self.trials=[]

    def cost_factors(self,states,actions,start):
        residuals=[];factors=[];cost=0.
        for j,state in enumerate(states):
            t=start+j
            r,L=physical_residual_factor(self.env,state,self.states[t],self.rotations[t],self.omega[t])
            self.geometry_queries+=1
            over=np.maximum(abs(state[:2])-2.7,0.)
            barrier=np.sqrt(2000)*over
            derivative=np.zeros((2,2*self.env.model.nv))
            derivative[:,:2]=np.diag(np.sqrt(2000)*np.sign(state[:2])*(over>0))
            r=np.r_[r,barrier];L=np.vstack([L,derivative])
            if self.terminal_value_root is not None:
                terminal_r=np.zeros(len(self.terminal_value_root))
                terminal_L=np.zeros((len(terminal_r),2*self.env.model.nv))
                if j==len(actions):
                    terminal_r,terminal_L=upright_value_residual_factor(self.env,state,self.terminal_value_root)
                r=np.r_[r,terminal_r];L=np.vstack([L,terminal_L])
            residuals.append(r);factors.append(L);cost+=.5*float(r@r)
            if j<len(actions):
                du=actions[j].astype(float)-self.actions[t]
                cost+=.5*float(du@self.R@du)
        return cost,np.array(residuals),np.array(factors)

    def rollout(self,initial,actions,feed=None,gains=None,centers=None,alpha=1.):
        states=[np.array(initial,copy=True)];delivered=[];raws=[];tilts=[];margins=[];warnings=[]
        failure=None
        for j,action in enumerate(actions):
            if self.env.native_steps+4>self.native_budget:
                raise NativePlanningBudgetError('native replanning budget exhausted')
            raw=action.astype(float)
            if feed is not None:
                raw=raw+alpha*feed[j]-gains[j]@self.env.difference(states[-1],centers[j])
            try:
                u=self.env.deliver(raw)
            except (ValueError,FloatingPointError) as exc:
                failure=str(exc);break
            self.env.min_plane_margin,self.env.warning_seen=self.env.config.plane_limit,False
            state=self.env.transition(states[-1],u)
            states.append(state);delivered.append(u);raws.append(raw)
            tilts.append(self.env.tilts());margins.append(self.env.min_plane_margin);warnings.append(self.env.warning_seen)
            if not np.isfinite(state).all() or self.env.warning_seen:
                failure='invalid native planning trajectory';break
        return dict(states=np.array(states),actions=np.array(delivered),raw_actions=np.array(raws),
                    tilts=np.array(tilts),plane_margins=np.array(margins),physics_warnings=np.array(warnings),
                    complete=failure is None and len(delivered)==len(actions),failure_reason=failure)

    def plan(self,initial,start,warm_actions,iterations=12,callback=None,admission=None):
        if len(warm_actions)!=self.horizon or start+self.horizon>len(self.actions):
            raise ValueError('complete fixed-window warm actions required')
        best=self.rollout(initial,np.asarray(warm_actions))
        if not best['complete']:raise FloatingPointError('initial native warm forecast is incomplete')
        cost,residuals,factors=self.cost_factors(best['states'],best['actions'],start)
        best['objective_cost']=cost
        self.initial_plan={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in best.items()};self.initial_cost=cost
        self.history=[];self.trials=[]
        trust=.001;self.last_models=None;self.last_gains=None;self.last_feed=None
        if admission is not None and admission(best):
            self.final_plan=best;self.final_cost=cost
            return best
        for iteration in range(iterations):
            estimate=(1+4*(2*self.env.model.nv+2))*self.horizon*self.env.config.frame_skip
            if self.env.native_steps+estimate+6*self.horizon*4>self.native_budget:
                raise NativePlanningBudgetError('insufficient budget for registered native planning update')
            self.last_model_states=best['states'].copy();self.last_model_actions=best['actions'].copy()
            models=linearize_many(self.env,best['states'][:-1],best['actions'],eps=.001,order=4)
            feed,gains,active=least_squares_backward(models,np.zeros((self.horizon,2*self.env.model.nv)),
                best['actions'].astype(float),factors[:-1],residuals[:-1],factors[-1],residuals[-1],
                self.R,self.env.config.force_limit,control_trust=trust,action_reference=self.actions[start:start+self.horizon])
            accepted=False;trials=[]
            for alpha in (1.,.5,.25,.1,.05,.01):
                if self.forward_mode=='feedback':
                    trial=self.rollout(initial,best['actions'],feed,gains,best['states'],alpha)
                else:
                    fixed=linear_control_rollout(self.env,models,best['actions'],feed,gains,alpha)
                    trial=self.rollout(initial,fixed['raw_actions'])
                    if trial['complete']:
                        exact_bytes(trial['actions'],fixed['actions'])
                    trial['linear_errors']=fixed['linear_errors']
                value,r,L=self.cost_factors(trial['states'],trial['actions'],start) if trial['complete'] else (None,None,None)
                trial['objective_cost']=value
                record=dict(iteration=iteration,alpha=alpha,cost=value,
                    maximum_world_tilt=float(np.max(trial['tilts'])) if len(trial['tilts']) else None,
                    minimum_plane_margin=float(np.min(trial['plane_margins'])) if len(trial['plane_margins']) else None,
                    native_steps=self.env.native_steps)
                trials.append(record);self.trials.append(dict(record,**trial))
                if value is not None and value<cost-1e-12*max(1.,cost):
                    best,cost,residuals,factors=trial,value,r,L;accepted=True
                    if alpha>=.5:trust=max(trust/2,1e-6)
                    break
            if not accepted:trust=min(trust*10,1e3)
            self.last_models=models;self.last_gains=gains;self.last_feed=feed
            row=dict(iteration=iteration,accepted=accepted,cost=cost,control_trust=trust,
                active_force_constraints=active,maximum_gain=float(np.max(abs(gains))),trials=trials,forward_mode=self.forward_mode,
                native_steps=self.env.native_steps)
            self.history.append(row)
            self.current_plan=best;self.current_cost=cost
            if callback:callback(row)
            if admission is not None and admission(best):break
        self.final_plan=best;self.final_cost=cost
        return best

    def close(self):
        self.env.close()


class PlanningAdmissionError(RuntimeError):
    pass


def exact_bytes(actual,expected):
    if actual.dtype!=expected.dtype or actual.shape!=expected.shape or actual.tobytes()!=expected.tobytes():
        raise RuntimeError('native MPC forecast/execution byte mismatch')


class NativeMPCPolicy:
    """Observe, extend a complete native plan, then execute eight verified forces."""
    def __init__(self,env,states,actions,initial_guess,native_budget=30000000,window_callback=None,strict_convergence=False,maximum_updates=None):
        if np.asarray(initial_guess).shape!=(64,2):
            raise ValueError('complete64-force initial planning guess required')
        self.env=env;self.initial_guess=np.asarray(initial_guess).copy()
        self.planner=NativeReplanner(env.config,states,actions,native_budget=native_budget)
        self.window_callback=window_callback;self.plan=None;self.start=None;self.pending=None
        self.strict_convergence=bool(strict_convergence)
        updates=(12 if self.strict_convergence else 4) if maximum_updates is None else maximum_updates
        if updates not in (4,12):
            self.planner.close();raise ValueError('registered MPC update caps are four or twelve')
        self.windows=[]
        self.metadata=dict(policy='native64-command MPC with eight-command admission',horizon=64,
            replan_every=8,maximum_updates_per_window=updates,
            admission_tracking_cost_limit=1e-6 if self.strict_convergence else None,predictor_native_steps=0,
            verified_selected_transitions=0,hardware_realtime_claim=False)

    @staticmethod
    def admitted(plan):
        return bool(plan['complete'] and np.max(plan['tilts'])<.14 and
            np.min(plan['plane_margins'])>.1 and not np.any(plan['physics_warnings']) and
            np.max(np.linalg.norm(plan['actions'].astype(float),axis=1))<=80)

    def admission(self,plan):
        return self.admitted(plan) and (not self.strict_convergence or plan.get('objective_cost',np.inf)<=1e-6)

    def __call__(self,state,step):
        if self.pending is not None:
            raise RuntimeError('native MPC transition observer was not called')
        if step%8==0:
            if self.plan is None:
                if step!=0:raise ValueError('MPC maintenance must start at commandzero')
                warm=self.initial_guess
            else:
                if step!=self.start+8:raise ValueError('MPC requires every selectedcommand')
                warm=np.concatenate([self.plan['actions'][8:],self.planner.actions[step+56:step+64]])
            before=state.copy();clock=self.env.data.time;count=self.env.native_steps
            plan=self.planner.plan(state,step,warm,iterations=self.metadata['maximum_updates_per_window'],admission=self.admission)
            exact_bytes(self.env.state(),before)
            if self.env.data.time!=clock or self.env.native_steps!=count:
                raise RuntimeError('MPC predictor changed the live clock or counters')
            self.metadata['predictor_native_steps']=self.planner.env.native_steps
            admitted=self.admission(plan)
            record=dict(start=step,admitted=admitted,updates=len(self.planner.history),
                initial_cost=self.planner.initial_cost,final_cost=self.planner.final_cost,
                maximum_world_tilt=float(np.max(plan['tilts'])),minimum_plane_margin=float(np.min(plan['plane_margins'])),
                predictor_native_steps=self.planner.env.native_steps)
            self.windows.append(record)
            if self.window_callback:self.window_callback(record,plan,self.planner)
            if not admitted:raise PlanningAdmissionError('no complete admitted64-command native continuation')
            self.plan,self.start=plan,step
        index=step-self.start
        exact_bytes(state,self.plan['states'][index])
        action=self.plan['actions'][index].copy()
        self.pending=(step,action,self.plan['states'][index+1],self.plan['tilts'][index])
        return action

    def observe_transition(self,state,step,action):
        if self.pending is None or self.pending[0]!=step:
            raise RuntimeError('native MPC observer clock mismatch')
        _,expected_action,expected_state,expected_tilts=self.pending
        exact_bytes(action,expected_action);exact_bytes(state,expected_state)
        exact_bytes(self.env.tilts(),expected_tilts)
        self.pending=None;self.metadata['verified_selected_transitions']+=1

    def close(self):
        self.planner.close()
