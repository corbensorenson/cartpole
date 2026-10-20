"""Causal native mid-swing repair with a matched finite continuation value."""
import json
import numpy as np
from .env import SpatialCartpole
from .aligned_value import aligned_next_value
from .immediate_lookahead import ImmediateLookaheadPolicy
from .rollout_value import NativeRolloutValuePolicy
from .native_mpc import NativeReplanner,exact_bytes
from .experiment import write_json

def save_trace(path,trace,**metadata):
    np.savez_compressed(path,**{k:v for k,v in trace.items() if isinstance(v,np.ndarray)},metadata=json.dumps(metadata))

class TerminalReplanner(NativeReplanner):
    def __init__(self,*args,terminal_reference,terminal_root,terminal_residual,**kwargs):
        self.terminal_reference=terminal_reference
        self.terminal_root=terminal_root
        self.terminal_residual=terminal_residual
        super().__init__(*args,**kwargs)

    def finite_residual(self,state):
        ref,root,_=aligned_next_value(self.env,self.terminal_reference,self.terminal_root,state)
        return root@self.env.difference(state,ref)+self.terminal_residual

    def cost_factors(self,states,actions,start):
        cost,residuals,factors=super().cost_factors(states,actions,start)
        terminal=self.finite_residual(states[-1]);nx=2*self.env.model.nv;eps=1e-5
        J=np.empty((len(terminal),nx))
        for j in range(nx):
            delta=np.zeros(nx);delta[j]=eps
            plus=self.finite_residual(self.env.integrate(states[-1],delta))
            minus=self.finite_residual(self.env.integrate(states[-1],-delta))
            J[:,j]=(plus-minus)/(2*eps)
        self.geometry_queries+=1+2*nx
        residuals=np.pad(residuals,((0,0),(0,len(terminal))))
        factors=np.pad(factors,((0,0),(0,len(terminal)),(0,0)))
        residuals[-1,-len(terminal):]=terminal;factors[-1,-len(terminal):]=J
        return cost+.5*float(terminal@terminal),residuals,factors

class OnlineRepair:
    def __init__(self,env,payload,folder):
        self.env,self.payload,self.folder=env,payload,folder
        self.base=ImmediateLookaheadPolicy(env,payload)
        self.extra_native=0;self.repair=None;self.warm_tapes={}
        self.metadata=dict(self.base.metadata,policy='observed-state immediate native repair prototype',
            repair_start=170,repair_stop=234,repair_admitted=False,verified_repair_transitions=0,
            loads_recorded_solution=False,hardware_realtime_claim=False)

    @staticmethod
    def admitted(plan):
        return bool(plan['complete'] and np.max(plan['tilts'][-16:])<.15 and
            np.min(plan['plane_margins'])>=0 and not np.any(plan['physics_warnings']))

    def synthesize(self,state):
        self.folder.mkdir(exist_ok=False)
        before=self.env.state().copy();clock=self.env.data.time
        counters=(self.env.steps,self.env.native_steps,self.env.best_streak)
        original=self.base.rollout;assert original is not None
        warm_env=SpatialCartpole(self.env.config)
        warm_policy=NativeRolloutValuePolicy(warm_env,original.states,original.actions,original.feed,
            original.gains,original.roots,original.residuals,original.R,
            {k:self.payload[k] for k in ('reference','K','P','Q','R')},
            stage_factors=original.stage_factors,horizon=4)
        try:
            x=state.copy();xs=[x];us=[];raws=[]
            for t in range(170,234):
                raw=warm_policy(x,t);u=warm_env.deliver(raw);x=warm_env.transition(x,u)
                xs.append(x);us.append(u);raws.append(raw)
            self.warm_tapes=dict(states=np.asarray(xs),actions=np.asarray(us),raw_actions=np.asarray(raws),**warm_policy.forecast_tapes)
            self.extra_native=warm_env.native_steps+warm_policy.metadata['predictor_native_steps']
            save_trace(self.folder/'warm_owned_forecast.npz',self.warm_tapes)
        finally:warm_policy.close();warm_env.close()
        planner=NativeReplanner(self.env.config,original.states,original.actions,native_budget=1600000)
        terminal=None;phases=[]
        def early(plan):return self.admitted(plan) and plan.get('objective_cost',np.inf)<=1e-10
        try:
            warm=np.asarray(us)
            for phase in range(3):
                if phase==2:
                    terminal=TerminalReplanner(self.env.config,original.states,original.actions,
                        native_budget=1450000,terminal_reference=original.states[234],
                        terminal_root=original.roots[234],terminal_residual=original.residuals[234])
                engine=planner if phase<2 else terminal
                before_native=engine.env.native_steps
                def checkpoint(row):
                    print(json.dumps(dict(case=self.folder.name,phase=phase,iteration=row['iteration'],cost=row['cost'],accepted=row['accepted'])),flush=True)
                    with (self.folder/f'phase{phase}_optimizer.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    save_trace(self.folder/f'phase{phase}_latest.npz',engine.current_plan,cost=engine.current_cost)
                plan=engine.plan(state,170,warm,iterations=12,callback=checkpoint,admission=early)
                save_trace(self.folder/f'phase{phase}_initial.npz',engine.initial_plan,cost=engine.initial_cost)
                save_trace(self.folder/f'phase{phase}_final.npz',plan,cost=engine.final_cost)
                for j,trial in enumerate(engine.trials):
                    save_trace(self.folder/f'phase{phase}_trial{j:03d}.npz',trial,
                               **{k:v for k,v in trial.items() if not isinstance(v,np.ndarray)})
                if engine.last_models is not None:
                    np.savez_compressed(self.folder/f'phase{phase}_last_models.npz',
                        A=np.array([A for A,B in engine.last_models]),B=np.array([B for A,B in engine.last_models]),
                        states=engine.last_model_states,actions=engine.last_model_actions,
                        feed=engine.last_feed,gains=engine.last_gains)
                phases.append(dict(phase=phase,native_steps=engine.env.native_steps-before_native,
                    initial_cost=engine.initial_cost,final_cost=engine.final_cost,
                    updates=len(engine.history),admitted=self.admitted(plan)))
                warm=plan['actions']
            if self.admitted(plan):self.repair=plan
            self.extra_native+=planner.env.native_steps+terminal.env.native_steps
            assert self.extra_native<=3100000
            self.metadata.update(repair_admitted=self.repair is not None,repair_planning_phases=phases,
                                 repair_and_warm_native_steps=self.extra_native)
            write_json(self.folder/'summary.json',dict(admitted=self.repair is not None,phases=phases,
                warm_and_planning_native_steps=self.extra_native,observed_state=state.tolist()))
        finally:
            planner.close()
            if terminal is not None:terminal.close()
        exact_bytes(before,self.env.state());assert self.env.data.time==clock
        assert counters==(self.env.steps,self.env.native_steps,self.env.best_streak)

    def __call__(self,state,step):
        if step==170:self.synthesize(state)
        if self.repair is not None and 170<=step<234:
            exact_bytes(state,self.repair['states'][step-170]);raw=self.repair['actions'][step-170]
        else:raw=self.base(state,step)
        self.metadata['predictor_native_steps']=self.extra_native+self.base.metadata['predictor_native_steps']
        return raw

    def observe_transition(self,state,step,action):
        if self.repair is not None and 170<=step<234:
            exact_bytes(state,self.repair['states'][step-169]);exact_bytes(action,self.repair['actions'][step-170])
            self.metadata['verified_repair_transitions']+=1

    @property
    def forecast_tapes(self):
        tape=dict(self.base.forecast_tapes,**{'repair_warm_'+k:v for k,v in self.warm_tapes.items()})
        if self.repair is not None:
            tape.update(repair_states=self.repair['states'],repair_actions=self.repair['actions'])
        return tape

    def close(self):self.base.close()
