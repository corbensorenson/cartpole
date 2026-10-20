"""Infer actuator bias from measured states and requested forces, never force sensing."""
import numpy as np
from .env import SpatialCartpole


class NativeForceObserver:
    def __init__(self,env,base_policy):
        self.base=base_policy;self.predictor=SpatialCartpole(env.config)
        self.bias=np.zeros(2);self.previous=None;self.queries=[];self.samples=[]
        self.metadata=dict(policy='native observed force-bias compensation',park_steps=0,park_seconds=0.,
            predictor_native_steps=0,inverse_updates=0,unidentifiable_updates=0,
            observed_delivered_force=False,hardware_realtime_claim=False)

    def __call__(self,state,step):
        p=self.predictor;nq=p.model.nq
        if self.previous is not None:
            previous,raw,previous_step=self.previous
            if step!=previous_step+1:raise ValueError('successive observed states required')
            def predict(bias):
                force=p.deliver(raw+bias);successor=p.transition(previous,force)
                self.queries.append((previous_step,bias.copy(),force.copy(),successor.copy()))
                return successor[nq:nq+2]
            predicted=predict(self.bias)
            for _ in range(3):
                residual=state[nq:nq+2]-predicted
                if np.linalg.norm(residual,np.inf)<=1e-10:break
                J=np.empty((2,2));h=.01
                for axis in range(2):
                    d=np.zeros(2);d[axis]=h
                    J[:,axis]=(predict(self.bias+d)-predict(self.bias-d))/(2*h)
                if np.linalg.svd(J,compute_uv=False)[-1]<1e-8:
                    self.metadata['unidentifiable_updates']+=1;break
                self.bias+=np.linalg.solve(J,residual)
                predicted=predict(self.bias);self.metadata['inverse_updates']+=1
        baseline=np.asarray(self.base(state,step),float);command=baseline-self.bias
        self.previous=(np.array(state,copy=True),command.copy(),step)
        self.samples.append((self.bias.copy(),baseline.copy(),command.copy()))
        self.metadata['predictor_native_steps']=p.native_steps
        return command

    @property
    def forecast_tapes(self):
        return dict(observer_bias_estimate=np.array([x[0] for x in self.samples]),
            observer_baseline_actions=np.array([x[1] for x in self.samples]),
            observer_commands=np.array([x[2] for x in self.samples]),
            observer_query_previous_step=np.array([x[0] for x in self.queries],dtype=int),
            observer_query_bias=np.array([x[1] for x in self.queries]),
            observer_query_force=np.array([x[2] for x in self.queries]),
            observer_query_successor=np.array([x[3] for x in self.queries]))

    def close(self):
        self.predictor.close()
        if hasattr(self.base,'close'):self.base.close()
