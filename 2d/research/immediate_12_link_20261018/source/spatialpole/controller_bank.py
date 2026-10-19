"""Fixed profile admission by complete owned native continuations, never seeds."""
import numpy as np
from .env import SpatialCartpole
from .affine_value import ParkedAffineValuePolicy


FIELDS=('states','actions','feed','gains','roots','residuals')


def exact_bytes(actual,expected):
    actual,expected=np.asarray(actual),np.asarray(expected)
    if actual.dtype!=expected.dtype or actual.shape!=expected.shape or actual.tobytes()!=expected.tobytes():
        raise RuntimeError('controller-bank forecast/execution byte mismatch')


class ControllerBankPolicy:
    """A separately identified policy; original plain/offset policies are unchanged."""
    def __init__(self,env,payload):
        if not bool(payload.get('controller_bank',False)):
            raise ValueError('explicit controller-bank marker required')
        if (any(bool(payload.get(k,False)) for k in ('native_search','online_parking','align_cart_origin',
                'world_coordinate_feedback','rollout_native_value','aligned_native_value')) or
            'launch_reference_profiles' in payload or np.any(payload.get('cart_reference_offset',np.zeros(2)))):
            raise ValueError('controller bank does not combine other search/admission/origin policies')
        if not bool(payload.get('online_gauge',False)) or not bool(payload.get('align_gauge',False)):
            raise ValueError('controller bank requires the declared online/launch frame law')
        self.env=env
        self.shared={k:np.array(payload[k],copy=True) for k in payload if not k.startswith('bank_')}
        self.tables={k:np.array(payload['bank_'+k],copy=True) for k in FIELDS}
        self.count=len(self.tables['states'])
        if self.count not in (3,5):
            raise ValueError('only explicitly registered three or five profile banks are supported')
        self.park_steps=int(payload['park_steps']);H=len(payload['actions'])
        shapes=dict(states=(H+1,env.model.nq+env.model.nv),actions=(H,2),feed=(H,2),
            gains=(H,2,2*env.model.nv),roots=payload['roots'].shape,residuals=payload['residuals'].shape)
        if not 0<=self.park_steps<env.config.horizon or env.config.horizon-self.park_steps>H:
            raise ValueError('controller bank requires a fully covered execution prefix')
        for key,table in self.tables.items():
            if table.shape!=(self.count,)+shapes[key] or not np.isfinite(table).all():
                raise ValueError('finite equal-shaped controller profiles required')
            exact_bytes(table[0],self.shared[key])
        for table in (*self.shared.values(),*self.tables.values()):
            table.setflags(write=False)
        self.policy=ParkedAffineValuePolicy(env,self.profile(0))
        self.metadata=dict(self.policy.metadata,policy='observed-launch native controller bank',
            controller_profiles=self.count,admission_order='fixed first complete pass',hardware_realtime_claim=False,
            predictor_native_steps=0,verified_selected_transitions=0)
        self.selected=None;self.forecast_tapes={};self.pending=None

    def profile(self,index):
        return dict(self.shared,**{key:value[index] for key,value in self.tables.items()})

    def __call__(self,state,step):
        if step<self.park_steps:
            return self.policy(state,step)
        observed_time=self.env.data.time
        if self.selected is None:
            if step!=self.park_steps:
                raise ValueError('bank admission must start at the declared launch command')
            records=[];selected=None
            for index in range(self.count):
                predictor=SpatialCartpole(self.env.config)
                candidate=None
                try:
                    predicted=np.array(state,copy=True);predictor.set_state(predicted)
                    predictor.data.time=observed_time
                    for key in ('steps','streak','best_streak','min_plane_margin','warning_seen'):
                        setattr(predictor,key,getattr(self.env,key))
                    candidate=ParkedAffineValuePolicy(predictor,self.profile(index))
                    states,actions,raws,tilts=[predicted],[],[],[]
                    for t in range(step,self.env.config.horizon):
                        raw=candidate(predicted,t);action=predictor.deliver(raw)
                        predicted,_,terminated,truncated,info=predictor.step_force(action)
                        states.append(predicted);actions.append(action);raws.append(raw);tilts.append(predictor.tilts())
                        if terminated or truncated:
                            break
                    for key,value in dict(states=states,actions=actions,raw_actions=raws,tilts=tilts).items():
                        self.forecast_tapes[f'bank_forecast_{index}_{key}']=np.asarray(value)
                    admitted=bool(info['success'] and predictor.steps==self.env.config.horizon and
                        not info['physics_warning'] and info['plane_margin']>=0)
                    records.append(dict(profile=index,**info,steps=predictor.steps,
                        native_steps=predictor.native_steps,admitted=admitted))
                    if admitted:
                        selected=index
                        break
                finally:
                    if candidate is not None:
                        candidate.close()
                    predictor.close()
            self.selected=0 if selected is None else selected
            self.policy.close();self.policy=ParkedAffineValuePolicy(self.env,self.profile(self.selected))
            self.metadata.update(launch_forecasts=records,selected_launch_profile=self.selected,
                launch_admitted=selected is not None,launch_forecast_start_step=step,
                predictor_native_steps=sum(r['native_steps'] for r in records))
        if self.pending is not None:
            raise RuntimeError('controller-bank transition observer was not called')
        i=step-self.park_steps;prefix=f'bank_forecast_{self.selected}_'
        exact_bytes(state,self.forecast_tapes[prefix+'states'][i])
        try:
            raw=self.policy(state,step)
        finally:
            self.env.data.time=observed_time
        action=self.env.deliver(raw)
        exact_bytes(raw,self.forecast_tapes[prefix+'raw_actions'][i])
        exact_bytes(action,self.forecast_tapes[prefix+'actions'][i])
        self.pending=(step,action.copy(),self.forecast_tapes[prefix+'states'][i+1],
            self.forecast_tapes[prefix+'tilts'][i])
        return raw

    def observe_transition(self,state,step,action):
        if step<self.park_steps:
            return
        if self.pending is None or self.pending[0]!=step:
            raise RuntimeError('controller-bank transition clock mismatch')
        _,expected_action,expected_state,expected_tilts=self.pending
        exact_bytes(action,expected_action);exact_bytes(state,expected_state)
        exact_bytes(self.env.tilts(),expected_tilts)
        self.pending=None
        self.metadata['verified_selected_transitions']+=1
        self.metadata['final_selected_transition_verified']=step==self.env.config.horizon-1

    def close(self):
        self.policy.close()
