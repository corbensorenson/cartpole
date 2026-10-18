"""Fixed admission over cheap feedback and causal observed-state native repair."""
import numpy as np
from .env import SpatialCartpole
from .immediate_bank import ImmediateBankPolicy
from .immediate_lookahead import ImmediateLookaheadPolicy
from .immediate_online_repair import OnlineRepair


class ImmediateRecoveryBankPolicy(ImmediateBankPolicy):
    def __init__(self,env,profiles,folder):
        if env.config.n_links!=9:
            raise ValueError('registered nine-link recovery bank required')
        super().__init__(env,profiles)
        if not (np.array_equal(self.profiles[0]['tracking_R'],np.eye(2)*.0001) and
                np.array_equal(self.profiles[1]['tracking_R'],np.eye(2)*.001)):
            raise ValueError('fixed low-effort then original finite-value profile order required')
        self.folder=folder;self.folder.mkdir(exist_ok=False)
        self.online_instances=0
        self.metadata.update(policy='immediate forecast bank with observed-state native recovery',
                             profile_types=['fixed feedback','causal online repair'])

    def _make(self,index,env):
        if index==0:return ImmediateLookaheadPolicy(env,self.profiles[0])
        folder=self.folder/f'online_instance{self.online_instances}'
        self.online_instances+=1
        return OnlineRepair(env,self.profiles[1],folder)

    def _select(self,state):
        records=[];selected=None
        for index in range(2):
            predictor=SpatialCartpole(self.env.config);candidate=None
            try:
                predicted=np.array(state,copy=True);predictor.set_state(predicted)
                predictor.data.time=self.env.data.time
                for key in ('steps','streak','best_streak','min_plane_margin','warning_seen'):
                    setattr(predictor,key,getattr(self.env,key))
                candidate=self._make(index,predictor)
                states,actions,raws,tilts=[predicted],[],[],[]
                for t in range(self.env.config.horizon):
                    raw=candidate(predicted,t);action=predictor.deliver(raw)
                    predicted,_,terminated,truncated,info=predictor.step_force(action)
                    if hasattr(candidate,'observe_transition'):
                        candidate.observe_transition(predicted,t,action)
                    states.append(predicted);actions.append(action);raws.append(raw);tilts.append(predictor.tilts())
                    if terminated or truncated:break
                tapes=dict(states=states,actions=actions,raw_actions=raws,tilts=tilts,**candidate.forecast_tapes)
                for key,value in tapes.items():self._forecasts[f'bank_forecast_{index}_{key}']=np.asarray(value)
                admitted=bool(info['success'] and predictor.steps==self.env.config.horizon and
                    not info['physics_warning'] and info['plane_margin']>=0)
                policy_native=candidate.metadata['predictor_native_steps']
                records.append(dict(profile=index,**info,steps=predictor.steps,
                    actual_forecast_native_steps=predictor.native_steps,
                    policy_native_steps=policy_native,native_steps=predictor.native_steps+policy_native,
                    policy_metadata=dict(candidate.metadata),admitted=admitted))
                if admitted:selected=index;break
            finally:
                if candidate is not None:candidate.close()
                predictor.close()
        self.selected=0 if selected is None else selected
        self.policy=self._make(self.selected,self.env)
        self.metadata.update(launch_forecasts=records,selected_launch_profile=self.selected,
            launch_admitted=selected is not None,launch_forecast_start_step=0,
            forecast_native_steps=sum(r['native_steps'] for r in records))

    def __call__(self,state,step):
        raw=super().__call__(state,step)
        self.metadata.update(execution_policy_native_steps=self.policy.metadata['predictor_native_steps'],
                             selected_policy_metadata=dict(self.policy.metadata))
        return raw

    def observe_transition(self,state,step,action):
        if hasattr(self.policy,'observe_transition'):self.policy.observe_transition(state,step,action)
        super().observe_transition(state,step,action)
        self.metadata['selected_policy_metadata']=dict(self.policy.metadata)


def immediate_recovery_bank_from_payload(env,payload,folder):
    if not bool(payload.get('immediate_recovery_bank',False)) or int(payload['profile_count'])!=2:
        raise ValueError('explicit two-profile immediate recovery bank payload required')
    profiles=[{key.split('__',1)[1]:value for key,value in payload.items()
               if key.startswith(f'profile{i}__')} for i in range(2)]
    return ImmediateRecoveryBankPolicy(env,profiles,folder)
