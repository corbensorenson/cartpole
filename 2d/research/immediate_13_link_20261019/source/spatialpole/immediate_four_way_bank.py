"""Cheap complete feedback forecasts precede either observed-state repair."""
import numpy as np
from .env import SpatialCartpole
from .immediate_lookahead import ImmediateLookaheadPolicy
from .immediate_three_way_bank import ThreeWayImmediateBankPolicy


class FourWayImmediateBankPolicy(ThreeWayImmediateBankPolicy):
    def __init__(self,env,profiles,folder):
        super().__init__(env,profiles,folder)
        self.metadata.update(policy='immediate forecast bank with two fixed values and two repair timings',
            controller_profiles=4,value_design_profiles=2,
            profile_types=['lower-effort fixed feedback','original fixed feedback',
                           'late causal repair','earlier causal repair'])

    def _make(self,index,env):
        if index==1:
            return ImmediateLookaheadPolicy(env,self.profiles[1])
        if index not in (0,2,3):
            raise ValueError('exactly four registered strategies required')
        return super()._make(index-1 if index>1 else index,env)

    def _select(self,state):
        records=[];selected=None
        for index in range(4):
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


def immediate_four_way_bank_from_payload(env,payload,folder):
    if (not bool(payload.get('immediate_four_way_recovery_bank',False))
            or bool(payload.get('immediate_three_way_recovery_bank',False))
            or bool(payload.get('immediate_recovery_bank',False))
            or int(payload['profile_count'])!=2 or int(payload['strategy_count'])!=4):
        raise ValueError('explicit four-strategy, two-value-design recovery payload required')
    if (int(payload['earlier_repair_start']),int(payload['earlier_repair_stop']))!=(150,214):
        raise ValueError('registered earlier repair window required')
    profiles=[{key.split('__',1)[1]:value for key,value in payload.items()
               if key.startswith(f'profile{i}__')} for i in range(2)]
    return FourWayImmediateBankPolicy(env,profiles,folder)
