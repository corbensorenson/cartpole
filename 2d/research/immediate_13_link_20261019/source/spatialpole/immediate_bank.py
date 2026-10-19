"""Select a fixed immediate controller by owned, complete native forecasts."""
import numpy as np
from .env import SpatialCartpole
from .controller_bank import exact_bytes
from .immediate_lookahead import ImmediateLookaheadPolicy
from .immediate_swing import ImmediateFiniteValuePolicy


def _profile_policy(env, payload):
    factory = ImmediateLookaheadPolicy if bool(payload.get('windowed_lookahead', False)) else ImmediateFiniteValuePolicy
    return factory(env, payload)


class ImmediateBankPolicy:
    """No seed input or live resets; selected feedback is executed again.

    Forecast time is computational work, not physical hanging preparation.
    The simulator is deterministic, and the final actual transition is checked
    against the chosen forecast just like every preceding transition.
    """
    def __init__(self, env, profiles):
        if len(profiles) != 2:
            raise ValueError('exactly two fixed immediate profiles required')
        self.env = env
        self.profiles = tuple({key: np.array(value, copy=True) for key, value in p.items()} for p in profiles)
        for p in self.profiles:
            for value in p.values():
                value.setflags(write=False)
        for p in profiles:
            if int(p['park_steps']) != 0 or len(p['actions']) != env.config.horizon:
                raise ValueError('complete immediate profiles required')
        self.policy = None
        self.selected = None
        self.pending = None
        self._forecasts = {}
        self.metadata = dict(policy='immediate native forecast bank', park_steps=0,
            park_seconds=0., controller_profiles=2,
            admission_order='fixed first complete pass', predictor_native_steps=0,
            hardware_realtime_claim=False, verified_selected_transitions=0)

    def _select(self, state):
        records = []
        selected = None
        for index, payload in enumerate(self.profiles):
            predictor = SpatialCartpole(self.env.config)
            candidate = None
            try:
                predicted = np.array(state, copy=True)
                predictor.set_state(predicted)
                predictor.data.time = self.env.data.time
                for key in ('steps', 'streak', 'best_streak', 'min_plane_margin', 'warning_seen'):
                    setattr(predictor, key, getattr(self.env, key))
                candidate = _profile_policy(predictor, payload)
                states, actions, raws, tilts = [predicted], [], [], []
                for t in range(self.env.config.horizon):
                    raw = candidate(predicted, t)
                    action = predictor.deliver(raw)
                    predicted, _, terminated, truncated, info = predictor.step_force(action)
                    states.append(predicted); actions.append(action)
                    raws.append(raw); tilts.append(predictor.tilts())
                    if terminated or truncated:
                        break
                tapes = dict(states=states, actions=actions, raw_actions=raws, tilts=tilts,
                             **candidate.forecast_tapes)
                for key, value in tapes.items():
                    self._forecasts[f'bank_forecast_{index}_{key}'] = np.asarray(value)
                admitted = bool(info['success'] and predictor.steps == self.env.config.horizon
                                and not info['physics_warning'] and info['plane_margin'] >= 0)
                records.append(dict(profile=index, **info, steps=predictor.steps,
                    actual_forecast_native_steps=predictor.native_steps,
                    search_native_steps=candidate.metadata['predictor_native_steps'],
                    native_steps=predictor.native_steps+candidate.metadata['predictor_native_steps'],
                    admitted=admitted))
                if admitted:
                    selected = index
                    break
            finally:
                if candidate is not None:
                    candidate.close()
                predictor.close()
        self.selected = 0 if selected is None else selected
        self.policy = _profile_policy(self.env, self.profiles[self.selected])
        self.metadata.update(launch_forecasts=records, selected_launch_profile=self.selected,
            launch_admitted=selected is not None, launch_forecast_start_step=0,
            forecast_native_steps=sum(r['native_steps'] for r in records))

    def __call__(self, state, step):
        if self.selected is None:
            if step != 0:
                raise ValueError('immediate admission must observe command zero')
            self._select(state)
        if self.pending is not None:
            raise RuntimeError('transition observer was not called')
        prefix = f'bank_forecast_{self.selected}_'
        exact_bytes(state, self._forecasts[prefix+'states'][step])
        raw = self.policy(state, step)
        action = self.env.deliver(raw)
        exact_bytes(raw, self._forecasts[prefix+'raw_actions'][step])
        exact_bytes(action, self._forecasts[prefix+'actions'][step])
        self.pending = (step, action.copy(), self._forecasts[prefix+'states'][step+1],
                        self._forecasts[prefix+'tilts'][step])
        self.metadata.update(
            execution_search_native_steps=self.policy.metadata['predictor_native_steps'],
            predictor_native_steps=self.metadata['forecast_native_steps']+self.policy.metadata['predictor_native_steps'])
        # Retain the actual selected-policy query ledgers as well as all forecasts.
        return raw

    def observe_transition(self, state, step, action):
        if self.pending is None or self.pending[0] != step:
            raise RuntimeError('immediate-bank transition clock mismatch')
        _, expected_action, expected_state, expected_tilts = self.pending
        exact_bytes(action, expected_action)
        exact_bytes(state, expected_state)
        exact_bytes(self.env.tilts(), expected_tilts)
        self.pending = None
        self.metadata['verified_selected_transitions'] += 1
        self.metadata['final_selected_transition_verified'] = step == self.env.config.horizon-1

    @property
    def forecast_tapes(self):
        return dict(self._forecasts, **(self.policy.forecast_tapes if self.policy is not None else {}))

    def close(self):
        if self.policy is not None:
            self.policy.close()


def immediate_bank_from_payload(env, payload):
    if not bool(payload.get('immediate_forecast_bank', False)) or int(payload['profile_count']) != 2:
        raise ValueError('explicit two-profile immediate bank payload required')
    profiles = [{key.split('__', 1)[1]: value for key, value in payload.items()
                 if key.startswith(f'profile{i}__')} for i in range(2)]
    return ImmediateBankPolicy(env, profiles)
