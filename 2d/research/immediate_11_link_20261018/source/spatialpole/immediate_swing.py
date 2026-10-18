"""Start an existing motion controller at reset, without executing its parking law."""
from .aligned_value import aligned_value_policy_from_payload
from .affine_value import affine_policy_from_payload
from .env import SpatialCartpole


class ImmediateSwingPolicy:
    """Offset only the controller's route clock; the live plant clock is untouched.

    This is an experimental controller, not evidence that a prepared controller
    generalizes to unprepared starts. Its normal terminal balance law is retained.
    """
    def __init__(self, env, payload):
        self.inner = aligned_value_policy_from_payload(env, payload)
        self.offset = int(payload['park_steps'])
        self.metadata = dict(policy='immediate spatial swing with native affine correction',
            park_steps=0, park_seconds=0., source_park_steps=self.offset,
            route_seconds=int(payload['swing_steps'])*env.config.dt,
            design_steps=len(payload['actions']), hardware_realtime_claim=False,
            predictor_native_steps=0)

    def __call__(self, state, step):
        force = self.inner(state, step+self.offset)
        for key in ('predictor_native_steps', 'native_search_commands', 'baseline_fallback_commands',
                    'gauge_angles', 'suffix_switched'):
            if key in self.inner.metadata:
                self.metadata[key] = self.inner.metadata[key]
        return force

    @property
    def forecast_tapes(self):
        return self.inner.forecast_tapes

    def close(self):
        self.inner.close()


class ImmediateFiniteValuePolicy:
    """Execute a complete zero-parking finite design using owned scratch state."""
    def __init__(self,env,payload):
        if int(payload['park_steps'])!=0 or len(payload['actions'])!=env.config.horizon:
            raise ValueError('immediate finite policy must cover the whole unprepared episode')
        self.scratch=SpatialCartpole(env.config)
        factory=aligned_value_policy_from_payload if bool(payload['aligned_native_value']) else affine_policy_from_payload
        try:self.inner=factory(self.scratch,payload)
        except BaseException:self.scratch.close();raise
        self.metadata=self.inner.metadata
        self.metadata['policy']='immediate full-motion native affine value' if bool(payload['aligned_native_value']) else 'immediate full-motion affine feedback'

    def __call__(self,state,step):
        return self.inner(state,step)

    @property
    def forecast_tapes(self):
        return getattr(self.inner,'forecast_tapes',{})

    def close(self):
        self.inner.close();self.scratch.close()
