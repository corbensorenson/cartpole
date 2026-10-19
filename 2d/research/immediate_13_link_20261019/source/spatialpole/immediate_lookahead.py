"""Immediate finite feedback with a fixed window of four-step native lookahead."""
import numpy as np
from .conditioning import axial_gauge_route
from .immediate_swing import ImmediateFiniteValuePolicy
from .rollout_value import NativeRolloutValuePolicy


class ImmediateLookaheadPolicy:
    def __init__(self, env, payload):
        self.start = int(payload['lookahead_start'])
        self.stop = int(payload['lookahead_stop'])
        if not bool(payload.get('windowed_lookahead', False)) or not 1 <= self.start < self.stop <= env.config.horizon-3:
            raise ValueError('explicit covered immediate lookahead window required')
        if any(bool(payload.get(k, False)) for k in ('protected_native_suffix','protected_native_schedule','controller_bank','momentum_parking')):
            raise ValueError('windowed lookahead requires the plain immediate finite family')
        nx = 2*env.model.nv
        self.factors = np.asarray(payload['tracking_stage_factors'])
        if self.factors.shape != (env.config.horizon+1,nx,nx) or not np.isfinite(self.factors).all():
            raise ValueError('full native stage-factor columns required')
        self.base = ImmediateFiniteValuePolicy(env, payload)
        self.payload = payload
        self.rollout = None
        self.metadata = dict(self.base.metadata, policy='immediate finite feedback with four-step lookahead window',
                             lookahead_start=self.start, lookahead_stop=self.stop, native_rollout_horizon=4,
                             hardware_realtime_claim=False)

    def _initialize_rollout(self):
        route = self.base.inner.route
        if route is None:
            raise RuntimeError('initialize the original launch frame at command zero')
        angles = np.asarray(self.base.metadata['gauge_angles'])
        _, factors = axial_gauge_route(self.base.scratch,self.payload['states'],self.factors,angles)
        design = {k:self.payload[k] for k in ('reference','K','P','Q','R')}
        self.rollout = NativeRolloutValuePolicy(self.base.scratch,route.states,route.actions,
            route.feed,route.gains,route.roots,route.residuals,route.R,design,stage_factors=factors,horizon=4)

    def __call__(self, state, step):
        if self.rollout is None:
            if step != 0:
                raise ValueError('an immediate policy must observe command zero before the window')
            force = self.base(state,step)
            self._initialize_rollout()
        elif self.start <= step < self.stop:
            force = self.rollout(state,step)
        else:
            force = self.base(state,step)
        self.metadata.update(gauge_angles=self.base.metadata['gauge_angles'],
            predictor_native_steps=self.base.inner.route.predictor.native_steps+self.rollout.predictor.native_steps,
            native_search_commands=len(self.base.inner.route.trace),native_rollout_commands=len(self.rollout.trace))
        return force

    @property
    def forecast_tapes(self):
        tapes = dict(self.base.forecast_tapes)
        if self.rollout is not None and self.rollout.trace:
            tapes.update(self.rollout.forecast_tapes)
        return tapes

    def close(self):
        if self.rollout is not None:
            self.rollout.close()
        self.base.close()
