"""Switch a native-search suffix without changing the protected prefix policy."""
import numpy as np
from .env import SpatialCartpole
from .conditioning import axial_gauge_route
from .aligned_value import ParkedAlignedNativeValuePolicy


class ProtectedNativeSuffixPolicy:
    """Preserve even the old next-value boundary until the declared switch."""
    def __init__(self,env,payload):
        if not bool(payload.get('protected_native_suffix',False)):
            raise ValueError('explicit protected native suffix marker required')
        self.env,self.payload=env,payload
        self.seam=int(payload['patch_route_step'])
        H=len(payload['actions'])
        if not 0<self.seam<H:
            raise ValueError('suffix switch must lie strictly inside the design')
        prefix=dict(payload)
        # Native action t uses value/reference t+1. Merely replacing arrays
        # from seam onward would silently change the LAST protected action.
        for name in ('states','roots','residuals'):
            key='protected_boundary_'+name
            if key not in payload:
                raise ValueError('original next-value boundary is required')
            boundary=np.asarray(payload[key])
            if boundary.shape!=payload[name][self.seam].shape or not np.isfinite(boundary).all():
                raise ValueError('finite original next-value boundary required')
            prefix[name]=np.array(payload[name],copy=True)
            prefix[name][self.seam]=boundary
        self.policy=ParkedAlignedNativeValuePolicy(env,prefix)
        self.metadata=self.policy.metadata
        self.metadata.update(protected_native_suffix=True,patch_route_step=self.seam,
            protected_commands=self.policy.park_steps+self.seam,suffix_switched=False)
        self.switched=False

    def __call__(self,state,step):
        switch=self.policy.park_steps+self.seam
        if step>=switch and not self.switched:
            if step!=switch or self.policy.route is None:
                raise RuntimeError('native suffix switch must follow its complete protected prefix')
            p=self.payload
            scratch=SpatialCartpole(self.env.config)
            try:
                angles=np.asarray(self.metadata['gauge_angles'])
                states,gains=axial_gauge_route(scratch,p['states'],p['gains'],angles)
                _,roots=axial_gauge_route(scratch,p['states'],p['roots'],angles)
            finally:
                scratch.close()
            route=self.policy.route
            route.states,route.gains,route.roots=states,gains,roots
            route.actions,route.feed,route.residuals=p['actions'],p['feed'],p['residuals']
            # Keep the same owned predictor and trace; every query is charged.
            self.switched=True
            self.metadata['suffix_switched']=True
        return self.policy(state,step)

    @property
    def forecast_tapes(self):
        return self.policy.forecast_tapes

    def close(self):
        self.policy.close()
