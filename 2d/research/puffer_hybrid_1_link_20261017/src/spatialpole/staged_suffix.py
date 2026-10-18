"""Two protected native-search boundaries for an observed capture-tail repair."""
import numpy as np
from .env import SpatialCartpole
from .conditioning import axial_gauge_route
from .aligned_value import ParkedAlignedNativeValuePolicy


class StagedNativeSuffixPolicy:
    def __init__(self,env,payload):
        if not bool(payload.get('protected_native_schedule',False)):
            raise ValueError('explicit staged native suffix marker required')
        self.env,self.payload=env,payload
        self.switches=np.asarray(payload['protected_switch_steps'])
        H=len(payload['actions'])
        if self.switches.shape!=(2,) or not np.issubdtype(self.switches.dtype,np.integer) or not 0<self.switches[0]<self.switches[1]<H:
            raise ValueError('two ordered interior native switch commands required')
        self.tables={}
        for key in ('states','roots','residuals'):
            table=np.asarray(payload['protected_boundary_table_'+key])
            if table.shape!=(2,)+payload[key][0].shape or not np.isfinite(table).all():
                raise ValueError('finite original native boundaries required')
            self.tables[key]=table.copy()
        self.completed=0
        self.policy=ParkedAlignedNativeValuePolicy(env,self.remaining_payload())
        self.metadata=self.policy.metadata
        self.metadata.update(protected_native_schedule=True,protected_switch_steps=self.switches.tolist(),
            completed_protected_switches=[],suffix_switched=False)

    def remaining_payload(self):
        active=dict(self.payload)
        for key,table in self.tables.items():
            active[key]=np.array(self.payload[key],copy=True)
            for index in range(self.completed,2):
                active[key][self.switches[index]]=table[index]
        return active

    def __call__(self,state,step):
        if self.completed<2 and step>=self.policy.park_steps+self.switches[self.completed]:
            if step!=self.policy.park_steps+self.switches[self.completed] or self.policy.route is None:
                raise RuntimeError('staged suffix requires every protected prefix command')
            self.completed+=1;p=self.remaining_payload()
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
            self.metadata.update(completed_protected_switches=self.switches[:self.completed].tolist(),
                suffix_switched=self.completed==2)
        return self.policy(state,step)

    @property
    def forecast_tapes(self):
        return self.policy.forecast_tapes

    def close(self):
        self.policy.close()
