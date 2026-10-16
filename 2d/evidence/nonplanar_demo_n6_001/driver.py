"""Controlled XY motion after capture, on the unchanged qualified plant."""
import argparse
import json
from pathlib import Path
import numpy as np
from spatialpole import Config
from spatialpole.control import WorldBalancePolicy
from spatialpole.conditioning import ParkedRoutePolicy
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=6)
design = {k:source[k] for k in ('K','P','Q','R','reference')}
parking = {'reference':source['parking_reference'],'K':source['parking_K']}

class OrbitPolicy:
    def __init__(self,env):
        self.env = env
        self.base = ParkedRoutePolicy(env,source['states'],source['actions'],source['gains'],
            design,parking,int(source['park_steps']),bool(source['align_gauge']))
        self.world = WorldBalancePolicy(env,design)
        self.metadata = dict(self.base.metadata,demonstration='controlled XY orbit after capture',
                             orbit_start_seconds=18.,orbit_radius=.35,orbit_angular_frequency=.6)

    def __call__(self,state,step):
        tau = step*config.dt-18.
        if tau<0:
            force = self.base(state,step)
            self.metadata.update(self.base.metadata)
            return force
        z = np.exp(-tau/2)
        r,rd,rdd = .35*(1-z)**2,.35*z*(1-z),.35*(-z/2+z*z)
        angle,w = .6*tau,.6
        c,s = np.cos(angle),np.sin(angle)
        desired = r*np.array([c-1,s])
        velocity = rd*np.array([c-1,s])+r*w*np.array([-s,c])
        acceleration = rdd*np.array([c-1,s])+2*rd*w*np.array([-s,c])-r*w*w*np.array([c,s])
        error = self.world.error(state)
        error[:2] -= desired
        error[self.env.model.nv:self.env.model.nv+2] -= velocity
        return -design['K']@error+(config.cart_mass+config.total_mass)*acceleration

rows = []
for seed,noise in ((20261016,0.),(20261116,.02),(20261117,.02),(20261118,.02)):
    tape = out/f'seed{seed}.npz'
    result = episode(config,OrbitPolicy,seed,'hanging',noise,tape)
    xy = np.load(tape)['states'][900:,:2]
    singular = np.linalg.svd(xy-xy.mean(axis=0),compute_uv=False)
    result['orbit_xy_spans'] = np.ptp(xy,axis=0).tolist()
    result['orbit_minor_major_ratio'] = float(singular[-1]/singular[0])
    result['nonplanar_motion_pass'] = bool(np.min(np.ptp(xy,axis=0))>.1 and singular[-1]/singular[0]>.1)
    rows.append(result)
    print(json.dumps(result),flush=True)
write_json(out/'summary.json',{'n_links':6,'passes':sum(r['success'] and r['replay_exact'] and r['nonplanar_motion_pass'] for r in rows),
    'results':rows,'scope':'controlled nonplanar audit demonstration; not an additional count qualification'})
write_json(out/'completed.json',{'status':'completed'})
