"""Bounded one-link energy-controller initializer search, fully native episodes."""
import json
from pathlib import Path
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.control import balance_design, BalancePolicy
from spatialpole.experiment import episode, write_json


class EnergyPolicy:
    def __init__(self, env, design, energy_gain, kp, kd, capture):
        self.env = env
        self.balance = BalancePolicy(env, design)
        self.gain, self.kp, self.kd, self.capture = energy_gain, kp, kd, capture
        self.captured = False

    def __call__(self, state, step):
        e = self.env
        e.set_state(state)
        direction = e.data.xmat[e.link_ids[0]].reshape(3,3)[:,2]
        theta = np.arctan2(direction[0], direction[2])
        omega = state[e.model.nq+3]
        if abs(theta) < self.capture and abs(omega) < 1.2:
            self.captured = True
        if self.captured:
            return self.balance(state)
        # Mechanical link energy about a prescribed moving pivot; use the
        # compiled capsule inertia (includes cap extent) rather than rod guess.
        body = e.model.body('link_0').id
        mass = e.model.body_mass[body]
        com = e.model.body_ipos[body,2]
        inertia = e.model.body_inertia[body,1] + mass * com**2
        energy = 0.5*inertia*omega**2 + mass*9.81*com*np.cos(theta)
        target = mass*9.81*com
        u = self.gain*(energy-target)*omega*np.cos(theta) - self.kp*state[0] - self.kd*state[e.model.nq]
        if step < 10:
            u += 15.
        return np.array([u, -10*state[1]-5*state[e.model.nq+1]])


if __name__ == '__main__':
    out = Path('output/energy_n1_001')
    out.mkdir(parents=True, exist_ok=False)
    config = Config()
    design = balance_design(SpatialCartpole(config))
    rows = []
    for gain in (0.5,1.,2.,4.):
        for kp,kd in ((2,3),(5,5),(10,8),(20,10)):
            for capture in (0.2,0.45):
                name = f'g{gain}_p{kp}_d{kd}_c{capture}'
                factory = lambda e: EnergyPolicy(e, design, gain, kp, kd, capture)
                r = episode(config, factory, 20261016, 'hanging', 0., out / (name+'.npz'))
                r.update(gain=gain,kp=kp,kd=kd,capture=capture,name=name)
                rows.append(r)
                print(json.dumps({k:r[k] for k in ('name','success','steps','best_hold_seconds')}),flush=True)
                write_json(out/'summary.json',rows)
