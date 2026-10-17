"""Freeze a declared actual prefix as a new, independently replayed reference."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True,help='source controller, retained as provenance')
p.add_argument('--tape',required=True)
p.add_argument('--commands',type=int,required=True)
p.add_argument('--swing-steps',type=int,required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
shutil.copy2(a.tape,out/'input_tape.npz')
tape = np.load(out/'input_tape.npz')
source = np.load(out/'input_route.npz')
config = Config(**json.loads(str(tape['config'])))
assert config==Config(n_links=config.n_links)==Config(**json.loads(str(source['config'])))
assert 0<a.swing_steps<=a.commands<=len(tape['actions'])
states,actions = tape['states'][:a.commands+1],tape['actions'][:a.commands]
assert actions.dtype==np.float32 and len(states)==a.commands+1
env = SpatialCartpole(config)
assert np.max(np.linalg.norm(actions.astype(float),axis=1))<=config.force_limit
for state,action,following in zip(states,actions,states[1:]):
    np.testing.assert_array_equal(env.deliver(action),action)
    np.testing.assert_array_equal(env.transition(state,action),following)
assert not env.warning_seen and env.min_plane_margin>=0
np.savez_compressed(out/'route.npz',states=states,actions=actions,swing_steps=a.swing_steps,
    config=str(tape['config']),physical_reference=True)
write_json(out/'summary.json',dict(n_links=config.n_links,commands=a.commands,swing_steps=a.swing_steps,
    source_tape_sha256=hashlib.sha256((out/'input_tape.npz').read_bytes()).hexdigest(),
    source_controller_sha256=hashlib.sha256((out/'input_route.npz').read_bytes()).hexdigest(),
    physical_reference_sha256=hashlib.sha256((out/'route.npz').read_bytes()).hexdigest(),
    native_replay_exact=True,maximum_reference_defect=0.,native_steps=env.native_steps,
    minimum_substep_margin=env.min_plane_margin,
    scope='actual prefix only; no implication that source episode or redesigned controller passes'))
write_json(out/'completed.json',dict(status='completed'))
env.close()
