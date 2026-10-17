"""Full native suffix forecasts for a bounded observed-state launch window."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import affine_policy_from_payload
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--tape',required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
shutil.copy2(a.tape,out/'input_tape.npz')
write_json(out/'tape_identity.json',dict(original=a.tape,
    sha256=hashlib.sha256((out/'input_tape.npz').read_bytes()).hexdigest()))
source,tape = np.load(out/'input_route.npz'),np.load(out/'input_tape.npz')
payload = {k:source[k] for k in source.files}
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=config.n_links)
assert Config(**json.loads(str(tape['config'])))==config
candidates = list(range(900,956,5))
start,original = candidates[0],int(source['park_steps'])
assert original in candidates and start<original and len(tape['actions'])>=original
prefix = SpatialCartpole(config)
prefix.set_state(tape['states'][0])
for action in tape['actions'][:start]:
    prefix.step_force(action)
np.testing.assert_array_equal(prefix.state(),tape['states'][start])
assert prefix.steps==start and prefix.min_plane_margin>=0 and not prefix.warning_seen
past = dict(streak=prefix.streak,best_streak=prefix.best_streak,margin=prefix.min_plane_margin)
prefix_steps = prefix.native_steps
prefix.close()
rows = []
for index,park_steps in enumerate(candidates):
    env = SpatialCartpole(config)
    state = tape['states'][start].copy()
    env.set_state(state)
    env.steps,env.streak,env.best_streak = start,past['streak'],past['best_streak']
    env.min_plane_margin = past['margin']
    policy = affine_policy_from_payload(env,dict(payload,park_steps=park_steps))
    states,actions,raw,tilts = [state],[],[],[]
    for step in range(start,config.horizon):
        force = policy(state,step)
        delivered = env.deliver(force)
        state,_,terminated,truncated,info = env.step_force(delivered)
        states.append(state)
        actions.append(delivered)
        raw.append(force)
        tilts.append(env.tilts())
        if terminated or truncated:
            break
    if park_steps==original:
        np.testing.assert_array_equal(states,tape['states'][start:])
        np.testing.assert_array_equal(actions,tape['actions'][start:])
    np.savez_compressed(out/f'profile_{index}.npz',states=states,actions=actions,
        raw_actions=raw,tilts=tilts,config=json.dumps(asdict(config)))
    row = dict(profile=index,park_steps=park_steps,park_seconds=park_steps*config.dt,
        **info,steps=env.steps,native_steps=env.native_steps,
        original_suffix_byte_exact=park_steps==original)
    rows.append(row)
    print(json.dumps(row),flush=True)
    policy.close()
    env.close()
total = prefix_steps+sum(r['native_steps'] for r in rows)
assert total<=33000
write_json(out/'summary.json',dict(n_links=config.n_links,passes=sum(r['success'] for r in rows),
    results=rows,prefix_proof_native_steps=prefix_steps,inclusive_native_steps=total,
    claim='bounded observed-state launch-time diagnosis; no new actual episode or qualified count'))
write_json(out/'completed.json',dict(status='completed'))
