"""Bounded native suffix forecasts from an already recorded observed launch.

This is development diagnosis, not a deployed selector or a new qualification.
"""
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
write_json(out/'tape_identity.json',dict(original=a.tape,sha256=hashlib.sha256((out/'input_tape.npz').read_bytes()).hexdigest()))
source = np.load(a.route)
payload = {k:source[k] for k in source.files}
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=config.n_links)
tape = np.load(a.tape)
assert Config(**json.loads(str(tape['config'])))==config
launch = int(source['park_steps'])
prefix = SpatialCartpole(config)
prefix.set_state(tape['states'][0])
prefix.min_plane_margin = config.plane_limit
for action in tape['actions'][:launch]:
    prefix.step_force(action)
np.testing.assert_array_equal(prefix.state(),tape['states'][launch])
assert prefix.steps==launch and prefix.min_plane_margin>=0 and not prefix.warning_seen
past = dict(streak=prefix.streak,best_streak=prefix.best_streak,margin=prefix.min_plane_margin)
prefix_steps = prefix.native_steps
prefix.close()
profiles = [(0.,0.),(.0001,0.),(-.0001,0.),(0.,.0001),(0.,-.0001),
            (.001,0.),(-.001,0.),(0.,.001),(0.,-.001)]
rows = []
for index,offset in enumerate(profiles):
    env = SpatialCartpole(config)
    state = tape['states'][launch].copy()
    env.set_state(state)
    env.steps,env.streak,env.best_streak = launch,past['streak'],past['best_streak']
    env.min_plane_margin = past['margin']
    candidate = dict(payload,cart_reference_offset=np.array(offset),align_cart_origin=False)
    policy = affine_policy_from_payload(env,candidate)
    states,actions,raw,tilts = [state],[],[],[]
    for step in range(launch,config.horizon):
        force = policy(state,step)
        delivered = env.deliver(force)
        state,_,terminated,truncated,info = env.step_force(delivered)
        states.append(state)
        actions.append(delivered)
        raw.append(force)
        tilts.append(env.tilts())
        if terminated or truncated:
            break
    if index==0:
        np.testing.assert_array_equal(states,tape['states'][launch:launch+len(states)])
        np.testing.assert_array_equal(actions,tape['actions'][launch:launch+len(actions)])
    np.savez_compressed(out/('profile_'+str(index)+'.npz'),states=states,actions=actions,
        raw_actions=raw,tilts=tilts,config=json.dumps(asdict(config)))
    row = dict(profile=index,reference_cart_offset=list(offset),**info,steps=env.steps,native_steps=env.native_steps,
               baseline_suffix_byte_exact=index==0)
    rows.append(row)
    print(json.dumps(row),flush=True)
    policy.close()
    env.close()
total = prefix_steps+sum(r['native_steps'] for r in rows)
assert total<=25000
write_json(out/'summary.json',dict(n_links=config.n_links,passes=sum(r['success'] for r in rows),results=rows,
    prefix_proof_native_steps=prefix_steps,inclusive_native_steps=total,
    claim='observed-launch native forecast diagnosis on reused development state; no new full episode or qualified count'))
write_json(out/'completed.json',dict(status='completed'))
