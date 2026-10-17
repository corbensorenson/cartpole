"""Independently rescore saved post-state streaks and inventory release gates.

This complements the actual native episodes/replays; it does not infer physics-
substep safety from command states or replace the native replay evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from spatialpole import Config
from spatialpole.experiment import write_json

p = argparse.ArgumentParser()
p.add_argument('--links',type=int,required=True)
p.add_argument('--qualification',required=True)
p.add_argument('--expected-seed',type=int,required=True)
a = p.parse_args()
folder = Path(a.qualification)
assert (folder/'completed.json').is_file()
summary = json.loads((folder/'summary.json').read_text())
rows = summary['results']
assert summary['n_links']==a.links and summary['passes']==32 and len(rows)==32
assert [r['seed'] for r in rows]==list(range(a.expected_seed,a.expected_seed+32))
config = Config(n_links=a.links)
initial_hashes,holds,max_positions,max_forces = [],[],[],[]
for row in rows:
    assert row['success'] and row['replay_exact'] and row['replay_max_error']==0
    assert not row['physics_warning'] and row['steps']==config.horizon
    assert row['native_steps']==row['proof_native_steps']==config.horizon*config.frame_skip
    tape = np.load(folder/('seed'+str(row['seed'])+'.npz'))
    assert Config(**json.loads(str(tape['config'])))==config
    states,actions,tilts = tape['states'],tape['actions'],tape['tilts']
    assert len(states)==config.horizon+1 and actions.shape==(config.horizon,2)
    assert tilts.shape==(config.horizon,a.links) and actions.dtype==np.float32
    assert np.isfinite(states).all() and np.isfinite(actions).all() and np.isfinite(tilts).all()
    assert np.max(np.abs(states[:,:2]))<=config.plane_limit and row['plane_margin']>=0
    force = float(np.max(np.linalg.norm(actions.astype(float),axis=1)))
    assert force<=config.force_limit
    streak,best = 0,0
    for upright in np.max(tilts,axis=1)<config.angle_threshold:
        streak = streak+1 if upright else 0
        best = max(best,streak)
    hold = best*config.dt
    assert hold>=config.sustain_seconds and abs(hold-row['best_hold_seconds'])<1e-12
    initial_hashes.append(hashlib.sha256(states[0].tobytes()).hexdigest())
    holds.append(hold)
    max_positions.append(float(np.max(np.abs(states[:,:2]))))
    max_forces.append(force)
assert len(set(initial_hashes))==32
report = dict(passes=32,distinct_initial_states=32,minimum_hold_seconds=min(holds),
    maximum_cart_coordinate=max(max_positions),minimum_substep_margin=min(r['plane_margin'] for r in rows),
    maximum_force_norm=max(max_forces),maximum_launch_tilt_from_down=max(r['policy_metadata']['launch_max_tilt_from_down'] for r in rows),
    inclusive_native_steps=sum(r['native_steps']+r['proof_native_steps'] for r in rows),
    all_full=True,all_replay_exact=True,all_no_warning=True,
    scope='saved-tape post-state rescore; substep margins and exact replay from actual native execution records')
write_json(folder/'release_stats.json',report)
print(json.dumps(report,indent=2))
