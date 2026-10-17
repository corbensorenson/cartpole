"""Frozen development pilot for axial-aligned native action improvement."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from spatialpole import Config
from spatialpole.aligned_value import aligned_value_policy_from_payload
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
p.add_argument('--development-seeds',type=int,nargs='+',default=[20261116,20261117,20261118,20261120,20261121])
p.add_argument('--physics-budget',type=int,default=285000)
a = p.parse_args()
if len(set(a.development_seeds))!=len(a.development_seeds) or any(20262516<=s<=20262547 for s in a.development_seeds):
    raise ValueError('development seeds must be distinct and cannot consume reserved fourteen labels')
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(out/'input_route.npz')
values = {k:source[k] for k in source.files}
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=14)
assert int(source['park_steps'])==960 and int(source['swing_steps'])==450 and len(source['actions'])==700
assert bool(source['reflection_projected_model']) and not bool(source.get('reflection_averaged_feedback',False))
changed = dict(aligned_native_value=True,native_search=True,online_gauge=True)
values.update(changed)
np.savez_compressed(out/'route.npz',**values)
for k in source.files:
    if k not in changed:
        np.testing.assert_array_equal(values[k],source[k])
write_json(out/'controller_identity.json',dict(source_sha256=hashlib.sha256((out/'input_route.npz').read_bytes()).hexdigest(),
    controller_sha256=hashlib.sha256((out/'route.npz').read_bytes()).hexdigest(),changed_keys=list(changed),
    scope='reused development starts; not qualification',design_commands=700,executed_tracking_commands=540))
factory = lambda e: aligned_value_policy_from_payload(e,values)
rows = []
for seed,noise in [(20261016,0.),*[(s,.02) for s in a.development_seeds]]:
    row = episode(config,factory,seed,'hanging',noise,out/('nominal.npz' if noise==0 else f'seed{seed}.npz'))
    rows.append(row)
    used = sum(r['native_steps']+r['proof_native_steps']+r['policy_metadata']['predictor_native_steps'] for r in rows)
    assert row['policy_metadata']['predictor_native_steps']<=34560 and used<=a.physics_budget
    write_json(out/'summary.json',dict(n_links=14,passes=sum(r['success'] and r['replay_exact'] for r in rows),
        results=rows,inclusive_native_steps=used,design_native_steps=0,scope='declared reused development starts; no reserved data'))
    print(json.dumps(dict(seed=seed,success=row['success'],hold=row['best_hold_seconds'],steps=row['steps'],
        replay_exact=row['replay_exact'],predictor_native_steps=row['policy_metadata']['predictor_native_steps'],inclusive_native_steps=used)),flush=True)
write_json(out/'completed.json',dict(status='completed',episodes=len(rows)))
