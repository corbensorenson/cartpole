"""Complete canonical 45-start development gate for a frozen affine payload."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config
from spatialpole.affine_value import affine_policy_from_payload
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
p.add_argument('--development-seeds',type=int,nargs='+')
p.add_argument('--physics-budget',type=int,default=540000)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
shutil.copy2(out/'input_route.npz',out/'frozen_route.npz')
source = np.load(out/'frozen_route.npz')
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=config.n_links)
assert bool(source['affine_value']) and not bool(source['native_search'])
development=a.development_seeds if a.development_seeds is not None else list(range(20261116,20261128))+list(range(20261416,20261448))
seeds=[20261016]+development
assert len(seeds)==len(set(seeds))
if config.n_links==14:
    assert not any(20262516<=s<=20262547 for s in development),'reserved fourteen labels are not development'
identity = hashlib.sha256((out/'frozen_route.npz').read_bytes()).hexdigest()
write_json(out/'cohort.json',dict(seeds=seeds,noise=[0.]+[.02]*len(development),controller_sha256=identity,
    scope='canonical development; no new qualification'))
factory = lambda e: affine_policy_from_payload(e,source)
rows=[]
for i,seed in enumerate(seeds):
    row = episode(config,factory,seed,'hanging',0. if i==0 else .02,
        out/('nominal.npz' if i==0 else f'seed{seed}.npz'))
    rows.append(row)
    inclusive = sum(r['native_steps']+r['proof_native_steps']+r['policy_metadata'].get('predictor_native_steps',0) for r in rows)
    assert inclusive<=a.physics_budget
    write_json(out/'summary.json',dict(n_links=config.n_links,passes=sum(r['success'] and r['replay_exact'] for r in rows),
        results=rows,design_native_steps=0,inclusive_native_steps=inclusive,
        controller_sha256=identity,scope='canonical45 development; not reserved qualification' if a.development_seeds is None else 'declared reused development subset; not qualification'))
    print(json.dumps(dict(completed=i+1,passes=sum(r['success'] and r['replay_exact'] for r in rows),seed=seed,
        success=row['success'],hold=row['best_hold_seconds'],steps=row['steps'],replay_exact=row['replay_exact'])),flush=True)
write_json(out/'completed.json',dict(status='completed',episodes=len(rows)))
