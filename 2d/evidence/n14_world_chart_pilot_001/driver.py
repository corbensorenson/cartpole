"""Frozen reused eight-start pilot of full-spin physical-coordinate feedback."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config
from spatialpole.world_coordinates import world_coordinate_policy_from_payload
from spatialpole.experiment import freeze_inputs,write_json,episode

p=argparse.ArgumentParser();p.add_argument('--route',required=True);p.add_argument('--out',required=True)
a=p.parse_args();out=Path(a.out);freeze_inputs(out,a)
shutil.copy2(out/'input_route.npz',out/'frozen_route.npz')
source=np.load(out/'frozen_route.npz');config=Config(**json.loads(str(source['config'])))
assert config==Config(n_links=14) and bool(source['world_coordinate_feedback'])
assert int(source['park_steps'])==960 and int(source['swing_steps'])==450 and len(source['actions'])==700
factory=lambda e:world_coordinate_policy_from_payload(e,source)
rows=[]
for seed,noise in [(20261016,0.),*[(s,.02) for s in (20261116,20261117,20261118,20261120,20261121,20261431,20261437)]]:
    r=episode(config,factory,seed,'hanging',noise,out/('nominal.npz' if noise==0 else f'seed{seed}.npz'))
    rows.append(r)
    steps=sum(x['native_steps']+x['proof_native_steps'] for x in rows)
    assert steps<=100000
    write_json(out/'summary.json',dict(n_links=14,passes=sum(x['success'] and x['replay_exact'] for x in rows),results=rows,
        inclusive_native_steps=steps,design_native_steps=0,
        coordinate_forward_queries=sum(x['policy_metadata']['coordinate_forward_queries'] for x in rows),
        controller_sha256=hashlib.sha256((out/'frozen_route.npz').read_bytes()).hexdigest(),
        scope='reused eight-start development pilot, not qualification'))
    print(json.dumps(dict(seed=seed,success=r['success'],hold=r['best_hold_seconds'],steps=r['steps'],replay_exact=r['replay_exact'])),flush=True)
write_json(out/'completed.json',dict(status='completed',episodes=len(rows)))
