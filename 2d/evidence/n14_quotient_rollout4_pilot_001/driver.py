"""Frozen eight-start four-command native value pilot, retaining every query."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config
from spatialpole.rollout_value import rollout_value_policy_from_payload
from spatialpole.experiment import freeze_inputs,episode,write_json

p=argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--quotient-design',required=True)
p.add_argument('--design-controller',required=True)
p.add_argument('--out',required=True)
a=p.parse_args()
out=Path(a.out)
freeze_inputs(out,a)
for original,name in ((a.quotient_design,'input_quotient_design.npz'),(a.design_controller,'input_design_controller.npz')):
    shutil.copy2(original,out/name)
source=np.load(out/'input_route.npz')
design=np.load(out/'input_design_controller.npz')
frames=np.load(out/'input_quotient_design.npz')
values={k:source[k] for k in source.files}
config=Config(**json.loads(str(source['config'])))
assert config==Config(n_links=14) and bool(source['axial_quotient_planning'])
assert int(source['park_steps'])==960 and int(source['swing_steps'])==450 and len(source['actions'])==700
assert not bool(source['native_search']) and not bool(source.get('online_parking',False))
for k in design.files:
    if k in source.files:
        np.testing.assert_array_equal(design[k],source[k])
E,F,G,L=frames['E'],frames['F'],frames['G'],frames['tracking_L']
assert E.shape==(701,88,74) and F.shape==(701,74,88) and L.shape==(88,88)
np.testing.assert_allclose(E@F,G,atol=3e-13,rtol=3e-13)
np.testing.assert_array_equal(frames['tracking_R'],source['tracking_R'])
factors=(L[None,:,:]@E)@F
assert np.isfinite(factors).all()
changes=dict(rollout_native_value=True,aligned_native_value=True,native_search=True,online_gauge=True,
    native_rollout_horizon=4,tracking_stage_factors=factors)
values.update(changes)
for k in source.files:
    if k not in changes:
        np.testing.assert_array_equal(values[k],source[k])
np.savez_compressed(out/'route.npz',**values)
write_json(out/'controller_identity.json',dict(
    input_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
        ('input_route.npz','input_design_controller.npz','input_quotient_design.npz')},
    controller_sha256=hashlib.sha256((out/'route.npz').read_bytes()).hexdigest(),
    added_or_changed_keys=list(changes),original_arrays_exact=True,
    stage_factor='(tracking_L @ quotient_E) @ quotient_F',horizon=4,max_candidates=16,
    future_feedback='plain online-aligned affine; no recursive search',scope='reused pilot only'))
factory=lambda e:rollout_value_policy_from_payload(e,values)
rows=[]
for seed,noise in [(20261016,0.),*[(s,.02) for s in (20261116,20261117,20261118,20261120,20261121,20261431,20261437)]]:
    print(json.dumps(dict(starting_seed=seed)),flush=True)
    row=episode(config,factory,seed,'hanging',noise,out/('nominal.npz' if noise==0 else f'seed{seed}.npz'))
    rows.append(row)
    used=sum(r['native_steps']+r['proof_native_steps']+r['policy_metadata']['predictor_native_steps'] for r in rows)
    assert row['policy_metadata']['predictor_native_steps']<=138240 and used<=1250000
    write_json(out/'summary.json',dict(n_links=14,passes=sum(r['success'] and r['replay_exact'] for r in rows),
        results=rows,inclusive_native_steps=used,design_native_steps=0,scope='eight reused starts; not qualification'))
    print(json.dumps(dict(seed=seed,success=row['success'],hold=row['best_hold_seconds'],steps=row['steps'],
        replay_exact=row['replay_exact'],predictor_native_steps=row['policy_metadata']['predictor_native_steps'],inclusive_native_steps=used)),flush=True)
write_json(out/'completed.json',dict(status='completed',episodes=len(rows)))
