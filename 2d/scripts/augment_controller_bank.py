"""Append reflected spatial suffix laws; keep the three original profiles exact."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.controller_bank import FIELDS
from spatialpole.symmetry import reflect_tracking_suffix
from spatialpole.experiment import freeze_inputs,write_json,episode
from spatialpole.affine_value import affine_policy_from_payload

p=argparse.ArgumentParser();p.add_argument('--route',required=True)
p.add_argument('--failed-qualification',required=True);p.add_argument('--out',required=True)
a=p.parse_args();out=Path(a.out);freeze_inputs(out,a)
old=Path(a.failed_qualification)
assert (old/'completed.json').is_file()
for name in ('summary.json','bank_audit.json','seed20262528.npz'):
    shutil.copy2(old/name,out/('previous_'+name))
q=json.loads((out/'previous_summary.json').read_text())
assert q['n_links']==14 and q['passes']==31 and len(q['results'])==32
assert [r['seed'] for r in q['results'] if not r['success']]==[20262528]
assert hashlib.sha256((out/'input_route.npz').read_bytes()).hexdigest()=='b2cd01d9770f194e82d329f471ae11d3a32f7063589a6c79625b5071de47112a'
source=np.load(out/'input_route.npz');values={k:source[k] for k in source.files}
config=Config(**json.loads(str(source['config'])));assert config==Config(n_links=14)
env=SpatialCartpole(config);mirrors=[]
for index in (1,2):
    candidate={key:source['bank_'+key][index] for key in FIELDS}
    mirrors.append(reflect_tracking_suffix(env,candidate,start=150))
for key in FIELDS:
    values['bank_'+key]=np.concatenate((source['bank_'+key],np.stack([m[key] for m in mirrors])))
    assert values['bank_'+key][:3].tobytes()==source['bank_'+key].tobytes()
    for index in (3,4):
        assert values['bank_'+key][index,:150].tobytes()==source[key][:150].tobytes()
values['bank_source_controller_sha256']=values.pop('bank_controller_sha256')
values.update(bank_schema=2,bank_reflection_projected_models=np.array([True,False,False,False,False]),
    bank_profile_origins=np.array(['original','repair431','repair437','mirror431','mirror437']))
np.savez_compressed(out/'route.npz',**values)
assert env.native_steps==0;env.close()
identity=hashlib.sha256((out/'route.npz').read_bytes()).hexdigest()
write_json(out/'design_summary.json',dict(controller_sha256=identity,design_native_steps=0,
    original_three_profiles_byte_exact=True,reflection_seam=150,
    previous_qualification_sha256=hashlib.sha256((out/'previous_summary.json').read_bytes()).hexdigest(),
    scope='version2 reflection-augmented family; previous reserved failure is now explicit development'))
rows=[]
for seed in (20261016,20261116,20261117,20261118,20261120,20261121,20261431,20261437,20262528):
    row=episode(config,lambda e:affine_policy_from_payload(e,values),seed,'hanging',
        0. if seed==20261016 else .02,out/('nominal.npz' if seed==20261016 else f'seed{seed}.npz'))
    rows.append(row)
    inclusive=sum(r['native_steps']+r['proof_native_steps']+r['policy_metadata']['predictor_native_steps'] for r in rows)
    assert inclusive<=210000
    write_json(out/'summary.json',dict(n_links=14,passes=sum(r['success'] and r['replay_exact'] for r in rows),
        results=rows,design_native_steps=0,inclusive_native_steps=inclusive,controller_sha256=identity,
        scope='reused development INCLUDING one explicitly consumed prior qualification failure; not fresh qualification'))
    print(json.dumps(dict(completed=len(rows),seed=seed,success=row['success'],hold=row['best_hold_seconds'],
        selected=row['policy_metadata']['selected_launch_profile'])),flush=True)
shutil.copy2(out/'route.npz',out/'frozen_route.npz')
write_json(out/'completed.json',dict(status='completed'))
