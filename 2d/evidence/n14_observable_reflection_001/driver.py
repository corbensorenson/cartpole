"""New reflection-averaged affine feedback; actual plant stays fully spatial."""
import argparse
import json
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import affine_policy_from_payload
from spatialpole.symmetry import reflection_average_feedback
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(a.route)
values = {k:source[k] for k in source.files}
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=config.n_links) and not bool(source['native_search'])
env = SpatialCartpole(config)
gains,feed,record = reflection_average_feedback(env,source['states'],source['actions'],source['gains'],source['feed'])
env.close()
values.update(gains=gains,feed=feed,reflection_averaged_feedback=True,native_search=False,
    value_role='original unsymmetrized design provenance; not used by averaged feedback')
assert all(np.array_equal(source[k],values[k]) for k in source.files if k not in ('gains','feed'))
write_json(out/'feedback_change.json',record)
np.savez_compressed(out/'route.npz',**values)
rows = []
for index,seed in enumerate([20261016]+list(range(20261116,20261128))):
    row = episode(config,lambda e:affine_policy_from_payload(e,values),seed,'hanging',0. if index==0 else .02,
        out/('nominal.npz' if index==0 else f'seed{seed}.npz'))
    rows.append(row)
    print(json.dumps({k:row[k] for k in ('seed','success','steps','best_hold_seconds','replay_exact')}),flush=True)
    write_json(out/'summary.json',dict(n_links=config.n_links,passes=sum(r['success'] and r['replay_exact'] for r in rows),
        results=rows,inclusive_native_steps=sum(r['native_steps']+r['proof_native_steps'] for r in rows),
        scope='new reflection-averaged force law on unrestricted native spatial plant; original values unused'))
    if index==0 and not(row['success'] and row['replay_exact']):
        break
assert sum(r['native_steps']+r['proof_native_steps'] for r in rows)<=160000
write_json(out/'completed.json',dict(status='completed'))
