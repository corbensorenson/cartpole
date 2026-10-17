"""Freeze three demonstrated development controllers into a causal forecast bank."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config
from spatialpole.controller_bank import FIELDS
from spatialpole.experiment import freeze_inputs,write_json

p=argparse.ArgumentParser();p.add_argument('--route',required=True)
p.add_argument('--repairs',nargs=2,required=True);p.add_argument('--out',required=True)
a=p.parse_args();out=Path(a.out);freeze_inputs(out,a)
expected=['3a3f643c4b8af5371c52b6f224393ad2c9c98656b20313ef568ca86274801c17',
    '65dcc0ef085ddf92e3d502cb797b0b814de642037b6978cff997df3eaed829f8',
    '33f5005aec74045b298190cc79506867b94785b0cf11771619d05df5c62c68d4']
paths=[out/'input_route.npz'];receipts={}
for index,folder in enumerate(map(Path,a.repairs),1):
    assert (folder/'completed.json').is_file()
    path=out/f'input_profile{index}.npz';shutil.copy2(folder/'route.npz',path);paths.append(path)
    for name in ('summary.json','design_summary.json','prefix_audit.json'):
        dest=out/f'profile{index}_{name}';shutil.copy2(folder/name,dest)
        receipts[dest.name]=hashlib.sha256(dest.read_bytes()).hexdigest()
    rows=json.loads((folder/'summary.json').read_text())['results']
    assert rows[0]['success'] and rows[0]['replay_exact']
    assert json.loads((folder/'prefix_audit.json').read_text())['all_pilot_prefixes_byte_exact']
hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths];assert hashes==expected
profiles=[np.load(path) for path in paths];base=profiles[0]
assert Config(**json.loads(str(base['config'])))==Config(n_links=14)
for other in profiles[1:]:
    for key in base.files:
        if key in FIELDS or key in ('defects','reflection_projected_model'):
            continue
        x,y=base[key],other[key]
        assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),key
    for key in (*FIELDS,'defects'):
        x,y=base[key][:150],other[key][:150]
        assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),key
values={k:base[k] for k in base.files}
values.update({'bank_'+key:np.stack([profile[key] for profile in profiles]) for key in FIELDS})
values.update(controller_bank=True,bank_schema=1,bank_controller_sha256=np.array(hashes),
    bank_patch_route_step=150,bank_reflection_projected_models=np.array([True,False,False]))
np.savez_compressed(out/'route.npz',**values)
report=dict(n_links=14,design_native_steps=0,profiles_sha256=hashes,repair_receipts_sha256=receipts,
    controller_sha256=hashlib.sha256((out/'route.npz').read_bytes()).hexdigest(),
    scope='fixed observed-state first-passing complete-native-continuation bank; no qualification yet')
write_json(out/'summary.json',report);write_json(out/'completed.json',dict(status='completed'))
print(json.dumps(report),flush=True)
