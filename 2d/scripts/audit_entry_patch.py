"""Check every retained patch pilot against its unchanged physical prefix."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser();p.add_argument('--run',required=True)
p.add_argument('--baseline',default='output/conditioned_n14_quotient_001')
a=p.parse_args();run=Path(a.run);baseline=Path(a.baseline)
assert (run/'completed.json').is_file()
summary=json.loads((run/'summary.json').read_text())
design=json.loads((run/'design_summary.json').read_text())
source=np.load(run/'input_route.npz');candidate=np.load(run/'route.npz')
seam=int(candidate['patch_route_step']);step=int(source['park_steps'])+seam
records=[]
for key in ('states','actions','feed','gains','roots','residuals','defects'):
    x,y=candidate[key][:seam],source[key][:seam]
    assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()
for row in summary['results']:
    name=f"seed{row['seed']}.npz"
    oldname='nominal.npz' if row['angle_noise']==0 else name
    current=np.load(run/name);old=np.load(baseline/oldname)
    for key,end in (('states',step+1),('actions',step),('raw_actions',step),('tilts',step)):
        x,y=current[key][:end],old[key][:end]
        assert len(x)==end and x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),(name,key)
    assert row['replay_exact'] and row['replay_max_error']==0.
    records.append(dict(seed=row['seed'],prefix_states_actions_raw_tilts_byte_exact=True,
        cold_exact=True,tape_sha256=hashlib.sha256((run/name).read_bytes()).hexdigest(),
        baseline_tape_sha256=hashlib.sha256((baseline/oldname).read_bytes()).hexdigest()))
report=dict(prefix_commands=step,all_pilot_prefixes_byte_exact=True,results=records,
    additional_native_steps=0,design_native_steps=design['design_native_steps'],
    pilot_native_steps=sum(r['native_steps']+r['proof_native_steps'] for r in summary['results']),
    scope='all candidate physical prefixes and stored cold proofs; no new episodes')
(run/'prefix_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='results'}))
