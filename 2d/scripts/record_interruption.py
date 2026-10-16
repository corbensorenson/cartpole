"""Record a verified stopped repair process without inventing exact call totals."""
import argparse
import json
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--experiment',required=True)
p.add_argument('--reason',required=True)
a = p.parse_args()
out = Path(a.experiment)
if (out/'completed.json').exists():
    raise ValueError('completed experiments must not be relabeled as interrupted')
rows = [json.loads(x) for x in (out/'optimizer.jsonl').read_text().splitlines()]
checkpoint = sorted((out/'checkpoints').glob('*.npz'))[-1]
index = int(checkpoint.stem.split('_')[-1])
row = max((r for r in rows if r['iteration']<=index),key=lambda r:r['iteration'])
source = np.load(out/'input_route.npz')
config = json.loads(str(source['config']))
nv,H = 2+3*config['n_links'],len(source['actions'])
budget = json.loads((out/'inputs.json').read_text())['args']['physics_budget']
lower = rows[-1]['native_steps']
upper = min(budget,lower+(1+4*nv+4+7)*H*config['frame_skip'])
note = {'status':'interrupted','reason':a.reason,'actual_validation_episodes':0,
        'native_steps_lower_bound':lower,'native_steps_upper_bound':upper,
        'accounting_scope':'physics steps; final native batch may have been interrupted',
        'checkpoint':str(checkpoint).replace('\\','/'),'checkpoint_penalty':row['penalty']}
(out/'interruption.json').write_text(json.dumps(note,indent=2),encoding='utf-8')
print(json.dumps(note,indent=2))
