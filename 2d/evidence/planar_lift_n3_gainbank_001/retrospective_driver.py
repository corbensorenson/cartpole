"""Small declared tracking-gain screen; full-episode success selects a candidate."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from spatialpole import Config
from spatialpole.control import RoutePolicy
from spatialpole.experiment import freeze_inputs,episode,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--out',required=True)
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
shutil.copy2(a.route,out/'source_route.npz')
data = np.load(a.route)
config = Config(**json.loads(str(data['config'])))
design = {k:data[k] for k in ('K','P','Q','R','reference')}
rows = []
for scale in (0.75,1.25,1.5,2.,3.):
    gains = data['gains'].copy()
    gains[:,0,:] *= scale
    factory = lambda e: RoutePolicy(e,data['states'],data['actions'],gains,design,True)
    r = episode(config,factory,20261016,'hanging',0.,out/f'nominal_scale_{scale}.npz')
    rows.append({'scale':scale,'nominal':r})
    print(json.dumps({'scale':scale,'success':r['success'],'steps':r['steps'],'hold':r['best_hold_seconds']}),flush=True)
    if r['success'] and r['replay_exact']:
        dev = [episode(config,factory,20261116+k,'hanging',0.02,out/f'scale_{scale}_validation_{k}.npz') for k in range(12)]
        rows[-1]['development'] = dev
        if all(x['success'] and x['replay_exact'] for x in dev):
            values = {k:data[k] for k in data.files}
            values['gains'],values['world_balance'] = gains,True
            np.savez_compressed(out/'route.npz',**values)
            write_json(out/'selected.json',{'scale':scale,'development_passes':12,'nominal_passes':1})
            break
    write_json(out/'summary.json',{'n_links':config.n_links,'candidates':rows})
write_json(out/'summary.json',{'n_links':config.n_links,'candidates':rows})
write_json(out/'completed.json',{'status':'completed','selected':(out/'selected.json').exists()})
