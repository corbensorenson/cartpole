"""Complete selected-suffix byte proof and optional independent policy reruns."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from spatialpole import Config
from spatialpole.affine_value import affine_policy_from_payload
from spatialpole.experiment import episode,write_json

p=argparse.ArgumentParser();p.add_argument('--run',required=True)
p.add_argument('--rerun-seeds',type=int,nargs='*',default=[])
p.add_argument('--physics-budget',type=int,default=60000)
a=p.parse_args();run=Path(a.run);assert (run/'completed.json').is_file()
summary=json.loads((run/'summary.json').read_text())
route=run/'frozen_route.npz';payload=np.load(route)
config=Config(**json.loads(str(payload['config'])))
assert config==Config(n_links=14) and bool(payload['controller_bank'])
assert len(set(a.rerun_seeds))==len(a.rerun_seeds)
rows={row['seed']:row for row in summary['results']}
assert all(seed in rows for seed in a.rerun_seeds)
records=[];rerun_native=0

def exact(x,y):
    assert x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes()

for row in summary['results']:
    name='nominal.npz' if row['angle_noise']==0 else f"seed{row['seed']}.npz"
    path=run/name
    if not path.is_file():
        path=run/f"seed{row['seed']}.npz"
    with np.load(path) as archive:
        data={key:archive[key] for key in archive.files}
    meta=json.loads(str(data['policy_metadata']));selected=meta['selected_launch_profile']
    forecasts=meta['launch_forecasts'];park=int(payload['park_steps'])
    assert [r['profile'] for r in forecasts]==list(range(len(forecasts)))
    good=[r['profile'] for r in forecasts if r['admitted']]
    assert meta['launch_admitted']==bool(good)
    assert selected==(good[0] if good else 0)
    if good:
        assert selected==len(forecasts)-1 and all(not r['admitted'] for r in forecasts[:-1])
    else:
        assert len(forecasts)==len(payload['bank_states'])
    for forecast in forecasts:
        prefix=f"bank_forecast_{forecast['profile']}_"
        count=len(data[prefix+'actions'])
        assert forecast['native_steps']==config.frame_skip*count
        assert len(data[prefix+'states'])==count+1
        assert forecast['steps']==park+count
        force=data[prefix+'actions'].astype(float)
        assert float(np.max(np.linalg.norm(force,axis=1)))<=80.
        if forecast['admitted']:
            assert count==config.horizon-park and forecast['success'] and not forecast['physics_warning']
            assert forecast['plane_margin']>=0 and forecast['best_hold_seconds']>=5.
    prefix=f'bank_forecast_{selected}_'
    for key in ('states','actions','raw_actions','tilts'):
        exact(data[key][park:],data[prefix+key])
    assert row['replay_exact'] and row['replay_max_error']==0.
    assert meta['verified_selected_transitions']==len(data['actions'])-park
    if row['steps']==config.horizon:
        assert meta['final_selected_transition_verified']
    assert meta['predictor_native_steps']==sum(r['native_steps'] for r in forecasts)
    record=dict(seed=row['seed'],selected=selected,admitted=bool(good),
        all_selected_suffix_arrays_byte_exact=True,final_transition_verified=meta.get('final_selected_transition_verified',False),
        predictor_native_steps=meta['predictor_native_steps'],tape_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if row['seed'] in a.rerun_seeds:
        rerun=run/f"audit_rerun{row['seed']}.npz"
        assert not rerun.exists(),'policy rerun outputs are create-only'
        fresh=episode(config,lambda e:affine_policy_from_payload(e,payload),row['seed'],
            'hanging',row['angle_noise'],rerun)
        rerun_native+=fresh['native_steps']+fresh['proof_native_steps']+fresh['policy_metadata']['predictor_native_steps']
        assert rerun_native<=a.physics_budget
        with np.load(rerun) as archive:
            for key,value in data.items():
                if np.issubdtype(value.dtype,np.number):
                    exact(archive[key],value)
        assert fresh['success']==row['success'] and fresh['replay_exact']
        record['independent_policy_rerun_all_numeric_bytes_exact']=True
    records.append(record)
inclusive=sum(r['native_steps']+r['proof_native_steps']+r['policy_metadata']['predictor_native_steps'] for r in summary['results'])
report=dict(controller_sha256=hashlib.sha256(route.read_bytes()).hexdigest(),episodes=len(records),
    all_selected_suffix_bytes_exact=True,all_final_successors_checked=True,results=records,
    original_inclusive_native_steps=inclusive,audit_rerun_native_steps=rerun_native,
    inclusive_native_steps=inclusive+rerun_native,scope='full conditional forecast/native execution proof, not arbitrary disturbance guarantee')
write_json(run/'bank_audit.json',report)
print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
