"""Offline native/chart remainder of the actually recorded affine backbone."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import affine_policy_from_payload
from spatialpole.aligned_value import aligned_next_value
from spatialpole.conditioning import axial_gauge_state
from spatialpole.experiment import freeze_inputs,write_json

p=argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--design-dir',required=True)
p.add_argument('--tapes',required=True,nargs='+')
p.add_argument('--out',required=True)
a=p.parse_args()
out=Path(a.out)
freeze_inputs(out,a)
design_dir=Path(a.design_dir)
for name in ('affine_controller.npz','quotient_design.npz','cache_and_quotient_preflight.json'):
    shutil.copy2(design_dir/name,out/('input_'+name))
source=np.load(out/'input_route.npz');control=np.load(out/'input_affine_controller.npz');design=np.load(out/'input_quotient_design.npz')
for key in control.files:
    if key in source.files:
        np.testing.assert_array_equal(control[key],source[key])
config=Config(**json.loads(str(source['config'])))
assert config==Config(n_links=14) and bool(source['axial_quotient_planning']) and not bool(source['native_search'])
assert int(source['park_steps'])==960
A,B,d,K,E,F=map(lambda k:design[k],('reduced_A','reduced_B','reduced_defects','reduced_gains','E','F'))
scales=np.r_[[3.,3.],np.full(42,.15),[4.,4.],np.full(42,15.)]
identities={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
    ('input_route.npz','input_affine_controller.npz','input_quotient_design.npz','input_cache_and_quotient_preflight.json')}
summaries=[]
for tape_path in a.tapes:
    target=out/('input_'+Path(tape_path).name)
    shutil.copy2(tape_path,target)
    identities[target.name]=hashlib.sha256(target.read_bytes()).hexdigest()
    tape=np.load(target)
    assert Config(**json.loads(str(tape['config'])))==config
    env=SpatialCartpole(config)
    policy=affine_policy_from_payload(env,source)
    count=len(tape['actions'])-960
    coordinates,raw_reconstructed,base_reconstructed=[],[],[]
    for t,state in enumerate(tape['states'][960:]):
        raw=policy(state,960+t)
        if t<count:
            assert raw.dtype==tape['raw_actions'][960+t].dtype and raw.tobytes()==tape['raw_actions'][960+t].tobytes()
        route=policy.route
        _,_,online=aligned_next_value(env,route.states[t],np.eye(88),state)
        total=np.asarray(policy.metadata['gauge_angles'])+online
        copied=axial_gauge_state(env,state,-total)
        error=env.difference(copied,source['states'][t])
        y=F[t]@error
        coordinates.append(y)
        raw_reconstructed.append(raw)
        base_reconstructed.append(source['actions'][t]+source['feed'][t]-K[t]@y)
    y=np.array(coordinates)
    action=tape['actions'][960:].astype(float)
    raw=np.array(raw_reconstructed)
    delivered_delta=action-source['actions'][:count]
    predicted=np.einsum('tij,tj->ti',A[:count],y[:-1])+np.einsum('tij,tj->ti',B[:count],delivered_delta)+d[:count]
    rho=y[1:]-predicted
    projected=np.einsum('tij,tj->ti',K[1:count+1],rho)
    local_error=np.max(abs(np.einsum('tij,tj->ti',E[:count],y[:-1]))/scales,axis=1)
    local=local_error<=.1
    predicted_raw=source['actions'][1:count+1]+source['feed'][1:count+1]-np.einsum('tij,tj->ti',K[1:count+1],predicted)
    observed_next_raw=raw[1:]
    raw_difference=np.array(base_reconstructed)-raw
    projected_norm=np.linalg.norm(projected,axis=1)
    actual_error=np.linalg.norm(observed_next_raw-predicted_raw,axis=1)
    full_rho=np.einsum('tij,tj->ti',E[1:count+1],rho)
    rho_scaled=np.max(abs(full_rho)/scales,axis=1)
    filename=Path(tape_path).stem
    np.savez_compressed(out/(filename+'_remainder.npz'),y=y,predicted_y_next=predicted,rho=rho,
        projected_future_force_N=projected,full_tangent_error_scaled_max=local_error,local_domain=local,
        actual_raw_force=raw,actual_delivered_force=action,reconstructed_raw_difference=raw_difference,
        predicted_next_raw_force=predicted_raw,actual_next_raw_force=observed_next_raw,
        normalized_remainder_max=rho_scaled,actual_next_raw_prediction_error_N=actual_error,
        delivery_error=action-raw[:-1],config=str(source['config']))
    crossing=np.flatnonzero((projected_norm>.1)&local)
    sample_ids=sorted(set(list(range(0,count,50))+([int(crossing[0])] if len(crossing) else [])))
    samples=[dict(command=960+i,route_command=i,time_seconds=(960+i)*config.dt,
        local_scaled_error=float(local_error[i]),projected_future_force_N=float(projected_norm[i]),
        normalized_remainder=float(rho_scaled[i]),raw_force_norm=float(np.linalg.norm(raw[i])),
        delivered_force_norm=float(np.linalg.norm(action[i])),local_domain=bool(local[i])) for i in sample_ids]
    summaries.append(dict(tape=Path(tape_path).name,tracking_commands=count,raw_forces_byte_exact=True,
        native_steps=env.native_steps,local_samples=int(sum(local)),future_force_threshold_N=.1,
        local_scaled_error_threshold=.1,maximum_projected_future_force_N=float(max(projected_norm)),
        maximum_local_projected_future_force_N=float(max(projected_norm[local])) if np.any(local) else None,
        maximum_local_normalized_remainder=float(max(rho_scaled[local])) if np.any(local) else None,
        maximum_raw_coordinate_reconstruction_difference_N=float(np.max(abs(raw_difference))),
        first_local_threshold_command=int(960+crossing[0]) if len(crossing) else None,samples=samples))
    assert env.native_steps==0
    policy.close();env.close()
write_json(out/'summary.json',dict(n_links=14,inputs_sha256=identities,results=summaries,native_steps=0,
    claim='offline affine-backbone remainder; mixes model/chart/arithmetic effects; not native-selected-law sensitivity'))
write_json(out/'completed.json',dict(status='completed'))
print(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in summaries]),flush=True)
