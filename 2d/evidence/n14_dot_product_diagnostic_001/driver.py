"""Offline comparison of equivalent axial-phase arithmetic on frozen tapes."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import mujoco as mj
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import affine_policy_from_payload
from spatialpole.conditioning import axial_gauge_route
from spatialpole.experiment import freeze_inputs,write_json

p = argparse.ArgumentParser()
p.add_argument('--route',required=True)
p.add_argument('--tapes',nargs='+',required=True)
p.add_argument('--out',required=True)
p.add_argument('--dot-oracle',action='store_true')
a = p.parse_args()
out = Path(a.out)
freeze_inputs(out,a)
source = np.load(out/'input_route.npz')
values = {k:source[k] for k in source.files}
config = Config(**json.loads(str(source['config'])))
assert config==Config(n_links=config.n_links) and bool(source['online_gauge']) and not bool(source['native_search'])
rows,identities,operands = [],[],[]
if a.dot_oracle:
    import mpmath as mp
for tape_path in a.tapes:
    target = out/('input_'+Path(tape_path).name)
    shutil.copy2(tape_path,target)
    identities.append(dict(source=tape_path,copy=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    tape = np.load(target)
    assert Config(**json.loads(str(tape['config'])))==config
    park = int(source['park_steps'])
    env = SpatialCartpole(config)
    policy = affine_policy_from_payload(env,values)
    for step in range(park,park+251,50):
        if step>=len(tape['actions']):
            break
        state = tape['states'][step]
        env.set_state(state)
        original = policy(state,step)
        np.testing.assert_array_equal(original,tape['raw_actions'][step])
        t = step-park
        route = policy.route
        reference,gain = route.states[t],route.gains[t]
        dot_record = {}
        if a.dot_oracle:
            env.set_state(reference)
            mr = env.data.xmat[env.link_ids].reshape(-1,3,3).copy()
            env.set_state(state)
            ma = env.data.xmat[env.link_ids].reshape(-1,3,3)
            relative = np.transpose(mr,(0,2,1))@ma
            phase = np.arctan2(relative[:,1,0]-relative[:,0,1],relative[:,0,0]+relative[:,1,1])
            mrefs,mgains = axial_gauge_route(env,reference[None,:],gain[None,:,:],phase)
            error = env.difference(state,mrefs[0])
            matrix_gain = mgains[0]
            np.testing.assert_array_equal(original,route.actions[t]+route.feed[t]-matrix_gain@error)
            fsum_raw = np.array([route.actions[t,j]+route.feed[t,j]-math.fsum((matrix_gain[j]*error).tolist()) for j in range(2)])
            precise = []
            for digits in (80,120):
                with mp.workdps(digits):
                    dots = [mp.fsum(mp.mpf(float(k))*mp.mpf(float(v)) for k,v in zip(row,error)) for row in matrix_gain]
                    precise.append(np.array([float(mp.mpf(float(route.actions[t,j]))+mp.mpf(float(route.feed[t,j]))-dots[j]) for j in range(2)]))
            np.testing.assert_array_equal(precise[0],precise[1])
            delivered,precise_delivered = env.deliver(original),env.deliver(precise[0])
            dot_record = dict(fsum_raw=fsum_raw.tolist(),mp_raw=precise[0].tolist(),
                mp_maximum_raw_change=float(np.max(abs(original-precise[0]))),
                mp_delivered=precise_delivered.tolist(),
                mp_maximum_delivered_change=float(np.max(abs(delivered.astype(float)-precise_delivered.astype(float)))),
                mp_changed_bits=int(np.count_nonzero(delivered.view(np.uint32)!=precise_delivered.view(np.uint32))),
                mp_changed_values=int(np.count_nonzero(delivered!=precise_delivered)))
            operands.append((error.copy(),matrix_gain.copy(),route.actions[t].copy(),route.feed[t].copy()))
        env.set_state(reference)
        nominal = env.data.xquat[env.link_ids].copy()
        env.set_state(state)
        actual = env.data.xquat[env.link_ids].copy()
        angles = []
        for ref,obs in zip(nominal,actual):
            inverse = ref.copy()
            inverse[1:] *= -1
            r = np.zeros(4)
            mj.mju_mulQuat(r,inverse,obs)
            angles.append(np.arctan2(2*r[0]*r[3],r[0]*r[0]-r[3]*r[3]))
        refs,gains = axial_gauge_route(env,reference[None,:],gain[None,:,:],np.array(angles))
        alternate = route.actions[t]+route.feed[t]-gains[0]@env.difference(state,refs[0])
        np.testing.assert_array_equal(env.state(),state)
        first,second = env.deliver(original),env.deliver(alternate)
        rows.append(dict(tape=target.name,step=step,original_raw=original.tolist(),
            quaternion_raw=alternate.tolist(),maximum_raw_change=float(np.max(abs(original-alternate))),
            original_delivered=first.tolist(),quaternion_delivered=second.tolist(),
            maximum_delivered_change=float(np.max(abs(first.astype(float)-second.astype(float)))),
            changed_delivered_components=int(np.count_nonzero(first.view(np.uint32)!=second.view(np.uint32))),
            baseline_raw_byte_exact=True,**dot_record))
    assert env.native_steps==0
    policy.close()
    env.close()
write_json(out/'tape_identities.json',identities)
if a.dot_oracle:
    np.savez_compressed(out/'dot_operands.npz',errors=np.array([x[0] for x in operands]),
        gains=np.array([x[1] for x in operands]),actions=np.array([x[2] for x in operands]),
        feed=np.array([x[3] for x in operands]))
    oracle = dict(mpmath=mp.__version__,precision_digits=[80,120],final_f64_precision_agreement=True,
        maximum_raw_change=max(r['mp_maximum_raw_change'] for r in rows),
        maximum_delivered_change=max(r['mp_maximum_delivered_change'] for r in rows),
        changed_bits=sum(r['mp_changed_bits'] for r in rows),changed_values=sum(r['mp_changed_values'] for r in rows))
    write_json(out/'dot_oracle.json',oracle)
    print(json.dumps(oracle),flush=True)
write_json(out/'summary.json',dict(n_links=config.n_links,results=rows,native_steps=0,
    samples=len(rows),maximum_raw_change=max(r['maximum_raw_change'] for r in rows),
    maximum_delivered_change=max(r['maximum_delivered_change'] for r in rows),
    changed_delivered_components=sum(r['changed_delivered_components'] for r in rows),
    baseline_all_byte_exact=True,
    claim='offline equivalent-phase arithmetic comparison only; no new actual episode or qualification'))
write_json(out/'completed.json',dict(status='completed'))
print(json.dumps(dict(samples=len(rows),maximum_raw_change=max(r['maximum_raw_change'] for r in rows),
    maximum_delivered_change=max(r['maximum_delivered_change'] for r in rows),
    changed_components=sum(r['changed_delivered_components'] for r in rows))))
