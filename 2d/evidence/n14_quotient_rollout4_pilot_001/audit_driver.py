"""Audit selected first successors and full candidate native lookahead paths."""
import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.affine_value import axial_aligned_affine_force
from spatialpole.aligned_value import aligned_next_value
from spatialpole.conditioning import axial_gauge_route,launch_gauge_angles
from spatialpole.rollout_value import rollout_value_policy_from_payload
from spatialpole.experiment import write_json,episode

p=argparse.ArgumentParser()
p.add_argument('--run',required=True)
a=p.parse_args()
out=Path(a.run)
assert (out/'completed.json').is_file(),'audit requires terminal pilot'
shutil.copy2(__file__,out/'audit_driver.py')
shutil.copy2('research/plan.md',out/'audit_plan.md')
summary=json.loads((out/'summary.json').read_text())
assert len(summary['results'])==8
for path in (out/'source').glob('*.py'):
    assert hashlib.sha256(path.read_bytes()).digest()==hashlib.sha256((Path('src/spatialpole')/path.name).read_bytes()).digest(),path
payload=np.load(out/'route.npz')
config=Config(**json.loads(str(payload['config'])))
assert config==Config(n_links=14) and int(payload['park_steps'])==960 and int(payload['native_rollout_horizon'])==4
oracle=SpatialCartpole(config)
identities,rows={},[]
samples=(960,1060,1160,1260,1360,1460)
for row in summary['results']:
    path=out/('nominal.npz' if row['angle_noise']==0 else f"seed{row['seed']}.npz")
    identities[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as archive:
        tape={k:archive[k] for k in archive.files}
    metadata=json.loads(str(tape['policy_metadata']))
    angles=launch_gauge_angles(oracle,payload['states'][0],tape['states'][960])
    np.testing.assert_array_equal(angles,metadata['gauge_angles'])
    refs,gains=axial_gauge_route(oracle,payload['states'],payload['gains'],angles)
    _,roots=axial_gauge_route(oracle,payload['states'],payload['roots'],angles)
    _,factors=axial_gauge_route(oracle,payload['states'],payload['tracking_stage_factors'],angles)
    def get(k):
        return tape['native_rollout_'+k]
    steps,offsets=get('step'),get('offsets')
    assert len(steps)==len(tape['actions'])-960
    assert sum(get('native_steps'))==row['policy_metadata']['predictor_native_steps']
    assert np.all(get('native_steps')==get('used_steps')*4)
    assert np.all((get('used_steps')>=1)&(get('used_steps')<=4))
    for command,t in enumerate(steps):
        actual=960+int(t)
        lo,hi=map(int,offsets[command:command+2])
        actions=get('actions')[lo:hi]
        paths=get('states')[lo:hi]
        assert get('horizon')[command]==4 and 1<=hi-lo<=16
        assert len({u[0].tobytes() for u in actions})==hi-lo
        np.testing.assert_array_equal(paths[:,0],np.repeat(tape['states'][actual][None,:],hi-lo,axis=0))
        baseline=axial_aligned_affine_force(oracle,tape['states'][actual],refs[t],payload['actions'][t],payload['feed'][t],gains[t])
        np.testing.assert_array_equal(baseline,get('raw')[lo,0])
        np.testing.assert_array_equal(oracle.deliver(baseline),actions[0,0])
        baseline_complete=bool(get('complete')[lo])
        anchors=[]
        if baseline_complete:
            for j in range(1,5):
                f=roots[t+j] if j==4 else factors[t+j]
                ref,factor,angle=aligned_next_value(oracle,refs[t+j],f,paths[0,j])
                np.testing.assert_array_equal(angle,get('anchor_angles')[command,j-1])
                residual=payload['residuals'][t+j] if j==4 else np.zeros(len(factor))
                e0=oracle.difference(paths[0,j],ref)
                anchors.append((ref,factor,residual,e0))
        for k in range(lo,hi):
            used=int(get('used_steps')[k])
            force=get('actions')[k,:used]
            assert force.dtype==np.float32 and np.isfinite(force).all()
            assert np.max(np.linalg.norm(force.astype(float),axis=1))<=80
            for j in range(used):
                np.testing.assert_array_equal(oracle.deliver(get('raw')[k,j]),force[j])
                physical_valid=bool(np.isfinite(get('states')[k,j+1]).all() and not get('physics_warnings')[k,j] and get('plane_margins')[k,j]>=0)
                assert physical_valid==bool(get('valid_steps')[k,j])
            if bool(get('valid')[k]):
                assert used==4 and bool(get('complete')[k]) and np.all(get('valid_steps')[k]) and baseline_complete
                state_diffs,input_diffs=[],[]
                for j,(ref,factor,residual,e0) in enumerate(anchors):
                    e=oracle.difference(get('states')[k,j+1],ref)
                    base=factor@e0+residual
                    change=factor@(e-e0)
                    state_diffs.append(math.fsum(((2*base+change)*change).tolist()))
                    u0=get('actions')[lo,j].astype(float)-payload['actions'][t+j]
                    u1=get('actions')[k,j].astype(float)-payload['actions'][t+j]
                    input_diffs.append(float((u1-u0)@payload['tracking_R']@(u1+u0)))
                np.testing.assert_array_equal(state_diffs,get('state_cost_differences')[k])
                np.testing.assert_array_equal(input_diffs,get('input_cost_differences')[k])
                assert math.fsum(state_diffs+input_diffs)==get('cost')[k]
            if actual in samples:
                x=tape['states'][actual].copy()
                for j in range(used):
                    if j:
                        raw=axial_aligned_affine_force(oracle,x,refs[t+j],payload['actions'][t+j],payload['feed'][t+j],gains[t+j])
                        np.testing.assert_array_equal(raw,get('raw')[k,j])
                    oracle.min_plane_margin,oracle.warning_seen=config.plane_limit,False
                    x=oracle.transition(x,force[j])
                    np.testing.assert_array_equal(x,get('states')[k,j+1])
                    assert oracle.min_plane_margin==get('plane_margins')[k,j]
                    assert oracle.warning_seen==get('physics_warnings')[k,j]
        selected=int(get('selected_index')[command])
        fallback=bool(get('baseline_fallback')[command])
        assert fallback==(selected==-1)
        if fallback:
            assert not np.any(get('valid')[lo:hi])
            chosen=lo
        else:
            valid=np.flatnonzero(get('valid')[lo:hi])+lo
            assert len(valid)>0
            chosen=lo+selected
            assert chosen==valid[np.argmin(get('cost')[valid])]
        np.testing.assert_array_equal(get('selected')[command],get('actions')[chosen,0])
        np.testing.assert_array_equal(tape['actions'][actual],get('actions')[chosen,0])
        np.testing.assert_array_equal(tape['states'][actual+1],get('states')[chosen,1])
    rows.append(dict(seed=row['seed'],commands=len(steps),distinct_first_forces=int(offsets[-1]),
        all_selected_first_successors_exact=True,final_actual_successor_exact=True,
        candidate_accounting_and_scores_exact=True,spot_native_paths_and_future_feedback_exact=True))
native_spot_steps=oracle.native_steps
assert native_spot_steps<=12288
oracle.close()
reexecution=episode(config,lambda e:rollout_value_policy_from_payload(e,payload),20261016,'hanging',0.,out/'nominal_reexecution.npz')
old,new=np.load(out/'nominal.npz'),np.load(out/'nominal_reexecution.npz')
parity={k:bool(old[k].dtype==new[k].dtype and old[k].shape==new[k].shape and old[k].tobytes()==new[k].tobytes()) for k in old.files if k not in ('config','policy_metadata')}
assert all(parity.values()),parity
repeat_steps=reexecution['native_steps']+reexecution['proof_native_steps']+reexecution['policy_metadata']['predictor_native_steps']
assert repeat_steps<=150240
total=summary['inclusive_native_steps']+native_spot_steps+repeat_steps
assert total<=1450000
write_json(out/'native_integrity_audit.json',dict(input_tape_sha256=identities,episodes=rows,
    nominal_reexecution=reexecution,nominal_reexecution_byte_parity=parity,native_spot_steps=native_spot_steps,
    reexecution_inclusive_native_steps=repeat_steps,pilot_inclusive_native_steps=summary['inclusive_native_steps'],
    total_inclusive_native_steps=total,scope='reused integrity only; only FIRST forecast successor is actual-parity claim'))
print(json.dumps(dict(audit='passed',total_inclusive_native_steps=total)),flush=True)
