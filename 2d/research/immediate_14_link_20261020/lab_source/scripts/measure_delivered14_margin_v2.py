"""Development-only margin of the unchanged common14 planner's saved policies."""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from spatialpole import Config, SpatialCartpole
from delivered_candidate14 import candidate
from delivered_planner14 import controller
from delivered_reference_bank14 import Profile
from delivered_fresh_gate14 import require_all45
from delivered_fresh_context14 import require_runtime
from audit_spatial_synthesis_models import load
from eleven_synthesis_protocol import sha, read

ROOT=Path(__file__).resolve().parents[1]
HOME=ROOT.parent/'experiments/2d'
STUDY=HOME/'immediate-disturbance-margin'
COUNT=HOME/'fourteen-link-immediate'
CANDIDATE=COUNT/'delivered_planner14_freeze_001'
PROOF=COUNT/'delivered_development14_all45_audit_001'
SEEDS=list(range(20261416,20261436))
SIGMAS=[1e-8,1e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3,1e-2]
MASSES=[1e-13,1e-11,1e-9,1e-7,1e-6,1e-4,1e-3]
CONTRACT='2D14/immediate/frozen-common59a380/full30s/native200Hz/commands50Hz/radial80N/F32/3m/iidXY-preclip/all-commands/dev20261416-435/Firth-log10-sigma/bootstrap2000/mass-both-signs-ge10of20'


def write(path,x):Path(path).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def source(seed,bank,cm,proof):
    row=next(row for row in proof['results'] if row['seed']==seed)
    if row['proof_kind']=='unchanged_exact_source_constructor':
        meta=next(meta for meta in bank if meta['seed']==seed)
        expected=load(meta['actual'])
        return lambda:Profile(Config(n_links=14),meta),expected,dict(kind='preserved_exact_profile',
            actual=meta['actual'],actual_sha256=sha(meta['actual']),source_inventory=meta['inventory'])
    inventory=Path(row['witness_inventory']);receipt=read(inventory)
    assert sha(inventory)==row['witness_inventory_sha256']
    # Historical development receipts bind the parent's digest, not its path.
    prep_path=inventory.with_name(inventory.name.replace('_witness_inventory_','_inventory_'))
    assert sha(prep_path)==receipt['parent_inventory_sha256']
    prep=read(prep_path);run=Path(prep['run'])
    assert prep['passed'] and prep['plan_found'] and sha(run/'summary.json')==prep['summary_sha256']
    assert read(run/'frozen_manifest.json')['controller_identity']==cm['controller_identity']
    report=read(run/'planning/summary.json');actual=Path(receipt['run'])/'actual.npz'
    assert sha(actual)==receipt['actual_sha256']==row['actual_sha256']
    expected=load(actual)
    return lambda:controller(Config(n_links=14),report,run/'planning',bank),expected,dict(
        kind='preserved_observed_state_plan',witness_inventory=str(inventory),
        witness_inventory_sha256=sha(inventory),planning=str(run/'planning'),
        actual=str(actual),actual_sha256=sha(actual),mode=report['mode'])


def predictor_steps(policy):
    if hasattr(policy,'predictor_native_steps'):return policy.predictor_native_steps
    return policy.predictor.native_steps if hasattr(policy,'predictor') else 0


def episode(factory,expected,seed,sigma,delta,out):
    nominal=SpatialCartpole(Config(n_links=14));cfg=Config(n_links=14,total_mass=1.+delta)
    plant,cold=SpatialCartpole(cfg),SpatialCartpole(cfg);law=None
    X=[];U=[];raw=[];noisy=[];tilts=[];exception=None;begun=time.monotonic()
    try:
        initial,_=plant.reset(seed=seed,options=dict(mode='hanging',angle_noise=.02))
        assert initial.tobytes()==expected['states'][0].tobytes()
        X.append(initial);law=factory()
        z=np.random.default_rng(np.random.SeedSequence([seed,0x51A7,2])).standard_normal((1500,2))
        for tick in range(1500):
            protected=(plant.state().tobytes(),plant.data.time,plant.native_steps)
            u=law(X[-1],tick)
            assert protected==(plant.state().tobytes(),plant.data.time,plant.native_steps)
            disturbed=u if sigma==0 else u+sigma*z[tick];force=plant.deliver(disturbed)
            x,_,done,truncated,info=plant.step_force(force)
            X.append(x);U.append(force);raw.append(u);noisy.append(disturbed);tilts.append(plant.tilts())
            if done or truncated:break
    except Exception as error:
        exception=dict(type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    replay=True
    if X:
        x=X[0].copy()
        for j,u in enumerate(U):
            x=cold.transition(x,u)
            if x.tobytes()!=X[j+1].tobytes():replay=False;break
    arrays=dict(states=np.array(X).reshape(-1,102),actions=np.array(U,dtype=np.float32).reshape(-1,2),
        raw_actions=np.array(raw).reshape(-1,2),noisy_raw_actions=np.array(noisy).reshape(-1,2),
        tilts=np.array(tilts).reshape(-1,14),plant_config=np.array(json.dumps(cfg.__dict__)),
        compiled_body_mass=plant.model.body_mass.copy(),compiled_body_inertia=plant.model.body_inertia.copy())
    arrays['noise_added']=arrays['noisy_raw_actions']-arrays['raw_actions']
    name=f'seed{seed}_s{sigma:.0e}_m{delta:+.0e}'.replace('+','p').replace('-','n')
    tape=out/'tapes'/f'{name}.npz';np.savez_compressed(tape,**arrays)
    info=plant.info();steps=len(U);predictor=predictor_steps(law) if law is not None else 0
    native=dict(plant=plant.native_steps,cold=cold.native_steps,predictor=predictor,nominal=nominal.native_steps)
    parity={key:int(np.sum(np.any(arrays[key]!=expected[key][:len(arrays[key])],axis=1)))
            for key in ('states','actions','raw_actions')}
    exact=bool(steps==1500 and all(arrays[k].tobytes()==expected[k].tobytes() for k in ('states','actions','raw_actions','tilts')))
    success=bool(exception is None and steps==1500 and info['success'] and info['best_hold_seconds']>=5
                 and info['plane_margin']>=0 and not info['physics_warning'] and replay)
    cause=None if success else ('exception' if exception else 'plane_exit' if info['plane_margin']<0
          else 'physics_warning' if info['physics_warning'] else 'replay_mismatch' if not replay else 'upright_hold_short')
    row=dict(n=14,seed=seed,mode='bypass',sigma=sigma,mass_delta=delta,success=success,cause=cause,steps=steps,
        best_hold_seconds=info['best_hold_seconds'],plane_margin=info['plane_margin'],physics_warning=info['physics_warning'],
        exception=exception,replay_exact=replay,reproduces_recorded_actual=exact if sigma==delta==0 else None,
        guard_abort=None,parity_counters=parity,native=native,inclusive_native_steps=sum(native.values()),
        max_force_norm=float(np.linalg.norm(arrays['actions'].astype(float),axis=1).max()) if steps else None,
        tape=tape.name,tape_sha256=sha(tape),wall_seconds=time.monotonic()-begun)
    if law is not None:law.close()
    nominal.close();plant.close();cold.close()
    assert row['inclusive_native_steps']<=84000 and predictor<=72000
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',required=True,type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--smoke-proof',type=Path);p.add_argument('--inherit-smoke',type=Path);a=p.parse_args();out=a.out.resolve();out.relative_to(STUDY)
    assert out.is_dir() and not (out/'inputs.json').exists()
    admission=require_all45(PROOF,CANDIDATE);bank,runtime,cm=candidate(CANDIDATE)
    require_runtime(read(CANDIDATE/'inputs.json')['runtime'])
    if not a.smoke:
        assert a.smoke_proof is not None
        previous=read(a.smoke_proof/'summary.json');assert previous['passed'] and previous['smoke']
        assert previous['all_smoke_reproduce_recorded_actual'] and previous['controller_identity']==cm['controller_identity']
        assert previous['driver_sha256']==sha(__file__)
    (out/'tapes').mkdir();begun=time.monotonic();budget=252000 if a.smoke else 40320000
    proof=read(PROOF/'summary.json');rows=[];inherited_native=0;inherited_sources=[]
    if a.inherit_smoke:
        assert a.smoke
        old=a.inherit_smoke.resolve();old_record=read(old/'exploration.json')
        assert old_record['status']=='failed' and old_record['returncode']==1
        assert "KeyError: 'parent_inventory'" in (old/'output.log').read_text()
        inherited=[json.loads(line) for line in (old/'rows.jsonl').read_text().splitlines()]
        assert [row['seed'] for row in inherited]==SEEDS[:2]
        for row in inherited:
            assert row['success'] and row['reproduces_recorded_actual'] and row['sigma']==row['mass_delta']==0
            assert row['inclusive_native_steps']<=84000
            tape=old/'tapes'/row['tape'];assert sha(tape)==row['tape_sha256']
            _,original,_=source(row['seed'],bank,cm,proof);saved=load(tape)
            for key in ('states','actions','raw_actions','tilts'):assert saved[key].tobytes()==original[key].tobytes()
            shutil.copyfile(tape,out/'tapes'/row['tape']);rows.append(row);inherited_native+=row['inclusive_native_steps']
            inherited_sources.append(dict(seed=row['seed'],source_run=str(old),row=row,tape_sha256=sha(tape)))
            with (out/'rows.jsonl').open('a') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n')
        write(out/'inherited_smoke_proof.json',dict(source_rows_sha256=sha(old/'rows.jsonl'),
            source_receipt_sha256=sha(old/'exploration.json'),source_driver_sha256=read(old/'inputs.json')['driver_sha256'],
            native_steps=inherited_native,rows=inherited_sources,new_physics=0))
    write(out/'inputs.json',dict(protocol_sha256=sha(ROOT/'research/delivered14_margin_measurement.md'),
        inherited_smoke=str(a.inherit_smoke.resolve()) if a.inherit_smoke else None,
        inherited_smoke_native_steps=inherited_native,driver_sha256=sha(__file__),candidate_manifest_sha256=sha(CANDIDATE/'frozen_manifest.json'),
        development_admission=admission,smoke=a.smoke,seeds=SEEDS[:3] if a.smoke else SEEDS,
        sigmas=SIGMAS,mass_magnitudes=MASSES,native_cap=budget,new_planning_native_steps=0,
        new_fresh_labels=0,original_runtime=read(CANDIDATE/'inputs.json')['runtime']))
    for seed in (SEEDS[:3] if a.smoke else SEEDS):
        if any(row['seed']==seed for row in rows):continue
        factory,expected,provenance=source(seed,bank,cm,proof)
        write(out/f'source_{seed}.json',provenance)
        conditions=[(0.,0.)] if a.smoke else [(s,0.) for s in SIGMAS]+[(0.,sign*d) for d in MASSES for sign in (-1,1)]
        for sigma,delta in conditions:
            assert sum(r['inclusive_native_steps'] for r in rows)+84000<=budget
            row=episode(factory,expected,seed,sigma,delta,out);rows.append(row)
            with (out/'rows.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            write(out/'progress.json',dict(episodes=len(rows),seed=seed,sigma=sigma,mass_delta=delta,
                native_steps=sum(r['inclusive_native_steps'] for r in rows),latest_success=row['success']))
            if a.smoke:assert row['success'] and row['reproduces_recorded_actual'],'Smoke mismatch; preserve failure, no sweep'
    summary=dict(passed=True,smoke=a.smoke,controller_identity=cm['controller_identity'],driver_sha256=sha(__file__),
        episodes=len(rows),successes=sum(r['success'] for r in rows),inclusive_native_steps=sum(r['inclusive_native_steps'] for r in rows),
        new_planning_native_steps=0,new_fresh_labels=0,inherited_native_steps=inherited_native,
        newly_executed_native_steps=sum(r['inclusive_native_steps'] for r in rows)-inherited_native,wall_seconds=time.monotonic()-begun,
        all_smoke_reproduce_recorded_actual=all(r['reproduces_recorded_actual'] for r in rows) if a.smoke else None,
        sigma50_N=None,mass_tolerance=None,qualified=False,
        scope='development diagnostic; frozen estimator analysis required for margins, fresh32 required forTierE')
    write(out/'summary.json',summary)
    write(out/'task_result.json',dict(metric='measured_force_noise_sigma_N',metric_contract=CONTRACT,
        task_valid=False,value=0.,reason='smoke adapter or raw grid; complete frozen-estimator analysis required'))
    print(summary,flush=True)


if __name__=='__main__':main()
