"""Zero-native union of preserved constructors and contiguous canonical proofs."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config
from spatialpole.experiment import freeze_inputs,write_json
from spatialpole.native_mpc import exact_bytes
from delivered_candidate14 import candidate
from audit_spatial_synthesis_models import load
from stage_six_source_synthesis14 import helper_closure
from eleven_synthesis_protocol import read,sha


ROOT=Path(__file__).resolve().parents[1]
SOURCE_SEEDS=[20261016]+list(range(20261116,20261128))+[20261416,20261417]


def studies(receipt,checkpoint):
    for folder,files in receipt['studies'].items():
        for name,item in files.items():
            path=Path(folder)/name
            assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],str(path)
        checkpoint()


def required_metric(result,key):
    nested=result.get('policy_metadata',{})
    if key in result:
        if key in nested:assert result[key]==nested[key]
        return result[key]
    return nested[key]  # Missing flat AND nested evidence is an error, never zero.


def score(tape,result,seed):
    config=Config(n_links=14)
    assert json.loads(str(tape['config']))==config.__dict__
    for key,shape,dtype in (('states',(1501,102),np.float64),('actions',(1500,2),np.float32),
                           ('raw_actions',(1500,2),np.float64),('tilts',(1500,14),np.float64)):
        assert tape[key].shape==shape and tape[key].dtype==dtype and np.isfinite(tape[key]).all()
    assert result['seed']==seed and result['initialization']=='hanging' and result['angle_noise']==(0. if seed==20261016 else .02)
    assert result['steps']==1500 and result['success'] and result['replay_exact'] and required_metric(result,'park_seconds')==0.
    assert required_metric(result,'physical_velocities')==44 and result['plane_margin']>=0. and not result['physics_warning']
    peak=float(np.linalg.norm(tape['actions'].astype(float),axis=1).max())
    stored_precision_peak=float(np.linalg.norm(tape['actions'],axis=1).max())
    assert peak<=80. and result['max_force_norm'] in (peak,stored_precision_peak)
    arithmetic='float64_norm_of_delivered_float32' if result['max_force_norm']==peak else 'legacy_float32_norm'
    assert float(3.-np.max(abs(tape['states'][:,:2])))>=0.
    streak=best=0
    for upright in np.max(tape['tilts'],axis=1)<.15:
        streak=streak+1 if upright else 0;best=max(best,streak)
    assert best*.02==result['best_hold_seconds'] and best*.02>=5.
    return dict(seed=seed,commands=1500,seconds=30.,best_hold_seconds=best*.02,
        minimum_native_substep_margin=result['plane_margin'],peak_delivered_force=peak,
        source_reported_peak=result['max_force_norm'],source_peak_arithmetic=arithmetic,
        initial_state_sha256=__import__('hashlib').sha256(tape['states'][0].tobytes()).hexdigest())


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--out',required=True)
    p.add_argument('--constructor-inventory',default=str(ROOT/'output/fourteen_delivered_constructor_inventory_001.json'))
    p.add_argument('--through-seed',type=int,default=20261447);p.add_argument('--require-all45',action='store_true');a=p.parse_args()
    assert 20261417<=a.through_seed<=20261447
    if a.require_all45:assert a.through_seed==20261447
    out=Path(a.out);freeze_inputs(out,a);started=time.monotonic()
    def checkpoint():assert time.monotonic()-started<2400.
    try:
        bank,runtime,cm=candidate(a.candidate);ci=Path(a.constructor_inventory);construction=read(ci)
        assert construction['passed'] and not construction['qualified'] and construction['new_native_steps']==0
        assert construction['controller_identity']==cm['controller_identity'] and construction['candidate_manifest_sha256']==sha(Path(a.candidate)/'frozen_manifest.json')
        assert construction['known_source_proofs_preserved']==construction['common_exact_initial_dispatches']==15
        assert [meta['seed'] for meta in bank]==SOURCE_SEEDS
        studies(construction,checkpoint)
        preserved=read(Path(construction['cached_proof'])/'summary.json');bp=read(Path(construction['bank'])/'summary.json')
        assert preserved['controller_identity']==cm['controller_identity'] and preserved['native_steps']==0
        assert [x['seed'] for x in preserved['profiles']]==SOURCE_SEEDS and all(x['exact_common_dispatch'] and x['physical_source_proof_inherited'] for x in preserved['profiles'])
        assert bp['old14_metadata_and_implementation_exact'] and bp['old14_physical_success_proofs_inherited_without_rerun']
        assert bp['added17_all1500_controls_and16473_query_arrays_byte_exact']
        for name in helper_closure({'audit_delivered_development14.py'}):shutil.copyfile(Path(__file__).with_name(name),out/name)
        write_json(out/'frozen_manifest.json',dict(protocol='common-delivered14-development-proof-union-v1',config=Config(n_links=14).__dict__,
            controller_identity=cm['controller_identity'],candidate=str(Path(a.candidate).resolve()),candidate_manifest_sha256=sha(Path(a.candidate)/'frozen_manifest.json'),
            constructor_inventory=str(ci.resolve()),constructor_inventory_sha256=sha(ci),through_seed=a.through_seed,require_all45=a.require_all45,
            max_native_steps=0,max_wall_seconds=2400,source_sha256=cm['source_sha256'],
            helper_sha256={name:sha(out/name) for name in helper_closure({'audit_delivered_development14.py'})}))
        rows=[]
        for meta in bank:
            tape=load(meta['actual']);exact_bytes(tape['states'][0],np.array(meta['initial_state'],np.float64))
            row=score(tape,meta['result'],meta['seed'])
            row.update(proof_kind='unchanged_exact_source_constructor',actual_sha256=sha(Path(meta['actual'])),
                source_inventory=meta['inventory'],source_inventory_sha256=sha(Path(meta['inventory'])))
            rows.append(row);checkpoint()
        parent=ci;ledger=construction['total_new_target_preparation_native_steps'];bounds=construction['total_new_target_preparation_native_steps_bounds']
        bindings=[]
        for seed in range(20261418,a.through_seed+1):
            short=f'6{seed % 100:02d}';prep_path=ROOT/f'output/fourteen_delivered_case{short}_inventory_001.json'
            witness_path=ROOT/f'output/fourteen_delivered_case{short}_witness_inventory_001.json'
            prep,witness=read(prep_path),read(witness_path)
            assert prep['passed'] and prep['plan_found'] and not prep['qualified'] and not prep['task_success']
            assert witness['passed'] and witness['task_success'] and not witness['qualified']
            assert prep['seed']==witness['seed']==seed and prep['controller_identity']==witness['controller_identity']==cm['controller_identity']
            assert prep['parent_inventory_sha256']==sha(parent) and witness['parent_inventory_sha256']==sha(prep_path)
            assert prep['actor_native_steps']<=9072324 and prep['audit_native_steps']<=384324
            assert prep['new_native_steps']==prep['actor_native_steps']+prep['audit_native_steps']
            assert witness['actor_native_steps']==witness['audit_native_steps']==12000 and witness['new_native_steps']==24000
            assert prep['total_new_target_preparation_native_steps']==ledger+prep['new_native_steps']
            assert prep['total_new_target_preparation_native_steps_bounds']==[x+prep['new_native_steps'] for x in bounds]
            assert witness['total_new_target_preparation_native_steps']==prep['total_new_target_preparation_native_steps']+24000
            assert witness['total_new_target_preparation_native_steps_bounds']==[x+24000 for x in prep['total_new_target_preparation_native_steps_bounds']]
            studies(prep,checkpoint);studies(witness,checkpoint)
            pr,pa=Path(prep['run']),Path(prep['audit']);wr,wa=Path(witness['run']),Path(witness['audit'])
            ps,proof=read(pr/'summary.json'),read(pa/'summary.json');ws,wp=read(wr/'summary.json'),read(wa/'summary.json')
            assert proof['passed'] and proof['plan_found'] and proof['source_summary_sha256']==sha(pr/'summary.json') and proof['source_manifest_sha256']==sha(pr/'frozen_manifest.json')
            assert ps['controller_identity']==ws['controller_identity']==wp['controller_identity']==cm['controller_identity']
            assert wp['canonical_reset_actual_controls_states_scores_cache_and_cold_independently_byte_exact']
            assert wp['source_summary_sha256']==sha(wr/'summary.json') and wp['source_actual_sha256']==sha(wr/'actual.npz')==witness['actual_sha256']
            assert read(wr/'frozen_manifest.json')['inventory_sha256']==sha(prep_path)
            assert sha(wr/'summary.json')==witness['summary_sha256'] and sha(wa/'summary.json')==witness['audit_summary_sha256']
            assert sha(pr/'summary.json')==prep['summary_sha256'] and sha(pa/'summary.json')==prep['audit_summary_sha256']
            tape,replica=load(wr/'actual.npz'),load(wa/'actual.npz');assert tape.keys()==replica.keys()
            for key in tape:exact_bytes(tape[key],replica[key])
            for key,value in ws.items():assert wp[key]==value
            exact_bytes(tape['states'][0],load(pr/'observed_state.npz')['state'])
            assert witness['result']==ws['result']
            row=score(tape,ws['result'],seed)
            row.update(proof_kind='new_common_plan_canonical_and_independent_reproduction',witness_inventory=str(witness_path),witness_inventory_sha256=sha(witness_path),
                actual_sha256=sha(wr/'actual.npz'),canonical_and_reproduction_arrays_byte_exact=True)
            rows.append(row);assert witness['common_development_states_with_constructor_or_physical_proofs']==len(rows)
            bindings.append(dict(seed=seed,preparation_inventory_sha256=sha(prep_path),witness_inventory_sha256=sha(witness_path)))
            parent=witness_path;ledger=witness['total_new_target_preparation_native_steps'];bounds=witness['total_new_target_preparation_native_steps_bounds']
            write_json(out/'progress.json',dict(verified_development_states=len(rows),native_steps=0));checkpoint()
        expected=[20261016]+list(range(20261116,20261128))+list(range(20261416,a.through_seed+1))
        assert [row['seed'] for row in rows]==expected and len(set(expected))==len(expected)
        all45=bool(expected==cm['registered_development_seeds'] and len(rows)==45 and a.through_seed==20261447)
        if a.require_all45:assert all45
        report=dict(passed=True,links=14,qualified=False,task_success=False,controller_identity=cm['controller_identity'],
            development_prefix_passed=True,verified_development_states=len(rows),all45_development_passed=all45,
            canonical_and_reproduction_proofs_joined=True,exact_source_constructor_proofs_inherited=15,results=rows,bindings=bindings,
            source_manifest_sha256=sha(out/'frozen_manifest.json'),native_steps=0,new_native_steps=0,new_solver_calls=0,new_controller_callbacks=0,
            total_new_target_preparation_native_steps=ledger,total_new_target_preparation_native_steps_bounds=bounds,
            original_reset_episodes=0,new_noisy_samples=0,reserved_cases_consumed=0,fresh_qualification_activated=False,
            fresh_gate_satisfied=all45,remaining_development_states=45-len(rows),
            scope='hash/identity/score/constructor-inheritance union of original task proofs; no physical rerun, new integration, fresh qualification or accepted-count claim')
        write_json(out/'summary.json',report);write_json(out/'completed.json',dict(status='completed'))
        print({key:value for key,value in report.items() if key not in ('results','bindings')},flush=True)
    except BaseException as exc:
        write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=0,accounting_complete=True));raise


if __name__=='__main__':main()
