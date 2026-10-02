#!/usr/bin/env python
"""Repeat the fixed thirteen-link procedure on one adjacent canonical count.

This development runner never consumes reserved seeds, freezes a release,
promotes a count or publishes. Failure is retained for subsequent diagnosis.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from gcartpole.config import dump_json, load_config
from gcartpole.evidence import file_metadata, runtime_metadata, utc_timestamp
from gcartpole.roadmap import validate_canonical_config
from scripts.synthesize_inverse_increment import exact_candidate_passed
from scripts.verify_frontier_release import check_episode, verify_manifest

ROOT = Path(__file__).resolve().parents[1]
HANDOFF_SECONDS = (8., 9., 10., 12.)
FIRST_PARAMETERS = dict(iterations=20, lqr_decimal_digits=100, qp_solver='osqp',
    qp_max_iterations=10000, qp_initial_tolerance=.001, qp_tolerance=1e-8,
    qp_inexact_dual_tolerance=.001, trust_policy='legacy', state_trust=.05,
    control_trust=.1, initial_regularization=10., defect_penalty=1e5)


def physical_replay_passed(result, cfg):
    """Judge executed physics; virtual-node feasibility is checked after rebuilding."""
    env = cfg['env']
    steps = round(env['episode_seconds']/(env['timestep']*env['frame_skip']))
    metrics = [result.get('max_cart_excursion', np.inf),
               result.get('max_upright_streak_seconds', -np.inf)]
    return bool(result.get('success') is True and result.get('trajectory_integrity') is True
        and result.get('termination_reason') == 'time_limit' and result.get('length') == steps
        and result.get('final_info', {}).get('simulation_error') is None
        and np.all(np.isfinite(metrics)) and metrics[0] <= env['rail_limit']
        and metrics[1] >= env['success_sustain_seconds'])


def validate_development_seeds(count, seed):
    cohorts = [set([seed]), set(range(seed+100, seed+120)), set(range(seed+200, seed+300))]
    base = 200000+1000*count
    reserved = set(range(base, base+20)) | set(range(base+100, base+220)) | {base+500}
    if any(cohort & reserved for cohort in cohorts):
        raise ValueError('development cohorts intersect reserved canonical seeds')
    if any(cohorts[i] & cohorts[j] for i in range(3) for j in range(i)):
        raise ValueError('development cohorts overlap')


def validate_adopted_repair(path, cfg_path, controller_path, seed):
    """An adopted first stage must be the same declared transfer and numerical recipe."""
    data = json.loads(path.read_text())
    recipe = data['shared_capture_recipe']
    params = recipe['parameters']
    if any(params.get(k) != v for k, v in FIRST_PARAMETERS.items()) or params['seed'] != seed:
        raise ValueError('adopted repair differs from the fixed first-stage recipe')
    if file_metadata(ROOT/params['config'])['sha256'] != file_metadata(cfg_path)['sha256']:
        raise ValueError('adopted repair config differs')
    init_path = ROOT/recipe['initialization']['path']
    if file_metadata(init_path)['sha256'] != recipe['initialization']['sha256']:
        raise ValueError('adopted repair initializer changed')
    init = json.loads(init_path.read_text())
    if init['source']['sha256'] != file_metadata(controller_path)['sha256']:
        raise ValueError('adopted repair came from a different predecessor')
    if init['config']['sha256'] != file_metadata(cfg_path)['sha256']:
        raise ValueError('adopted initializer target differs')
    if init['controller'].get('virtual_tail_seconds') != 5.:
        raise ValueError('adopted initializer tail differs')
    execution = json.loads((path.parent/'execution.json').read_text())
    if execution['status'] != 'finished' or execution['returncode'] != 0:
        raise ValueError('adopted repair execution did not finish cleanly')
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest', required=True)
    p.add_argument('--config', required=True)
    p.add_argument('--directory', required=True)
    p.add_argument('--seed', type=int)
    p.add_argument('--adopt-first-repair', help='Explicitly reuse a recorded identical completed twenty-iteration stage; disclosed in the journal.')
    args = p.parse_args()
    cfg_path, manifest_path = Path(args.config), Path(args.source_manifest)
    cfg = load_config(cfg_path)
    n = int(cfg['env']['n_links'])
    if n not in range(12,21):
        raise ValueError('target count must be twelve through twenty')
    reference = copy.deepcopy(cfg); reference['env']['n_links'] = 7
    errors = validate_canonical_config(reference) + verify_manifest(manifest_path)
    if errors:
        raise ValueError('canonical source/config verification failed: '+repr(errors))
    manifest = json.loads(manifest_path.read_text())
    policy_path = Path(manifest['policy']['path'])
    policy = json.loads(policy_path.read_text())
    controller_path, source_cfg_path = Path(policy['controller']['path']), Path(policy['config']['path'])
    if manifest['n_links'] != n-1:
        raise ValueError('source must be the accepted adjacent predecessor')
    seed = args.seed if args.seed is not None else 20250000+1000*n+100
    validate_development_seeds(n,seed)
    adopted_path = Path(args.adopt_first_repair) if args.adopt_first_repair else None
    if adopted_path:
        validate_adopted_repair(adopted_path,cfg_path,controller_path,seed)
    directory = Path(args.directory)
    directory.mkdir(parents=True,exist_ok=False)
    journal = directory/'development_pipeline.json'
    record = dict(schema_version=1,started_at=utc_timestamp(),status='running',n_links=n,
        not_solution=True,release_promotion=False,reserved_seeds_used=False,
        source_manifest=file_metadata(manifest_path),config=file_metadata(cfg_path),
        runner=file_metadata(Path(__file__).resolve()),runtime=runtime_metadata(),
        seed=seed,handoff_screen_seconds=list(HANDOFF_SECONDS),stages=[],
        adopted_first_repair=file_metadata(adopted_path) if adopted_path else None,
        protocol='Fixed thirteen-link 20 OSQP / 40 Clarabel agreement / earliest passing declared handoff / physical rebuild / unchanged launch / noisy20 and100.')

    def save():
        dump_json(record,journal)

    def stage(label, script, values, inputs):
        out = directory/label
        command = [sys.executable,'scripts/'+script]
        for key,value in values.items():
            command.extend([key,str(value)])
        command.extend(['--out',str(out/'result.json')] if script in ('transfer_physical_capture_route.py','evaluate_fddp_parked_route.py') else
                       ['--out-directory',str(out)] if script == 'synthesize_frontier_capture.py' else ['--directory',str(out)])
        row = dict(label=label,status='running',command=command,inputs=[file_metadata(x) for x in inputs])
        record['stages'].append(row);save()
        run = [sys.executable,'scripts/run_frontier_experiment.py','--directory',str(out)]
        for x in inputs:
            run.extend(['--input',str(x)])
        run.extend(['--',*command])
        try:
            subprocess.run(run,cwd=ROOT,check=True)
        except BaseException as error:
            row.update(status='failed_or_interrupted',error=f'{type(error).__name__}: {error}')
            if (out/'execution.json').is_file():
                row['execution']=file_metadata(out/'execution.json')
            save()
            raise
        row.update(status='finished',execution=file_metadata(out/'execution.json'),result=file_metadata(out/'result.json'))
        save()
        return out/'result.json',json.loads((out/'result.json').read_text())

    save()
    try:
        if adopted_path:
            first_path = adopted_path
        else:
            init_path,_ = stage('01_physical_transfer','transfer_physical_capture_route.py',
                {'--source-controller':controller_path,'--source-config':source_cfg_path,'--config':cfg_path,'--capture-tail-seconds':5},
                [controller_path,source_cfg_path,cfg_path])
            first_path,_ = stage('02_scvx20','synthesize_frontier_capture.py',
                {'--config':cfg_path,'--initializer':init_path,'--iterations':20,'--lqr-decimal-digits':100,'--seed':seed},[cfg_path,init_path])
        refined_path,refined = stage('03_agreement_scvx40','synthesize_frontier_capture.py',
            {'--config':cfg_path,'--initializer':first_path,'--iterations':40,'--lqr-decimal-digits':100,'--seed':seed,
             '--qp-solver':'clarabel','--qp-max-iterations':200,'--qp-initial-tolerance':1e-5,'--trust-policy':'agreement'},[cfg_path,first_path])
        screen = []
        for seconds in HANDOFF_SECONDS:
            path,data = stage(f'04_handoff{seconds:g}','replay_frontier_handoff.py',
                {'--source':refined_path,'--config':cfg_path,'--handoff-seconds':seconds},[refined_path,cfg_path])
            screen.append((seconds,path,physical_replay_passed(data['result'],cfg)))
        record['handoff_results'] = [dict(seconds=t,result=file_metadata(path),full_physical_pass=passed) for t,path,passed in screen]
        selected = next(((t,path) for t,path,passed in screen if passed),None)
        save()
        if selected is None:
            raise ValueError('no declared handoff passed the full physical episode')
        record['selected_handoff_seconds']=selected[0]
        rebuilt_path,rebuilt = stage('05_physical_rebuild','rebuild_frontier_physical_route.py',
            {'--source':selected[1],'--config':cfg_path},[selected[1],cfg_path])
        if not exact_candidate_passed(rebuilt,cfg) or not rebuilt['physical_rebuild_comparison']['all_physical_states_identical']:
            raise ValueError('rebuilt route lacks passed exact physical calibration')
        for episodes,offset in ((20,100),(100,200)):
            path,data = stage(f'06_noisy{episodes}','evaluate_fddp_parked_route.py',
                {'--config':cfg_path,'--controller':rebuilt_path,'--episodes':episodes,'--seed':seed+offset,
                 '--park-seconds':16,'--cart-target':-.05,'--settle-control-cost':1000,
                 '--settle-cart-position-cost':10,'--settle-cart-velocity-cost':5,'--tracking-gain-scale':1,'--phase-window':12},
                [cfg_path,rebuilt_path])
            if (data.get('full_episode_successes') != episodes or len(data.get('episode_results',[])) != episodes
                    or not all(check_episode(x,cfg) for x in data['episode_results'])):
                raise ValueError(f'development noisy{episodes} cohort failed; no release promotion')
        record.update(status='passed_development',candidate=file_metadata(rebuilt_path),
            next_step='Freeze the candidate before reserved validation; no canonical promotion has occurred.')
    except BaseException as error:
        record.update(status='failed_or_interrupted',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        record['finished_at']=utc_timestamp();save()
    print(json.dumps({'status':record['status'],'journal':str(journal),'release_promotion':False}))


if __name__ == '__main__':
    main()
