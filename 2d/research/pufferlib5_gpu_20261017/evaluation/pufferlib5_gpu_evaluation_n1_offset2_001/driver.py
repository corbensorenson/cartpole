"""Build a recorded GPU evaluator, replay it, and repeat the first closed loop."""
import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import mujoco
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.experiment import freeze_inputs, write_json
from puffer_training_support import verify_integration, sha256
from puffer_eval_io import create_plan, import_traces
from run_puffer_training import run_bounded


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkout',required=True);p.add_argument('--training-run',required=True)
    p.add_argument('--out',required=True);p.add_argument('--cohort',choices=['pilot13','development45'],default='pilot13')
    args=p.parse_args()
    if platform.system()!='Linux':raise ValueError('Recorded GPU evaluation requires Linux')
    checkout=Path(args.checkout).resolve();training=Path(args.training_run).resolve();out=Path(args.out).resolve()
    integration=verify_integration(checkout)
    if not integration['evaluation_io']:raise ValueError('Evaluator must have capture enabled')
    trained=json.loads((training/'result.json').read_text());trained_manifest=json.loads((training/'manifest.json').read_text())
    checkpoint=training/'frozen_policy.bin'
    if not trained['training_completed'] or sha256(checkpoint)!=trained['checkpoint_sha256']:
        raise ValueError('A completed training run and its frozen checkpoint are required')
    if integration['links']!=trained_manifest['links']:raise ValueError('Checkpoint link count differs from evaluator')
    if integration.get('reward_offset',0.)!=trained_manifest.get('training_reward_offset',0.):
        raise ValueError('Training/evaluation reward settings differ')
    if mujoco.__version__!='3.3.5':raise ValueError('Use MuJoCo 3.3.5')
    library=Path(mujoco.__file__).parent/'libmujoco.so.3.3.5'
    if sha256(library)!=trained['mujoco_library_sha256']:raise ValueError('Training/evaluation physics library differs')
    freeze_inputs(out,args)
    for name in ('result.json','manifest.json'):shutil.copy2(training/name,out/('training_'+name))
    for name in integration['files']:
        target=out/'build_inputs'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(checkout/name,target)
    n=integration['links'];config=Config(n_links=n)
    seeds=[(20261016,0.)]+[(s,.02) for s in range(20261116,20261128)]
    if args.cohort=='development45':seeds += [(s,.02) for s in range(20261416,20261448)]
    env=SpatialCartpole(config);cases=[]
    try:
        for seed,noise in seeds:
            state,_=env.reset(seed=seed,options=dict(mode='hanging',angle_noise=noise));cases.append((seed,noise,0,state))
    finally:env.close()
    create_plan(out/'plan',n,cases,'development',checkpoint)
    create_plan(out/'repeat_plan',n,cases[:1],'development',checkpoint)
    maximum=(len(cases)+1)*6000*2
    report=dict(schema=1,qualified=False,cohort=args.cohort,links=n,checkpoint_sha256=sha256(checkpoint),
                inclusive_native_budget=maximum,frozen_closed_loop_reexecution=False,reserved_cases_consumed=0)
    write_json(out/'result.json',report)
    envvars={**os.environ,'MUJOCO_DIR':str(library.parent),'MUJOCO_LIBRARY':str(library),
             'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    binary=out/'spatial_eval'
    try:
        build=['bash','build_spatial.sh','spatial_cartpole',str(binary),'--float']
        report['build']=run_bounded(build,checkout,envvars,out/'build.log',600,'evaluation_build')
        if report['build']['exit_code']:raise RuntimeError('Evaluation build failed')
        report['binary_sha256']=sha256(binary)
        linked=subprocess.check_output(['ldd',str(binary)],text=True,timeout=30)
        (out/'ldd.txt').write_text(linked,encoding='utf-8')
        line=[x for x in linked.splitlines() if 'libmujoco.so' in x]
        if len(line)!=1 or '=>' not in line[0]:raise ValueError('Cannot resolve evaluator physics library')
        resolved=Path(line[0].split('=>',1)[1].split('(',1)[0].strip()).resolve()
        if sha256(resolved)!=sha256(library):raise ValueError('Evaluator/Python shared-library mismatch')
        summaries=[]
        for label,plan,count in [('evaluation',out/'plan',len(cases)),('repeat',out/'repeat_plan',1)]:
            traces=out/(label+'_traces');traces.mkdir()
            command=[str(binary),'eval',str(plan/'frozen_policy.bin'),'--headless',
                     '--base.seed=73',f'--base.eval_episodes={count}','--base.eval_agents=1',
                     '--vec.total_agents=1','--vec.num_buffers=1','--vec.num_threads=1',
                     '--train.horizon=64','--train.minibatch_size=64',
                     f'--env.evaluation_plan={plan/"plan.bin"}',f'--env.trace_directory={traces}',
                     f'--env.accounting_file={traces/"accounting.json"}']
            report[label]=run_bounded(command,checkout,envvars,out/(label+'.log'),600,label)
            if report[label]['exit_code']:raise RuntimeError(f'{label} process failed')
            result=import_traces(plan,traces,out/(label+'_import'))
            if not result['native_shutdown_accounting_verified']:raise ValueError('Missing native accounting')
            summaries.append(result);write_json(out/'result.json',report)
        original=out/'evaluation_traces/episode0000.sptrace';repeated=out/'repeat_traces/episode0000.sptrace'
        if original.read_bytes()!=repeated.read_bytes():raise ValueError('Frozen GPU first-case closed loop differs')
        verify_integration(checkout)
        if sha256(binary)!=report['binary_sha256']:raise ValueError('Evaluator binary changed')
        report.update(cases=len(cases),passes=summaries[0]['passes'],
                      actual_native_steps=sum(x['actual_native_steps'] for x in summaries),
                      cold_proof_native_steps=sum(x['additional_native_steps'] for x in summaries),
                      inclusive_native_steps=sum(x['inclusive_native_steps'] for x in summaries),
                      frozen_closed_loop_reexecution=True,first_case_trace_sha256=sha256(original),
                      every_state_and_command_byte_exact=True,completed=True)
        if report['inclusive_native_steps']>maximum:raise ValueError('Evaluation budget exceeded')
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}';report['completed']=False
        # Preserve actual completed process accounting even when cold import fails.
        report['captured_native_steps']=sum(json.loads(x.read_text())['native_steps'] for x in out.glob('*_traces/accounting.json'))
        write_json(out/'result.json',report);raise
    write_json(out/'result.json',report);write_json(out/'completed.json',dict(status='completed'))
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
