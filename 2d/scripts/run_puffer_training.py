"""Prepare, or build and execute, one bounded native PufferLib 5 experiment.

Default: provenance/command dry run. --execute requires Linux and builds --float
from the frozen integration before training. No automatic evaluation or sweep.
"""
import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback

from puffer_training_support import verify_integration, sha256, float_weight_layout, REF


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')


def run_bounded(command, cwd, env, log, seconds, phase):
    started=time.monotonic(); expired=False
    with Path(log).open('xb') as stream:
        process=subprocess.Popen(command,cwd=cwd,env=env,stdout=stream,
                                 stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps(dict(phase=phase,pid=process.pid,started=True)),flush=True)
        try:
            while process.poll() is None:
                if time.monotonic()-started>=seconds:
                    expired=True;os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    break
                try:process.wait(timeout=min(20,max(.1,seconds-(time.monotonic()-started))))
                except subprocess.TimeoutExpired:
                    print(json.dumps(dict(phase=phase,pid=process.pid,elapsed_seconds=round(time.monotonic()-started,1))),flush=True)
        finally:
            if process.poll() is None:
                os.killpg(process.pid,signal.SIGTERM)
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
    return dict(command=command,exit_code=process.returncode,timed_out=expired,
                elapsed_seconds=time.monotonic()-started)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkout',required=True);p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=4096);p.add_argument('--seed',type=int,default=73)
    p.add_argument('--max-seconds',type=int,default=600);p.add_argument('--execute',action='store_true')
    args=p.parse_args();checkout=Path(args.checkout).resolve();out=Path(args.out).resolve()
    integration=verify_integration(checkout)
    if integration['evaluation_io']:raise ValueError('Training requires capture disabled at compile time')
    if args.execute and platform.system()!='Linux':raise ValueError('--execute requires the Linux CUDA toolchain')
    config=configparser.ConfigParser(interpolation=None)
    config.read([checkout/'config/default.ini',checkout/'config/spatial_cartpole.ini'],encoding='utf-8')
    for section,key,value in [('vec','total_agents',64),('vec','num_buffers',2),('vec','num_threads',4),
                              ('vec','num_policies',1),('train','horizon',64),('train','gpus',1),
                              ('policy','hidden_size',64),('policy','num_layers',2),('selfplay','enabled',0)]:
        if config.getint(section,key)!=value:raise ValueError(f'Unexpected initial experiment setting: {section}.{key}')
    batch=config.getint('vec','total_agents')*config.getint('train','horizon')
    if not (batch<=args.steps<=999424 and args.steps%batch==0):
        raise ValueError('Use complete 4096-transition batches, at most 999424 transitions')
    if args.seed!=73:raise ValueError('First registered training seed is 73; register a new experiment before adding seeds')
    if not 1<=args.max_seconds<=3600:raise ValueError('Wall-time cap must be 1..3600 seconds')
    out.mkdir(parents=True,exist_ok=False);frozen=out/'frozen_inputs';frozen.mkdir()
    for name in integration['files']:
        target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(checkout/name,target)
    for name in ['src','config/default.ini','resources/spatial_cartpole/integration.json']:
        source=checkout/name;target=frozen/name
        if source.is_dir():shutil.copytree(source,target)
        else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    for name in ('run_puffer_training.py','puffer_training_support.py','prepare_puffer_environment.py'):
        shutil.copy2(Path(__file__).parent/name,frozen/name)
    (out/'checkpoints').mkdir();(out/'logs').mkdir()
    binary=out/'spatial_trainer';accounting=out/'native_accounting.json'
    build=['bash','build_spatial.sh','spatial_cartpole',str(binary),'--float']
    run_id=f'n{integration["links"]}_s{args.seed}_t{args.steps}'
    overrides={'base.seed':args.seed,'base.run_id':run_id,'base.eval_episodes':0,
               'base.checkpoint_dir':str(out/'checkpoints'),'base.log_dir':str(out/'logs'),
               'base.load_model_path':'None','base.load_enemy_model_path':'None',
               'train.total_timesteps':args.steps,'train.gpus':1,'env.accounting_file':str(accounting)}
    command=[str(binary),'train',*[f'--{key}={value}' for key,value in overrides.items()]]
    manifest=dict(schema=1,puffer_ref=REF,trainer_patch=integration['trainer_patch'],links=integration['links'],
                  training_started=False,training_completed=False,qualified=False,dry_run=not args.execute,
                  checkout=str(checkout),control_transition_budget=args.steps,native_step_budget=4*args.steps,
                  batch_size=batch,epochs=args.steps//batch,seed=args.seed,automatic_evaluation_episodes=0,
                  reserved_cases_consumed=0,wall_time_cap_per_phase_seconds=args.max_seconds,
                  build_command=build,training_command=command,overrides=overrides,
                  expected_checkpoint_layout=float_weight_layout(integration['links']),
                  frozen_sha256={x.relative_to(frozen).as_posix():sha256(x) for x in frozen.rglob('*') if x.is_file()})
    write_json(out/'manifest.json',manifest)
    if not args.execute:
        print(json.dumps(dict(dry_run=True,training_started=False,steps=args.steps,epochs=args.steps//batch,out=str(out))),flush=True)
        return
    report=dict(training_started=False,training_completed=False,qualified=False,native_steps=None)
    env={**os.environ,'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    try:
        import mujoco
        if mujoco.__version__!='3.3.5':raise ValueError('Use MuJoCo 3.3.5')
        sdk=Path(mujoco.__file__).parent
        candidates=list(sdk.glob('libmujoco.so.3.3.5'))
        if len(candidates)!=1:raise ValueError('Cannot identify the Python MuJoCo 3.3.5 shared library')
        library=candidates[0].resolve()
        env.update(MUJOCO_DIR=str(sdk),MUJOCO_LIBRARY=str(library))
        report['mujoco_library']=str(library);report['mujoco_library_sha256']=sha256(library)
        for label,cmd in [('nvcc',['nvcc','--version']),('nvidia-smi',['nvidia-smi'])]:
            diagnostic=subprocess.run(cmd,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=30)
            (out/(label+'.txt')).write_text(diagnostic.stdout,encoding='utf-8')
            if diagnostic.returncode:raise RuntimeError(f'{label} failed')
        report['build']=run_bounded(build,checkout,env,out/'build.log',args.max_seconds,'build')
        if report['build']['exit_code'] or report['build']['timed_out']:raise RuntimeError('CUDA build failed or timed out')
        verify_integration(checkout)
        report['binary_sha256']=sha256(binary)
        linked=subprocess.check_output(['ldd',str(binary)],text=True,timeout=30)
        (out/'ldd.txt').write_text(linked,encoding='utf-8')
        linked_mujoco=[line for line in linked.splitlines() if 'libmujoco.so' in line]
        if len(linked_mujoco)!=1 or '=>' not in linked_mujoco[0]:raise ValueError('Cannot resolve linked MuJoCo')
        resolved=Path(linked_mujoco[0].split('=>',1)[1].split('(',1)[0].strip()).resolve()
        if sha256(resolved)!=sha256(library):raise ValueError('Trainer/Python MuJoCo library mismatch')
        report['training_started']=manifest['training_started']=True
        write_json(out/'manifest.json',manifest);write_json(out/'result.json',report)
        report['train']=run_bounded(command,checkout,env,out/'train.log',args.max_seconds,'train')
        if report['train']['exit_code'] or report['train']['timed_out']:raise RuntimeError('Training failed or timed out')
        cost=json.loads(accounting.read_text(encoding='utf-8'));report['native_accounting']=cost
        report['native_steps']=cost['native_steps']
        if cost!={'schema':1,'links':integration['links'],'closed_environments':64,
                  'native_steps':4*args.steps,'control_transitions':args.steps}:
            raise ValueError('Native shutdown accounting differs from the complete finite-action budget')
        checkpoint=out/'checkpoints/spatial_cartpole'/run_id/f'{args.steps:016d}.bin'
        if checkpoint.stat().st_size!=manifest['expected_checkpoint_layout']['bytes']:
            raise ValueError('Checkpoint does not contain the complete aligned float32 slab')
        import numpy as np
        if not np.isfinite(np.fromfile(checkpoint,dtype='<f4')).all():raise ValueError('Nonfinite checkpoint weights')
        shutil.copy2(checkpoint,out/'frozen_policy.bin')
        report.update(checkpoint_sha256=sha256(checkpoint),
                      checkpoint_bytes=checkpoint.stat().st_size,
                      control_transitions_per_second=args.steps/report['train']['elapsed_seconds'])
        verify_integration(checkout)
        if sha256(binary)!=report['binary_sha256']:raise ValueError('Trainer binary changed during execution')
        report['training_completed']=True
    except BaseException as error:
        report['error']=f'{type(error).__name__}: {error}'
        report['traceback']=traceback.format_exc()
        if accounting.exists():
            report['native_accounting']=json.loads(accounting.read_text(encoding='utf-8'))
            report['native_steps']=report['native_accounting']['native_steps']
        report['native_cost_unknown']=bool(report['training_started'] and report['native_steps'] is None)
        write_json(out/'result.json',report)
        raise
    write_json(out/'result.json',report)
    manifest['training_completed']=report['training_completed']
    write_json(out/'manifest.json',manifest)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
