"""At most six independent fresh cases; unchanged numerical stages, no retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from delivered_fresh_context14_parallel import ROOT, FRESH, CASE_NATIVE_CAP, MAX_COHORT_WALL, cohort, previous, read, sha

CLI = dict(register='register_fresh_delivered14_parallel.py',actor='prepare_fresh_delivered14_parallel.py',
 audit='audit_fresh_delivered14_parallel.py',plan_inventory='inventory_fresh_delivered14_parallel.py',
 witness='witness_fresh_delivered14_parallel.py',reproduce='witness_fresh_delivered14_parallel.py',
 witness_inventory='inventory_fresh_delivered14_witness_parallel.py',cohort_audit='audit_fresh_delivered14_cohort_parallel.py')
STAGE_WALL = dict(register=300,actor=15300,audit=7500,plan_inventory=300,witness=540,reproduce=540,witness_inventory=300,cohort_audit=2400)


def main():
 p=argparse.ArgumentParser();p.add_argument('--cohort',required=True,type=Path);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
 assert 1 <= a.workers <= 6
 a.cohort=a.cohort.resolve();bank,runtime,cm,m,development=cohort(a.cohort)
 assert m['max_worker_processes']==6 and m['independent_case_ledgers']
 assert read(a.cohort/'consumed_cases.json')['seeds']==[]
 assert not (a.cohort/'cases').exists() and not (a.cohort/'execution').exists()
 with (a.cohort/'execution_activation.json').open('x') as f:
  json.dump(dict(pid=os.getpid(),max_worker_processes=a.workers,cohort_manifest_sha256=sha(a.cohort/'frozen_manifest.json')),f,indent=2)
 from spatialpole.experiment import freeze_inputs,write_json
 out=a.cohort/'execution';freeze_inputs(out,argparse.Namespace(cohort=str(a.cohort),workers=a.workers))
 assert read(out/'inputs.json')['runtime']==m['original_runtime']
 write_json(out/'worker.json',dict(pid=os.getpid(),max_worker_processes=a.workers))
 (a.cohort/'cases').mkdir();deadline=time.monotonic()+MAX_COHORT_WALL;started=time.monotonic()
 lock=threading.Lock();registration_lock=threading.Lock();stopped=threading.Event();rows={};active={};failures=[]
 base=development['total_new_target_preparation_native_steps'];base_bounds=development['total_new_target_preparation_native_steps_bounds']
 def progress():
  increments=sum(row.get('incremental_native_steps',0) for row in rows.values())
  value=dict(status='running' if active else 'idle',completed_cases=sum(row.get('passed',False) for row in rows.values()),
   fresh_labels_consumed=len(read(a.cohort/'consumed_cases.json')['seeds']),active=list(active.values()),
   incremental_closed_native_steps=increments,known_closed_native_ledger=base+increments,
   known_closed_native_bounds=[x+increments for x in base_bounds],
   unsealed_case_native_upper_bound=len(active)*CASE_NATIVE_CAP,failures=failures,
   max_worker_processes=a.workers,wall_seconds=time.monotonic()-started)
  pending=out/'progress.pending';write_json(pending,value);os.replace(pending,out/'progress.json')
 def execute(seed,stage,args):
  for name,digest in m['helper_sha256'].items():assert sha(ROOT/'scripts'/name)==digest
  remaining=deadline-time.monotonic();assert remaining>0,'Frozen cohort wall cap reached'
  cmd=[sys.executable,'-B','-u',str(ROOT/'scripts'/CLI[stage]),*map(str,args)]
  env=dict(os.environ);env['PYTHONPATH']=os.pathsep.join((str(ROOT/'src'),str(ROOT/'scripts')))
  log_path=out/f'{seed}_{stage}.log'
  with log_path.open('x',encoding='utf-8') as log:
   options=dict(cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env)
   if os.name=='nt':options['creationflags']=subprocess.CREATE_NO_WINDOW
   else:options['start_new_session']=True
   child=subprocess.Popen(cmd,**options)
   with lock:
    active[seed]=dict(seed=seed,stage=stage,child_pid=child.pid,command=cmd,log=str(log_path));progress()
   print(dict(seed=seed,stage=stage,pid=child.pid),flush=True)
   try:code=child.wait(timeout=min(STAGE_WALL[stage],remaining))
   except subprocess.TimeoutExpired:
    # Only this owned frozen stage's predeclared deadline may stop its process tree.
    if os.name=='nt':subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=log,stderr=subprocess.STDOUT)
    else:
     import signal
     os.killpg(child.pid,signal.SIGKILL)
    child.wait(timeout=30);raise
  assert code==0,f'{stage} exited {code}; preserve {log_path}'
 def run_case(seed):
  case=a.cohort/'cases'/str(seed);wall=time.monotonic()
  try:
   with registration_lock:
    if stopped.is_set():return
    assert deadline-time.monotonic()>=1200,'No new case below1200s remaining'
    case.mkdir()
    with lock:active[seed]=dict(seed=seed,stage='registration_pending',child_pid=None);progress()
    execute(seed,'register',['--cohort',a.cohort,'--seed',seed,'--out',case/'registration'])
   execute(seed,'actor',['--registration',case/'registration','--out',case/'prepare'])
   execute(seed,'audit',['--run',case/'prepare','--out',case/'audit'])
   inventory=case/'preparation_inventory.json';witness_inventory=case/'witness_inventory.json'
   execute(seed,'plan_inventory',['--run',case/'prepare','--audit',case/'audit','--out',inventory])
   plan=read(inventory)
   assert plan['plan_found'],'Preparation rejected; no retry or redraw'
   execute(seed,'witness',['--inventory',inventory,'--out',case/'witness'])
   execute(seed,'reproduce',['--reproduce',case/'witness','--out',case/'reproduction'])
   execute(seed,'witness_inventory',['--parent',inventory,'--run',case/'witness','--audit',case/'reproduction','--out',witness_inventory])
   proof=previous(witness_inventory,cm,seed+1)
   native=plan['new_native_steps']+proof['new_native_steps'];assert native<=CASE_NATIVE_CAP
   result=dict(seed=seed,passed=True,witness_inventory=str(witness_inventory),witness_inventory_sha256=sha(witness_inventory),
    result=proof['result'],incremental_native_steps=native,planning_actor_native_steps=plan['actor_native_steps'],
    planning_audit_native_steps=plan['audit_native_steps'],wall_seconds=time.monotonic()-wall,
    runtime_class='offline per-episode planning with simulation paused')
   with lock:rows[seed]=result;active.pop(seed);progress()
   print(dict(seed=seed,canonical_and_reproduction_passed=True,incremental_native_steps=native),flush=True)
  except BaseException as exc:
   stopped.set()
   failure=dict(seed=seed,type=type(exc).__name__,error=str(exc),case_native_upper_bound=CASE_NATIVE_CAP,
    wall_seconds=time.monotonic()-wall,spent_label_must_not_be_retried=True)
   write_json(case/'scheduler_failure.json' if case.exists() else out/f'{seed}_scheduler_failure.json',failure)
   with lock:failures.append(failure);rows[seed]=dict(seed=seed,passed=False);active.pop(seed,None);progress()
   print(failure,flush=True)
 # Register in exact seed order. Numerical work continues independently after registration.
 # Initial workers are admitted one by one to avoid thread scheduling changing label order.
 with ThreadPoolExecutor(max_workers=a.workers) as pool:
  pending={};index=0
  while index<len(FRESH) or pending:
   while index<len(FRESH) and len(pending)<a.workers and not stopped.is_set():
    seed=FRESH[index]
    future=pool.submit(run_case,seed);pending[future]=seed;index+=1
    # Wait only for this label to be consumed, not its planning result, before next dispatch.
    while not stopped.is_set() and seed not in read(a.cohort/'consumed_cases.json')['seeds']:
     if future.done():break
     time.sleep(.05)
   if not pending:break
   done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
   for future in done:future.result();pending.pop(future)
   if stopped.is_set() and not pending:break
  # Already-running cases finish even after another case fails. No new labels are admitted.
 qualified=False
 if not failures and len(rows)==32 and all(row['passed'] for row in rows.values()):
  execute('cohort','cohort_audit',['--cohort',a.cohort,'--out',a.cohort/'qualification-proof'])
  proof=read(a.cohort/'qualification-proof/summary.json');assert proof['qualified'] and proof['noisy_passes']==32
  qualified=True
 increment=sum(row.get('incremental_native_steps',0) for row in rows.values())
 write_json(out/'summary.json',dict(passed=qualified,qualified=qualified,task_success=qualified,links=14,
  status='all32_fresh_cases_and_cohort_proof_passed' if qualified else 'stopped_no_retry',
  results=[rows[k] for k in sorted(rows)],failures=failures,max_worker_processes=a.workers,
  all32_cases_completed=len(rows)==32,incremental_closed_native_steps=increment,
  known_closed_native_ledger=base+increment,known_closed_native_bounds=[x+increment for x in base_bounds],
  unsealed_failed_native_upper_bound=len(failures)*CASE_NATIVE_CAP,
  fresh_labels_consumed=len(read(a.cohort/'consumed_cases.json')['seeds']),wall_seconds=time.monotonic()-started,
  source_cohort_manifest_sha256=sha(a.cohort/'frozen_manifest.json'),runtime_class='offline per-episode planning with simulation paused',
  scope='fresh physical/model/reproduction qualification; runtime extraction/resynthesis, margins, video and publication still required'))
 write_json(out/'completed.json',dict(status='completed' if qualified else 'failed'))
 if not qualified:raise SystemExit(1)

if __name__=='__main__':main()
