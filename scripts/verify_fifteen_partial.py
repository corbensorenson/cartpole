"""Audit the saved15-link partial result without integrating physics.

Successful verification means evidence integrity, not cart-pole task success.
"""
import argparse,hashlib,json,tarfile
from pathlib import Path
import numpy as np

def longest(mask):
 best=run=0;end=None
 for i,ok in enumerate(mask):
  run=run+1 if ok else 0
  if run>best:best=run;end=i+1
 return best,end

def verify(checkout):
 root=Path(checkout);base=root/'runs/fifteen_link_partial_20261015';m=json.loads((base/'manifest.json').read_text());assert m['not_solution'] and not m['canonical_release_evidence'] and m['status']=='partial_result_unsolved'
 for row in m['files']:
  meta=row['published'];p=root/meta['path'];assert p.stat().st_size==meta['bytes'];assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
 reports=[]
 for case in m['cases']:
  with np.load(root/case['tape'],allow_pickle=False) as z:xs=z['states'];actions=z['actions'];raw=z['raw_actions']
  assert xs.shape==(len(actions)+1,32) and raw.shape==actions.shape and np.all(np.isfinite(xs)) and np.all(np.isfinite(actions)) and np.all(np.isfinite(raw))
  assert xs.dtype==np.float64 and actions.dtype==np.float64
  expected=np.zeros(32);expected[1]=-np.pi;assert np.array_equal(xs[0],expected)
  assert np.array_equal(actions,np.asarray(np.clip(raw,-1,1),np.float32).astype(np.float64)) and np.max(abs(actions))<=1
  absolute=np.cumsum(xs[1:,1:16],axis=1);wrapped=np.remainder(absolute+np.pi,2*np.pi)-np.pi;angles=np.where((absolute>=-np.pi)&(absolute<np.pi),absolute,wrapped)
  upright=np.max(abs(angles),axis=1)<m['upright_absolute_angle_limit_rad'];ticks,end=longest(upright);safe_ticks,_=longest(upright&(abs(xs[1:,0])<=m['rail_limit_m']));hold=ticks/m['command_rate_hz'];cart=float(np.max(abs(xs[:,0])))
  task_passed=len(actions)==1500 and ticks>=250 and cart<=3
  assert len(actions)==case['reported_ticks'] and hold==case['reported_hold_seconds']==3.0 and cart==case['reported_maximum_cart_m']
  assert abs(xs[-1,0])>3 and np.all(abs(xs[:-1,0])<=3) and case['reported_termination']=='rail_violation' and not task_passed
  if case['selected_forecast']:
   with np.load(root/case['selected_forecast'],allow_pickle=False) as z:assert np.array_equal(xs[:442],z['states']) and np.array_equal(actions[:441],z['actions'])
  reports.append(dict(label=case['label'],ticks=len(actions),elapsed_seconds=len(actions)/50,upright_ticks=ticks,upright_seconds=hold,in_rail_upright_ticks=safe_ticks,maximum_cart_m=cart,termination='rail_violation',nominal_task_passed=False,longest_streak_end_poststate=end))
 with np.load(base/'native_design.npz',allow_pickle=False) as z:assert z['gains'].shape==(441,32)
 with tarfile.open(base/'original_synthesis_source.tar.gz') as archive:
  names=archive.getnames();assert 'scripts/solve_canonical15_active_continuation.py' in names
  assert all(not p.startswith('/') and '..' not in Path(p).parts for p in names)
 return dict(evidence_integrity_passed=True,canonical15_solved=False,best_observed_hold_seconds=3.,highest_accepted_n_links=14,files_verified=len(m['files']),source_members=len(names),cases=reports,new_native_transition_calls=0,scope='Whole saved tapes, hashes, recordedF32 delivery, gate metrics and selected441-tick prefixes audited. No fresh dynamics integration, noisy cohort or robustness claim.')

def main():
 p=argparse.ArgumentParser();p.add_argument('--checkout',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output');a=p.parse_args();r=verify(a.checkout);text=json.dumps(r,indent=2)
 if a.output:Path(a.output).write_text(text+'\n')
 print(text)
if __name__=='__main__':main()
