import json
from pathlib import Path
import shutil
import subprocess
import sys

root=Path(__file__).resolve().parents[1]
out=root/'output/pufferlib5_plan_protocol_002';out.mkdir(exist_ok=False)
checkout=root/'output/pufferlib5_training_003'
driver=root/'scripts/run_puffer_training.py'
cases=[]
for label,extra,expected in [
    ('partial_batch',['--steps','4095'],'complete 4096-transition batches'),
    ('over_budget',['--steps','1003520'],'complete 4096-transition batches'),
    ('unregistered_seed',['--seed','74'],'First registered training seed'),
    ('unpatched_checkout',['--checkout',str(root/'output/pufferlib5_training_002')],'flat-layout correction'),
    ('windows_execution',['--execute'],'requires the Linux CUDA toolchain'),
]:
    target=out/label
    result=subprocess.run([sys.executable,str(driver),'--checkout',str(checkout),'--out',str(target),*extra],text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    (out/(label+'.log')).write_text(result.stdout,encoding='utf-8')
    assert result.returncode!=0 and expected in result.stdout and not target.exists()
    cases.append(dict(case=label,rejected_before_output_creation=True))
sys.path.insert(0,str(root/'scripts'))
from puffer_training_support import verify_integration,float_weight_layout
manifest=verify_integration(checkout)
# An isolated sparse clone tests the actual integrity check on a modified header.
copy=out/'tamper_checkout'
for args in [['clone','--shared','--no-checkout',str(checkout),str(copy)],
             ['-C',str(copy),'sparse-checkout','init','--cone'],
             ['-C',str(copy),'sparse-checkout','set','src','config','vendor'],
             ['-C',str(copy),'checkout',manifest['puffer_ref']]]:
    subprocess.run(['git',*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
for name in [*manifest['files'],'resources/spatial_cartpole/integration.json']:
    destination=copy/name;destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(checkout/name,destination)
verify_integration(copy)
header=copy/'ocean/spatial_cartpole/spatial_core.h'
header.write_bytes(header.read_bytes()+b'\n// diagnostic corruption\n')
try:verify_integration(copy)
except ValueError as e:assert 'Integration file changed' in str(e)
else:raise AssertionError('Modified native source accepted')
cases.append(dict(case='modified_native_source',rejected_before_execution=True))
for folder,steps in [('pufferlib5_smoke_plan_001',4096),('pufferlib5_learning_plan_001',999424)]:
    m=json.loads((root/'output'/folder/'manifest.json').read_text())
    assert m['dry_run'] and not m['training_started'] and not m['training_completed']
    assert m['control_transition_budget']==steps and m['native_step_budget']==4*steps
    assert m['expected_checkpoint_layout']['bytes']==102672
    assert m['automatic_evaluation_episodes']==0 and m['reserved_cases_consumed']==0
    assert m['training_command'][1]=='train' and f'--train.total_timesteps={steps}' in m['training_command']
    assert '--base.eval_episodes=0' in m['training_command']
    cases.append(dict(case=folder,valid_dry_run=True))
(out/'summary.json').write_text(json.dumps(dict(passed=True,cases=cases,native_steps=0,gpu_training=False),indent=2),encoding='utf-8')
shutil.copy2(__file__,out/'driver.py')
print(json.dumps(dict(passed=True,cases=len(cases),native_steps=0)))
