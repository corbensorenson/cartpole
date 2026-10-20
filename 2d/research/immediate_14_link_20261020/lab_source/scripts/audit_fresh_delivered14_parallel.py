"""Independent unchanged model/value/control audit for a registered start."""
import argparse
from pathlib import Path
import shutil
import time
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import freeze_inputs,write_json
from spatialpole.native_mpc import exact_bytes
from spatial_synthesis_geometry import error
from delivered_fresh_context14_parallel import registered,serialized_args
from delivered_reference_bank14 import choose
from delivered_planner14 import MAX_MODELS,ACTOR_NATIVE,AUDIT_NATIVE
from audit_spatial_synthesis_models import load
from audit_delivered_case14_margin import same_dict,value_plan,mean,model,forecast
from stage_six_source_synthesis14 import helper_closure
from eleven_synthesis_protocol import read,sha


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args()
    assert (a.run/'completed.json').exists();s,m=read(a.run/'summary.json'),read(a.run/'frozen_manifest.json')
    registration=Path(m['registration']);bank,runtime,cm,rm,expected,prior=registered(registration)
    assert sha(registration/'frozen_manifest.json')==m['registration_manifest_sha256']
    claim=read(registration/'activation.json');assert Path(claim['run']).resolve()==a.run.resolve()
    assert claim['registration_manifest_sha256']==m['registration_manifest_sha256']
    assert s['passed'] and not s['qualified'] and not s['task_success'] and s['preparation_only']
    assert s['controller_identity']==m['controller_identity']==cm['controller_identity'] and m['seed']==rm['seed']==s['seed']
    assert m['noise']==.02 and m['conditioning_steps']==0 and m['config']==Config(n_links=14).__dict__
    freeze_inputs(a.out,serialized_args(a));assert read(a.out/'inputs.json')['source_sha256']==cm['source_sha256']
    for name in helper_closure({'audit_fresh_delivered14_parallel.py'}):shutil.copyfile(Path(__file__).with_name(name),a.out/name)
    write_json(a.out/'frozen_manifest.json',dict(registration=str(registration),registration_manifest_sha256=sha(registration/'frozen_manifest.json'),
        source_run=str(a.run),source_manifest_sha256=sha(a.run/'frozen_manifest.json'),source_summary_sha256=sha(a.run/'summary.json'),
        controller_identity=cm['controller_identity'],cohort_manifest_sha256=rm['cohort_manifest_sha256'],max_native_steps=AUDIT_NATIVE,max_wall_seconds=7200))
    geometry=SpatialCartpole(Config(n_links=14));native=SpatialCartpole(geometry.config);native.derivative_workers=4
    started=time.monotonic();closed=actor=0;rows=[]
    try:
        observed,_=geometry.reset(seed=m['seed'],options=dict(mode='hanging',angle_noise=.02));exact_bytes(observed,expected)
        exact_bytes(observed,load(a.run/'observed_state.npz')['state']);planning=a.run/'planning'
        exact_bytes(observed,load(planning/'observed_state.npz')['state'])
        exact,order=choose(geometry,observed,bank);assert read(planning/'selection.json')==dict(exact_index=exact,ranking=order)
        assert s['selected_profile_index']==order[0]['index']
        if exact is not None:
            assert s['mode']=='cached_profile' and s['plan_found'] and s['native_steps']==0 and s['source_profile']==bank[exact]
            found=True
        else:
            source=Path(bank[order[0]['index']]['guide']);v,r,c=[load(source/name) for name in ('virtual_reference.npz','native_models.npz','coordinates.npz')]
            initial=error(geometry,v,r['F'],observed,0);exact_bytes(initial,load(planning/'warm_plan/model_plan.npz')['initial_error'])
            plan,warmvalue=value_plan(planning/'warm_plan',v,r,c,initial)
            v,warm=mean(planning/'warm_mean',v,r,c,initial,plan['w'],observed,geometry,native,True);actor+=warm['native_steps']
            found=False
            for index,row in enumerate(s['models']):
                assert v is not None and index==row['index'] and index<MAX_MODELS
                stage=planning/f'iteration{index}';r,c,plan,proof=model(stage,v,geometry,native,a.out,index);actor+=2166080
                forecasts=[]
                if not np.any(plan['w']):
                    assert row['fixed'] is not None
                    tape,f=forecast(stage,v,r,c,plan,row['fixed'],'fixed',geometry,native,a.out)
                    forecasts.append(f);closed+=f['cold_native_steps'];actor+=f['actor_native_steps'];found=f['feasible']
                    if not found:
                        assert row['native_control'] is not None
                        tape,f=forecast(stage/'native_control',v,r,c,plan,row['native_control'],'native_control',geometry,native,a.out)
                        forecasts.append(f);closed+=f['cold_native_steps'];actor+=f['actor_native_steps'];found=f['feasible']
                    else:assert row['native_control'] is None
                else:assert row['fixed'] is None and row['native_control'] is None
                rows.append(dict(index=index,model=proof,forecasts=forecasts))
                if found:
                    assert row['update'] is None and s['selected_iteration']==index and index==len(s['models'])-1
                    assert s['mode']==forecasts[-1]['kind'];same_dict(tape,load(planning/'accepted_anchor.npz'));break
                if row['update'] is not None:
                    v,update=mean(stage/'next_mean',v,r,c,np.zeros(74),plan['w'],observed,geometry,native,False)
                    actor+=update['native_steps'];rows[-1]['update']=update
                else:assert index==MAX_MODELS-1
                write_json(a.out/'audit_progress.json',dict(checked_models=len(rows),native_steps=native.native_steps+closed))
            assert found==s['plan_found']
            if not found:
                if not s['models']:assert v is None and s['stage']=='warm_mean_rejected'
                elif v is None:assert s['stage']=='updated_mean_rejected'
                else:assert len(rows)==MAX_MODELS and s['stage']=='model_budget_exhausted'
        assert actor==s['native_steps']<=ACTOR_NATIVE and native.native_steps+closed<=AUDIT_NATIVE
        assert geometry.native_steps==0 and time.monotonic()-started<7200
        result=dict(passed=True,links=14,qualified=False,task_success=False,plan_found=found,seed=m['seed'],controller_identity=cm['controller_identity'],
            native_steps=native.native_steps+closed,actor_native_steps=actor,models=rows,
            source_summary_sha256=sha(a.run/'summary.json'),source_manifest_sha256=sha(a.run/'frozen_manifest.json'),
            registration_manifest_sha256=sha(registration/'frozen_manifest.json'),original_reset_episodes=0,new_noisy_samples=0,reserved_cases_consumed=0,
            scope='fresh case: unchanged corrected model/value/mean/query/cold audit functions; canonical witness required')
        write_json(a.out/'summary.json',result);write_json(a.out/'completed.json',dict(status='completed'));print(result,flush=True)
    except BaseException as exc:
        failure=a.out/'forecast_audit_failure.json';extra=read(failure)['cold_native_steps'] if failure.exists() else 0
        write_json(a.out/'failure.json',dict(type=type(exc).__name__,error=str(exc),native_steps=native.native_steps+closed+extra,accounting_complete=True));raise
    finally:geometry.close();native.close()


if __name__=='__main__':main()
