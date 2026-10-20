"""Apply the preserved standard estimator to the complete frozen14 dev grid."""
import argparse
import importlib.util
import json
from pathlib import Path
import numpy as np
from measure_delivered14_margin_v2 import ROOT,STUDY,SEEDS,SIGMAS,MASSES,CONTRACT,sha,read,write


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out=a.out.resolve();a.run=a.run.resolve();a.out.relative_to(STUDY);a.run.relative_to(STUDY)
    a.out.mkdir(exist_ok=False)
    s=read(a.run/'summary.json');assert s['passed'] and not s['smoke'] and s['episodes']==480
    rows=[json.loads(line) for line in (a.run/'rows.jsonl').read_text().splitlines()]
    assert len(rows)==480 and sorted({row['seed'] for row in rows})==SEEDS
    assert len({(r['seed'],r['sigma'],r['mass_delta']) for r in rows})==480
    for seed in SEEDS:
        got={(r['sigma'],r['mass_delta']) for r in rows if r['seed']==seed}
        want={(sigma,0.) for sigma in SIGMAS}|{(0.,sign*d) for d in MASSES for sign in (-1,1)}
        assert got==want
    for row in rows:assert sha(a.run/'tapes'/row['tape'])==row['tape_sha256']
    archive=read(ROOT/'research/sources/immediate-disturbance-margin-2026-10-19/manifest.json')
    estimator=STUDY/'tools/analyze_immediate_margin.py'
    assert sha(estimator)==archive['files']['tools/analyze_immediate_margin.py']['sha256']
    write(a.out/'inputs.json',dict(run=str(a.run),source_summary_sha256=sha(a.run/'summary.json'),
        source_rows_sha256=sha(a.run/'rows.jsonl'),estimator_sha256=sha(estimator),driver_sha256=sha(__file__),native_cap=0))
    spec=importlib.util.spec_from_file_location('preserved_2d_margin_estimator',estimator)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    force=[r for r in rows if r['sigma']>0 and r['mass_delta']==0]
    mass=[r for r in rows if r['sigma']==0 and r['mass_delta']!=0]
    noise=module.noise_analysis(force,'2D14 frozen common immediate / guard bypass')
    tolerance=module.mass_analysis(mass,'2D14 frozen common immediate / both signs')
    rates=np.array([cell['successes']/cell['trials'] for cell in noise['cells']])
    bracketed=bool(rates.max()>=.5 and rates.min()<=.5)
    sigma50=noise['firth']['sigma50_N'] if bracketed and noise['firth']['usable'] else None
    mass_limit=tolerance['largest_both_signs_ge_10_of_20']
    tiers={key:bool(sigma50 is not None and sigma50>=sigma and mass_limit is not None and mass_limit>=delta)
           for key,sigma,delta in [('M1',1e-4,1e-6),('M2',1e-3,1e-4),('M3',1e-2,1e-3)]}
    result=dict(completed=True,qualified=False,controller_identity=s['controller_identity'],
        sigma50_N=sigma50,sigma50_bracketed=bracketed,mass_tolerance=mass_limit,
        force=noise,mass=tolerance,development_margin_tiers_met=tiers,
        force_failure_timing=module.failure_timing(force),mass_failure_timing=module.failure_timing(mass),
        referenced_episodes=480,referenced_native_steps=s['inclusive_native_steps'],
        native_steps=0,new_labels=0,source_summary_sha256=sha(a.run/'summary.json'),
        scope='complete dev20 diagnostic margin of frozen candidate; fresh32 E and publication gates separate')
    write(a.out/'summary.json',result)
    write(a.out/'task_result.json',dict(metric='measured_force_noise_sigma_N',metric_contract=CONTRACT,
        task_valid=bool(sigma50 is not None and mass_limit is not None),value=sigma50 or 0.,
        reason='complete frozen480-row estimator analysis; mass tolerance alongside; no fresh qualification inferred'))
    print(dict(sigma50_N=sigma50,mass_tolerance=mass_limit,development_tiers=tiers,native_steps=0),flush=True)


if __name__=='__main__':main()
