#!/usr/bin/env python
"""Replay inherited controls/nodes with independently rebuilt LTV QR feedback."""
import argparse,json,subprocess,sys
from pathlib import Path
import numpy as np
from gcartpole.config import dump_json
from gcartpole.evidence import file_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--directory',required=True)
    p.add_argument('--regularization',type=float,required=True)
    p.add_argument('--decimal-digits',type=int,default=0)
    p.add_argument('--state-epsilon',type=float,default=None)
    p.add_argument('--export-value-factors',default=None)
    args=p.parse_args()
    source=Path(args.source);out=Path(args.directory)
    if (out/'synthesis').exists() or (out/'result.json').exists():raise FileExistsError(out)
    reference=source.parent/'synthesis/execution.json'
    command=json.loads(reference.read_text())['command']
    if '--export-tracking-value-factors' in command:
        i=command.index('--export-tracking-value-factors');del command[i:i+2]
    if args.state_epsilon is not None:
        if not np.isfinite(args.state_epsilon) or args.state_epsilon <= 0:
            raise ValueError('state derivative increment must be finite and positive')
        command[command.index('--state-epsilon')+1]=str(args.state_epsilon)
    for key,value in {'--initial-controller':str(source),'--state-json':str(source),'--optimizer':'sqrt-ilqr','--out':str(out/'synthesis/result.json')}.items():
        command[command.index(key)+1]=value
    for key in ['--sparse-capture-angle-limit']:
        if key in command:
            i=command.index(key);del command[i:i+2]
    if '--enforce-rail-during-search' in command:
        command.remove('--enforce-rail-during-search')  # Unused optimizer flag; native replay still enforces the canonical rail.
    command.extend(['--replay-only','--recompute-tracking-feedback','--tracking-feedback-regularization',str(args.regularization),
                    '--tracking-feedback-decimal-digits',str(args.decimal_digits)])
    if args.export_value_factors:
        command.extend(['--export-tracking-value-factors',args.export_value_factors])
    subprocess.run([sys.executable,'scripts/run_frontier_experiment.py','--directory',str(out/'synthesis'),'--input',str(source),'--input',str(reference),'--',*command],check=True)
    d=json.loads((out/'synthesis/result.json').read_text());prior=json.loads(source.read_text())
    if not np.array_equal(d['controller']['controls'],prior['controller']['controls']) or not np.array_equal(d['search']['nominal_coordinate_states'],prior['search']['nominal_coordinate_states']):
        raise ValueError('tracking-only comparison changed inherited controls or nodes')
    d['tracking_only_comparison']=dict(source=file_metadata(source),regularization=args.regularization,
        decimal_digits=args.decimal_digits,state_epsilon_override=args.state_epsilon,
        controls_identical=True,nominal_states_identical=True,
        feedback_recomputed='pure_LTV_MP_Riccati' if args.decimal_digits else 'pure_LTV_QR',
        reserved_seeds_used=False,no_runtime_state_projection=True)
    dump_json(d,out/'result.json');print(json.dumps({k:v for k,v in d['result'].items() if not isinstance(v,(dict,list))}))


if __name__=='__main__':main()
