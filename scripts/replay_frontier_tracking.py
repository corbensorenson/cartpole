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
    args=p.parse_args()
    source=Path(args.source);out=Path(args.directory)
    if (out/'synthesis').exists() or (out/'result.json').exists():raise FileExistsError(out)
    reference=source.parent/'synthesis/execution.json'
    command=json.loads(reference.read_text())['command']
    for key,value in {'--initial-controller':str(source),'--state-json':str(source),'--optimizer':'sqrt-ilqr','--out':str(out/'synthesis/result.json')}.items():
        command[command.index(key)+1]=value
    for key in ['--sparse-capture-angle-limit']:
        if key in command:
            i=command.index(key);del command[i:i+2]
    command.extend(['--replay-only','--recompute-tracking-feedback','--tracking-feedback-regularization',str(args.regularization)])
    subprocess.run([sys.executable,'scripts/run_frontier_experiment.py','--directory',str(out/'synthesis'),'--input',str(source),'--input',str(reference),'--',*command],check=True)
    d=json.loads((out/'synthesis/result.json').read_text());prior=json.loads(source.read_text())
    if not np.array_equal(d['controller']['controls'],prior['controller']['controls']) or not np.array_equal(d['search']['nominal_coordinate_states'],prior['search']['nominal_coordinate_states']):
        raise ValueError('tracking-only comparison changed inherited controls or nodes')
    d['tracking_only_comparison']=dict(source=file_metadata(source),regularization=args.regularization,controls_identical=True,nominal_states_identical=True,feedback_recomputed='pure_LTV_QR',reserved_seeds_used=False,no_runtime_state_projection=True)
    dump_json(d,out/'result.json');print(json.dumps({k:v for k,v in d['result'].items() if not isinstance(v,(dict,list))}))


if __name__=='__main__':main()
