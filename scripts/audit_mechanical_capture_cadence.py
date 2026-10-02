#!/usr/bin/env python
"""Matched control-cadence components with the same native physics timestep.

Changing cadence here never changes the canonical promotion configuration.
"""
import argparse,json
from pathlib import Path
import numpy as np
from gcartpole.config import dump_json,load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata,runtime_metadata,utc_timestamp
from gcartpole.lqr_design import high_precision_discrete_lqr
from gcartpole.upright_mechanics import upright_rk4_matrices
from scripts.audit_link_count_frontier import WEIGHTS,exact_capture
from scripts.make_lqr_checkpoint import absolute_angle_cost


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',required=True)
    p.add_argument('--frame-skips',type=int,nargs='+',default=[2,1])
    p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    baseline=json.loads(Path(args.baseline).read_text())
    payload=dict(schema_version=1,generated_at=utc_timestamp(),not_solution=True,benchmark_evidence=False,
        baseline=file_metadata(args.baseline),runtime=runtime_metadata(),records=[],parameters=vars(args),
        scope='Local upright component comparison only. Same 0.005 s native RK4 timestep, canonical parsed geometry, ±80 N and ±3 m, delivered float32 actions, matched directions and eight-second duration. Only feedback cadence and its corresponding LQR design change; promotion configs remain 50 Hz.')
    for old in baseline['records']:
        n=old['n_links'];cfg_path=f'configs/swingup{n}_uniform.yaml'
        base=load_config(cfg_path)
        if base['env']['timestep']!=.005 or base['env']['frame_skip']!=4:
            raise ValueError('requires the declared canonical 50 Hz reference')
        directions=np.array(old['directions'])
        previous=next(x for x in old['cases'] if x['label']=='mechanical_preserved_mp')
        payload['records'].append(dict(n_links=n,frame_skip=4,control_hz=50,baseline_reused=True,
            diagnostics=previous['diagnostics'],local_capture=previous['local_capture']))
        for skip in args.frame_skips:
            cfg=load_config(cfg_path);cfg['env']['frame_skip']=skip
            env=NLinkCartPoleEnv(cfg,progress=1.,seed=0)
            try:a,b,mechanics=upright_rk4_matrices(env,decimal_digits=100)
            finally:env.close()
            row=dict(n_links=n,frame_skip=skip,control_hz=1/(.005*skip),config=file_metadata(cfg_path),
                component_override={'env.frame_skip':skip},mechanics=mechanics,local_capture={})
            payload['records'].append(row)
            try:
                gain,_,diag=high_precision_discrete_lqr(a.tolist(),b.tolist(),absolute_angle_cost(n,WEIGHTS),np.array([[1000.]]),decimal_digits=100,preserve_input_precision=True)
                row.update(gain=gain.ravel().tolist(),diagnostics=diag)
                for amp in (1e-13,1e-15):
                    episodes=[exact_capture(cfg,dict(qpos=np.r_[0.,amp*v[:n]].tolist(),qvel=np.r_[0.,amp*v[n:]].tolist()),gain.ravel()) for v in directions]
                    row['local_capture'][str(amp)]=dict(successes=sum(x['success'] for x in episodes),episodes=episodes)
            except (ValueError,np.linalg.LinAlgError,ZeroDivisionError) as error:row['error']=f'{type(error).__name__}: {error}'
            dump_json(payload,str(out)+'.progress.json')
            print(json.dumps(dict(n_links=n,hz=row['control_hz'],error=row.get('error'),successes={k:v['successes'] for k,v in row['local_capture'].items()})),flush=True)
    dump_json(payload,out)


if __name__=='__main__':main()
