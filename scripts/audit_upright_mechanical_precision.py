#!/usr/bin/env python
"""Matched local-capture component audit of FD, rounded and MP mechanical inputs."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from gcartpole.config import dump_json,load_config
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.evidence import file_metadata,runtime_metadata,utc_timestamp
from gcartpole.lqr_design import high_precision_discrete_lqr
from gcartpole.upright_mechanics import upright_rk4_matrices
from scripts.audit_link_count_frontier import WEIGHTS,exact_capture
from scripts.make_lqr_checkpoint import absolute_angle_cost,finite_difference_dynamics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts',type=int,nargs='+',default=[14,20])
    p.add_argument('--decimal-digits',type=int,default=100)
    p.add_argument('--amplitudes',type=float,nargs='+',default=[1e-11,1e-13,1e-15])
    p.add_argument('--directions',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261010)
    p.add_argument('--fd-design-json',required=True)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    prior=json.loads(Path(args.fd_design_json).read_text())
    prior_by_count={r['n_links']:r for r in prior['records']}
    payload=dict(schema_version=1,generated_at=utc_timestamp(),not_solution=True,
        scope='Local upright components only; no hanging-start evidence or nonlinear certificate.',
        parameters=vars(args),runtime=runtime_metadata(),fd_source=file_metadata(args.fd_design_json),records=[],
        references=['https://mujoco.readthedocs.io/en/3.3.5/computation/',
                    'https://github.com/google-deepmind/mujoco/blob/3.3.5/src/engine/engine_forward.c'],
        fidelity='Same parsed canonical target, native binary64 MuJoCo physics and delivered float32 actions for every case. MP is used only for local design. Concurrent search timings are not serial performance comparisons.')
    for n in args.counts:
        cfg_path=f'configs/swingup{n}_uniform.yaml'
        cfg=load_config(cfg_path)
        env=NLinkCartPoleEnv(cfg,progress=1.,seed=0)
        try:
            a,b,mechanics=upright_rk4_matrices(env,decimal_digits=args.decimal_digits)
        finally:
            env.close()
        a64,b64=np.array(a.tolist(),dtype=float),np.array(b.tolist(),dtype=float)
        fd_a,fd_b=finite_difference_dynamics(cfg,1.,1e-7)
        q,r=absolute_angle_cost(n,WEIGHTS),np.array([[1000.]])
        prev=prior_by_count[n]
        for key,matrix in zip(('a','b','q','r'),(fd_a,fd_b,q,r)):
            if not np.array_equal(np.asarray(prev['input_matrices'][key]),matrix):
                raise ValueError(f'cached FD design differs at count {n}: {key}')
        directions=np.random.default_rng(args.seed+n).uniform(-1,1,(args.directions,2*n))
        record=dict(n_links=n,config=file_metadata(cfg_path),mechanics=mechanics,directions=directions.tolist(),cases=[],
            relative_a_difference_from_fd=float(np.linalg.norm(a64-fd_a)/np.linalg.norm(fd_a)),
            relative_b_difference_from_fd=float(np.linalg.norm(b64-fd_b)/np.linalg.norm(fd_b)),
            mechanical_a=[[str(v) for v in row] for row in a.tolist()],
            mechanical_b=[[str(v) for v in row] for row in b.tolist()])
        payload['records'].append(record)
        for label,preserve in [('finite_difference_cached',False),('mechanical_rounded_binary64',False),('mechanical_preserved_mp',True)]:
            started=time.monotonic()
            case=dict(label=label,preserved_input_precision=preserve)
            record['cases'].append(case)
            try:
                if label=='finite_difference_cached':
                    gain=np.asarray(prev['gain'])[None,:]
                    diagnostics=prev['high_precision_diagnostics']
                    case['reused_gain']=file_metadata(args.fd_design_json)
                else:
                    gain,_,diagnostics=high_precision_discrete_lqr(a.tolist() if preserve else a64,b.tolist() if preserve else b64,q,r,
                        decimal_digits=args.decimal_digits,preserve_input_precision=preserve)
                case.update(gain=gain.ravel().tolist(),diagnostics=diagnostics,
                            design_wall_time_seconds=time.monotonic()-started,local_capture={})
                for amp in args.amplitudes:
                    episodes=[exact_capture(cfg,dict(qpos=np.r_[0.,amp*v[:n]].tolist(),qvel=np.r_[0.,amp*v[n:]].tolist()),gain.ravel()) for v in directions]
                    case['local_capture'][str(amp)]=dict(successes=sum(x['success'] for x in episodes),episodes=episodes)
            except (ValueError,np.linalg.LinAlgError,ZeroDivisionError) as error:
                case.update(error=f'{type(error).__name__}: {error}',design_wall_time_seconds=time.monotonic()-started)
            dump_json(payload,str(out)+'.progress.json')
            print(json.dumps(dict(n_links=n,label=label,error=case.get('error'),
                successes={k:v['successes'] for k,v in case.get('local_capture',{}).items()})),flush=True)
    dump_json(payload,out)


if __name__=='__main__':
    main()
