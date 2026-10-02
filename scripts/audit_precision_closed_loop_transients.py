#!/usr/bin/env python
"""Isolate bounded-action and runtime arithmetic effects on a saved MP linear model.

All rollouts here are linear-model components, never MuJoCo benchmark evidence.
The native nonlinear comparison remains in the input component audit.
"""
import argparse,json
from pathlib import Path
import mpmath
import numpy as np
from gcartpole.config import dump_json
from gcartpole.evidence import file_metadata,runtime_metadata,utc_timestamp


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--amplitude',type=float,default=1e-15)
    p.add_argument('--steps',type=int,default=400)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    ctx=mpmath.mp.clone();ctx.dps=100
    prior=json.loads(Path(args.source).read_text())
    payload=dict(schema_version=1,generated_at=utc_timestamp(),not_solution=True,benchmark_evidence=False,
        source=file_metadata(args.source),parameters=vars(args),runtime=runtime_metadata(),records=[],
        scope='Linear-model components only: fixed saved rounded gain, same MP plant, matched directions. Mathematical-model arithmetic changes are disclosed, and native MuJoCo benchmark physics remains unchanged.',
        note='A stable linear spectral radius is not a bounded-input or nonlinear robustness certificate. Large linear excursions invalidate a local nonlinear approximation.')
    for record in prior['records']:
        n=record['n_links'];d=n+1
        saved=next(c for c in record['cases'] if c['label']=='mechanical_preserved_mp')
        a,b=ctx.matrix(record['mechanical_a']),ctx.matrix(record['mechanical_b'])
        k64=np.array(saved['gain']);k=ctx.matrix([k64.tolist()])
        row=dict(n_links=n,rounded_gain_spectral_radius=saved['diagnostics']['rounded_gain_high_precision_spectral_radius'],cases=[])
        payload['records'].append(row)
        for mode in ('mp_unsaturated','mp_bounded','mp_bounded_float32_action','mp_bounded_float64_feedback_float32_action'):
            case=dict(mode=mode,episodes=[]);row['cases'].append(case)
            for direction in record['directions']:
                initial=np.r_[0.,args.amplitude*np.array(direction[:n]),0.,args.amplitude*np.array(direction[n:])]
                x=ctx.matrix(initial.tolist())
                max_raw=ctx.mpf(0);max_cart=ctx.mpf(0);max_angle=ctx.mpf(0);saturated=0;quantized=0
                max_action_error=ctx.mpf(0);terminated='step_budget'
                for step in range(args.steps):
                    raw_mp=-(k*x)[0]
                    raw=ctx.mpf(float(-k64@np.array(x.tolist(),dtype=float).ravel())) if 'float64_feedback' in mode else raw_mp
                    max_raw=max(max_raw,abs(raw));saturated+=int(abs(raw)>1)
                    applied=max(ctx.mpf(-1),min(ctx.mpf(1),raw)) if 'bounded' in mode else raw
                    if 'float32_action' in mode:
                        before=applied;applied=ctx.mpf(float(np.float32(float(applied))))
                        max_action_error=max(max_action_error,abs(applied-before));quantized+=int(before!=applied)
                    x=a*x+b*applied
                    max_cart=max(max_cart,abs(x[0]));max_angle=max(max_angle,max(abs(sum(x[j+1] for j in range(i+1))) for i in range(n)))
                    if 'bounded' in mode and abs(x[0])>3:
                        terminated='linear_cart_crossed_3m';break
                case['episodes'].append(dict(steps=step+1,termination_reason=terminated,saturated_steps=saturated,
                    maximum_raw_action=str(max_raw),maximum_cart_excursion=str(max_cart),maximum_absolute_angle=str(max_angle),
                    quantized_steps=quantized,maximum_action_quantization_error=str(max_action_error)))
            dump_json(payload,str(out)+'.progress.json')
            print(json.dumps(dict(n_links=n,mode=mode,steps=[e['steps'] for e in case['episodes']],
                max_raw_action=max(float(e['maximum_raw_action']) for e in case['episodes']),
                max_cart=max(float(e['maximum_cart_excursion']) for e in case['episodes']))),flush=True)
    dump_json(payload,out)


if __name__=='__main__':main()
