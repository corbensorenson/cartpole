#!/usr/bin/env python
"""Inspect local tracking derivatives and forced error components offline.

Finite-difference linear components do not establish nonlinear causality.
The recorded nonlinear rollout remains the execution evidence.
"""
import argparse,json
from pathlib import Path
import numpy as np
from gcartpole.config import load_config,dump_json
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import MujocoTransition
from gcartpole.high_order_dynamics import FourthOrderMujocoTransition
from gcartpole.evidence import file_metadata,utc_timestamp,runtime_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',required=True);p.add_argument('--seconds',type=float,default=6)
    p.add_argument('--out',required=True);args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    if not np.isfinite(args.seconds) or args.seconds<=0:raise ValueError('positive finite prefix duration required')
    audit=json.loads(Path(args.audit).read_text());records=[]
    for prior in audit['records']:
        source_path=prior['source']['path'];config_path=prior['config']['path']
        if file_metadata(source_path)['sha256']!=prior['source']['sha256'] or file_metadata(config_path)['sha256']!=prior['config']['sha256']:raise ValueError('Audit source or config changed')
        source=json.loads(Path(source_path).read_text());c=source['controller'];cfg=load_config(config_path)
        env=NLinkCartPoleEnv(cfg,progress=1.,seed=source['seed']);env.reset(seed=source['seed'])
        cls=FourthOrderMujocoTransition if c['derivative_order']==4 else MujocoTransition
        transition=cls(env,np.asarray(c['coordinate_transform']),continuous_angles=True)
        x=np.asarray(source['search']['nominal_coordinate_states']);u=np.asarray(c['controls']);k=np.asarray(c['feedback_gains'])
        errors=np.asarray(prior['actual_nominal_errors']);gaps=np.asarray(prior['gaps']);delivered=np.asarray(prior['delivered_actions'])
        count=min(round(args.seconds/env.dt),len(delivered));nx=x.shape[1]
        gap_component=np.zeros(nx);action_component=np.zeros(nx);remainder_component=np.zeros(nx)
        rows=[];matrices=[];exact_actual_gap=0.
        for i in range(count):
            a,b=transition.linearize(x[i],float(u[i]),state_epsilon=c['state_epsilon'],action_epsilon=c['action_epsilon']);b=b[:,0]
            closed=a+np.outer(b,k[i])
            feedback=float(k[i]@errors[i])
            action_noise=float(delivered[i]-u[i]-feedback)
            remainder=errors[i+1]-gaps[i]-a@errors[i]-b*(delivered[i]-u[i])
            actual_state=x[i]+errors[i]
            # For the native replay identity, use the saved physical state
            # directly instead of adding the error back to a nominal node.
            physical=np.r_[source['selected_state']['qpos'],source['selected_state']['qvel']] if i==0 else np.array(source['result']['trajectory'][i-1]['qpos']+source['result']['trajectory'][i-1]['qvel'])
            actual_state=transition.to_coordinates(physical)
            next_physical=np.array(source['result']['trajectory'][i]['qpos']+source['result']['trajectory'][i]['qvel'])
            native_gap=float(np.max(np.abs(transition(actual_state,float(delivered[i]))-transition.to_coordinates(next_physical))))
            exact_actual_gap=max(exact_actual_gap,native_gap)
            gap_component=closed@gap_component+gaps[i]
            action_component=closed@action_component+b*action_noise
            remainder_component=closed@remainder_component+remainder
            sum_error=float(np.max(np.abs(gap_component+action_component+remainder_component-errors[i+1])))
            rows.append(dict(step=i,time_seconds=i*env.dt,actual_error=float(np.max(np.abs(errors[i+1]))),
                native_actual_step_gap=native_gap,nonlinear_and_derivative_remainder=float(np.max(np.abs(remainder))),
                action_rounding_or_clipping_residual=action_noise,raw_action=float(u[i]+feedback),
                local_closed_loop_radius=float(np.max(np.abs(np.linalg.eigvals(closed)))),
                local_closed_loop_spectral_norm=float(np.linalg.norm(closed,2)),
                gap_component=float(np.max(np.abs(gap_component))),action_component=float(np.max(np.abs(action_component))),
                remainder_component=float(np.max(np.abs(remainder_component))),component_sum_rounding_error=sum_error))
            matrices.append(dict(A=a.tolist(),B=b.tolist()))
            if (i+1)%50==0:print(json.dumps(dict(n_links=env.n,processed_steps=i+1,latest=rows[-1])),flush=True)
        records.append(dict(source=prior['source'],config=prior['config'],n_links=env.n,
            derivative_order=c['derivative_order'],state_epsilon=c['state_epsilon'],action_epsilon=c['action_epsilon'],
            maximum_native_actual_step_gap=exact_actual_gap,rows=rows,linearizations=matrices))
        env.close()
    dump_json(dict(not_solution=True,benchmark_evidence=False,generated_at=utc_timestamp(),runtime=runtime_metadata(),
        audit=file_metadata(args.audit),parameters=vars(args),records=records,
        scope='Approximate finite-difference local tracking models at saved nominal nodes. Additive components use the recorded error and delivered action to define the remainder; they are not independent nonlinear counterfactuals. Radius is per-node and does not establish finite-horizon stability. Native recorded-step gaps are independently recomputed with saved physical states. No live state projection.'),out)
if __name__=='__main__':main()
