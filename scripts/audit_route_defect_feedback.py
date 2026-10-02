#!/usr/bin/env python
"""Measure exact native nominal gaps and their projection onto saved feedback.

This is an offline diagnostic, not a physical feasibility certificate or a
causal decomposition of a nonlinear failed rollout.
"""
import argparse,json
from pathlib import Path
import numpy as np
from gcartpole.config import load_config,dump_json
from gcartpole.env import NLinkCartPoleEnv
from gcartpole.ilqr import MujocoTransition
from gcartpole.evidence import file_metadata,utc_timestamp,runtime_metadata


def analyze(source_path,config_path):
    source=json.loads(Path(source_path).read_text());c=source['controller'];cfg=load_config(config_path)
    if not c['continuous_angles'] or c['phase_adaptive']:
        raise ValueError('Requires continuous angles and fixed-time saved feedback')
    nodes=np.asarray(source['search']['nominal_coordinate_states']);controls=np.asarray(c['controls']);gains=np.asarray(c['feedback_gains']);transform=np.asarray(c['coordinate_transform'])
    env=NLinkCartPoleEnv(cfg,progress=1.,seed=source['seed']);env.reset(seed=source['seed'])
    transition=MujocoTransition(env,transform,continuous_angles=True)
    gaps=np.asarray([transition(nodes[i],float(u))-nodes[i+1] for i,u in enumerate(controls)])
    checked=float(np.max(np.abs(gaps)));saved=c['final_trajectory_diagnostics']['maximum_dynamics_defect']
    if checked!=saved:raise ValueError(f'Recomputed defect differs: {checked} vs {saved}')
    count=min(len(controls),len(source['result']['trajectory']))
    rows=source['result']['trajectory'][:count]
    initial=np.r_[source['selected_state']['qpos'],source['selected_state']['qvel']]
    actual=np.vstack([initial,[row['qpos']+row['qvel'] for row in rows]])@transform.T
    errors=actual-nodes[:count+1]
    raw=np.array([controls[i]+float(gains[i]@errors[i]) for i in range(count)])
    requested=np.asarray([row['action'] for row in rows])
    # The trajectory's action field records the pre-interface request. Native
    # stepping casts that request to float32 before applying the held force.
    delivered=np.clip(requested.astype(np.float32),-1,1).astype(float)
    reconstructed=np.clip(raw.astype(np.float32),-1,1).astype(float)
    if not np.array_equal(reconstructed,delivered):raise ValueError('Saved feedback does not reconstruct delivered physical controls')
    # Gap i injects an error at node i+1, where the next saved gain acts.
    projections=np.array([float(gains[i+1]@gaps[i]) for i in range(len(controls)-1)])
    dt=env.dt
    def crossing(values,threshold):
        index=np.flatnonzero(values>threshold)
        return float(index[0]*dt) if len(index) else None
    summary=dict(n_links=env.n,maximum_native_gap=checked,
        maximum_next_action_projection=float(np.max(np.abs(projections))),
        median_next_action_projection=float(np.median(np.abs(projections))),
        maximum_feedback_norm=float(np.max(np.linalg.norm(gains,axis=1))),
        first_raw_action_saturation_time=crossing(np.abs(raw),1),
        saturated_feedback_ticks=int(np.count_nonzero(np.abs(raw)>1)),
        error_crossing_times={str(x):crossing(np.max(np.abs(errors),axis=1),x) for x in [1e-8,1e-6,1e-4,.01,.1]},
        delivered_actions_reconstructed_exactly=True,
        physical_episode={k:v for k,v in source['result'].items() if not isinstance(v,(dict,list))})
    env.close()
    return dict(source=file_metadata(source_path),config=file_metadata(config_path),summary=summary,
        dt=dt,gaps=gaps.tolist(),next_action_gap_projections=projections.tolist(),
        feedback_norms=np.linalg.norm(gains,axis=1).tolist(),actual_nominal_errors=errors.tolist(),
        raw_tracking_actions=raw.tolist(),requested_actions=requested.tolist(),
        delivered_actions=delivered.tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',action='append',required=True);p.add_argument('--config',action='append',required=True)
    p.add_argument('--out',required=True);args=p.parse_args()
    if len(args.source)!=len(args.config):raise ValueError('Each source requires a corresponding config')
    out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    records=[]
    for source,config in zip(args.source,args.config):
        r=analyze(source,config);records.append(r);print(json.dumps(r['summary']),flush=True)
    dump_json(dict(generated_at=utc_timestamp(),not_solution=True,benchmark_evidence=False,runtime=runtime_metadata(),records=records,
        scope='Exact native nominal gaps and next-tick saved-feedback projections; recorded hanging-start errors/actions. A next-action gap projection is a one-tick local sensitivity, not an accumulated nonlinear counterfactual or proof of causality. No plant state is projected during live execution.'),out)
if __name__=='__main__':main()
