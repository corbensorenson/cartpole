#!/usr/bin/env python
"""Probe hypothetical nominal arrival nodes without claiming physical reachability."""
import argparse,json
from pathlib import Path
import numpy as np
from gcartpole.config import dump_json,load_config
from gcartpole.evidence import file_metadata,data_sha256,runtime_metadata,utc_timestamp
from scripts.audit_link_count_frontier import exact_capture
from scripts.search_swingup_capture import lqr_gain


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--source',required=True)
    p.add_argument('--times',type=float,nargs='+',default=[8,9,10,12])
    p.add_argument('--gain-cache');p.add_argument('--out',required=True)
    args=p.parse_args();out=Path(args.out)
    if out.exists():raise FileExistsError(out)
    cfg=load_config(args.config);source=json.loads(Path(args.source).read_text());c=source['controller']
    gain=lqr_gain(cfg,progress=c['lqr_progress'],fd_eps=1e-7,control_cost=c['lqr_control_cost'],q_weights=c['lqr_weights'],decimal_digits=c['lqr_decimal_digits'])
    cache_matches=None
    if args.gain_cache:
        cached=next(r for r in json.loads(Path(args.gain_cache).read_text())['records'] if r['n_links']==cfg['env']['n_links'])
        cache_matches=bool(np.array_equal(gain,np.asarray(cached['gain'])))
        if not cache_matches:raise ValueError('cache gain differs from recomputed source design')
    nodes=np.asarray(source['search']['nominal_coordinate_states']);transform=np.asarray(c['coordinate_transform'])
    physical=nodes@np.linalg.inv(transform).T;dt=cfg['env']['timestep']*cfg['env']['frame_skip'];d=cfg['env']['n_links']+1
    payload=dict(generated_at=utc_timestamp(),not_solution=True,benchmark_evidence=False,
        source=file_metadata(args.source),config=file_metadata(args.config),runtime=runtime_metadata(),
        gain=gain.tolist(),gain_sha256=data_sha256(gain.tolist()),cache_matches_recomputed_gain=cache_matches,
        scope='Hypothetical nominal-node upright capture, not the executed hanging-start prefix. No live plant is projected or reset during an episode.',records=[])
    for seconds in args.times:
        index=round(seconds/dt)
        if not 0<index<len(nodes) or not np.isclose(index*dt,seconds):raise ValueError('time must select a nominal interior control tick')
        state=dict(qpos=physical[index,:d].tolist(),qvel=physical[index,d:].tolist())
        result=exact_capture(cfg,state,gain*c['lqr_scale'])
        passed=bool(result['success'] and result['trajectory_clean'] and result['steps']==round(8/dt) and result['termination_reason']=='time_limit')
        payload['records'].append(dict(seconds=seconds,initial_state=state,result=result,full_eight_second_capture_passed=passed))
        print(json.dumps({'seconds':seconds,'full_capture_passed':passed,'result':result}),flush=True)
    dump_json(payload,out)


if __name__=='__main__':main()
