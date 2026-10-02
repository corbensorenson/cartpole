#!/usr/bin/env python
"""Append explicit virtual capture nodes to an offline optimizer initializer.

This does not inherit physical feasibility or provide executable evidence.
"""
import argparse,json
from pathlib import Path
import numpy as np
from gcartpole.config import dump_json,load_config
from gcartpole.evidence import file_metadata


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--config',required=True)
    p.add_argument('--tail-seconds',type=float,default=5.)
    p.add_argument('--out',required=True)
    args=p.parse_args()
    if Path(args.out).exists():raise FileExistsError(args.out)
    source=json.loads(Path(args.source).read_text());cfg=load_config(args.config)
    dt=cfg['env']['timestep']*cfg['env']['frame_skip'];n=cfg['env']['n_links'];dim=2*(n+1)
    if not np.isfinite(args.tail_seconds) or args.tail_seconds<=0 or not np.isclose(args.tail_seconds/dt,round(args.tail_seconds/dt),rtol=0,atol=1e-10):
        raise ValueError('positive tick-aligned virtual tail required')
    controls=np.asarray(source['controller']['controls'],dtype=np.float32).astype(float)
    transform=np.asarray(source['controller']['coordinate_transform'])
    coordinates=np.asarray(source['search']['nominal_coordinate_states'])
    if transform.shape!=(dim,dim) or coordinates.shape!=(len(controls)+1,dim):
        raise ValueError('source dimensions must match the target config')
    states=np.linalg.solve(transform,coordinates.T).T
    if not np.all(np.isfinite(states)) or not np.all(np.isfinite(controls)):
        raise ValueError('finite initialization required')
    tail=round(args.tail_seconds/dt);target=np.zeros(dim)
    target[1:n+1]=2*np.pi*np.round(states[-1,1:n+1]/(2*np.pi))
    states=np.vstack([states,np.repeat(target[None],tail,axis=0)])
    controls=np.r_[controls,np.zeros(tail)]
    dump_json(dict(not_solution=True,initialization_only=True,source=file_metadata(args.source),
        config=file_metadata(args.config),selected_state=source['selected_state'],seed=source.get('seed',0),
        controller=dict(type='offline_virtual_capture_initializer',continuous_angles=True,
            physical_shooting=True,coordinate_transform=np.eye(dim).tolist(),controls=controls.tolist(),
            feedback_gains=np.zeros((len(controls),dim)).tolist(),feedback_placeholder=True,
            capture_start_seconds=(len(controls)-tail)*dt,virtual_tail_seconds=args.tail_seconds,
            angle_branch_terminal_target=target.tolist()),
        search=dict(nominal_coordinate_states=states.tolist()),
        note='Source optimizer nodes transformed to physical coordinates, followed by virtual upright nodes. Nonzero target dynamics gaps are expected. No node may overwrite the runtime plant.'),args.out)


if __name__=='__main__':main()
