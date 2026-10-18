"""Native-defect line search of the existing full-noise corrector's state step."""
import argparse
import json
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.direct_shooting import native_successors
from spatialpole.world_coordinates import native_geometry
from spatialpole.experiment import freeze_inputs,write_json
from eleven_compression_features import physical_features
from eleven_initial_continuation import load
from trajectory_expert_support import sha


def defects(geometry,predicted,targets):
    residual=[]
    for successor,target in zip(predicted,targets):
        rotation,omega=native_geometry(geometry,target)
        residual.append(physical_features(geometry,successor,target,rotation,omega))
    residual=np.array(residual)
    return .5*float(np.sum(residual*residual)),float(np.max(abs(residual))),residual


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();source,out=Path(a.attempt),Path(a.out);freeze_inputs(out,a)
    assert (source/'completed.json').exists()
    v,m,plan=[load(source/name) for name in ('virtual_reference.npz','native_models.npz','model_plan.npz')]
    assert float(v['fraction'])==1.
    geometry=SpatialCartpole(Config(n_links=11));native=SpatialCartpole(geometry.config);native.derivative_workers=4
    try:
        baseline,bmax,residual=defects(geometry,m['predicted'],v['states'][1:])
        np.savez_compressed(out/'source_defects.npz',residual=residual)
        rows=[];admitted=False
        delta=np.einsum('tij,tj->ti',m['E'],plan['errors'])
        for index,alpha in enumerate((1.,.5,.25)):
            states=np.array([geometry.integrate(x,alpha*d) for x,d in zip(v['states'],delta)])
            states[0]=v['states'][0].copy()
            actions=np.array([geometry.deliver(u) for u in v['actions'].astype(float)+alpha*(plan['controls']-v['actions'].astype(float))])
            geometry_ok=bool(np.max(abs(states[:,:2]))<=3.)
            last=[]
            for x in states[-251:]:geometry.set_state(x);last.append(geometry.tilts())
            last_tilt=float(np.max(last));geometry_ok=bool(geometry_ok and last_tilt<.15)
            if not geometry_ok:
                rows.append(dict(alpha=alpha,geometry_passed=False,last5_max_tilt=last_tilt));continue
            predicted=native_successors(native,states[:-1],actions)
            score,maximum,residual=defects(geometry,predicted,states[1:])
            admitted=bool(score<=.8*baseline)
            row=dict(alpha=alpha,geometry_passed=True,physical_defect_cost=score,physical_defect_max=maximum,
                reduction=(baseline-score)/baseline,admitted=admitted,native_steps=native.native_steps,last5_max_tilt=last_tilt)
            rows.append(row);np.savez_compressed(out/f'trial{index}.npz',states=states,actions=actions,predicted=predicted,residual=residual)
            print(json.dumps(row),flush=True)
            if admitted:
                np.savez_compressed(out/'accepted_predictor.npz',states=states,actions=actions,fraction=1.);break
        assert native.native_steps<=18000 and geometry.native_steps==0
        write_json(out/'summary.json',dict(completed=True,predictor_admitted=admitted,native_steps=native.native_steps,
            baseline_physical_defect_cost=baseline,baseline_physical_defect_max=bmax,results=rows,
            source_model_sha256=sha(source/'native_models.npz'),qualified=False,reserved_cases_consumed=0,
            scope='full-noise virtual Newton seed, independent intervals only; no physical task success'))
        write_json(out/'completed.json',dict(status='completed'))
    finally:geometry.close();native.close()


if __name__=='__main__':main()
