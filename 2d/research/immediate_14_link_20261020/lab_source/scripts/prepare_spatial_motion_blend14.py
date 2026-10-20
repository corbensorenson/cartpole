"""One observed-state convex physical blend of four existing successful motions."""
import argparse
import itertools
from pathlib import Path
import shutil
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.circular import state_from_physical
from spatialpole.world_coordinates import native_geometry,world_coordinate_error
from spatialpole.direct_shooting import native_successors
from spatialpole.experiment import freeze_inputs,write_json
from spatialpole.native_mpc import exact_bytes
from spatial_synthesis_geometry import defects
from eleven_synthesis_protocol import read,sha

SCALE=np.r_[[.05,.05],np.full(28,.02),np.full(44,.04)]


def simplex_fit(C,target):
    """Solve all faces of the four-variable convex QP; independently check KKT."""
    G,b=C.T@C,C.T@target;answers=[]
    for k in range(1,5):
        for active in itertools.combinations(range(4),k):
            active=list(active);K=np.block([[G[np.ix_(active,active)],np.ones((k,1))],[np.ones((1,k)),np.zeros((1,1))]])
            try:x=np.linalg.solve(K,np.r_[b[active],1.])
            except np.linalg.LinAlgError:continue
            w=np.zeros(4);w[active]=x[:k]
            if np.min(w)<-1e-12:continue
            g=G@w-b;inactive=[j for j in range(4) if j not in active]
            if inactive and np.min(g[inactive]-np.mean(g[active])) < -1e-10:continue
            answers.append((float(np.sum((C@w-target)**2)),w))
    assert answers;cost,w=min(answers,key=lambda row:row[0])
    gradient=C.T@(C@w-target);active=w>1e-10;lagrange=float(np.mean(gradient[active]))
    assert abs(np.sum(w)-1.)<1e-12 and np.min(w)>=-1e-12
    assert np.max(abs(gradient[active]-lagrange))<1e-9
    assert np.all(gradient[~active]-lagrange>=-1e-9)
    return w,cost,dict(sum_error=abs(float(w.sum())-1.),active_stationarity=float(np.max(abs(gradient[active]-lagrange))),
        minimum_inactive_dual=float(np.min(gradient[~active]-lagrange)) if np.any(~active) else None)


def tilt_vectors(directions):
    r=np.linalg.norm(directions[:,:2],axis=1)
    assert np.all((r>1e-12)|(directions[:,2]>0.)), 'Ambiguous hanging antipode in virtual chart'
    angle=np.arctan2(r,directions[:,2]);return directions[:,:2]*(angle/np.maximum(r,1e-30))[:,None]


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-inventory',action='append',required=True)
    for name in ('failed-inventory','out'):p.add_argument('--'+name,required=True)
    a=p.parse_args();sources=list(map(Path,a.source_inventory));out=Path(a.out);assert len(sources)==4
    failed_path=Path(a.failed_inventory);failed=read(failed_path);case=Path(failed['run'])
    assert failed['passed'] and failed['links']==14 and not failed['task_success']
    assert sha(case/'summary.json')==failed['summary_sha256']
    with np.load(case/'observed_state.npz') as z:initial=z['state']
    geometry,native=SpatialCartpole(Config(n_links=14)),SpatialCartpole(Config(n_links=14));native.derivative_workers=4
    try:
        observed,_=geometry.reset(seed=20261120,options=dict(mode='hanging',angle_noise=.02));exact_bytes(initial,observed)
        nominal,_=geometry.reset(seed=20261016,options=dict(mode='hanging',angle_noise=0.));R0,w0=native_geometry(geometry,nominal)
        tapes=[];lineage=[];features=[]
        for seed,path in zip(range(20261116,20261120),sources):
            inventory=read(path);run=Path(inventory['run']);s=read(run/'summary.json')
            assert inventory['passed'] and inventory['task_success'] and inventory['links']==14 and not inventory['qualified']
            assert sha(run/'summary.json')==inventory['summary_sha256'] and (run/'completed.json').exists()
            with np.load(run/'actual.npz') as z:t={k:z[k] for k in z.files}
            result=s['result'];assert result['seed']==seed and result['success'] and result['replay_exact'] and result['steps']==1500
            assert result['angle_noise']==.02 and result['policy_metadata']['park_seconds']==0.
            assert result['policy_metadata']['physical_velocities']==44 and result['best_hold_seconds']>=5.
            assert t['states'].shape==(1501,102) and t['actions'].shape==(1500,2)
            if 'audit' in inventory:
                audit=Path(inventory['audit']);g=read(audit/'summary.json')
                assert g['passed'] and sha(audit/'summary.json')==inventory['audit_summary_sha256']
                assert sha(run/'actual.npz')==inventory.get('actual_sha256',g.get('actual_sha256'))
                assert g.get('case_summary_sha256',g.get('source_summary_sha256'))==sha(run/'summary.json')
                if 'actual_sha256' not in g:
                    assert g['actual_cold_byte_exact_via_identical_forecast'] and g['all_forecasts_cold_byte_exact']
                    refinement=run/'planning/refinement'
                    last=read(refinement/'summary.json')['results'][-1]['iteration']
                    with np.load(refinement/f'iteration{last}'/'native_forecast.npz') as forecast:
                        for key in ('states','actions','raw_actions','tilts'):exact_bytes(t[key],forecast[key])
            else:assert sha(run/'actual.npz')==inventory['actual_sha256']
            features.append(world_coordinate_error(geometry,t['states'][0],nominal,R0,w0)/SCALE)
            tapes.append(t);lineage.append(dict(inventory=str(path.resolve()),inventory_sha256=sha(path),run=str(run.resolve()),
                summary_sha256=sha(run/'summary.json'),actual_sha256=sha(run/'actual.npz'),seed=seed))
        C=np.array(features).T;target=world_coordinate_error(geometry,initial,nominal,R0,w0)/SCALE
        weights,cost,kkt=simplex_fit(C,target);single=np.sum((C-target[:,None])**2,axis=0)
        assert cost<=.8*min(single) and np.count_nonzero(weights>1e-10)>=2,'No distinct improved blend; do not refine'
        freeze_inputs(out,a);np.savez_compressed(out/'weight_fit.npz',C=C,target=target,weights=weights,scale=SCALE)
        for name in ('prepare_spatial_motion_blend14.py','spatial_synthesis_geometry.py','eleven_synthesis_protocol.py'):
            shutil.copyfile(Path(__file__).with_name(name),out/name)
        states=[];direction_errors=[];rate_errors=[];directions=[];omegas=[];tilts=[]
        for phase in range(1501):
            geometry_rows=[native_geometry(geometry,t['states'][phase]) for t in tapes]
            direction=np.einsum('k,kij->ij',weights,np.array([r[:,:,2] for r,w in geometry_rows]))
            direction/=np.linalg.norm(direction,axis=1)[:,None]
            omega=np.einsum('k,kij->ij',weights,np.array([w for r,w in geometry_rows]))
            xy=sum(weight*t['states'][phase,:2] for weight,t in zip(weights,tapes))
            velocity=sum(weight*t['states'][phase,58:60] for weight,t in zip(weights,tapes))
            vectors=tilt_vectors(direction);x=state_from_physical(geometry,xy,vectors,velocity,omega)
            R,w=native_geometry(geometry,x)
            direction_errors.append(float(np.max(abs(R[:,:,2]-direction))));rate_errors.append(float(np.max(abs(w-omega))))
            states.append(x);directions.append(direction);omegas.append(omega);tilts.append(np.linalg.norm(vectors,axis=1))
        states=np.asarray(states);assert max(direction_errors)<1e-12 and max(rate_errors)<1e-10
        checks=[]
        for phase in (0,160,325,1000,1500):
            columns=[];base=states[phase];vectors=tilt_vectors(directions[phase]);eps=1e-5
            for j in range(44):
                values=[]
                for sign in (-1,1):
                    v=base[58:60].copy();w=omegas[phase].copy()
                    if j<2:v[j]+=sign*eps
                    else:w.flat[j-2]+=sign*eps
                    values.append(state_from_physical(geometry,base[:2],vectors,v,w)[58:])
                columns.append((values[1]-values[0])/(2*eps))
            rank=int(np.linalg.matrix_rank(np.asarray(columns).T));assert rank==44;checks.append(dict(phase=phase,rank=rank))
        actions=np.asarray([geometry.deliver(sum(weight*t['actions'][phase].astype(float) for weight,t in zip(weights,tapes))) for phase in range(1500)])
        states[0]=initial.copy();margin=float(3.-np.max(abs(states[:,:2])));tail=float(np.max(tilts[-251:]))
        assert margin>=0. and tail<.15 and np.max(np.linalg.norm(actions.astype(float),axis=1))<=80.
        np.savez_compressed(out/'virtual_reference.npz',states=states,actions=actions)
        np.savez_compressed(out/'observed_state.npz',state=initial)
        manifest=dict(config=Config(n_links=14).__dict__,seed=20261120,protocol='observed-state-convex-physical-motion14-v1',
            source_lineage=lineage,failed_inventory_sha256=sha(failed_path),weight_fit_sha256=sha(out/'weight_fit.npz'),
            reference_sha256=sha(out/'virtual_reference.npz'),helper_sha256={p.name:sha(p) for p in out.glob('*.py')},
            source_sha256=read(out/'inputs.json')['source_sha256'],max_native_steps=6000,
            qualified=False,new_noisy_samples=0,reserved_cases_consumed=0)
        write_json(out/'frozen_manifest.json',manifest)
        predicted=native_successors(native,states[:-1],actions);defect,maximum,residual=defects(geometry,predicted,states[1:])
        assert native.native_steps==6000 and geometry.native_steps==0 and np.isfinite(defect)
        np.savez_compressed(out/'interval_check.npz',predicted=predicted,residual=residual)
        report=dict(passed=True,links=14,qualified=False,task_success=False,native_steps=native.native_steps,weights=weights.tolist(),
            initial_scaled_squared_error=cost,single_source_scaled_squared_errors=single.tolist(),simplex_kkt=kkt,
            all_physical_rates=44,rate_chart_checks=checks,maximum_direction_roundtrip_error=max(direction_errors),
            maximum_world_rate_roundtrip_error=max(rate_errors),virtual_plane_margin=margin,last5_virtual_max_tilt=tail,
            physical_defect_cost=defect,physical_defect_max=maximum,reference_sha256=manifest['reference_sha256'],
            original_reset_episodes=0,new_noisy_samples=0,reserved_cases_consumed=0,
            scope='one new state-conditioned blend of existing positive14 motions and native interval check; proposal only, no executable capture, task or qualification claim')
        write_json(out/'summary.json',report);write_json(out/'completed.json',dict(status='completed'))
        print({k:report[k] for k in ('passed','native_steps','weights','physical_defect_cost','physical_defect_max','virtual_plane_margin','last5_virtual_max_tilt')},flush=True)
    except BaseException as error:
        if out.exists():write_json(out/'failure.json',dict(type=type(error).__name__,error=str(error),native_steps=native.native_steps,accounting_complete=True))
        raise
    finally:geometry.close();native.close()


if __name__=='__main__':main()
