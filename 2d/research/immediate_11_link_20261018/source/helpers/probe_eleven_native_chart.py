"""Native local error dynamics at saved observed eleven-link noisy states."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.aligned_value import aligned_next_value
from spatialpole.control import linearize_many
from spatialpole.direct_shooting import native_successors
from spatialpole.quotient import axial_quotient_frame
from spatialpole.experiment import freeze_inputs,write_json
from spatialpole.native_mpc import exact_bytes
from eleven_sequence_qp import reduced_cache
from trajectory_expert_support import sha


def load(path):
    with np.load(path) as data:return {k:data[k] for k in data.files}


def error(env,v,cache,state,t):
    ref,F,_=aligned_next_value(env,v['states'][t],cache['F'][t],state)
    return F@env.difference(state,ref)


def derivative(env,v,cache,state,t,eps):
    result=np.empty((59,70))
    for j in range(70):
        delta=np.zeros(70);delta[j]=eps
        result[:,j]=(error(env,v,cache,env.integrate(state,delta),t)-error(env,v,cache,env.integrate(state,-delta),t))/(2*eps)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();run,out=Path(a.run),Path(a.out);assert (run/'completed.json').exists();freeze_inputs(out,a)
    for source,target in [('input_route.npz','input_route.npz'),('input_cache.npz','input_cache.npz'),
                          ('coordinates.npz','coordinates.npz'),('seed20261116.npz','input_actual.npz')]:
        shutil.copyfile(run/source,out/target)
    assert sha(out/'input_route.npz')=='2df4ade7bd626df977289e527db7d1e6e1add701e31ee4ce76a3f7f36f711bf3'
    v,cache,c,data=[load(out/name) for name in ('input_route.npz','input_cache.npz','coordinates.npz','input_actual.npz')]
    reduced=reduced_cache(cache);phases=np.array([160,175,180,184,185]);xs=data['states'][phases];us=data['actions'][phases]
    native=SpatialCartpole(Config(n_links=11));native.derivative_workers=4;geometry=SpatialCartpole(Config(n_links=11))
    rows=[]
    try:
        models,predicted=linearize_many(native,xs,us.astype(float),eps=.0003,order=4,return_next=True)
        exact_bytes(predicted,data['states'][phases+1]);rng=np.random.default_rng(7300185)
        for index,t in enumerate(phases):
            t=int(t);state=xs[index];successor=predicted[index];A,B=models[index]
            e=error(geometry,v,cache,state,t);en=error(geometry,v,cache,successor,t+1)
            J=derivative(geometry,v,cache,state,t,1e-5);Jn=derivative(geometry,v,cache,successor,t+1,1e-5)
            Jcheck=derivative(geometry,v,cache,state,t,5e-6)
            jacobian_error=float(np.max(abs(J-Jcheck)));assert jacobian_error<1e-6
            assert np.linalg.matrix_rank(J[:,35:])==np.linalg.matrix_rank(Jn[:,35:])==35
            basis=axial_quotient_frame(geometry,state)['E'];T=J@basis
            anew=np.linalg.solve(T.T,(Jn@A@basis).T).T;bnew=Jn@B
            offset=en-anew@e-bnew@(us[index].astype(float)-v['actions'][t])
            old=reduced['A'][t]@e+reduced['B'][t]@(us[index].astype(float)-v['actions'][t])+reduced['d'][t]
            bias=en-old;checks=[]
            for number in range(2):
                direction=rng.normal(size=59);direction/=np.linalg.norm(direction);dx=basis@direction
                du=-us[index].astype(float)/max(1.,float(np.linalg.norm(us[index].astype(float))))
                eps=1e-4;states=np.array([geometry.integrate(state,k*eps*dx) for k in range(1,5)])
                actions=np.array([us[index].astype(float)+k*eps*du for k in range(1,5)])
                assert np.linalg.norm(actions,axis=1).max()<=80.
                successors=native_successors(native,states,actions)
                output=np.array([error(geometry,v,cache,x,t+1) for x in successors])
                secant=(-25.*en+48.*output[0]-36.*output[1]+16.*output[2]-3.*output[3])/(12.*eps)
                analytic=anew@(J@dx)+bnew@du
                defect=float(np.max(abs(secant-analytic)));assert defect<1e-5
                checks.append(dict(error=defect,physical_state_direction=dx.tolist(),force_direction=du.tolist()))
                np.savez_compressed(out/f'p{t}_probe{number}.npz',states=states,actions=actions,successors=successors,
                    errors=output,secant=secant,analytic=analytic)
            np.savez_compressed(out/f'p{t}_model.npz',state=state,action=us[index],successor=successor,
                raw_A=A,raw_B=B,J=J,Jnext=Jn,Jcheck=Jcheck,basis=basis,chart=T,
                error=e,next_error=en,A=anew,B=bnew,offset=offset,cached_prediction=old,bias=bias)
            rows.append(dict(phase=t,jacobian_difference=jacobian_error,chart_condition=float(np.linalg.cond(T)),
                cached_bias_max=float(np.max(abs(bias))),cached_bias_value_norm=float(np.linalg.norm(c['S'][t+1]@bias)),
                A_change_norm=float(np.linalg.norm(anew-reduced['A'][t])),B_change_norm=float(np.linalg.norm(bnew-reduced['B'][t])),
                maximum_local_derivative_error=max(check['error'] for check in checks),checks=checks))
            write_json(out/'progress.json',dict(results=rows,native_steps=native.native_steps))
            print(json.dumps({k:value for k,value in rows[-1].items() if k!='checks'}),flush=True)
        assert native.native_steps==5940 and geometry.native_steps==0
        write_json(out/'summary.json',dict(passed=True,results=rows,native_steps=native.native_steps,qualified=False,reserved_cases_consumed=0,
            source_actual_sha256=sha(out/'input_actual.npz'),all_centers_byte_exact=True,physical_velocities=35,
            scope='observed-state native/chart derivative component only, no new full task'))
        write_json(out/'completed.json',dict(status='completed'))
    except BaseException as exception:
        write_json(out/'failure.json',dict(type=type(exception).__name__,error=str(exception),native_steps=native.native_steps));raise
    finally:native.close();geometry.close()


if __name__=='__main__':main()
