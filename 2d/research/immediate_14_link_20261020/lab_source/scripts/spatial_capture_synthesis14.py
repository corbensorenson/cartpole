"""State-only capture continuation, built when the live measured event occurs."""
from pathlib import Path
import json
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.direct_shooting import native_successors
from spatialpole.experiment import write_json
from spatialpole.native_mpc import exact_bytes
from spatial_synthesis_geometry import CorrectorPolicy,error,defects
from spatial_suffix_feedback import SegmentFeedback,build_model
from spatial_physical_sequence import solve_physical
from eleven_value_sequence import coordinates,solve_value_sequence,mapped_error,physical_cost,forward


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def source_data(folder):
    folder=Path(folder)
    v,m,c,p=[load(folder/name) for name in ('virtual_reference.npz','native_models.npz','coordinates.npz','model_plan.npz')]
    return v,m,c,p['w']


def sliced_model(model,phase):
    assert 1<=phase<=1251
    return {k:value[phase:].copy() for k,value in model.items()}


def entry_phase(states,geometry):
    for phase,state in enumerate(states):
        geometry.set_state(state)
        if geometry.tilts().max()<.15:return phase
    return None


def observed_mean(geometry,r,errors,controls,observed):
    baseline=dict(states=r['reference_states'].copy(),actions=r['reference_actions'].copy())
    baseline['states'][0]=observed.copy()
    delta=np.einsum('tij,tj->ti',r['E'],errors)
    states=np.asarray([geometry.integrate(x,d) for x,d in zip(r['reference_states'],delta)])
    states[0]=observed.copy();actions=np.asarray([geometry.deliver(u) for u in controls])
    return baseline,dict(states=states,actions=actions)


def changed_inputs(left,right):
    return np.asarray([t for t in range(len(left['actions']))
        if left['states'][t].tobytes()!=right['states'][t].tobytes() or
           left['actions'][t].tobytes()!=right['actions'][t].tobytes()],dtype=int)


def probe_phases(H):
    return (0,160,205,325,1000) if H>1000 else tuple(np.linspace(0,H-1,5).astype(int))


def forecast(config,initial,phase,streak,clock,v,F,c,w,path):
    env,cold=SpatialCartpole(config),SpatialCartpole(config);policy=SegmentFeedback(config,v,F,c,w)
    states=[initial.copy()];actions=[];raw=[];tilts=[];H=len(v['actions'])
    try:
        env.set_state(initial);env.steps=phase;env.streak=env.best_streak=streak;env.data.time=clock
        for t in range(H):
            before,now=env.state().tobytes(),env.data.time;request=policy(states[-1],t)
            assert env.state().tobytes()==before and env.data.time==now
            force=env.deliver(request);state,_,done,truncated,info=env.step_force(force)
            states.append(state);actions.append(force);raw.append(request);tilts.append(env.tilts())
            if done or truncated:break
        state=initial.copy()
        for t,u in enumerate(actions):state=cold.transition(state,u);exact_bytes(state,states[t+1]);exact_bytes(cold.tilts(),tilts[t])
        assert cold.min_plane_margin==info['plane_margin'] and cold.warning_seen==info['physics_warning']
        np.savez_compressed(path,states=np.asarray(states),actions=np.asarray(actions,np.float32),raw_actions=np.asarray(raw),tilts=np.asarray(tilts),
            config=np.array(json.dumps(config.__dict__)),start_phase=np.array(phase),prefix_upright_streak=np.array(streak))
        return dict(continuation_feasible=bool(len(actions)==H and info['success'] and not info['physics_warning'] and info['plane_margin']>=0),
            commands=len(actions),best_hold_seconds=info['best_hold_seconds'],plane_margin=info['plane_margin'],physics_warning=info['physics_warning'],
            cold_replay_byte_exact=True,inclusive_native_steps=env.native_steps+cold.native_steps,rollout_native_steps=env.native_steps,cold_native_steps=cold.native_steps,
            benchmark_success=False,start_phase=phase,prefix_upright_streak=streak)
    except BaseException as exc:
        write_json(path.with_suffix('.failure.json'),dict(type=type(exc).__name__,error=str(exc),rollout_native_steps=env.native_steps,cold_native_steps=cold.native_steps,accounting_complete=True));raise
    finally:policy.close();env.close();cold.close()


def build_remaining(config,observed,phase,clock,source,runtime,out):
    """No live environment, seed, event schedule or future observations are inputs."""
    assert config==Config(n_links=14) and 1<=phase<=1251
    out=Path(out);out.mkdir(exist_ok=False);H=1500-phase
    r=sliced_model(source,phase);v=dict(states=r['reference_states'],actions=r['reference_actions'])
    np.savez_compressed(out/'source_model.npz',**r);np.savez_compressed(out/'observed_state.npz',state=observed)
    write_json(out/'event.json',dict(phase=phase,remaining_commands=H,prefix_upright_streak=1,physical_clock=clock))
    geometry,native=SpatialCartpole(config),SpatialCartpole(config);native.derivative_workers=4;closed=0
    def finish(stage,plan=None,**extra):
        summary=dict(passed=True,stage=stage,component_plan_found=plan is not None,native_steps=native.native_steps+closed,
            native_design_steps=native.native_steps,forecast_native_steps=closed,phase=phase,suffix_commands=H,
            new_noisy_samples=0,reserved_cases_consumed=0,**extra)
        assert summary['native_steps']<=4+1456*H+80
        write_json(out/'summary.json',summary);write_json(out/'completed.json',dict(status='completed'))
        return plan,summary
    try:
        initial=error(geometry,v,r['F'],observed,0)
        screen=out/'screen';screen.mkdir()
        solution,formulation=solve_physical(r,initial,v['states'],v['actions'],runtime,screen)
        write_json(screen/'summary.json',dict(formulation=formulation))
        if solution is None:return finish('screen_rejected',formulation=formulation)
        controls,errors=solution;np.savez_compressed(screen/'model_plan.npz',controls=controls,errors=errors)
        baseline,proposal=observed_mean(geometry,r,errors,controls,observed)
        mean=out/'mean';mean.mkdir();np.savez_compressed(mean/'baseline_reference.npz',**baseline);np.savez_compressed(mean/'virtual_proposal.npz',**proposal)
        source_ref=dict(states=r['reference_states'],actions=r['reference_actions']);base_indices=changed_inputs(baseline,source_ref)
        assert set(base_indices.tolist())<= {0}
        centers=r['predicted'].copy()
        if len(base_indices):centers[base_indices]=native_successors(native,baseline['states'][:-1][base_indices],baseline['actions'][base_indices])
        base,bmax,bres=defects(geometry,centers,baseline['states'][1:]);margin=float(3.-np.max(abs(proposal['states'][:,:2])));tail=[]
        for state in proposal['states'][-251:]:geometry.set_state(state);tail.append(geometry.tilts())
        tail=float(np.max(tail));admitted=bool(margin>=0 and tail<.15 and np.linalg.norm(proposal['actions'].astype(float),axis=1).max()<=80)
        indices=changed_inputs(proposal,baseline);cost=maximum=None;predicted=residual=None;ready=False
        if admitted:
            predicted=centers.copy()
            if len(indices):predicted[indices]=native_successors(native,proposal['states'][:-1][indices],proposal['actions'][indices])
            cost,maximum,residual=defects(geometry,predicted,proposal['states'][1:]);ready=bool(cost<=.8*base)
        np.savez_compressed(mean/'interval_check.npz',baseline_predicted=centers,baseline_residual=bres,changed_baseline_inputs=base_indices,changed_inputs=indices,
            **({} if predicted is None else dict(predicted=predicted,residual=residual)))
        write_json(mean/'summary.json',dict(geometry_admitted=admitted,ready_for_new_model=ready,native_steps=native.native_steps,
            baseline_native_defect=base,baseline_native_max=bmax,proposal_native_defect=cost,proposal_native_max=maximum,virtual_plane_margin=margin,last5_virtual_max_tilt=tail))
        if not ready:return finish('mean_rejected',formulation=formulation)
        model=out/'model';model.mkdir();np.savez_compressed(model/'virtual_reference.npz',**proposal)
        before=native.native_steps;nr,F,E,predicted=build_model(native,geometry,proposal,model)
        assert native.native_steps-before==1444*H
        exact_bytes(predicted,load(mean/'interval_check.npz')['predicted'])
        rng=np.random.default_rng(142563);checks=[]
        for t in probe_phases(H):
            d=rng.normal(size=74);d/=np.linalg.norm(d);u=proposal['actions'][t].astype(float);du=-u/max(1.,float(np.linalg.norm(u)));eps=1e-4
            xs=np.asarray([geometry.integrate(proposal['states'][t],k*eps*(E[t]@d)) for k in range(1,5)]);us=np.asarray([u+k*eps*du for k in range(1,5)])
            assert np.linalg.norm(us,axis=1).max()<=80
            successors=native_successors(native,xs,us);ys=np.asarray([error(geometry,proposal,F,x,t+1) for x in successors])
            measured=(-25*nr['d'][t]+48*ys[0]-36*ys[1]+16*ys[2]-3*ys[3])/(12*eps);expected=nr['A'][t]@d+nr['B'][t]@du
            discrepancy=float(np.max(abs(measured-expected)));assert discrepancy<1e-5;checks.append(dict(phase=int(t),error=discrepancy))
            np.savez_compressed(model/f'probe{t}.npz',states=xs,actions=us,successors=successors,measured=measured,expected=expected)
        c=coordinates(nr,np.eye(2)*.001,terminal_scale=1.);np.savez_compressed(model/'coordinates.npz',**c)
        try:controls,e,w,proof=solve_value_sequence(c,np.zeros(74),proposal['states'],proposal['actions'].astype(float),runtime,model/'certificate.npz',time_limit=120.)
        except (AssertionError,RuntimeError) as exc:
            write_json(model/'rejection.json',dict(type=type(exc).__name__,error=str(exc),checks=checks));return finish('model_certificate_rejected')
        assert mapped_error(nr,e,controls-proposal['actions'])<1e-6
        score=physical_cost(nr,e,controls-proposal['actions'],np.eye(2)*.001,terminal_scale=1.)
        _,be,bu=forward(c,np.zeros(74),np.zeros((H,2)));constant=physical_cost(nr,be,bu,np.eye(2)*.001,terminal_scale=1.)
        assert abs(score-constant-.5*np.sum(w*w))/max(1.,abs(score))<1e-6
        np.savez_compressed(model/'model_plan.npz',controls=controls,errors=e,w=w)
        result=forecast(config,observed,phase,1,clock,proposal,F,c,w,model/'native_forecast.npz');closed=result['inclusive_native_steps']
        write_json(model/'summary.json',dict(iteration=0,result=result,checks=checks,proof=proof,model_cost=score))
        plan=(proposal,F,c,w) if result['continuation_feasible'] else None
        assert geometry.native_steps==0
        return finish('forecast_passed' if plan is not None else 'forecast_failed',plan,result=result)
    except BaseException as exc:
        write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),owned_native_steps=native.native_steps,completed_forecast_native_steps=closed,
            partial_forecast_counters='model/native_forecast.failure.json',accounting_complete=False));raise
    finally:geometry.close();native.close()


class CaptureRejected(RuntimeError):pass


class MeasuredCapturePolicy:
    def __init__(self,config,source,builder):
        v,m,c,w=source;self.geometry=SpatialCartpole(config)
        self.prefix=CorrectorPolicy(self.geometry,v,m['F'],c,w)
        self.builder=builder;self.suffix=None;self.phase=None;self.report=None;self.config=config

    def __call__(self,state,step):
        if self.suffix is None and step>0:
            self.geometry.set_state(state)
            if self.geometry.tilts().max()<.15:
                self.phase=step
                if step>1251:raise CaptureRejected('Insufficient original task time after first entry')
                plan,self.report=self.builder(np.array(state,copy=True),step)
                if plan is None:raise CaptureRejected('Measured continuation did not pass admission')
                v,F,c,w=plan;self.suffix=SegmentFeedback(self.config,v,F,c,w)
        return self.prefix(state,step) if self.suffix is None else self.suffix(state,step-self.phase)

    def close(self):
        assert self.geometry.native_steps==0
        self.geometry.close();self.prefix.close()
        if self.suffix is not None:self.suffix.close()
