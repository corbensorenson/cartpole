"""One causal measured-error uncertainty refresh on the original spatial plant."""
import hashlib
from pathlib import Path
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.direct_shooting import native_successors
from spatialpole.experiment import write_json
from spatialpole.native_mpc import exact_bytes
from spatial_capture_synthesis14 import observed_mean, changed_inputs, probe_phases
from spatial_physical_sequence import solve_physical
from spatial_suffix_feedback import build_model
from spatial_synthesis_geometry import error, defects
from eleven_value_sequence import right
from eleven_synthesis_protocol import read, sha
from audit_spatial_synthesis_models import load
from multiplicative_critic14 import design, DesignRangeError
from screen_multiplicative_critic14 import envelope
from multiplicative_critic_forecast14 import forecast
from multiplicative_control_bellman14 import MultiplicativeControlBellman
from native_control_bellman14 import NativeControlBellman
from witness_multiplicative_critic14 import AuditedQueryCache


def digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()


def execution(source,v,F,c,phase):
    ov,om,oc,_ = source
    H = len(v['actions'])
    assert phase+H==1500
    K = np.array([right(G,S) for G,S in zip(c['gain'],c['S'][:-1])])
    future = dict(S=c['S'],K=K,feed=c['feed'],value_residual=c['value_residual'],Cinv=np.zeros((H,2,2)))
    joined = {key:np.concatenate([oc[key][:phase],value]) for key,value in future.items()}
    joined['uncertainty_weight'] = np.r_[np.zeros(phase),c['uncertainty_weight']]
    ref = dict(states=np.concatenate([ov['states'][:phase],v['states']]),actions=np.concatenate([ov['actions'][:phase],v['actions']]))
    return ref,dict(F=np.concatenate([om['F'][:phase],F])),joined,np.zeros((1500,2))


def save_execution(out,bundle):
    for name,value in zip(('execution_reference.npz','execution_geometry.npz','execution_coordinates.npz'),bundle[:3]):
        np.savez_compressed(out/name,**value)
    np.savez_compressed(out/'execution_plan.npz',w=bundle[3])


def project_envelope(mapped_model,basis,proposal,critic,directions,geometry):
    """Forward existing quotient basis metadata; no model or physics change."""
    assert basis.shape==(len(proposal['states']),88,74)
    return envelope(dict(mapped_model,E=basis),proposal,critic,directions,geometry)


def build(config,observed,event,source,source_folder,C,runtime,out,caches=()):
    """Only current observation, past scores and fixed design inputs are accepted."""
    assert config==Config(n_links=14)
    phase = event['phase']
    assert 50<phase<=1250
    out = Path(out)
    out.mkdir(exist_ok=False)
    H = 1500-phase
    v,full,_,_ = source
    r = {key:value[phase:].copy() for key,value in full.items()}
    np.savez_compressed(out/'source_model.npz',**r)
    np.savez_compressed(out/'observed_state.npz',state=observed)
    write_json(out/'event.json',event)
    cache_key = [sha(Path(source_folder)/'native_models.npz'),str(phase),digest(observed),digest(C),sha(Path(runtime)/'receipt.json')]
    geometry,native = SpatialCartpole(config),SpatialCartpole(config)
    native.derivative_workers = 4
    closed = 0
    def finish(stage,bundle=None,**extra):
        report = dict(passed=True,stage=stage,component_plan_found=bundle is not None,native_steps=native.native_steps+closed,
            design_native_steps=native.native_steps,forecast_native_steps=closed,event=event,cache_used=False,cache_key=cache_key,**extra)
        assert report['native_steps']<=4+1504*H+80 and geometry.native_steps==0
        write_json(out/'summary.json',report)
        write_json(out/'completed.json',dict(status='completed'))
        return bundle,report
    try:
        initial = error(geometry,v,full['F'],observed,phase)
        np.savez_compressed(out/'initial_error.npz',error=initial)
        for cached in caches:
            if cached['key']!=cache_key:
                continue
            for path,value in cached['proof_files_sha256'].items():
                assert sha(Path(path))==value
            for name,value in cached['helper_sha256'].items():
                assert sha(Path(__file__).with_name(name))==value
            package = Path(__import__('spatialpole').__file__).parent
            assert {p.name:sha(p) for p in package.glob('*.py')}==cached['source_sha256']
            screen,mean,model,critic,old_forecast = map(Path,(cached[k] for k in ('screen','mean','model','critic','forecast')))
            expected = load(screen/'source_model.npz')
            assert set(r)==set(expected)
            for key in r:
                exact_bytes(r[key],expected[key])
            exact_bytes(initial,load(screen/'event.npz')['initial_error'])
            assert read(screen/'summary.json')['event']==event and read(screen/'summary.json')['linear_plan_found']
            plan = load(screen/'model_plan.npz')
            baseline,proposal = observed_mean(geometry,r,plan['errors'],plan['controls'],observed)
            for actual,name in ((baseline,'baseline_reference.npz'),(proposal,'virtual_proposal.npz')):
                expected = load(mean/name)
                for key in actual:
                    exact_bytes(actual[key],expected[key])
            nr = load(model/'native_models.npz')
            exact_bytes(nr['reference_states'],proposal['states'])
            exact_bytes(nr['reference_actions'],proposal['actions'])
            c = design(nr,C)
            expected = load(critic/'uncertain_critic.npz')
            for key in c:
                exact_bytes(c[key],expected[key])
            result,arrays = envelope(nr,proposal,c,C,geometry)
            assert result==read(critic/'summary.json')['result'] and result['passed']
            expected = load(critic/'model_envelope.npz')
            for key in arrays:
                exact_bytes(arrays[key],expected[key])
            bundle = execution(source,proposal,nr['F'],c,phase)
            for actual,name in zip(bundle[:3],('execution_reference.npz','execution_geometry.npz','execution_coordinates.npz')):
                expected = load(old_forecast/name)
                for key in actual:
                    exact_bytes(actual[key],expected[key])
            assert not np.any(bundle[3]) and read(old_forecast/'summary.json')['component_plan_found']
            save_execution(out,bundle)
            np.savez_compressed(out/'virtual_proposal.npz',**proposal)
            np.savez_compressed(out/'uncertain_critic.npz',**c)
            report = dict(passed=True,stage='cached_forecast_passed',component_plan_found=True,native_steps=0,design_native_steps=0,
                forecast_native_steps=0,event=event,cache_used=True,cache_key=cache_key,cache_inventory=cached['inventory'],
                inherited_native_steps=cached['inherited_native_steps'],forecast_folder=str(old_forecast),
                source_model_folder=str(model),all_constructor_mean_critic_envelope_and_execution_arrays_byte_exact=True)
            write_json(out/'summary.json',report)
            write_json(out/'completed.json',dict(status='completed'))
            return bundle,report
        screen = out/'screen'
        screen.mkdir()
        sv = dict(states=r['reference_states'],actions=r['reference_actions'])
        solution,formulation = solve_physical(r,initial,sv['states'],sv['actions'],runtime,screen)
        write_json(screen/'summary.json',dict(formulation=formulation))
        if solution is None:
            return finish('physical_screen_rejected')
        controls,errors = solution
        np.savez_compressed(screen/'model_plan.npz',controls=controls,errors=errors)
        baseline,proposal = observed_mean(geometry,r,errors,controls,observed)
        mean = out/'mean'
        mean.mkdir()
        np.savez_compressed(mean/'baseline_reference.npz',**baseline)
        np.savez_compressed(mean/'virtual_proposal.npz',**proposal)
        indices = changed_inputs(baseline,sv)
        assert set(indices.tolist())<={0}
        centers = r['predicted'].copy()
        if len(indices):
            centers[indices] = native_successors(native,baseline['states'][:-1][indices],baseline['actions'][indices])
        base,bmax,bres = defects(geometry,centers,baseline['states'][1:])
        margin = float(3.-np.max(abs(proposal['states'][:,:2])))
        tail = []
        for x in proposal['states'][-251:]:
            geometry.set_state(x)
            tail.append(geometry.tilts())
        tail = float(np.max(tail))
        admitted = bool(margin>=0 and tail<.15 and np.linalg.norm(proposal['actions'].astype(float),axis=1).max()<=80.)
        changed = changed_inputs(proposal,baseline)
        merit,maximum,predicted,residual = None,None,None,None
        if admitted:
            predicted = centers.copy()
            if len(changed):
                predicted[changed] = native_successors(native,proposal['states'][:-1][changed],proposal['actions'][changed])
            merit,maximum,residual = defects(geometry,predicted,proposal['states'][1:])
        ready = bool(admitted and merit<=.8*base)
        np.savez_compressed(mean/'interval_check.npz',baseline_predicted=centers,baseline_residual=bres,changed_baseline_inputs=indices,
            changed_inputs=changed,**({} if predicted is None else dict(predicted=predicted,residual=residual)))
        write_json(mean/'summary.json',dict(ready_for_new_model=ready,geometry_admitted=admitted,native_steps=native.native_steps,
            baseline_native_defect=base,baseline_native_max=bmax,proposal_native_defect=merit,proposal_native_max=maximum,
            virtual_plane_margin=margin,last251_virtual_max_tilt=tail))
        if not ready:
            return finish('mean_rejected')
        model = out/'model'
        model.mkdir()
        np.savez_compressed(model/'virtual_reference.npz',**proposal)
        nr,F,E,centers = build_model(native,geometry,proposal,model)
        exact_bytes(centers,predicted)
        checks = []
        rng = np.random.default_rng(143703)
        for t in probe_phases(H):
            d = rng.normal(size=74)
            d /= np.linalg.norm(d)
            u = proposal['actions'][t].astype(float)
            du = -u/max(1.,float(np.linalg.norm(u)))
            eps = 1e-4
            xs = np.array([geometry.integrate(proposal['states'][t],k*eps*(E[t]@d)) for k in range(1,5)])
            us = np.array([u+k*eps*du for k in range(1,5)])
            assert np.linalg.norm(us,axis=1).max()<=80.
            successors = native_successors(native,xs,us)
            ys = np.array([error(geometry,proposal,F,x,t+1) for x in successors])
            measured = (-25*nr['d'][t]+48*ys[0]-36*ys[1]+16*ys[2]-3*ys[3])/(12*eps)
            expected = nr['A'][t]@d+nr['B'][t]@du
            discrepancy = float(np.max(abs(measured-expected)))
            assert discrepancy<1e-5
            checks.append(dict(phase=int(t),error=discrepancy))
            np.savez_compressed(model/f'probe{t}.npz',states=xs,actions=us,successors=successors,measured=measured,expected=expected)
        try:
            c = design(nr,C)
        except DesignRangeError as exc:
            np.savez_compressed(out/'uncertain_critic.npz',**exc.partial)
            return finish('critic_range_rejected',checks=checks,reason=str(exc),rejected_phase=exc.phase)
        np.savez_compressed(out/'uncertain_critic.npz',**c)
        er,ea = project_envelope(nr,E,proposal,c,C,geometry)
        np.savez_compressed(out/'model_envelope.npz',**ea)
        write_json(out/'envelope.json',er)
        if not er['passed']:
            return finish('envelope_rejected',checks=checks)
        bundle = execution(source,proposal,F,c,phase)
        save_execution(out,bundle)
        result = forecast(config,observed,event,bundle,out)
        closed = result['inclusive_native_steps']
        return finish('forecast_passed' if result['continuation_feasible'] else 'forecast_rejected',
            bundle if result['continuation_feasible'] else None,checks=checks,result=result,forecast_folder=str(out),source_model_folder=str(model))
    except BaseException as exc:
        write_json(out/'failure.json',dict(type=type(exc).__name__,error=str(exc),owned_design_native_steps=native.native_steps,
            completed_forecast_native_steps=closed,partial_forecast_counters='adaptive_forecast.failure.json',accounting_complete=False))
        raise
    finally:
        geometry.close()
        native.close()


class RefreshRejected(RuntimeError):
    pass


class MeasuredUncertaintyPolicy:
    def __init__(self,config,source,builder):
        self.config,self.source,self.builder = config,source,builder
        v,m,c,w = source
        self.geometry = SpatialCartpole(config)
        self.prefix = NativeControlBellman(config,v,m,c,w)
        self.suffix,self.phase,self.report,self.event = None,None,None,None
        self.streak,self.best,self.margin,self.warning = 0,0,3.,False
        self.future_cache = None

    def __call__(self,state,phase):
        if self.suffix is None and phase>50:
            v,m,_,_ = self.source
            e = error(self.geometry,v,m['F'],state,phase)
            magnitude = float(np.linalg.norm(m['factors'][phase]@e))
            if magnitude>=1e-4:
                self.phase = phase
                self.event = dict(phase=phase,physical_error=magnitude,threshold=1e-4,after_command=50,remaining_commands=1500-phase,
                    logical_seconds=phase*.02,prefix_streak=self.streak,prefix_best_streak=self.best,prefix_plane_margin=self.margin,prefix_warning=self.warning)
                if phase>1250:
                    raise RefreshRejected('Insufficient original task time for one measured refresh')
                execution,self.report = self.builder(state.copy(),self.event)
                if execution is None:
                    raise RefreshRejected('Measured uncertainty continuation rejected')
                self.suffix = MultiplicativeControlBellman(self.config,*execution)
                future = Path(self.report['forecast_folder'])
                self.future_cache = AuditedQueryCache(self.suffix.predictor,load(future/'adaptive_forecast.npz'),load(future/'adaptive_native_queries.npz'),phase,1500)
        policy = self.prefix if self.suffix is None else self.suffix
        return policy(state,phase)

    @property
    def selected(self):
        return (self.prefix if self.suffix is None else self.suffix).selected

    def observe(self,state):
        self.geometry.set_state(state)
        self.streak = self.streak+1 if self.geometry.tilts().max()<.15 else 0
        self.best = max(self.best,self.streak)
        self.margin = min(self.margin,float(self.selected['margin']))
        self.warning = bool(self.warning or self.selected['warning'])

    def close(self):
        assert self.geometry.native_steps==0
        self.geometry.close()
        self.prefix.close()
        if self.suffix is not None:
            self.suffix.close()
