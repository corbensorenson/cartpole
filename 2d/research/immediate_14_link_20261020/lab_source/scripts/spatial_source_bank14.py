"""Frozen 14-link two-source motion selection; original numerical helpers reused.

Only native-checked physical proposal defects rank the two existing sources.
Exactly one source is refined, with the original four-model Newton schedule.
There is no seed input, fallback-source grid, cache optimization, or physics edit.
"""
from pathlib import Path
import numpy as np
from spatialpole import Config,SpatialCartpole
from spatialpole.experiment import write_json
from spatialpole.direct_shooting import native_successors
from spatialpole.native_mpc import exact_bytes
from spatial_synthesis_geometry import Dimensions,corrector_model,CorrectorPolicy,defects,error
from spatial_observed_synthesis import Plan,ForecastFailure,forecast,prepare,planning_budget,MAX_MODELS
from eleven_value_sequence import coordinates,solve_value_sequence,mapped_error,physical_cost,forward
from eleven_synthesis_protocol import read,sha
from working_source14_support import bundle as working_bundle


def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}


def bank(path):
    path=Path(path);m=read(path/'frozen_manifest.json')
    assert m['protocol']=='state-only-two-source14-v1'
    for directory,group in (('source0',m['source0_files']),('source1',m['source1_files'])):
        for name,digest in group.items():assert sha(path/directory/name)==digest
    first=working_bundle(path/'source0');folder=path/'source1'
    second=dict(v=load(folder/'virtual_reference.npz'),m=load(folder/'native_models.npz'),
        c=load(folder/'coordinates.npz'),w=load(folder/'model_plan.npz')['w'],expected=load(folder/'actual.npz'))
    return [first,second]


def rank(rows):
    admitted=[(row['physical_defect_cost'],i) for i,row in enumerate(rows)
              if row['geometry_passed'] and row['predictor_admitted']]
    return min(admitted)[1] if admitted else None


def budget(config):
    assert config==Config(n_links=14)
    return 2*6004+MAX_MODELS*(Dimensions.from_config(config).model_native_steps+80+12000)+(MAX_MODELS-1)*18000


def refine_prepared(config,initial,reference,runtime,out):
    """Literal original Newton-loop body; accepts a checked prepared reference."""
    dimensions=Dimensions.from_config(config);max_native=planning_budget(config)
    out=Path(out);out.mkdir(exist_ok=False)
    np.savez_compressed(out/'observed_state.npz',state=initial)
    v=reference;exact_bytes(v['states'][0],initial)
    native=SpatialCartpole(config);native.derivative_workers=4
    geometry=SpatialCartpole(config);closed=0;rows=[];result_plan=None
    def total():return native.native_steps+closed
    try:
        for iteration in range(MAX_MODELS):
            assert total()+dimensions.model_native_steps+80+2*config.horizon*config.frame_skip <= max_native
            folder = out/f'iteration{iteration}'; folder.mkdir()
            np.savez_compressed(folder/'virtual_reference.npz', **v)
            before = native.native_steps
            r, F, E, predicted, _ = corrector_model(native, geometry, v, folder)
            assert native.native_steps-before == dimensions.model_native_steps
            baseline, bmax, _ = defects(geometry, predicted, v['states'][1:])
            rng = np.random.default_rng(1001950+iteration); checks = []
            for t in (0, 160, 205, 325, 1000):
                d = rng.normal(size=dimensions.quotient); d /= np.linalg.norm(d); dx = E[t]@d
                u = v['actions'][t].astype(float); du = -u/max(1., float(np.linalg.norm(u))); eps = 1e-4
                xs = np.array([geometry.integrate(v['states'][t], k*eps*dx) for k in range(1, 5)])
                us = np.array([u+k*eps*du for k in range(1, 5)])
                assert np.linalg.norm(us, axis=1).max() <= 80.
                xn = native_successors(native, xs, us)
                ys = np.array([error(geometry, v, F, x, t+1) for x in xn])
                measured = (-25.*r['d'][t]+48.*ys[0]-36.*ys[1]+16.*ys[2]-3.*ys[3])/(12.*eps)
                expected = r['A'][t]@d+r['B'][t]@du
                discrepancy = float(np.max(abs(measured-expected))); assert discrepancy < 1e-5
                checks.append(dict(phase=t, error=discrepancy))
                np.savez_compressed(folder/f'probe{t}.npz', states=xs, actions=us,
                                    successors=xn, measured=measured, expected=expected)
            c = coordinates(r, np.eye(2)*.001, terminal_scale=1.)
            np.savez_compressed(folder/'coordinates.npz', **c)
            try:
                u, e, w, proof = solve_value_sequence(c, np.zeros(dimensions.quotient), v['states'],
                    v['actions'].astype(float), runtime, folder/'certificate.npz', time_limit=120.)
            except (AssertionError, RuntimeError) as exception:
                row = dict(iteration=iteration, model_rejected=True, error=str(exception),
                           checks=checks, inclusive_native_steps=total())
                rows.append(row); write_json(folder/'summary.json', row); break
            assert mapped_error(r, e, u-v['actions']) < 1e-6
            score = physical_cost(r, e, u-v['actions'], np.eye(2)*.001, terminal_scale=1.)
            _, be, bu = forward(c, np.zeros(dimensions.quotient), np.zeros((1500, 2)))
            base = physical_cost(r, be, bu, np.eye(2)*.001, terminal_scale=1.)
            assert abs(score-base-.5*np.sum(w*w))/max(1., abs(score)) < 1e-6
            np.savez_compressed(folder/'model_plan.npz', controls=u, errors=e, w=w)
            try:
                tape, result = forecast(config, initial, lambda env:CorrectorPolicy(env, v, F, c, w),
                                        folder/'native_forecast.npz')
            except ForecastFailure as failure:
                closed += failure.native_steps
                raise
            closed += result['inclusive_native_steps']
            row = dict(iteration=iteration, result=result, checks=checks, proof=proof,
                model_cost=score, baseline_physical_defect=baseline,
                baseline_physical_max=bmax, inclusive_native_steps=total())
            rows.append(row); write_json(folder/'summary.json', row)
            write_json(folder/'completed.json', dict(status='completed'))
            print(f"planner iteration={iteration} feasible={result['continuation_feasible']} "
                  f"commands={result['commands']} native_steps={total()}", flush=True)
            if result['continuation_feasible']:
                exact_bytes(tape['states'][0], initial)
                np.savez_compressed(out/'accepted_anchor.npz', **tape)
                result_plan = Plan(config, v, F, c, w, tape, total()); break
            if iteration == MAX_MODELS-1:
                break
            delta = np.einsum('tij,tj->ti', E, e); selected = False; updates = []
            for index, alpha in enumerate((1., .5, .25)):
                assert total()+6000 <= max_native
                xs = np.array([geometry.integrate(x, alpha*d) for x, d in zip(v['states'], delta)])
                xs[0] = initial.copy()
                us = np.array([geometry.deliver(z) for z in
                    v['actions'].astype(float)+alpha*(u-v['actions'].astype(float))])
                if np.max(abs(xs[:, :2])) > 3.:
                    continue
                last = []
                for x in xs[-251:]:
                    geometry.set_state(x); last.append(geometry.tilts())
                if np.max(last) >= .15:
                    continue
                xn = native_successors(native, xs[:-1], us)
                cost, maximum, residual = defects(geometry, xn, xs[1:])
                selected = bool(cost <= .8*baseline)
                updates.append(dict(alpha=alpha, physical_defect=cost,
                                    maximum=maximum, accepted=selected))
                np.savez_compressed(folder/f'update{index}.npz', states=xs, actions=us,
                                    predicted=xn, residual=residual)
                if selected:
                    v = dict(states=xs, actions=us, fraction=np.array(1.)); break
            write_json(folder/'updates.json', dict(results=updates))
            write_json(out/'progress.json', dict(results=rows, inclusive_native_steps=total()))
            if not selected:
                break
        assert total()<=max_native and geometry.native_steps==0
        write_json(out/'summary.json',dict(completed=True,plan_found=result_plan is not None,
            results=rows,model_and_interval_native_steps=native.native_steps,
            forecast_native_steps=closed,inclusive_native_steps=total(),
            actual_benchmark_episodes=0,reserved_cases_consumed=0,qualified=False,max_models=MAX_MODELS))
        write_json(out/'completed.json',dict(status='completed'))
        return result_plan
    except BaseException as exception:
        write_json(out/'failure.json',dict(type=type(exception).__name__,error=str(exception),
            completed_forecast_native_steps=closed,owned_native_steps=native.native_steps,accounting_complete=False))
        raise
    finally:native.close();geometry.close()


def synthesize_bank(config,observed,sources,runtime,out):
    assert config==Config(n_links=14) and len(sources)==2 and observed.shape==(102,)
    out=Path(out);out.mkdir(exist_ok=False)
    np.savez_compressed(out/'observed_state.npz',state=observed)
    for i,b in enumerate(sources):
        if np.array_equal(observed,b['v']['states'][0]):
            write_json(out/'summary.json',dict(completed=True,plan_found=True,reused_source=True,
                selected_source=i,inclusive_native_steps=0,actual_benchmark_episodes=0,qualified=False))
            write_json(out/'completed.json',dict(status='completed'))
            return Plan(config,b['v'],b['m']['F'],b['c'],b['w'],b['expected'],0,True)
    geometry,native=SpatialCartpole(config),SpatialCartpole(config)
    rows=[];refine_steps=0
    try:
        for i,b in enumerate(sources):
            folder=out/f'warm{i}';folder.mkdir();before=native.native_steps
            row=prepare(geometry,native,b['v'],b['m'],b['c'],observed,runtime,folder)
            assert native.native_steps-before<=6004
            row['native_steps']=native.native_steps-before;rows.append(row)
            write_json(folder/'summary.json',row)
        chosen=rank(rows)
        write_json(out/'selection.json',dict(results=rows,selected_source=chosen,
            rule='minimum absolute native-checked physical defect among ordinarily admissible proposals, source index breaks ties',
            native_steps=native.native_steps))
        plan=None
        if chosen is not None:
            v=load(out/f'warm{chosen}'/'accepted_predictor.npz')
            plan=refine_prepared(config,observed,v,runtime,out/'refinement')
            refine_steps=read(out/'refinement/summary.json')['inclusive_native_steps']
            if plan is not None:plan.native_steps+=native.native_steps
        total=native.native_steps+refine_steps
        assert total<=budget(config) and geometry.native_steps==0
        write_json(out/'summary.json',dict(completed=True,plan_found=plan is not None,reused_source=False,
            selected_source=chosen,warm_native_steps=native.native_steps,refinement_native_steps=refine_steps,
            inclusive_native_steps=total,max_planning_native=budget(config),max_models=MAX_MODELS,
            original_reset_episodes=0,actual_benchmark_episodes=0,qualified=False,reserved_cases_consumed=0))
        write_json(out/'completed.json',dict(status='completed'))
        return plan
    except BaseException as exception:
        write_json(out/'failure.json',dict(type=type(exception).__name__,error=str(exception),
            warm_native_steps=native.native_steps,completed_refinement_native_steps=refine_steps,
            owned_partial_refinement_counter='refinement/failure.json',accounting_complete=False))
        raise
    finally:native.close();geometry.close()
