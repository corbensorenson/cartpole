"""Count-aware bounded planning for future immediate spatial swing-up frontiers."""
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.direct_shooting import native_successors
from spatialpole.experiment import write_json
from spatialpole.native_mpc import exact_bytes
from spatial_synthesis_geometry import Dimensions, corrector_model, CorrectorPolicy, defects, error
from eleven_value_sequence import coordinates, solve_value_sequence, mapped_error, physical_cost, forward

MAX_MODELS = 4
def planning_budget(config, models=MAX_MODELS):
    d = Dimensions.from_config(config); interval = config.frame_skip
    horizon_cost = config.horizon*interval
    return interval+horizon_cost+models*(d.model_native_steps+80+2*horizon_cost)+(models-1)*3*horizon_cost


class ForecastFailure(RuntimeError):
    def __init__(self, native_steps, message):
        self.native_steps = native_steps
        super().__init__(message)


def forecast(config,initial,factory,path):
    env=SpatialCartpole(config);cold=SpatialCartpole(env.config);policy=None
    try:
        env.set_state(initial);policy=factory(env);states=[initial.copy()];actions=[];raws=[];tilts=[]
        for t in range(config.horizon):
            before,clock=env.state().tobytes(),env.data.time
            force=policy(states[-1],t);assert before==env.state().tobytes() and clock==env.data.time
            u=env.deliver(force);state,_,done,truncated,info=env.step_force(u)
            states.append(state);actions.append(u);raws.append(force);tilts.append(env.tilts())
            if done or truncated:break
        state=initial.copy()
        for t,u in enumerate(actions):state=cold.transition(state,u);exact_bytes(state,states[t+1])
        data=dict(states=np.array(states),actions=np.array(actions,dtype=np.float32),raw_actions=np.array(raws),
            tilts=np.array(tilts),config=np.array(json.dumps(asdict(env.config))))
        np.savez_compressed(path,**data)
        feasible=bool(info['success'] and len(actions)==config.horizon and not info['physics_warning'] and info['plane_margin']>=0)
        return data,dict(continuation_feasible=feasible,commands=len(actions),best_hold_seconds=info['best_hold_seconds'],
            plane_margin=info['plane_margin'],physics_warning=info['physics_warning'],cold_replay_byte_exact=True,
            rollout_native_steps=env.native_steps,cold_native_steps=cold.native_steps,
            inclusive_native_steps=env.native_steps+cold.native_steps,
            benchmark_success=False,scope='internal continuation predictor, not original-noise task evidence')
    except BaseException as exception:
        cost = env.native_steps+cold.native_steps
        write_json(Path(path).with_suffix('.failure.json'), dict(type=type(exception).__name__,
            error=str(exception), rollout_native_steps=env.native_steps, cold_native_steps=cold.native_steps,
            counter_native_steps=cost, accounting_complete=False))
        raise ForecastFailure(cost, str(exception)) from exception
    finally:
        if policy is not None:policy.close()
        env.close();cold.close()


def prepare(geometry, native, v, m, c, observed_initial, runtime, out):
    """No seed/case label enters the proposal; all physical rates are retained."""
    initial_error = error(geometry, v, m['F'], observed_initial, 0)
    controls, errors, w, proof = solve_value_sequence(c, initial_error, v['states'],
        v['actions'].astype(float), runtime, out/'certificate.npz', time_limit=120.)
    assert mapped_error(m, errors, controls-v['actions']) < 1e-6
    np.savez_compressed(out/'model_plan.npz', controls=controls, errors=errors, w=w,
                        initial_error=initial_error)
    baseline_predicted = m['predicted'].copy()
    baseline_predicted[0] = native_successors(native, observed_initial[None], v['actions'][:1])[0]
    baseline, bmax, _ = defects(geometry, baseline_predicted, v['states'][1:])
    delta = np.einsum('tij,tj->ti', m['E'], errors)
    states = np.array([geometry.integrate(x, d) for x, d in zip(v['states'], delta)])
    states[0] = observed_initial.copy()
    actions = np.array([geometry.deliver(u) for u in controls])
    last = []
    for state in states[-251:]:
        geometry.set_state(state); last.append(geometry.tilts())
    margin = float(3.-np.max(abs(states[:, :2]))); tilt = float(np.max(last))
    geometry_passed = bool(margin >= 0. and tilt < .15)
    score = maximum = None; admitted = False
    if geometry_passed:
        predicted = native_successors(native, states[:-1], actions)
        score, maximum, residual = defects(geometry, predicted, states[1:])
        admitted = bool(score <= .8*baseline)
        np.savez_compressed(out/'interval_check.npz', predicted=predicted, residual=residual)
    np.savez_compressed(out/'virtual_proposal.npz', states=states, actions=actions,
                        fraction=np.array(1.))
    if admitted:
        np.savez_compressed(out/'accepted_predictor.npz', states=states, actions=actions,
                            fraction=np.array(1.))
        exact_bytes(states[0], observed_initial)
    return dict(predictor_admitted=admitted, geometry_passed=geometry_passed,
        baseline_physical_defect_cost=baseline, baseline_physical_defect_max=bmax,
        physical_defect_cost=score, physical_defect_max=maximum,
        virtual_plane_margin=margin, last5_virtual_max_tilt=tilt, proof=proof,
        original_target_bytes_preserved=True, physical_velocities=geometry.model.nv)


@dataclass
class Plan:
    config: Config
    reference: dict
    frames: np.ndarray
    coordinates: dict
    residual_controls: np.ndarray
    expected: dict
    native_steps: int
    reused_source: bool = False


def synthesize(config, observed_initial, bundle, runtime, out):
    """Only the observed state and one frozen source bundle guide synthesis."""
    dimensions = Dimensions.from_config(config); max_native = planning_budget(config)
    assert observed_initial.shape == (dimensions.state,)
    assert bundle['v']['states'].shape == (config.horizon+1, dimensions.state)
    assert bundle['m']['F'].shape == (config.horizon+1, dimensions.quotient, dimensions.tangent)
    out = Path(out); out.mkdir(exist_ok=False)
    initial = observed_initial.copy()
    np.savez_compressed(out/'observed_state.npz', state=initial)
    if np.array_equal(initial, bundle['v']['states'][0]):
        # Reuse the already audited source law without repeating its design.
        write_json(out/'summary.json', dict(completed=True, plan_found=True, reused_source=True,
            inclusive_native_steps=0, actual_benchmark_episodes=0, qualified=False))
        write_json(out/'completed.json', dict(status='completed'))
        return Plan(config, bundle['v'], bundle['m']['F'], bundle['c'], bundle['w'],
                    bundle['expected'], 0, True)
    native = SpatialCartpole(config); native.derivative_workers = 4
    geometry = SpatialCartpole(native.config); closed = 0; rows = []; result_plan = None
    def total():
        return native.native_steps+closed
    try:
        seed = out/'warm_start'; seed.mkdir()
        warm = prepare(geometry, native, bundle['v'], bundle['m'], bundle['c'], initial, runtime, seed)
        write_json(seed/'summary.json', dict(**warm, native_steps=native.native_steps))
        if warm['predictor_admitted']:
            with np.load(seed/'accepted_predictor.npz') as z:
                v = {k:z[k] for k in z.files}
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
        assert total() <= max_native and geometry.native_steps == 0
        write_json(out/'summary.json', dict(completed=True, plan_found=result_plan is not None,
            reused_source=False, warm_start=warm, results=rows,
            model_and_interval_native_steps=native.native_steps, forecast_native_steps=closed,
            inclusive_native_steps=total(), actual_benchmark_episodes=0, qualified=False,
            reserved_cases_consumed=0, max_models=MAX_MODELS,
            scope='frozen state-only initial-state planning; native forecasts precede separate original-reset evaluation'))
        write_json(out/'completed.json', dict(status='completed'))
        return result_plan
    except BaseException as exception:
        write_json(out/'failure.json', dict(type=type(exception).__name__, error=str(exception),
            completed_forecast_native_steps=closed, owned_native_steps=native.native_steps,
            accounting_complete=False))
        raise
    finally:
        native.close(); geometry.close()


class PlannedPolicy:
    """Execute the unchanged feedback after state-only offline planning."""
    def __init__(self, env, plan):
        assert env.config == plan.config
        exact_bytes(env.state(), plan.expected['states'][0])
        self.env, self.plan = env, plan
        self.inner = CorrectorPolicy(env, plan.reference, plan.frames,
                                     plan.coordinates, plan.residual_controls)
        nonupright = np.flatnonzero(np.max(plan.expected['tilts'], axis=1) >= env.config.angle_threshold)
        route_seconds = float(nonupright[-1]+1)*env.config.dt if len(nonupright) else 0.
        self.metadata = dict(park_seconds=0., route_seconds=route_seconds, physical_velocities=env.model.nv,
            predictor_native_steps=plan.native_steps, hardware_realtime_claim=False,
            policy='frozen state-only full native Newton synthesis and observed-state feedback')

    def __call__(self, state, t):
        before, clock = self.env.state().tobytes(), self.env.data.time
        force = self.inner(state, t)
        assert before == self.env.state().tobytes() and clock == self.env.data.time
        exact_bytes(force, self.plan.expected['raw_actions'][t])
        return force

    def observe_transition(self, state, t, action):
        exact_bytes(action, self.plan.expected['actions'][t])
        exact_bytes(state, self.plan.expected['states'][t+1])

    def close(self):
        self.inner.close()
