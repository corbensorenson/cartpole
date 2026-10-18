"""A bounded, state-only full-native planner for noisy immediate 11-link swing-up."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from spatialpole import Config, SpatialCartpole
from spatialpole.direct_shooting import native_successors
from spatialpole.experiment import write_json
from spatialpole.native_mpc import exact_bytes
from eleven_initial_continuation import forecast
from eleven_secant_corrector import corrector_model, CorrectorPolicy
from eleven_value_sequence import coordinates, solve_value_sequence, mapped_error, physical_cost, forward
from prepare_eleven_observed_target import prepare
from prepare_eleven_newton_seed import defects
from probe_eleven_native_chart import error

MAX_MODELS = 4
MAX_NATIVE = 7044324


@dataclass
class Plan:
    reference: dict
    frames: np.ndarray
    coordinates: dict
    residual_controls: np.ndarray
    expected: dict
    native_steps: int
    reused_source: bool = False


def synthesize(observed_initial, bundle, runtime, out):
    """Only the observed state and one frozen source bundle guide synthesis."""
    out = Path(out); out.mkdir(exist_ok=False)
    initial = observed_initial.copy()
    np.savez_compressed(out/'observed_state.npz', state=initial)
    if np.array_equal(initial, bundle['v']['states'][0]):
        # Reuse the already audited source law without repeating its design.
        write_json(out/'summary.json', dict(completed=True, plan_found=True, reused_source=True,
            inclusive_native_steps=0, actual_benchmark_episodes=0, qualified=False))
        write_json(out/'completed.json', dict(status='completed'))
        return Plan(bundle['v'], bundle['m']['F'], bundle['c'], bundle['w'],
                    bundle['expected'], 0, True)
    native = SpatialCartpole(Config(n_links=11)); native.derivative_workers = 4
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
                assert total()+1746080 <= MAX_NATIVE
                folder = out/f'iteration{iteration}'; folder.mkdir()
                np.savez_compressed(folder/'virtual_reference.npz', **v)
                before = native.native_steps
                r, F, E, predicted, _ = corrector_model(native, geometry, v, folder)
                assert native.native_steps-before == 1734000
                baseline, bmax, _ = defects(geometry, predicted, v['states'][1:])
                rng = np.random.default_rng(1001950+iteration); checks = []
                for t in (0, 160, 205, 325, 1000):
                    d = rng.normal(size=59); d /= np.linalg.norm(d); dx = E[t]@d
                    u = v['actions'][t].astype(float); du = -u/max(1., float(np.linalg.norm(u))); eps = 1e-4
                    xs = np.array([geometry.integrate(v['states'][t], k*eps*dx) for k in range(1, 5)])
                    us = np.array([u+k*eps*du for k in range(1, 5)])
                    assert np.linalg.norm(us, axis=1).max() <= 80.
                    xn = native_successors(native, xs, us)
                    ys = np.array([error(geometry, v, dict(F=F), x, t+1) for x in xn])
                    measured = (-25.*r['d'][t]+48.*ys[0]-36.*ys[1]+16.*ys[2]-3.*ys[3])/(12.*eps)
                    expected = r['A'][t]@d+r['B'][t]@du
                    discrepancy = float(np.max(abs(measured-expected))); assert discrepancy < 1e-5
                    checks.append(dict(phase=t, error=discrepancy))
                    np.savez_compressed(folder/f'probe{t}.npz', states=xs, actions=us,
                                        successors=xn, measured=measured, expected=expected)
                c = coordinates(r, np.eye(2)*.001, terminal_scale=1.)
                np.savez_compressed(folder/'coordinates.npz', **c)
                try:
                    u, e, w, proof = solve_value_sequence(c, np.zeros(59), v['states'],
                        v['actions'].astype(float), runtime, folder/'certificate.npz', time_limit=120.)
                except (AssertionError, RuntimeError) as exception:
                    row = dict(iteration=iteration, model_rejected=True, error=str(exception),
                               checks=checks, inclusive_native_steps=total())
                    rows.append(row); write_json(folder/'summary.json', row); break
                assert mapped_error(r, e, u-v['actions']) < 1e-6
                score = physical_cost(r, e, u-v['actions'], np.eye(2)*.001, terminal_scale=1.)
                _, be, bu = forward(c, np.zeros(59), np.zeros((1500, 2)))
                base = physical_cost(r, be, bu, np.eye(2)*.001, terminal_scale=1.)
                assert abs(score-base-.5*np.sum(w*w))/max(1., abs(score)) < 1e-6
                np.savez_compressed(folder/'model_plan.npz', controls=u, errors=e, w=w)
                tape, result = forecast(initial, lambda env:CorrectorPolicy(env, v, F, c, w),
                                        folder/'native_forecast.npz')
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
                    result_plan = Plan(v, F, c, w, tape, total()); break
                if iteration == MAX_MODELS-1:
                    break
                delta = np.einsum('tij,tj->ti', E, e); selected = False; updates = []
                for index, alpha in enumerate((1., .5, .25)):
                    assert total()+6000 <= MAX_NATIVE
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
        assert total() <= MAX_NATIVE and geometry.native_steps == 0
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
        exact_bytes(env.state(), plan.expected['states'][0])
        self.env, self.plan = env, plan
        self.inner = CorrectorPolicy(env, plan.reference, plan.frames,
                                     plan.coordinates, plan.residual_controls)
        self.metadata = dict(park_seconds=0., route_seconds=6.5, physical_velocities=35,
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
