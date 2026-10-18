"""Earlier observed-state repair; exact independently audited development class."""
import json
import numpy as np
from .env import SpatialCartpole
from .immediate_online_repair import OnlineRepair,TerminalReplanner,save_trace
from .native_mpc import NativeReplanner,exact_bytes
from .rollout_value import NativeRolloutValuePolicy
from .experiment import write_json


class EarlierRepair(OnlineRepair):
    start, stop = 150, 214

    def __init__(self, env, payload, folder):
        super().__init__(env, payload, folder)
        self.metadata.update(policy='observed-state earlier immediate repair development',
                             repair_start=self.start, repair_stop=self.stop)

    def synthesize(self, state):
        self.folder.mkdir(exist_ok=False)
        before, clock = self.env.state().copy(), self.env.data.time
        counters = (self.env.steps, self.env.native_steps, self.env.best_streak)
        original = self.base.rollout
        assert original is not None
        warm_env = SpatialCartpole(self.env.config)
        warm_policy = NativeRolloutValuePolicy(warm_env, original.states, original.actions,
            original.feed, original.gains, original.roots, original.residuals, original.R,
            {k: self.payload[k] for k in ('reference', 'K', 'P', 'Q', 'R')},
            stage_factors=original.stage_factors, horizon=4)
        try:
            x, xs, us, raws = state.copy(), [state.copy()], [], []
            for t in range(self.start, self.stop):
                raw = warm_policy(x, t)
                u = warm_env.deliver(raw)
                x = warm_env.transition(x, u)
                xs.append(x); us.append(u); raws.append(raw)
            self.warm_tapes = dict(states=np.asarray(xs), actions=np.asarray(us),
                raw_actions=np.asarray(raws), **warm_policy.forecast_tapes)
            self.extra_native = warm_env.native_steps + warm_policy.metadata['predictor_native_steps']
            save_trace(self.folder / 'warm_owned_forecast.npz', self.warm_tapes)
        finally:
            warm_policy.close()
            warm_env.close()
        planner = NativeReplanner(self.env.config, original.states, original.actions,
                                  native_budget=1600000)
        terminal, phases = None, []
        try:
            warm = np.asarray(us)
            for phase in range(3):
                if phase == 2:
                    terminal = TerminalReplanner(self.env.config, original.states, original.actions,
                        native_budget=1450000, terminal_reference=original.states[self.stop],
                        terminal_root=original.roots[self.stop], terminal_residual=original.residuals[self.stop])
                engine = planner if phase < 2 else terminal
                before_native = engine.env.native_steps

                def checkpoint(row):
                    print(json.dumps(dict(phase=phase, iteration=row['iteration'], cost=row['cost'],
                                          accepted=row['accepted'])), flush=True)
                    with (self.folder / f'phase{phase}_optimizer.jsonl').open('a') as stream:
                        stream.write(json.dumps(row) + '\n')
                    save_trace(self.folder / f'phase{phase}_latest.npz', engine.current_plan,
                               cost=engine.current_cost)

                plan = engine.plan(state, self.start, warm, iterations=12, callback=checkpoint,
                    admission=lambda p: self.admitted(p) and p.get('objective_cost', np.inf) <= 1e-10)
                save_trace(self.folder / f'phase{phase}_initial.npz', engine.initial_plan, cost=engine.initial_cost)
                save_trace(self.folder / f'phase{phase}_final.npz', plan, cost=engine.final_cost)
                for j, trial in enumerate(engine.trials):
                    save_trace(self.folder / f'phase{phase}_trial{j:03d}.npz', trial,
                        **{k: v for k, v in trial.items() if not isinstance(v, np.ndarray)})
                if engine.last_models is not None:
                    np.savez_compressed(self.folder / f'phase{phase}_last_models.npz',
                        A=np.array([a for a, b in engine.last_models]),
                        B=np.array([b for a, b in engine.last_models]),
                        states=engine.last_model_states, actions=engine.last_model_actions,
                        feed=engine.last_feed, gains=engine.last_gains)
                phases.append(dict(phase=phase, native_steps=engine.env.native_steps - before_native,
                    initial_cost=engine.initial_cost, final_cost=engine.final_cost,
                    updates=len(engine.history), admitted=self.admitted(plan)))
                warm = plan['actions']
            if self.admitted(plan):
                self.repair = plan
            self.extra_native += planner.env.native_steps + terminal.env.native_steps
            assert self.extra_native <= 3100000
            self.metadata.update(repair_admitted=self.repair is not None, repair_planning_phases=phases,
                                 repair_and_warm_native_steps=self.extra_native)
            write_json(self.folder / 'summary.json', dict(admitted=self.repair is not None,
                phases=phases, warm_and_planning_native_steps=self.extra_native,
                observed_state=state.tolist(), repair_start=self.start, repair_stop=self.stop))
        except Exception:
            write_json(self.folder / 'failure_accounting.json', dict(
                observed_native_steps=self.extra_native + planner.env.native_steps
                    + (0 if terminal is None else terminal.env.native_steps)))
            raise
        finally:
            planner.close()
            if terminal is not None:
                terminal.close()
        exact_bytes(before, self.env.state())
        assert self.env.data.time == clock
        assert counters == (self.env.steps, self.env.native_steps, self.env.best_streak)

    def __call__(self, state, step):
        if step == self.start:
            self.synthesize(state)
        if self.repair is not None and self.start <= step < self.stop:
            exact_bytes(state, self.repair['states'][step - self.start])
            raw = self.repair['actions'][step - self.start]
        else:
            raw = self.base(state, step)
        self.metadata['predictor_native_steps'] = self.extra_native + self.base.metadata['predictor_native_steps']
        return raw

    def observe_transition(self, state, step, action):
        if self.repair is not None and self.start <= step < self.stop:
            exact_bytes(state, self.repair['states'][step - self.start + 1])
            exact_bytes(action, self.repair['actions'][step - self.start])
            self.metadata['verified_repair_transitions'] += 1
