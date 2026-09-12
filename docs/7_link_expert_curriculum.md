# Seven-Link Expert Curriculum

The seven-link work is now ordered as a dependency chain with two runtime
experts: a swing-up expert and a capture/stabilize expert. A policy that only
holds an already-upright chain is not a swing-up result, but maintenance is a
required training prerequisite for the capture expert; without capture, a
swing-up crossing is not useful.

## Canonical Gate Result (2026-09-12)

The current canonical seven-link result uses a deterministic hanging-equilibrium
conditioning prelude before the two runtime experts. For 10 seconds, a
hanging-state LQR returns cart position and velocity toward the center and
damps the initial link noise. The measured settled cart position translates
the swing route's nominal cart coordinate without changing simulator state.
The Box-FDDP swing route then runs with tracking gain scale `2.0`, followed by
terminal upright LQR capture at the route horizon.

This chain passes `20/20` and `100/100` canonical noisy hanging-start episodes.
The complete manifest is
`runs/swingup7_uniform/seven_link_swingup_manifest.json`; the final evaluation
files and held-out video are under `runs/swingup7_uniform/`. The earlier PPO,
synthetic capture, and morphology curriculum notes below remain valid as
training diagnostics and calibration history, but they are no longer the
strongest final-control evidence.

## Phase Order

1. `swingup7_maintenance.yaml` trains the maintenance stage from upright,
   using exact equilibrium as a deterministic sanity check and beginning PPO
   at the first nontrivial disturbance stage, then progressively larger angle,
   velocity, cart-position, and cart-velocity disturbances.
2. `swingup7_capture_curriculum.yaml` trains the capture/stabilize runtime
   expert from the maintenance checkpoint and widens the upright-start
   disturbance envelope. Its job is recovery and capture, not hanging-start
   swing-up. Before training it, the real maintenance rollout states are
   exported and passed through `env.init_mode=state_list`; synthetic upright
   resets are only a smoke test. The final manifest may fold the maintenance
   teacher into this expert, but must retain the separate maintenance gate.
3. `swingup7_swing_curriculum.yaml` continues from the best capture checkpoint
   and uses `hanging_curriculum` to move the first relative link angle from
   upright to hanging. Its progress-1 evaluation is still only a training
   diagnostic; the final claim must use the frozen canonical config with
   `init_mode: hanging`.

The phases share the same 7-link uniform plant, observation layout, force
limit, rail, and network shape. Each phase produces a checkpoint and a
phase-specific evaluation record. A later phase may not replace a missing
earlier gate with a better shaped return.

For the exact iLQR terminal-state branch, use
`configs/swingup7_capture_ilqr_terminal.yaml`. It restores the saved qpos/qvel
handoff through an explicit zero-to-one state homotopy and includes the
packet's length/mass/damping/friction gradient. A controlled variant may pin
`env.plant_progress=0.0` to finish the handoff on the full p0 training-wheel
morphology before annealing each morphology axis back to the canonical
uniform plant. Any checkpoint from this branch must be replayed with
`init_qpos_scale=init_qvel_scale=1.0` before it can be called a capture
result.

## Commands

```bash
PYTHONPATH=src:scripts python3 scripts/train_torch_ppo.py \
  --config configs/swingup7_maintenance.yaml

# Optional verified maintenance warm start. Disable the residual teacher when
# materializing this direct MPC checkpoint; the residual branch is enabled by
# the maintenance config only for PPO training.
PYTHONPATH=src:scripts python3 scripts/make_torch_mpc_checkpoint.py \
  --config configs/swingup7_maintenance.yaml \
  --checkpoint runs/swingup7_maintenance_mpc0025_verified/checkpoints/best.pt \
  --progress 0.0025 --horizon 50 --control-cost 1000 \
  --terminal-q-factor 100 \
  --override 'ppo.hidden_sizes=[]' \
  --override 'env.action_lqr_residual.enabled=false' \
  --override 'experiment.out_dir=runs/swingup7_maintenance_mpc0025_verified'

MASTERED_PROGRESS=0.0025  # replace with the checkpoint's recorded mastered stage
PYTHONPATH=src:scripts python3 scripts/export_torch_policy_states.py \
  --config configs/swingup7_maintenance.yaml \
  --checkpoint runs/swingup7_maintenance_mpc0025_verified/checkpoints/best.pt \
  --progress "$MASTERED_PROGRESS" --episodes 256 --seconds 8.0 --require-success \
  --max-angle 0.35 --max-hinge-rms 1.50 --max-cart-velocity 1.0 \
  --max-cart-abs 2.5 --one-per-episode \
  --override 'ppo.hidden_sizes=[]' \
  --override 'env.action_lqr_residual.enabled=false' \
  --override 'env.episode_seconds=5.0' \
  --out runs/swingup7_policy_handoff/maintenance_states.json

# Use the exact mastered progress recorded in the checkpoint metadata; do not
# substitute a later or easier curriculum stage.

PYTHONPATH=src:scripts python3 scripts/evaluate_torch.py \
  --config configs/swingup7_capture_curriculum.yaml \
  --checkpoint runs/swingup7_maintenance_mpc0025_verified/checkpoints/best.pt \
  --episodes 32 --seed 0 --progress "$MASTERED_PROGRESS" \
  --states-path runs/swingup7_policy_handoff/maintenance_states.json \
  --override 'ppo.hidden_sizes=[]' \
  --override 'env.init_qvel_scale_start=1.0' \
  --override 'env.init_qvel_scale_end=1.0' \
  --override 'env.init_qpos_scale_start=1.0' \
  --override 'env.init_qpos_scale_end=1.0' \
  --override 'env.init_cart_noise_start=0.0' \
  --override 'env.init_cart_noise_end=0.0' \
  --override 'env.init_cart_vel_noise_start=0.0' \
  --override 'env.init_cart_vel_noise_end=0.0' \
  --override 'env.init_angle_noise_start=0.0' \
  --override 'env.init_angle_noise_end=0.0' \
  --override 'env.init_vel_noise_start=0.0' \
  --override 'env.init_vel_noise_end=0.0' \
  --override 'env.episode_seconds=5.0' \
  --out runs/swingup7_capture_mpc0025_handoff_eval32.json

PYTHONPATH=src:scripts python3 scripts/evaluate_torch.py \
  --config configs/swingup7_maintenance.yaml \
  --checkpoint runs/swingup7_maintenance/checkpoints/best.pt \
  --episodes 20 --progress 1.0 \
  --override env.episode_seconds=30.0 \
  --out runs/swingup7_maintenance/eval_20.json

PYTHONPATH=src:scripts python3 scripts/train_torch_ppo.py \
  --config configs/swingup7_capture_curriculum.yaml \
  --override env.init_mode=state_list \
  --override env.init_states_path=runs/swingup7_policy_handoff/maintenance_states.json \
  --override env.init_qvel_scale_start=1.0 \
  --override env.init_qvel_scale_end=1.0

PYTHONPATH=src:scripts python3 scripts/train_torch_ppo.py \
  --config configs/swingup7_swing_curriculum.yaml \
  --init-checkpoint runs/swingup7_capture_curriculum/checkpoints/best.pt
```

For a short smoke run, override `experiment.out_dir`,
`ppo.total_updates`, `ppo.num_envs`, and `ppo.eval_episodes` so the smoke
artifacts cannot be mistaken for phase evidence.

The reset-free discovery tools are `scripts/search_swingup_tail_capture_value.py`
and `scripts/search_two_phase_linear_policy.py`. The first ranks exact tail
endpoints by replaying the downstream capture checkpoint from their emitted
states. The second holds a swing actor fixed while optimizing a separate
capture actor after a declared phase switch. Both write `not_solution` JSON
artifacts until a canonical feedback replay passes the roadmap gates.

## Gates

- Maintenance: deterministic upright-start episodes hold for the configured
  five-second success window without rail termination at the mastered stage,
  and the gate passes three consecutive evaluations.
- Capture: deterministic rollouts from the saved real maintenance states meet
  the same hold and low-momentum criteria across the widest mastered handoff
  stage. Arrival velocity is restored from zero to its measured value during
  training; it is never silently replaced by a fresh upright reset.
- Swing: progress is advanced only after the previous stage passes; at
  progress 1.0 the policy must be evaluated from true hanging starts before
  any final evidence is considered.

The canonical seven-link 20/100 episode gates, reset-free video, hashes, and
fresh-clone reproduction remain the only completion evidence. This packet is
an execution protocol, not a relaxed benchmark.

## Current Maintenance Blocker

The first maintenance probes are diagnostic only. On the longer-rail,
graded-morphology stage at progress `0.01`, the best CPU policy reached a
`0.25` success rate on one 16-episode check and a `5.02 s` maximum upright
streak, but a fresh deterministic replay exported `0/32` successful source
episodes under the handoff selection bounds. The uniform-plant-only probe did
not pass its first nontrivial stage either. Consequently, the capture phase
must remain closed; the diagnostic state-list smoke test verifies plumbing only
and is not a capture result.

A fixed-stage time/phase-feature variant was also tested to allow periodic cart
excitation of the internal modes. It produced no deterministic five-second
successes, so the extra observation features are not part of the working
configuration. A second fixed-stage probe with a wider temporary rail and
additional damping also produced `0.00` deterministic success at progress
`0.01`; rail length and damping alone are therefore not the missing
maintenance capability.

The condensed upright-MPC teacher is the first reproducible maintenance
checkpoint: with its real MuJoCo feedback law, three independent deterministic
5-second replay sets each pass `30/32` episodes at progress `0.0025`, with
median hold `5.02 s` and mean cart excursion below `0.012 m`. At progress
`0.005` it passes only `16/32`, so the maintenance curriculum advances in
`0.0025` increments and treats `0.005` as an unsolved recovery stage. A
residual PPO run around that teacher reached `0.75` once but failed the
repeated-evaluation gate; it is not a mastered checkpoint. Later direct
continuation and residual trust-region trials also failed to preserve the
teacher: the best tiny-cap residual check was `7/32` before collapsing to
`0/32`. Capture training therefore remains an optimization/interface problem,
not a passed phase.

The saved real handoff currently contains `122/128` successful source states.
Running the same teacher from a seeded no-replacement sample of 32 of those
states gives `29/32` five-second captures at progress `0.0025`; the evaluator
records the exact state indices and three states still lose the hold. A direct
PPO continuation overwrote the teacher and fell to `6/32` at its best
checkpoint, so the capture config now keeps the teacher action active and
trains a zero-initialized residual branch. This is a valid handoff diagnostic,
not a passed capture expert, and capture training remains open. A first
energy-homotopy swing probe and an exact-MuJoCo global force CEM search have
also failed to produce a capture-ready hanging-start state; their results are
recorded in [`docs/levers_and_pitfalls.md`](levers_and_pitfalls.md). The next
swing attempt must use a new phase/energy or planner-to-capture interface
rather than simply increasing PPO duration.

The latest seven-link discovery branch now has a concrete reset-free planner
handoff, but not a capture result. A corrected exact-hanging cart-PD source
plus hinge-heavy tail search reached a real state with `0.142 rad` maximum
absolute angle, `0.524 rad/s` hinge RMS, `0.739 m` cart position, and
`0.436 m/s` cart speed under the same base-heavy/damped long-rail plant. When
replayed continuously, the existing maintenance teacher and online nonlinear
MPC each lost it after a `0.02 s` upright interval. Open-loop tail CEM,
tail iLQR, fixed-state iLQR, a dedicated one-state PPO probe, and longer or
earlier tail searches have not recovered the state. This establishes that the
current bottleneck is a robust low-velocity arrival basin plus capture policy,
not merely reaching the angle threshold. All artifacts and failure details
are in [`docs/levers_and_pitfalls.md`](levers_and_pitfalls.md).

The reason to retain the rail and morphology curricula is now measurable, not
assumed: the uniform seven-link upright linearization has deficient
controllability for the static local feedback test, so a static LQR checkpoint
is not a maintenance teacher under realistic perturbations. The working
maintenance search therefore needs nonlinear, exploratory upright feedback
before its states can be treated as a real handoff distribution.

The current two-expert diagnostic uses the same design at the swing side: a
state-feedback swing actor is held fixed while a separate capture actor is
optimized after a reset-free phase switch. The best linear probe at a `9.6 s`
switch reduced the joint handoff score to `82.54` and hinge RMS to
`1.975 rad/s`, but held only `0.04 s`. This validates the architecture
direction without passing the capture gate; the next capture actor must be
nonlinear or trained from a broader, measured state curriculum.

The latest capture branch adds a second momentum coordinate. Relative hinge
velocity is `qvel[1:1+n]`; physical link angular velocity is its cumulative
sum. A constrained tail search on the discovery plant produced an
`absolute004` handoff with `0.0015 rad` maximum angle, `0.040 rad/s` absolute
angular-velocity RMS, and `0.060 m/s` cart speed. A nonlinear feedback CEM
from that exact state reached `6.78 s` maximum centered upright and `6.76 s`
whole-chain low-momentum streak in an uninterrupted `8 s` diagnostic, but the
same actor held only `0.78 s` and `0.88 s` on two neighboring measured states.
A three-state optimization therefore succeeded on only `1/3` states. This is
the strongest current capture component, not a passed P4/P5 result: it uses
the base-heavy/damped p0 morphology and a `+/-12 m` rail, and the next step is
hard-negative/state-set expansion before using it as a swing objective.

The capture score now includes whole-horizon low-momentum time. With that
objective, the exact `absolute004` actor holds `8.02 s` upright and `8.00 s`
with low absolute link rates, but its 30-second replay reaches the rail after
about `9.08 s`. A reset-free planner-to-capture replay nevertheless reproduces
the saved handoff with zero `qpos/qvel` error and holds for the full `8 s`
diagnostic. This is the first integrated seven-link discovery chain. It is not
the canonical result: the plant is p0 base-heavy/damped, the rail is `+/-12
m`, and the long-horizon stabilizer still needs to be solved before the route
can become a P5 component.

Direct morphology transfer is not currently valid. Replaying the same chain at
`plant_progress=0.0025` hit the `12 m` rail before the handoff, while
`plant_progress=0.0001` reached a state with `0.901` maximum position error and
`3.757` maximum velocity error relative to the p0 handoff. The homotopy must
therefore regenerate planner labels and capture training states at each step;
the p0 checkpoint cannot simply be replayed into the uniform plant.
