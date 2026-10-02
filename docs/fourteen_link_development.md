# Fourteen-link development

Fourteen has now passed the complete canonical reserved bundle and full video;
GitHub publication is being prepared. Statements below about unused reserved
seeds describe the historical development stage. Final evidence is in
[the canonical appendix](fourteen_link_swingup_paper.md).
The source is the released thirteen-link physical controller, whose full manifest
passes verification. The benchmark and 50 Hz control rate are unchanged.

The fixed initial protocol repeats the thirteen procedure: transfer the executed
nine-second physical curve in material coordinates, append an explicitly virtual
five-second offline capture tail, run twenty full-horizon residual-SCvx iterations
with the legacy OSQP settings, then forty native Clarabel iterations with nonlinear
merit agreement and initial QP tolerance cap 1e-5. Riccati design uses 100 decimal
digits; nonlinear MuJoCo execution remains binary64 with the delivered float32
action interface. The first synthesis uses development seed 20264100.

The predeclared unchanged-prefix handoff screen is 8, 9, 10 and 12 seconds.
Select the earliest member that finishes a valid full physical episode in rail
with at least five continuous seconds upright. Preserve every failed replay.
Rebuild delivered actions and physical nodes, require exact dynamics feasibility
and identical independent replay, then test the unchanged sixteen-second parked
launch with cart position/velocity costs 10/5, target -0.05 m and control cost
1000. Development noisy cohorts use 20264200 (20) and 20264300 (100).

Only passing development cohorts authorize freezing a candidate and using the
reserved gates. Every intervention beyond this initial protocol must be recorded.
A saved-controller result is separate from automatic generalist re-synthesis.
The campaign journal and source snapshots are under
`runs/frontier_campaign_20261001/n14_adjacent_*`.

The bounded count-agnostic runner is `scripts/solve_frontier_capture_increment.py`.
It verifies the predecessor's canonical manifest, rejects any development cohort
that intersects reserved seeds, runs the declared stages with snapshots and
journals, and stops on failure. It cannot promote or publish a count. Reusing an
already completed first twenty-iteration stage is optional and explicitly
disclosed; the guard requires the same target config, predecessor, five-second
tail, development seed and numerical parameters. All 404 repository tests pass,
including tests of cohort rejection, physical-episode integrity and stage reuse.

A fresh complete development invocation is:

```sh
PYTHONPATH=.:src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.conda-aligator/bin/python scripts/solve_frontier_capture_increment.py \
  --source-manifest runs/swingup13_uniform/thirteen_link_swingup_manifest.json \
  --config configs/swingup14_uniform.yaml \
  --directory runs/frontier_campaign_20261001/n14_adjacent_fixed_recipe_increment
```

For the currently running fourteen experiment, the first twenty-iteration stage
was already launched separately with an identical recipe before this runner was
created. Its complete source/input snapshots and execution journal are retained.
The remaining stages will disclose adoption of that artifact rather than claim
a fresh automatic end-to-end re-synthesis.

First-stage result: twenty iterations leave maximum physical dynamics gap
0.0006356704126. Actual full replay violates
the rail at 5.10 s with zero upright hold. This
negative candidate is retained. The bounded runner now explicitly adopts this
completed stage and runs the declared forty-iteration agreement continuation
from its saved state. No fourteen reserved seeds have been used.

The fixed forty-iteration agreement continuation ends with maximum physical
dynamics gap 4.73368544e-8. The actual replay still violates the rail at 6.38 s
with only 0.14 s upright. All four handoff screens are identical up to this
failure, which occurs before the earliest declared eight-second switch. The
bounded runner stops as required; no development noisy gate or reserved cohort
is attempted. Small virtual defects are again insufficient execution evidence.

A same-controls/same-nodes pure LTV QR feedback replay with zero feedback
regularization fails at 7.04 s with 0.12 s hold. Two matched forty-iteration
continuations now start from the failed agreement source: both use Clarabel,
initial QP tolerance cap 1e-8, floor 1e-11, initial state/control trust
5e-5 / 1e-4, regularization 10 and nonlinear agreement. Their defect penalties
are 1e5 and 1e8 respectively. The initial trust and QP settings both differ
from the original fixed continuation, so this is not a single-factor comparison
against that original run. The two new continuations differ only in the defect
penalty and permit a matched test of stricter physical feasibility weighting.
All interventions, source snapshots and failures are retained.

Separate local components compare cached finite-difference inputs, mechanical
inputs rounded to binary64, and mechanics carried through 100-digit RK4 and
Riccati design. The mechanics include the parsed capsule inertias, COM offsets,
armature, damping and held-force cadence; independent native force/transition
checks pass at counts 1, 7, 14 and 20. All three fourteen designs fail the four
1e-11 and 1e-13 perturbations and pass the four 1e-15 perturbations. All three
twenty designs fail every tested amplitude. Input precision alone is not a
sufficient recovery for these tested local components. All 409 tests pass.

The matched linear diagnostic retains the rounded gain and the same MP plant
and directions, then adds force saturation, action quantization, and binary64
feedback arithmetic separately. At count twenty and perturbation amplitude
1e-15, the unsaturated mathematical model has a stable spectral radius but
demands normalized action up to 5.604 and cart excursion up to 10.997 m.
The bounded linear variants cross 3 m in 17–19 steps, even with full MP
arithmetic. This establishes a force/rail and transient problem for this
particular local controller and these directions; it does not establish
impossibility of twenty-link nonlinear swing-up or of a different controller.
These are linear-model components, not canonical MuJoCo promotion evidence.
The sources are `n14_n20_mechanical_input_precision_component` and
`n14_n20_linear_runtime_precision_component` under the campaign directory.

The inspected failure comparison is
`runs/frontier_campaign_20261001/n14_adjacent_fixed_recipe_replay_failure_report/nominal_vs_physical.png`.
It overlays recorded physical angles, cart motion and delivered force on the
saved offline nominal route. The feedback variants approach upright, then
saturate force and leave the rail; their nominal nodes remain quiet. Force
samples are aligned at the evaluator's post-step timestamps.

The two tighter continuations finish negatively: defect penalty 1e5 ends at
maximum gap 3.03279957e-8 and rail failure at 7.74 s (0.14 s hold); penalty
1e8 ends at gap 3.75825229e-8 and rail failure at 7.02 s (0.32 s hold).
The matched penalty increase does not restore physical execution.

The local cadence audit recomputes the mechanical MP100 LQR at 100 and 200 Hz
while retaining RK4 internal dt 0.005 s, parsed geometry, force/rail limits,
float32 delivered actions, matched directions and eight-second duration.
Fourteen passes the four 1e-15 directions but fails the four 1e-13 directions
at both rates, as at 50 Hz. Twenty fails both amplitudes at both rates.
This does not rule out benefit from a different controller or full swing
re-synthesis; it shows that increased rate alone does not recover these cases.

A separate offline nominal-arrival capture probe recomputes the exact source
MP100 gain and checks bit identity with the cached gain. The saved nine-,
ten- and twelve-second nominal nodes each survive all 400 native steps of
an eight-second capture replay without saturation. The eight-second node
fails after 71 steps. These hypothetical initial states are not hanging-start
benchmark evidence, and no executed plant state is replaced. The distinction
narrows the observed boundary: local stabilization at the intended arrival
works, whereas connection from the hanging state remains physically invalid.

Native Box-FDDP now starts from the saved original forty-iteration agreement
source, with a 100-iteration budget and nonzero defects explicitly declared
infeasible. All original objective and derivative settings are retained;
SCvx-only hard search constraints are removed because Box-FDDP uses the
existing soft rail objective. Actual replay retains the canonical hard rail,
force bound and float32 delivered action interface. This is a new development
variant, not an unchanged application of the successful thirteen-link recipe.

Native Box-FDDP terminates with zero accepted optimization iterations and
unchanged maximum gap 4.73368544e-8. Its feedback replay fails on the rail at
13.10 s, without any upright hold. This is not a trajectory repair.

The next declared experiment retimes the accepted thirteen-link physical
nine-second curve to ten and eleven seconds, retaining the five-second
virtual tail. Cubic Hermite interpolation uses saved positions and physical
velocities; target velocities are the derivative under the changed time scale.
Delivered source controls are interpolated only as warm-start variables.
The method preserves winding, endpoints, and the unchanged default arrays,
but neither target timing inherits physical feasibility. The two new timings
use the same development seed 20264100 and twenty-iteration SCvx first repair.
There is no online projection, and the release configuration remains 50 Hz.
All 415 tests pass, including analytic derivative and winding checks for the
time interpolation and exact default preservation. Search times during
concurrent execution must not be presented as serial speed comparisons.

Before any retimed repair completes, an eight-second initializer is added
to bracket the failed nine-second timing in both directions. A shorter swing
reduces the time over which errors can grow but may require greater force;
these are hypotheses, not observed recovery. The declared duration set is
8, 10 and 11 s. Each receives the same twenty-iteration first repair followed
by forty Clarabel agreement iterations and the established 8/9/10/12 s
handoff screens. The earliest fully passing declared handoff is selected per
candidate; if more than one candidate passes, the shortest swing candidate
is considered first for independent noisy development gates. This protocol
is recorded before the first-stage outcomes, and does not use reserved seeds.

The retimed continuation helper is frozen with all three initializer inputs
in `n14_adjacent_retimed_continuation_execution`. It explicitly adopts each
completed first repair, records its source/execution hashes, runs forty
agreement iterations and all four unchanged-prefix handoff comparisons, then
stops with either a negative result or a physical candidate requiring exact
rebuild and independent gates. It never freezes a release, consumes reserved
seeds, or publishes a candidate. Per-duration development journals preserve
all stages and exceptions.

A matched thirteen/fourteen diagnostic independently remeasures native gaps
and reconstructs every delivered prefix action from saved feedback and
recorded physical states. Both checks pass exactly after correcting the
diagnostic to distinguish recorded pre-interface requests from float32
delivery. The original diagnostic assertion failure is retained separately;
it was not a failed physical search. Thirteen/fourteen maximum next-tick
feedback projections of a single nominal gap are 2.3468e-7 / 2.1126e-7
normalized action, despite maximum gain norms 1.1386e8 / 1.0003e9. A large
gain norm or scalar maximum gap alone therefore does not explain the
different tracking outcome. Fourteen error grows past 1e-4 at 5.68 s and
first requested action saturation occurs at 5.84 s; thirteen remains below
1e-4 throughout the recorded fourteen-second route prefix without saturation.

Independent native transitions from every recorded physical pre-step state
reproduce both six-second prefixes with exactly zero gap. Finite-difference
local tracking models separate nominal-gap forcing, delivered-action
rounding/clipping and a remainder combining nonlinearity with derivative
error. These are algebraic linear components, not independent nonlinear
counterfactuals. Their accumulated binary64 sum becomes unreliable near the
fourteen failure, and per-node closed-loop spectral radii are large even on
the thirteen route that succeeds under earlier handoff. Neither diagnostic
is a stability certificate or a precise causal attribution. The source
records include the decomposition rounding discrepancy for each tick.

An optional high-precision finite-horizon tracking design now carries the
same native binary64 derivative inputs and saved binary64 cost factors
through a 100-digit Riccati recurrence, then rounds feedback to binary64
for physical replay. The native plant and float32 action interface remain
unchanged. The comparison against the previous pure QR replay changes both
backward arithmetic and the algebraic factorization method, so it is not a
single-factor precision ablation. Controls and nominal nodes must remain
bit-identical. Independent dense-horizon optimality and a weak-cost-direction
test pass before this replay begins; no fourteen reserved seeds are used.

The 100-digit tracking replay completes negatively at 352 steps / 7.04 s
with 0.12 s hold. All 352 physical post-step states and delivered float32
actions are bit-identical to the previous pure QR replay. Therefore this
feedback design precision change does not alter the executed failing prefix.
The full-horizon gains do differ, and the matched observation separately
reports differences over the actually executed prefix. All 417 tests pass;
frozen eleven, twelve and thirteen release verifiers also pass.

Handoff-only replay and physical route rebuild now suppress a source command's
feedback-recomputation flag so inherited gains are actually retained. The
existing array and physical-prefix identity assertions remain required.
This enables unchanged-gain followups from feedback-design experiments and
preserves the earlier default path.

The feedback-inheritance regression replays the earlier pure QR source with
an unchanged nine-second prefix and verifies bit-identical inherited controls,
feedback gains, nominal states and all physical prefix states. It retains the
same 352-step negative outcome; this validates comparison semantics rather
than claiming a new solve.

Two feedback-only state-derivative tests now use increments 1e-4 and 5e-5,
compared with the original 2e-4. All use fourth-order native finite differences,
action increment 5e-4, pure binary64 QR and regularization zero, with identical
controls/nodes/objective/canonical replay. These comparisons change one
derivative increment each. Neither search uses fourteen reserved seeds.

The inspected scientific figure `n14_adjacent_n13_n14_feedback_boundary_figure/feedback_boundary.png`
compares one-gap action projections, dimensionless physical tracking errors
and nominal/delivered forces over the first six seconds. It distinguishes
the negative fourteen source from the thirteen source's separately verified
earlier handoff and exact rebuild, and states the limits of one-tick sensitivity.

Both smaller state-derivative feedback increments finish negatively. Increment
1e-4 fails on the rail at 7.68 s with 0.02 s hold; increment 5e-5 fails at
7.20 s with 0.24 s hold. Controls and nominal nodes remain bit-identical
to the original source in both comparisons. These derivative refinements
alter physical execution but do not recover a valid route.

The eight-second retimed first repair finishes with maximum gap 0.0171340
and rail failure at 4.92 s with zero upright hold. It enters the declared
forty-iteration agreement continuation without any release advancement.
The living ledger now also discovers frozen execution records under each
shared synthesis directory, so these bounded stages and their per-duration
continuation journals are included explicitly rather than only being
reachable through the parent experiment. Frozen publication ledgers are
not rewritten.

The ten-second first repair also finishes negatively: maximum gap
0.00158663518, rail failure after 311 steps / 6.22 s, zero upright hold.
It proceeds into its declared forty-iteration agreement continuation.

A separate twenty-iteration square-root FDDP repair starts from the original
near-feasible nine-second agreement source. It keeps the objective, fourth-order
native derivatives, initial regularization 10, 100-digit capture design and
hard nominal rail rejection. It replaces the SCvx optimizer with square-root
FDDP, uses raw physical-coordinate L1 gap merit with penalty 1e8, and removes
the SCvx-only hard nominal capture-angle constraint. The actual canonical
upright threshold, force/rail limits, cadence, delivered action precision and
full replay gates remain unchanged. This is a structural solver variant with
several declared differences, not a single-factor ablation. Unit trial steps
are full physical trajectories; fractional trial steps explicitly retain
optimizer gaps. Those virtual states never replace live plant states.

The eleven-second first repair finishes with maximum gap 0.00156729197
and rail failure at 5.22 s with zero upright hold, then enters the declared
forty-iteration continuation. The square-root FDDP variant accepts zero
optimization iterations; its final feedback replay fails at 10.26 s with
0.10 s hold. The near-feasible source gap remains unchanged.

The native next-step value probe uses separately predicted physical next
states, a factored finite-horizon guide, an action effort penalty, 80-digit
cost arithmetic and bounded float32 scalar candidates. Its first three
seconds retain the source feedback; corrections continue through the
nine-second handoff. The actual thirty-second episode attempt fails after
473 steps / 9.46 s with 2.38 s hold and zero fallback decisions. Its
recorded 450-step physical prefix independently reproduces with exactly
zero native gaps. The emitted static initializer contains the delivered tape
and inherited feedback, and explicitly requires separate static replay;
it is not the controller used in this episode. No canonical solve is claimed.

The retimed-eight initializer's forty-iteration agreement continuation
finishes with maximum gap 4.63267331e-8. Handoff eight fails at 9.66 s with
2.96 s hold. Handoffs nine and ten both complete all 1,500 physical steps,
hold upright for 24.16 s and limit cart excursion to 1.91309675 m.
Handoff twelve fails at 14.34 s despite 6.50 s earlier hold. Every comparison
verifies unchanged controls, feedback, nominal nodes and physical prefix.
The declared earliest-full-pass rule therefore selects nine seconds.
This is a positive exact-start development candidate, not a release.
The delivered route is now being rebuilt before independent noisy gates.
The source curve was retimed to eight seconds as an initializer; the
selected executed tracking route is nine seconds long. No higher cadence,
extra actuator, live state projection or fourteen reserved seed is used.

The delivered-action rebuild has exactly zero initial and final native gaps
and reproduces all 1,500 states of the passing handoff episode bit for bit.
The unchanged parked launch passes fresh development noisy20/20 (20264200)
and disjoint noisy100/100 (20264300). Every episode completes 1,500 steps
and holds for 8.16 s, with maximum cart excursion below 1.864 m. All 419
tests pass. The source and policy are now being frozen before any fourteen
reserved cohort is used. This development result does not yet constitute
canonical release or GitHub publication.


Final validation: source3277b4173a9333819bf35acb848e384f52a227ec passes
reserved20/20,100/100,exact20/20 and the held-out video. All141 episodes
complete1500 steps and hold8.16 s; maximum cart excursion1.863168715 m.
All419 publication-source tests and119 focused frozen-source tests pass.
The full state-faithful video and manifest verify. Ten- and eleven-second
retiming alternatives also finish negatively at5.40 and7.06 s, respectively.
Fifteen may start only after fourteen publication has been verified.
