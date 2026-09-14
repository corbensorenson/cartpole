# Generalized n-link swing-up and hold

## Objective and claim boundary

The objective is one solver **recipe** that accepts an arbitrary serial-chain
cart-pole morphology and returns a swing-up-and-hold controller. The same code
path should handle different link counts, link lengths, masses, damping, cart
mass, force authority, control rate, and rail length. It does not mean that the
same recorded force samples can be replayed unchanged on every plant.

The frozen seven-link release and the newer internal-canonical eight-link
promotion remain benchmark-specific results, with external-record comparison
explicitly open. The bottom-up ladder below is development evidence for a
reusable solver recipe; it does not retroactively change either released
controller. Files produced by
`generalized_swingup_solver.py transfer` are warm starts, explicitly labelled
`not_solution`, and are not evidence of success.

## The dimensionless plant

Let

- \(L=\sum_i l_i\) be total chain length;
- \(M=m_c+\sum_i m_i\) be total moving mass;
- \(t_0=\sqrt{L/g}\) be the gravitational time scale; and
- \(v_0=\sqrt{gL}\) be the translational velocity scale.

The solver describes a setup using the following dimensionless quantities:

| Group | Definition | Meaning |
|---|---:|---|
| Link geometry | \(l_i/L\) | where each joint lies along the chain |
| Link mass | \(m_i/\sum_jm_j\) | mass distribution along the chain |
| Cart ratio | \(m_c/\sum_i m_i\) | cart inertia relative to the chain |
| Force authority | \(\phi=F_{max}/(Mg)\) | maximum horizontal force relative to weight |
| Rail ratio | \(\rho=R/L\) | rail half-length relative to chain length |
| Usable rail | \((R-r_c)/L\) | rail after allowing for cart half-length |
| Control period | \(\Delta t/t_0\) | controller rate on the gravitational clock |
| Cart damping | \(c_xt_0/M\) | dimensionless translational damping |
| Joint damping | \(d_it_0/(ML^2)\) | dimensionless rotational damping |
| Armature | \(J_a/(ML^2)\) | dimensionless joint inertia |
| Link radius | \(r_l/L\) | finite-body geometry scale |

Plants with matching groups are dynamically similar. Length scaling therefore
changes trajectory duration by \(\sqrt{L}\), cart position by \(L\), cart speed
by \(\sqrt{L}\), angular speed by \(1/\sqrt{L}\), and physical force by mass.

## Solver architecture

```text
measured morphology
        |
        v
dimensionless plant + upright controllability audit
        |
        v
nearest solved anchor -- arc-length state/feedback transfer
        |                         |
        |                         +-- natural-time resampling
        |                         +-- force-authority scaling
        v
exact MuJoCo feasible rollout (warm start only)
        |
        v
bounded Box-FDDP / direct trajectory optimization
        |
        +-- rail continuation: measure max |x|, solve for minimum rho
        |
        v
terminal Riccati funnel + sustained-upright gate
        |
        v
bounded online force calibration; replan if mismatch is structural
```

The promotion order is deliberately `1 -> 2 -> 3 -> ...`, never a search that
starts at six or seven. At every count the process is:

1. solve the simplest deterministic energy/phase problem available;
2. compute an exact target-plant trajectory and time-varying feedback;
3. add its exact planar mirror rather than searching both directions;
4. select a route by exact forward simulation from the measured settled state;
5. require a full noisy swing-and-five-second-hold gate;
6. back-check all previously accepted link counts before promoting the new one.

The exact route selector is intentionally narrow. It evaluates a small,
symmetry-derived library with the plant model; it is not an RL policy. The
bounded actuator RLS adapter is narrower still: it can estimate force gain and
bias, but structural mismatch must trigger replanning.

### Count continuation by deterministic locked splits

Arc-length interpolation is the right generic seed when both plants are already
free chains, but it is a poor way to cross a link-count boundary: a tiny
`ghost` link creates an ill-conditioned plant and does not exactly contain the
old controller. The generalized solver therefore has a second, deterministic
count-continuation map.

For source link \(i\), a dynamic program assigns one or more consecutive target
segments. It minimizes a dimensionless length-and-mass profile mismatch, with a
fixed distal-first tie break, and then divides the source link across that block
while preserving its linear density. New internal joints are initially locked.
The result has the target link count and target total length and mass, but its
locked endpoint represents the source chain without any near-zero masses or
lengths. There are no constants indexed by link count.

The generated continuation opts into a rigid-split inertia model. At full lock,
each split group carries one combined source-equivalent capsule; only a
`1e-8` fraction of the group mass remains on the inserted bodies to keep the
compiled coordinates numerically valid, and inserted-joint armature is zero.
As the lock releases, mass moves continuously from the combined capsule into
the separate target segments and armature grows to the target value. Historical
benchmark configurations do not enable this option. The equality itself
continues through MuJoCo constraint impedance from `0.9999` to its minimum
`0.0001`, rather than changing the spring time constant and leaving a material
constraint at the end of the schedule. New split continuations interpolate
`1 - impedance` logarithmically across that four-order-of-magnitude range.
This prevents a tiny first morphology step from multiplying constraint
compliance, while the historical linear-impedance schedule remains available
for reproducing earlier artifacts.

If \(E\) lifts source route coordinates onto that locked manifold, the solver
constructs a length-weighted left inverse \(P\), satisfying

\[
PE=I, \qquad K_{target}=s_F K_{source}P,
\qquad K_{target}E\,\delta z=s_FK_{source}\delta z,
\]

where \(s_F\) is the dimensionless force-authority conversion. Thus the lifted
feedback law is preserved exactly on the locked manifold; duplicating gain
columns would incorrectly double feedback on a split link. Cart position and
all velocities are also scaled on the gravitational time clock. A continuation
configuration then releases the inserted joints while moving length, mass,
damping, friction, and stiffness profiles to the requested target morphology.

The public entry point writes both that configuration and its algebraic warm
start:

```bash
PYTHONPATH=src:scripts python scripts/generalized_swingup_solver.py split \
  --source-config SOURCE.yaml --source-links N \
  --target-config TARGET_N_PLUS_K.yaml --controller SOURCE_ROUTE.json \
  --out-config CONTINUATION.yaml --out LOCKED_SPLIT_WARM_START.json

PYTHONPATH=src:scripts python scripts/run_split_count_homotopy.py \
  --continuation-config CONTINUATION.yaml \
  --source-controller LOCKED_SPLIT_WARM_START.json \
  --output-dir runs/generalized_solver/SPLIT_RELEASE

PYTHONPATH=src:scripts python scripts/verify_split_count_homotopy.py \
  --ledger runs/generalized_solver/SPLIT_RELEASE/continuation.json
```

If the exact equality-removal boundary is the blocker, generate the three-stage
supported handoff from the last accepted frozen configuration:

```bash
PYTHONPATH=src:. python scripts/build_supported_unlock_continuations.py \
  --locked-config LAST_ACCEPTED.yaml --target-config TARGET.yaml \
  --ramp-out SUPPORT_RAMP.yaml --release-out EQUALITY_RELEASE.yaml \
  --relaxation-out SUPPORT_RELAXATION.yaml
```

Each emitted file runs through the same adaptive driver and exact replay gate;
the ramp stage explicitly uses `--allow-locked-target` because retaining the
constraint is the intended intermediate contract.

If the first supported release still rejects the exact topology switch, the
same builder continues from an already verified support level without returning
to zero:

```bash
PYTHONPATH=src:. python scripts/build_supported_unlock_continuations.py \
  --locked-config LAST_ACCEPTED.yaml --target-config TARGET.yaml \
  --initial-stiffness-ratio 1 --initial-damping-ratio 0.01 \
  --stiffness-ratio 10 --damping-ratio 0.1 \
  --ramp-out SUPPORT_1_TO_10.yaml --release-out RELEASE_AT_10.yaml \
  --relaxation-out RELAX_FROM_10.yaml \
  --stiffness-relaxation-out RELAX_STIFFNESS_FROM_10.yaml \
  --damping-relaxation-out RELAX_DAMPING_FROM_10.yaml
```

Thus support strength is a measured homotopy coordinate, not a link-count
constant. The ratios continue to scale stiffness by `M g L` and damping by
`M L^2 / t0`; the smallest exact-verified support that crosses the topology
boundary is retained, then annealed away on the fully unlocked plant.
The optional axis-separated files remove stiffness at fixed damping first and
then remove damping at zero support stiffness. This deterministic two-axis path
avoids forcing the controller through a resonance that can make a coupled
linear anneal arbitrarily step-sensitive.

The artifact records the assignment, locks, lift and projection matrices,
dimensionless compatibility errors, and numerical feedback-invariance error.
It is always marked `not_solution`. Constraint tolerances and chaotic divergence
mean even a mathematically exact locked embedding must be reoptimized through
short continuation steps; every released target still has to pass exact replay,
rail measurement, sustained hold, and the independent noisy gate.

The release driver carries forward the last waypoint horizon whose exact
refinement passed. It tries that deterministic horizon first on the next step,
then adjacent declared horizons in nearest-first order. This changes only solve
ordering—not the controller class, acceptance gate, or continuation proposal—
and avoids repeatedly paying for local windows already shown too short on the
current route branch.

The first executable back-check uses the accepted uniform two-link route and a
three-link target whose final lengths are `[0.75, 1.0, 1.25] m` and masses are
`[0.2, 0.3, 0.5] kg`. On the exactly locked start, 20 deterministic waypoint
segments required at most `0.023` normalized action correction; the subsequent
Box-FDDP replay held upright for `20.54 s` with `2.797 m` peak cart travel. The
historical linear-impedance back-check accepted two nonzero releases through
`p=0.00026`. The logarithmic-compliance schedule has now carried the same route
through `p=0.99954875`. Trial 50 held upright for `20.58 s`, used `4.500052 m`
peak cart-center travel, and had body-aware rail demand `rho=1.560017`. The
continuation scheduler retains a sorted stack of failed upper boundaries, so a
successful midpoint cannot forget an enclosing failed proposal. It also treats
the `4.5 m` waypoint rail as an optimization target rather than a hard physical
limit: a finite seed inside the exact `5.0 m` rail may proceed to Box-FDDP, but
only the unchanged replay gate can accept it. A rank-deficient dense waypoint
SVD retries deterministically with LSMR, and a failed child process no longer
prevents the remaining declared horizons from running.

The direct `p=1` equality-removal proposal remains rejected. Its final replay
reached `5.014187 m` and never entered the upright hold; a separate 12-step
waypoint probe also failed. This boundary is qualitatively different from an
ordinary small morphology step: every positive lock value emits a MuJoCo
equality constraint, whereas `p=1` removes that constraint.

The deterministic supported-unlock construction has now crossed that topology
boundary. After the dimensionless support ladder completed from `kappa=1`,
`d=0.01` to `kappa=10`, `d=0.1`, the equality release reached exact `p=1`.
Its unchanged nonlinear replay has zero remaining lock strength, held upright
for `19.38 s`, and used `4.507433 m` peak cart-center travel. This establishes
an exact free three-link topology at the declared support; it does **not**
establish the ordinary unsupported target plant.

The coupled support-removal path has exact passes through `p=0.966604614`, where
the controller held for `20.52 s` with `4.507392 m` peak cart travel. Because
spring and damping removal together became micro-step sensitive, the generic
builder now emits an axis-separated schedule. Its first axis removed `99.375%`
of the added spring stiffness at fixed `d=0.1`; the accepted replay held for
`19.96 s`, used `3.917783 m` peak cart travel, and required body-aware rail ratio
`1.365928`. The zero-stiffness endpoint remains rejected, with the next solve
also exposing severe capture/Riccati conditioning. The second axis—damping
removal at zero stiffness—therefore has not begun. These `p` values are
homotopy coordinates, not estimates of percent task completion. These stages
remain development evidence. The
[current resumable ledger](../runs/generalized_solver/n2_to_n3_split_logcompliance_homotopy/continuation.json),
[support-ramp ledger](../runs/generalized_solver/n2_to_n3_supported_unlock_ramp/continuation.json),
[first release ledger](../runs/generalized_solver/n2_to_n3_supported_unlock_k054256_release/continuation.json),
[unit-support release ledger](../runs/generalized_solver/n2_to_n3_supported_unlock_k1_release/continuation.json),
[compact hash-bound support frontier](../runs/generalized_solver/supported_unlock_frontier.json),
[generated continuation](../configs/generalized_n2_to_n3_split_logcompliance.yaml),
and [historical ledger](../runs/generalized_solver/n2_to_n3_split_homotopy/continuation.json)
are development evidence. The ledgers are explicitly `not_solution`; the exact
unsupported target, its mirror, and its independent noisy gate remain unsolved
for this count-plus-morphology path.

### Morphology-conditioned terminal set

A fixed componentwise handoff box does not scale with link count: it ignores
weakly controllable directions in the exact upright plant and does not say
whether feedback immediately saturates. The solver now constructs terminal
geometry from each measured morphology. In dimensionless coordinates
$z=Tx$, let the exact discrete upright linearization under feedback be

\[
A_c=TAT^{-1}-TB\,sKT^{-1}.
\]

For stable $A_c$, it solves

\[
A_c^\top P A_c-P=-I,
\qquad
\rho_{sat}=\frac{u_{max}^2}{K_zP^{-1}K_z^\top},
\qquad K_z=sKT^{-1}.
\]

Thus $V(z)=z^\top Pz/\rho_{sat}\leq1$ is the largest sublevel of
this Lyapunov function whose *linear* feedback never exceeds the normalized
actuator limit. It is invariant for the unsaturated linearization. It is not,
by itself, a nonlinear certificate: exact MuJoCo feedback rollout may only
shrink the accepted set. The endpoint refiner can optimize either the readable
componentwise residual or this invariant residual, and can append a short
reset-free clipped-LQR rollout to its objective. Promotion still requires the
full exact hold and noisy gates.

The standalone capture audit now makes the actuator boundary directly
falsifiable. It computes the smallest scalar multiplier on the fixed LQR gain
direction that stabilizes the exact discrete linearization, the largest
multiplier that does not saturate at the proposed state, and then runs clipped
feedback on exact MuJoCo dynamics. Fresh held-out route replays provide the
handoff state—including MuJoCo's acceleration warm start—so this is not a
nominal optimizer-state comparison.

| Handoff | Gain norm | Initial raw action | Stable/action scale gap | Saturation in 5 s | Exact result |
|---|---:|---:|---:|---:|---|
| proven n=7 route | `1,974.29` | `0.00443` | `0.00374` | `0%` | requested hold completed |
| proven n=8 route | `9,183.37` | `-0.17890` | `0.15900` | `0%` | requested hold completed |
| proven n=9 route | `47,126.1` | `-0.06487` | `0.05985` | `0%` | requested hold completed |
| current n=10 arrival proposal | `265,135` | `738.673` | `699.44` | `100%` | rail violation at `1.04 s` |

A gap below one means some scale on that fixed gain direction is both linearly
stabilizing and initially within the actuator bound; it is not by itself a
nonlinear proof. The exact rollouts supply that second check. The n=10 result
shows why gain shrinkage is not the remedy: its minimum stabilizing scale is
`0.946884`, while initial nonsaturation requires at most `0.00135378`. The
arrival trajectory must instead be optimized into the much smaller
actuator-feasible capture set. See the [audit tool](../scripts/diagnose_capture_geometry.py),
[n=7](../runs/generalized_solver/n7_release_actual_handoff_capture_geometry.json),
[n=8](../runs/generalized_solver/n8_release_actual_handoff_capture_geometry.json),
[n=9](../runs/generalized_solver/n9_release_actual_handoff_capture_geometry.json),
and [n=10 negative control](../runs/generalized_solver/n10_centered_terminal8_capture_geometry.json).

The numerical implementation uses an SVD damped minimum-norm solve and scales
the complete correction direction into its trust region. This avoids squaring
the condition number of the increasingly ill-conditioned endpoint Jacobian
and does not rotate the proposed descent direction by clipping coordinates
independently.

Absolute link orientation is treated as a field over normalized chain arc
length. Transfers convert relative joints to that field, interpolate at the
target link centers, and convert back. This lets unequal-length chains transfer
without pretending that “link 4” has the same physical meaning everywhere.

### Why total energy stops being enough

For one link, normalized energy error and angular phase describe the swing
well enough for exact partial feedback linearization followed by LQR. With two
or more links, the same total energy can be distributed among internal modes
that cancel in aggregate momentum. The two-link ladder therefore adds an
explicit deterministic phase term. At three links, an exact modal objective
(absolute-angle coherence plus absolute-rate damping) is needed before Box-FDDP
can arrest the chain. At four links, one aggregate phase variable reaches the
upright neighborhood but leaves too much energy in the distal modes. Exact
morphology-derived normal modes resolve that ambiguity: ranking measured PFL
states by collective and internal modal energy identifies a better handoff,
and full-horizon Box-FDDP then closes the route.

Five links exposed a second relationship: the three-metre rail used at lower
counts was too short for the deterministic approach that Box-FDDP could
capture. Continuing the rail to 4.5 m let the same fixed-size 13-parameter PFL
proposal produce a handoff that exact tail optimization could arrest. A final
full-horizon Box-FDDP pass from hanging supplied stabilizing feedback for the
entire route. Allowing phase-adaptive skips reduced robustness; strict
time-order replay with local feedback passed. This promotes five links without
adding per-link learned parameters. Six links then required an analytic modal
controllability seed and an exact endpoint sensitivity correction; it did not
require a neural policy.

The online adapter estimates only actuator effectiveness and bias. It projects
observed one-step model error onto the exact model's action Jacobian, fits
`equivalent_action = gain * command + bias` with bounded recursive least
squares, and compensates conservatively. Mass, length, timing, or contact errors
that cannot be explained along the action direction require short-horizon
re-optimization; they must not be hidden by an unbounded adaptive controller.

The update has three explicit guards. It rejects commands near saturation,
action-equivalent innovations outside a fixed bound, and transitions for which
more than half the dimensionless innovation is orthogonal to the modeled action
direction. The compensating command is always clipped to the actuator range
and may differ from the deterministic command by at most `0.35`. A short,
zero-mean deterministic excitation runs during the existing hanging prelude;
the estimate is then frozen before swing-up so high-energy nonlinear residuals
cannot silently rewrite the controller.

### Bounded-adaptation ladder

The [paired n=1..7 artifact](../runs/generalized_solver/adaptation_ladder_n1_n7.json)
tests exactly one narrow residual family: an unannounced affine actuator map

\[
u_{delivered}=1.18u_{commanded}+0.06.
\]

For each link count, three noisy seeds were replayed both without compensation
and with the same four-second projected-RLS calibration. The deterministic
energy law or route, feedback gains, capture law, rail, and all solver
parameters were held fixed. The perturbation produced `0/21` unadapted
successes; bounded compensation produced `21/21`. Across the adapted trials,
the largest command correction was `0.20333`, the largest gain error was
`0.00259`, and the largest bias error was `0.00052`. The verifier checks paired
seeds, full sustained-hold success, unchanged controller parameters, estimator
tolerance, and the hard correction bound.

This is development evidence that the residual layer is reusable across the
existing ladder, not a general robustness claim. In particular, a weaker
actuator can make a saturated nominal request physically unattainable. Early
underactuation probes identified the gain accurately but still failed swing-up;
that is a feasibility boundary, not an estimator failure and not something an
RL residual should conceal.

### First unequal-morphology promotion

The first target outside the uniform-chain ladder has two links with physical
lengths `[1.2, 1.8] m` and masses `[0.35, 0.65] kg`; total chain length and mass
remain 3 m and 1 kg. This changes both the dimensionless length fractions from
`[0.5, 0.5]` to `[0.4, 0.6]` and the mass fractions from `[0.5, 0.5]` to
`[0.35, 0.65]`.

The experiment preserves the intended division of labor. Arc-length state and
feedback transfer plus natural-time/force scaling supplies the starting route,
but exact replay of that route fails `0/5` noisy trials. One target-morphology
Box-FDDP refinement makes the trajectory dynamically feasible and reaches a
`20.54 s` exact upright hold. Packaging that route with its analytic mirror and
running the unchanged exact-model selector then passes `20/20` independent
noisy, uninterrupted episodes; prediction agrees with execution in all 20.

The [verified frontier record](../runs/generalized_solver/n2_unequal_frontier.json)
binds the configuration, failed warm-start control, optimizer output, packaged
routes, and gate by hash. The 4.5 m half-rail is intentionally generous for
this first morphology promotion: the observed maximum body-aware requirement
is `rho=1.19289`, versus the configured `rho=1.5`. This proves the recipe on one
nonuniform plant. It does not establish an arbitrary-morphology success rate or
the minimum rail for this plant; systematic morphology and rail continuation
remain required.

### Parameterized morphology pipeline and next boundary

`solve_generalized_morphology.py` makes that division of labor executable as
one command. It infers the source count from the controller state dimension,
reads every target physical parameter from the target config, performs
arc-length/natural-time transfer, checks the raw transfer, refines on the exact
target plant, creates the analytic mirror, runs the noisy selector gate, and
writes a hash-bound manifest. It stops immediately if exact refinement or the
gate fails; failed routes cannot fall through to packaging or promotion.

```bash
PYTHONPATH=src:scripts python scripts/solve_generalized_morphology.py \
  --source-config configs/swingup7_uniform.yaml \
  --source-controller runs/generalized_solver/n2_route.json \
  --target-config configs/generalized_n2_unequal.yaml \
  --output-dir runs/generalized_solver/reproduction/n2_unequal \
  --name n2_unequal --require-transfer-failure
```

The same command was then applied without architecture changes to a stronger
three-link target with lengths `[0.75, 1.0, 1.25] m` and masses
`[0.2, 0.3, 0.5] kg`. Its deterministic transferred route failed `0/5`; the
first exact-target Box-FDDP pass also failed the uninterrupted hold, so the new
pipeline correctly stopped before packaging. The
[n=3 pipeline manifest](../runs/generalized_solver/n3_unequal_pipeline.json)
records this as `exact_refinement_failed`, and the
[negative gate record](../runs/generalized_solver/n3_unequal_frontier.json)
remains explicitly `passed=false`. This does not invalidate the uniform n=3
result. It identifies the next algorithmic requirement: bounded morphology
continuation or a more robust multiple-shooting/receding-horizon seed, followed
by the same exact gate. The actuator adapter is intentionally not widened to
hide this structural error.

The first such deterministic repair is now implemented. Transformed-coordinate
differences are evaluated on the wrapped angle manifold, so crossing `-pi/pi`
cannot masquerade as a large discontinuity. The continuation driver divides a
neighboring accepted route into bounded exact-target waypoint problems, starts
with 24 controls (`0.48 s`), and tries deterministic `2x` and `4x` lookahead
only if the cheaper horizon and its exact refinement fail. Every retained
waypoint trajectory must be finite and rail-safe. Full-horizon Box-FDDP then
builds time-varying feedback, and only uninterrupted exact replay can advance
the homotopy. This is a deterministic multiple-shooting-style bridge; it does
not learn a route or relax the final gate.

That adaptive horizon crossed the earlier local branch wall. The latest
[verified n=3 adaptive checkpoint](../runs/generalized_solver/n3_unequal_adaptive_checkpoint.json)
advances the strong-target homotopy from `p=0.02500` to
`p=0.03602337317097499` over 45 accepted/rejected trials. At the last accepted
step, the generic 48-control repair produced a rail-safe exact trajectory and
the full-horizon feedback replay held upright for `18.78 s`. The
[packaged route](../runs/generalized_solver/n3_unequal_p036023_route.json), its
analytic mirror, and the exact gate are published independently of the bulky
local campaign traces.

The route pair by itself passed 19 of 20 noisy starts; one state lay outside
both immediate-launch basins. Feedback-gain and phase-window sweeps did not
repair it, so the solver does not disguise the issue as retuning. Instead, the
new bounded selector evaluates the analytic pair after a fixed grid of
hanging-LQR durations normalized by morphology natural time:

\[
t_c/\tau \in \{0, 0.25, 0.5, 1, 2, 4\}, \qquad
\tau=\sqrt{L/g}.
\]

It executes the shortest predicted success and stops evaluating longer rungs
as soon as either symmetric route passes. This is deterministic measured-state
model prediction, not policy learning: learned parameter count is zero, the
grid is link-count independent, and all durations scale with the plant. The
[adaptive 20-episode gate](../runs/generalized_solver/n3_unequal_p036023_adaptive20.json)
passed **20/20**, and the disjoint
[100-episode gate](../runs/generalized_solver/n3_unequal_p036023_adaptive100.json)
passed **100/100**, with prediction matching execution in all 120 episodes.
In the 100-episode gate, 99 starts launched immediately and one used only the
first `0.25 tau` rung; no longer duration was needed. The original and mirror
routes were selected 51 and 49 times, respectively, and the maximum body-aware
required rail ratio was `1.36949`. The immediate-pair 19/20 artifact remains
published as a negative control. This robust partial checkpoint is still not
evidence that the full `p=1` morphology is solved.

## Rail-length relationship

There is no morphology-only constant guaranteeing swing-up: rail demand also
depends on force authority, control rate, damping, and the chosen route. For a
specific successful trajectory, the exact requirement is

\[
\rho_{required}=\frac{\max_t |x(t)|+r_c+c}{L},
\]

where \(r_c\) is cart half-length and \(c\) is desired clearance. The solver
records this value for every route. A rail continuation then tightens \(R\),
re-optimizes, and brackets the smallest passing \(\rho\). Reporting this curve
against link count and the other dimensionless groups is meaningful; fitting a
single rail/link-count line before those controls are fixed is not.

The MuJoCo slide limit constrains the **cart center**. The body-aware ratio
reported here additionally includes the 0.18 m cart half-length, so a successful
run can have center travel below the configured `3.0 m` limit while reporting
`rho_required` slightly above `1.0` for a 3 m chain. This distinction is
intentional and must accompany any physical-rail comparison.

The unequal n=3 continuation supplies a useful local comparison under the same
3 m chain, masses, force limit, controller rate, damping, and 5 m cart-center
limit. At `p=0.0349353`, the 100-episode selector needed at most
`rho_required=1.19806`; after deterministic continuation to `p=0.0360234`, the
corresponding disjoint gate needed `1.36949`. That sharp increase over a small
morphology step is evidence that rail demand is route-branch dependent. It is
therefore carried as an optimization coordinate and acceptance metric, rather
than inferred from link count or total chain length alone.

The compact [n=3 morphology/rail frontier](../runs/generalized_solver/n3_unequal_rail_frontier.json)
now exposes this relationship across the full deterministic campaign. Of 50
trials, 27 exact replays were accepted and all 23 rejected final replays ended
in rail violations. Required ratios on accepted routes ranged from `1.07470`
to `1.36641`. Trial 50 advanced the exact-only continuation to
`p=0.036035462` with `rho_required=1.36490`, but it has not replaced the
`p=0.036023373` noisy 20/100 checkpoint. Failed-route excursions are retained
separately and are not interpreted as sufficient rail estimates.

`run_joint_morphology_rail_homotopy.py` now makes the two-coordinate procedure
explicit and resumable. It advances morphology at the current rail; only an
exact replay whose termination is `rail_violation` may trigger a deterministic
rail expansion based on the measured excursion plus dimensionless clearance.
After the first expansion, the dimensionless deficit
`rho_required-rho_configured` must decrease before another expansion is
allowed. This detects a failed trajectory that simply rides the soft boundary
outward as the rail grows. Such a result is an optimizer-basin failure, not a
measurement of necessary rail, so the driver instead rejects the proposal and
shrinks its morphology step.
After morphology reaches `p=1`, it reverses direction on the rail coordinate
and contracts toward the requested target, re-optimizing and replaying every
proposal. Expanded-rail passes remain development evidence; only `p=1` at the
target rail can be labeled `target_solved`.

The first live proposal from the previous checkpoint advanced to
`p=0.036054805242284965` without expanding the configured `rho=1.666667` rail.
The packaged route's independent exact replay held upright for `18.78 s`, used
at most `3.930247 m` of cart-center excursion, and measured the body-aware
requirement `rho_required=1.370082`. This small but real advance shows that the
previous cluster of rail-terminated proposals marked a narrow route branch,
not a proven physical minimum. The
[joint ledger](../runs/generalized_solver/n3_unequal_joint_homotopy/continuation.json),
[exact replay](../runs/generalized_solver/n3_unequal_joint_homotopy/packaged_exact_replay.json),
and [packaged route](../runs/generalized_solver/n3_unequal_p036055_route.json)
are public development artifacts; the robust noisy checkpoint remains
`p=0.036023373`. The newer analytic route pair passed a fresh `20/20` cohort
but only `96/100` on a disjoint larger gate. All four misses were predicted by
the exact selector and all terminated at the rail, with failed-route required
ratios between `1.73415` and `1.75779`. Extending the same conditioning grid
from `4 tau` through `8 tau` and `16 tau` did not change those failures. The
[20-run](../runs/generalized_solver/n3_unequal_p036055_adaptive20.json) and
[100-run negative control](../runs/generalized_solver/n3_unequal_p036055_adaptive100.json)
therefore remain public, and this point is not promoted as the robust checkpoint.

Continuing the same ledger without changing controller architecture accepted
three more exact proposals on the original rail: `p=0.036085754`,
`p=0.036135271`, and `p=0.036214500`. The latest
[full optimizer artifact](../runs/generalized_solver/n3_unequal_p036214_optimizer.json),
[packaged route](../runs/generalized_solver/n3_unequal_p036214_route.json) and
[independent exact replay](../runs/generalized_solver/n3_unequal_p036214_exact1.json)
hold upright for `18.78 s`, use `4.598134 m` of cart-center excursion, and
measure `rho_required=1.592711`. This is exact-only development evidence; it
has not passed a fresh noisy 20/100 promotion gate.

The next proposal at `p=0.036277882` failed at `rho=1.666667` with normalized
rail deficit `0.06131`. Expanding to `rho=1.8` worsened that deficit to
`0.07525`, because the failed route followed the boundary outward. The new
clearance-improvement guard stopped there, halved the morphology step, and
preserved `p=0.036214500` as the accepted frontier. This negative result is why
failed-route excursion is never treated as a minimum-rail estimate.

## Verified bottom-up checkpoint

All rows use uniform 3 m, 1 kg chains, a 1 kg cart, 80 N authority, 50 Hz
control, the declared noisy hanging start, no state resets, and a five-second
upright requirement.

| Links | Result | Deterministic backbone | Required physical rail ratio |
|---:|---:|---|---:|
| 1 | 20/20 | energy PFL -> exact LQR | 0.9594 |
| 2 | 20/20 | phase-aware PFL -> Box-FDDP -> exact LQR; mirror selection | 1.0127 |
| 3 | 20/20 | modal route -> full-horizon Box-FDDP -> exact LQR; mirror selection | 1.0008 |
| 4 | 20/20 | modal-ranked PFL handoff -> full-horizon Box-FDDP -> exact LQR; mirror selection | 0.9953 |
| 5 | 20/20 | PFL handoff -> rail continuation -> tail/full-horizon Box-FDDP -> exact LQR; mirror selection | 1.1706 |
| 6 | 20/20 | analytic modal seed -> bounded residual -> endpoint Gauss--Newton -> Box-FDDP -> exact LQR; mirror selection | 1.0929 |
| 7 | 20/20 shared evaluator; 100/100 release | frozen Box-FDDP reference route -> exact LQR; mirror selection in shared gate | 0.8501 |

Evidence: [n=1 gate](../runs/generalized_solver/energy_n1_noisy20_v2.json),
[n=2 gate](../runs/generalized_solver/n2_gate.json),
[n=3 gate](../runs/generalized_solver/n3_gate.json),
[n=4 gate](../runs/generalized_solver/n4_gate_20.json),
[n=5 gate](../runs/generalized_solver/n5_gate_20.json),
[n=6 gate](../runs/generalized_solver/n6_gate_20.json), and
[n=7 shared-architecture gate](../runs/generalized_solver/n7_gate_20.json).
Prediction and uninterrupted execution agree on every accepted n=2 through
n=7 episode. The [n=5 frontier artifact](../runs/generalized_solver/frontier_n5.json)
records the three-metre failure boundary, rail continuation, deterministic
repair, and accepted gate. The [n=6 promotion record](../runs/generalized_solver/frontier_n6.json)
records the new full-rank endpoint correction and accepted gate.

## Current reproducible commands

```bash
make generalized-gates
make generalized-adaptation-gates
make generalized-unequal-n2

PYTHONPATH=src:scripts python scripts/solve_generalized_morphology.py \
  --source-config configs/swingup7_uniform.yaml \
  --source-controller runs/generalized_solver/n2_route.json \
  --target-config configs/generalized_n2_unequal.yaml \
  --output-dir runs/generalized_solver/reproduction/n2_unequal \
  --name n2_unequal --require-transfer-failure

PYTHONPATH=src python scripts/generalized_swingup_solver.py analyze \
  --config configs/swingup7_uniform.yaml --min-links 1 --max-links 20 \
  --out runs/generalized_solver/morphology_n1_n20.json

PYTHONPATH=src python scripts/evaluate_generalized_energy.py \
  --config configs/swingup7_uniform.yaml --n-links 1 --episodes 20 \
  --out runs/generalized_solver/energy_n1_noisy20.json

PYTHONPATH=src python scripts/generalized_swingup_solver.py transfer \
  --source-config runs/generalized_solver/n3_uniform_exact.yaml \
  --source-links 3 --target-links 4 \
  --controller runs/generalized_solver/n3_route.json \
  --out runs/generalized_solver/n3_to_n4_warm_start.json

# Resume bounded morphology continuation with adaptive deterministic lookahead.
PYTHONPATH=src:scripts python scripts/run_generalized_homotopy.py \
  --source-config runs/generalized_solver/n3_uniform_exact.yaml \
  --source-controller runs/generalized_solver/n3_route.json \
  --target-config configs/generalized_n3_unequal.yaml \
  --output-dir runs/generalized_solver/n3_unequal_waypoint_homotopy \
  --waypoint-segment-steps 24 \
  --waypoint-segment-multipliers 1 2 4 --resume

PYTHONPATH=src:scripts python \
  scripts/verify_generalized_homotopy_checkpoint.py \
  runs/generalized_solver/n3_unequal_adaptive_checkpoint.json

PYTHONPATH=src:scripts python scripts/summarize_generalized_homotopy.py \
  --continuation runs/generalized_solver/n3_unequal_waypoint_homotopy/continuation.json \
  --out runs/generalized_solver/n3_unequal_rail_frontier.json

# Continue morphology and rail as independent deterministic coordinates.
PYTHONPATH=src:scripts python scripts/run_joint_morphology_rail_homotopy.py \
  --source-config runs/generalized_solver/n3_uniform_exact.yaml \
  --target-config configs/generalized_n3_unequal.yaml \
  --source-controller runs/generalized_solver/n3_unequal_p036055_route.json \
  --output-dir runs/generalized_solver/n3_unequal_joint_homotopy \
  --maximum-rail-ratio 2.0 --max-rail-rescues 3 \
  --minimum-rail-deficit-improvement 0 \
  --waypoint-segment-multipliers 1 2 4 --resume

PYTHONPATH=src:scripts python scripts/evaluate_generalized_adaptive_library.py \
  --config configs/generalized_n3_unequal_p036023.yaml \
  --controller runs/generalized_solver/n3_unequal_p036023_route.json \
  --controller runs/generalized_solver/n3_unequal_p036023_route_mirror.json \
  --episodes 100 --seed 85901 --tracking-gain-scale 1 --phase-window 0 \
  --out runs/generalized_solver/reproduction/n3_p036023_adaptive100.json

PYTHONPATH=src python scripts/evaluate_generalized_route_library.py \
  --config configs/swingup7_uniform.yaml --n-links 5 \
  --override env.rail_limit=4.5 \
  --controller runs/generalized_solver/n5_route_solver.json \
  --controller runs/generalized_solver/n5_route_solver_mirror.json \
  --episodes 20 --seed 78001 --conditioning-seconds 15 \
  --tracking-gain-scale 1 --phase-window 0 \
  --out runs/generalized_solver/n5_gate_20.json

# Rebuild the n=6 deterministic/modal route and thin exact-model refinements.
make generalized-endpoint-refine6
```

The morphology table and transfer output are analysis/warm starts, not success
claims. Promotion requires exact optimization followed by independent noisy
evaluation with rail checks and a five-second upright hold.

## Six-link deterministic promotion

Six links now passes the bottom-up 20-episode development gate. Earlier
arc-length transfers from both n=5 and the frozen n=7 release failed even with
large diagnostic rail headroom, so the successful branch starts from the
morphology itself rather than copying a neighboring route.

The fixed-size 13-parameter PFL search reached 0.936 rad from upright at its
best exact-modal handoff, but retained too much absolute-rate and internal-mode
energy for the tested tail optimizer. A three-scalar extension adds exact
collective-mode pumping, gated internal-mode damping, and a modal acceleration
limit while keeping parameter count independent of links. It reduced unwanted
motion but did not improve the complete handoff score.

The decisive deterministic layer builds the exact hanging
small-oscillation model from the measured mass, stiffness, damping, and cart
coupling matrices, discretizes it on the plant's natural clock, and solves a
regularized finite-horizon controllability-Gramian problem. The resulting
acceleration schedule has no learned or per-link tuning parameters. At n=6, a
3.9 s schedule placed the *linear* model within `5.36e-4` of its complete target
and the exact nonlinear plant passed within `0.1783 rad` of upright at 2.56 s.
That analytic approach reached six links, but its
`3.9028 rad/s` absolute angular-rate RMS and body-aware required rail ratio
`1.9464` were far outside the capture envelope, so this stage remained only a
warm start.

This experiment also exposed an action-precision trap: tail CEM had evaluated
float64 actions while `env.step` accepts float32 actions. The chaotic n=6
trajectory diverged under real replay. Tail search now quantizes every batched
candidate at the public action boundary and emits a mandatory independent
serial verification. The invalid earlier tail proposal is not evidence and is
not published as a controller. A fresh 100-iteration search under the corrected
precision contract produced no feasible handoff; its best proposal crossed the
12 m diagnostic rail at 12.046 m and was still 0.699 rad from upright.

Reproduce the analytic warm start with:

```bash
PYTHONPATH=src python scripts/materialize_modal_phase_seed.py \
  --config configs/swingup7_uniform.yaml --n-links 6 --seconds 3.9 \
  --override env.rail_limit=12.0 \
  --out runs/generalized_solver/n6_analytic_modal_seed_h3p9.json
```

See the [promotion record](../runs/generalized_solver/frontier_n6.json) and
[analytic checkpoint](../runs/generalized_solver/frontier_n6_analytic_phase.json).

### Whole-route nonlinear continuation

The analytic schedule is now preserved at full policy resolution while a
serial exact-MuJoCo CEM searches only bounded, low-dimensional additive
residual knots. This is the intended thin optimization layer: it cannot replace
the deterministic controller, and every candidate is evaluated through the
same float32 `env.step` path used by replay.

A conventional weighted sum reduced route cost from `5815.25` to `3709.99`
and cart-center travel from `5.659 m` to `5.165 m`, but it exposed an important
failure mode: the optimizer could buy lower terminal cost by moving away from
upright. The search therefore supports a capture-envelope barrier that chooses
the phase with the smallest explicit violation of angle, hinge-rate,
absolute-rate, cart-position, and cart-velocity limits before considering the
smooth quality score.

On n=6, 100 barrier iterations reduced violation from `8.7709` to `4.1654`.
The best serial state at 4.18 s had `0.1772 rad` maximum angle, `1.0144 rad/s`
hinge RMS, `2.2515 rad/s` absolute-rate RMS, cart position `0.3435 m`, and cart
velocity `-0.4601 m/s`. Maximum cart-center travel fell to `4.6148 m`, a
body-aware rail ratio of `1.5983`. That intermediate was still not a valid
handoff.

Exact upright modal decomposition localizes the remaining error: `98.05%` of
the measured modal energy is in the first collective mode and only `1.95%` in
all internal modes combined. The core solver now also exposes a finite-horizon
minimum-energy modal transition from any measured state, not only the hanging
equilibrium. Direct linear capture-tail probes were not nonlinear-feasible at
this energy. Instead, the endpoint refiner finite-differenced all 14 terminal
coordinates against 48 smooth correction knots through the ordinary float32
`env.step` path. Its Jacobian retained full rank 14. Damped minimum-norm
Gauss--Newton reduced the endpoint to `0.09335 rad` maximum angle,
`0.38612 rad/s` hinge-rate RMS, `0.73249 rad/s` absolute-rate RMS, `0.52950 m`
cart offset, and `0.30060 m/s` cart speed. Every declared handoff limit passed.

Box-FDDP then supplied time-varying feedback around that exact route and the
upright Riccati controller held for the remainder of the 30-second episode.
The route plus its exact planar mirror passed `20/20` independent noisy,
uninterrupted episodes on a `+/-4.0 m` cart-center rail. Every prediction
matched execution, all episodes reached the time limit, minimum hold was
`26.06 s`, maximum cart-center travel was `3.09881 m`, and the worst
body-aware rail ratio was `1.09294`.

This closes uniform 3 m/1 kg link-count coverage from n=1 through n=7 under one
runtime recipe and the explicitly declared per-rung rails. It does not yet
prove automatic synthesis for arbitrary unequal lengths/masses or n>=8. In
particular, the n=7 shared gate consumes the already frozen record route;
regenerating that route from the new analytic modal seed remains a useful
back-check, while n=8 is the next synthesis test.

## Eight-link direct modal endpoint frontier

The morphology-derived branch has now been run directly on the uniform
eight-link plant; it does not pad, lock, spring, or replay the seven-link
record route. A deterministic horizon scan selected a `3.8 s` analytic normal-
mode seed. A bounded 56-knot exact-model residual search then reduced the
capture-envelope violation from `6.69` to `3.23` while keeping all candidates
valid and peak cart-center travel below `4.93 m`.

Exact endpoint Gauss--Newton subsequently differentiated all 18 terminal
coordinates against the smooth correction knots through the ordinary float32
MuJoCo step path. Every stage retained rank 18. Competing rate and cart
objectives were resolved by a reproducible convex blend of two exact routes,
followed by another bounded correction. The refiner exposes this generally as
`--secondary-controller` plus `--blend-alpha`; it is deterministic route-space
continuation, not learned policy interpolation.

The strongest endpoint at `3.94 s` has:

| Quantity | Exact value | Common handoff limit |
| --- | ---: | ---: |
| Maximum absolute angle | `0.114742 rad` | `< 0.15 rad` |
| Hinge velocity RMS | `0.243317 rad/s` | `< 0.75 rad/s` |
| Absolute angular velocity RMS | `0.669792 rad/s` | `< 0.75 rad/s` |
| Cart position | `-1.064930 m` | `|x| < 1.25 m` |
| Cart velocity | `0.460546 m/s` | `< 0.50 m/s` |
| Peak cart-center excursion | `3.716165 m` | diagnostic `+/-12 m` rail |

The [exact frontier artifact](../runs/generalized_solver/n8_gn_stage13.json)
contains the route, trace, source hashes, endpoint, and full optimization
history. It is marked `not_solution`. Exact local-LQR and Box-FDDP replay from
the gate-valid endpoint latched but held upright for only `0.08-0.14 s` before
leaving the nonlinear basin and violating the diagnostic rail. Alternative
LQR control and state weightings did not improve that boundary. Therefore the
direct branch has deterministically solved eight-link swing-up *to the common
capture envelope*, but it has not solved swing-up-and-hold.

This sharpens the generalized design requirement. As link count rises, the
linear upright controller's practical nonlinear basin contracts faster than
the earlier componentwise capture box. The next generalized layer should
compute or approximate a link-count- and morphology-conditioned invariant
terminal set, then drive the endpoint into that set with the same full-rank
exact correction. Promotion still requires an uninterrupted noisy 20-episode
gate and then the canonical evidence bundle.

The morphology-conditioned metric has now been exercised on this same direct
route. The common-box endpoint requested a raw normalized LQR action of
`-309.54` and had linear invariant value `1.2138e8`, explaining why its
componentwise success did not translate into capture. Three bounded exact
continuations reduced that value to `16.30` and the raw handoff request to
`0.134`, with all 18 endpoint directions retained in the numerical Jacobian
and peak cart-center travel `3.9215 m`. The
[invariant-set frontier](../runs/generalized_solver/n8_invariant_frontier.json)
is also marked `not_solution`; its exact clipped-LQR replay still leaves the
nonlinear basin. A ten-step exact-feedback terminal objective reduced its mean
rollout value by 57% (`107642` to `46077`), as recorded in the
[feedback-rollout probe](../runs/generalized_solver/n8_invariant_feedback10_stage1.json),
but did not yet move the endpoint
inside the empirically verified nonlinear set. This establishes the next
optimization target without promoting an eight-link hold claim.
