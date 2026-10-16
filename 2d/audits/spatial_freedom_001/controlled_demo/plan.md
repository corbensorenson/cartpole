# Spatial chain environment and first curriculum — 2026-10-16

Question: can a two-force cart balance and swing up a fully spatial chain, and
where does the first fixed-parameter controller recipe fail as links are added?

This is a new benchmark. The original one-force hinge benchmark is preserved at
D:/ChatGPT/cartpole. Baseline: total length 3m, rod mass 1kg, cart 1kg, uniform
links, radius 0.025m, armature 0.0005 per rotational DOF, total damping 0.0075
per rotational axis distributed across joints, cart damping 0.02 per slide.
Two unlimited cart slides on XY; every passive joint is an unlimited ball joint.
Gravity -9.81 Z; RK4 0.005s, actions 50Hz; Euclidean force <=80N; float32 action
delivery; square boundary |x|,|y|<=3m. No physical floor contacts or joint locks.
Axial twist is free, not scored. State uses quaternions, differences use tangent
rotations. Physics is float64.

Separate milestones: near-upright balance versus complete hanging-start swing-up.
Success: complete 30s valid episode, no physics warnings or boundary exit even at
physics substeps, all WORLD rod tilts <0.15rad for >=5 consecutive seconds.
These match the spirit of the original gate, extended to a 3D world tilt and XY
boundary. They do not claim infinite stabilization or hardware real-time control.

First experiment: native finite-difference discrete LQR at upright, projected
away from unactuated axial spin. Sequential 1..8 links, 12 seeds beginning
20261016, local joint rotations uniform +/-0.02rad including twist; joint and
cart velocities +/-0.04, cart positions +/-0.05m. Stop on first incomplete
cohort or DARE failure. Prediction: one and two links should pass; eventually
locality/conditioning will cause failure. No universal frontier inferred.
Budget <=8*12*1500 actual and equal proof control transitions, plus linearization.

Second experiment: start hanging (first ball rotated pi about Y; other joints
identity), optimize a smooth 8s initial force route using iLQR, native centered
finite differences and feedback rollout. Initial sinusoidal force 12N at 0.45Hz
in X, Y initially zero. Up to 60 iterations, six line search scales, about 4M
physics steps at one link. Complete nominal delivered-action rollout first,
then 12 fresh seeds 20261116..20261127 with full 3D perturbations. A nominal
merit improvement is not a passing controller. Warm-start revisions receive
new directories, source snapshots and optimizer history. Later comparison
changes will be added BEFORE execution. Full tapes include failures and replay.

Freeze source/config/runtime and this memo in every experiment directory. Retain
delivered/raw actions, all states, tilts, gains, optimizer trace, timing, warnings,
and independent replay equality. Successful videos show complete 30s tapes with
3D camera, timing and XY trace. Keep development and final qualification separate.

## Amendment before energy initializer screen

The first iLQR trajectory is reducing merit but still far from upright after
14 iterations. In parallel, test a bounded one-link energy-shaping initializer:
energy gains 0.5/1/2/4, cart PD (2,3)/(5,5)/(10,8)/(20,10), capture angles
0.2/0.45rad, capture rate <1.2rad/s. A fixed 15N 0.2s initial kick breaks
hanging symmetry. Total 32 exact hanging starts, <=48000 actual and proof
control transitions each. This uses planar proposals ONLY; the joints remain
ball joints and a selected route must pass separate spatially perturbed starts.
The paired initialization is deliberate development selection, not 32 independent
validation trials. Preserve each full failed or successful actual tape. Prediction:
energy shaping may give an initializer with a better arrival than a sinusoid.

## Amendment before world-coordinate balance comparison

Baseline passes 12/12 at counts 1..3, 9/12 at four. Hypothesis: using quaternion
log X/Y as independent tilt errors couples free twist into nonlinear feedback.
Test the SAME gains using physical world rod directions and angular velocities,
mapped to cumulative tilt and then relative upright-linearized coordinates. This
is a nonlinear-controller intervention, not a physics change or robustness proof.
Run paired 12-seed cohorts 20261016 onward, counts 1..10 until first incomplete
cohort. Hold initialization and all plant/acceptance parameters fixed. Budget
<=180000 actual+equal proof control transitions. Native C batch derivatives must
agree with the independent Python centered stencil before optimizer use.

## Amendment before two-link route transfer

One-link run swing_n1_001 passes nominal and 12 spatial starts, 21.74s hold
each. Transfer its XY force sequence unchanged as an INITIALIZER to two links,
then regenerate all native dynamics and gains and optimize up to 100 iterations.
No inference that a copied policy solves two links. Initial state remains root
pi about Y, all other relative balls identity. All mass/length totals and gates
fixed. Budget <=15M physics steps design, plus nominal+12 spatial validation
starts (seeds 20261116..27). Original source is frozen per run. If incomplete,
diagnose actual failure and preserve the run before another design intervention.

## Amendment before one-link final qualification

Freeze swing_n1_001/route.npz without further tuning. Test 32 disjoint seeds
20261216..20261247, same hanging perturbation distribution as development.
Budget 48000 actual and 48000 independent proof transitions, all full tapes.
Require 32/32 complete success and exact replay for this first empirical release.
No parameter selection using this final cohort. This distribution tests reset
noise, not arbitrary capture disturbances or hardware robustness.

## Amendment before quaternion-cost correction comparison

Two-link baseline merit approaches ~445 but requires large regularization. The
old cost gradient treated finite quaternion-log coordinates as tangent increments.
Correct it with SO(3) right-Jacobian inverses and a Gauss-Newton Hessian, retaining
the SAME actual scalar merit, dynamics stencil, one-link force initializer,
100-iteration ceiling and acceptance/validation seeds. Independent directional
finite differences pass at a nontrivial 2-link spatial rotation. Run a distinct
swing_n2_002; retain baseline results. Hypothesis: a gradient consistent with the
actual scalar merit improves nonlinear agreement/convergence. A pass would not
isolate a general robustness advantage, only this paired initializer comparison.

## Amendment before two-link handoff-coordinate test

Corrected route n2_002 reaches world tilts 0.01469/0.07227rad at 8s; actual and
nominal differ by only 2.02e-8 tangent norm. One passive relative twist is
-1.643rad; stationary feedback then requests up to 2513N and fails. Test only
the stationary feedback coordinate map: same route/gains/dynamics/actions and
same development starts, but WorldBalancePolicy after the 8s handoff. No new
trajectory optimization or gain selection. Prediction: eliminating the yaw
coordinate artifact improves the continuation. Budget 12*1500 actual and equal
proof transitions. This is development; a pass requires a separate final cohort.

## Amendment before two-link final qualification

World-coordinate handoff passes all 12 declared development starts. Freeze the
n2_002 reference/gains with world_balance=true; no trajectory or gain changes.
Test 32 fresh seeds 20261316..20261347 with the same reset distribution and
30s/5s/plane/force/native replay gates. Budget 48000 actual and equal proof
control transitions. Publish a second explicitly 2D release only after 32/32.

## Amendment before three-link transfer

Two-link world-coordinate handoff qualifies 32/32 fresh starts, 21.68s hold.
Transfer only its XY force sequence to three links and resynthesize using the
corrected SO(3) cost gradient, native batch centered differences, 8s horizon,
world-coordinate stationary handoff, up to 120 iterations. Keep all plant totals,
scoring and spatial reset noise fixed. Budget <=15M design physics steps, plus
nominal+12 development episodes and independent proofs. Development seeds remain
20261116..27; final qualification, if earned, will use a new count-specific cohort.
Prediction: transfer gives a useful initializer, but a pass is not guaranteed;
failure may require joint arrival/maintenance optimization or longer route timing.

## Amendment before three-link physical terminal metric

Three-link n3_001 completes 120 iterations, 0/13 passes. At the 8s nominal
endpoint world tilts are 0.289/0.260/0.335rad and substantial transverse rates
remain; the actual continuation exits the boundary at 8.32s. The stationary
controller cannot capture this arrival. Its quaternion-log terminal metric also
depends on cyclic joint twist, unlike the successful physical stationary law.

Warm-start n3_001/actions, change ONLY the terminal residual coordinates to the
same physical world tilt/rate error as the stationary policy, retain its upright
Riccati weight, running cost, native dynamics, 8s timing, 120-iteration limit,
and world-feedback handoff. Terminal derivatives use centered manifold secants
of that residual (no simulation steps) and a Gauss-Newton Hessian. Budget <=15M
design physics steps, plus nominal+12 development actual tapes and proofs.
Hypothesis: a coordinate-consistent arrival metric yields a more capturable
arrival. If failure persists, time horizon and initializer are the next factors.
Save warm-start checkpoints every ten iterations for interruption recovery;
they are optimizer proposals, not accepted controllers.

Implementation note during n3_002: the physical error's exact south-pole case
was corrected in the working source to assign pi tilt, rather than zero, and
covered by a regression test. The already-running n3_002 retains its frozen
source; do not retroactively replace that snapshot. Its exact physical success
gate uses world rod tilts independently, so an optimizer-metric corner case
cannot itself pass an inverted rod. Qualify any candidate through that gate.

## Amendment before joined three-link swing/hold optimization

n3_002 stops at 106 iterations, 0/13 complete passes. Nominal exits at 9.26s
with zero upright hold despite lower physical terminal merit. A terminal value
alone has not produced a capturable continuation in these initializations.

Extend its 8s XY force sequence by 5s of zero force. Optimize the entire 13s
route (no fixed prefix), with the previous weak running cost before 8s and a
physical upright running cost during 8..13s: angle weight 20, cart weight 1,
velocity weight 0.3, same force cost 0.0006 and boundary penalty. Retain physical
terminal Riccati metric, then world-coordinate stationary feedback after 13s.
This explicitly joins arrival and a five-second maintenance segment. It is a
new objective/horizon, not a cost-comparable continuation of the previous run.

The physical residual and centered tangent Jacobians use batched quaternion
kinematics checked against native MuJoCo body directions and angular velocities
at random full spatial states for 1,3,6 links. A native directional cost-gradient
check also passes. Native MuJoCo remains the execution and success authority.
Budget: 100 iterations, <=25M design physics steps, then nominal+12 development
episodes and independent replay. Same plant and reset/scoring gates, no joint
constraints or extra actuators. Prediction: shaping a full five-second segment
improves capture/survival; failure would favor a new route initializer or timing.

## Amendment before planar-source route lift

Inspecting the original project found `runs/generalized_solver/n3_route.json`,
backed by n3_full_fddp_controller.json and n3_gate.json. Its 11.28s planar route
has a quiet upright endpoint. It used total damping 0.015 and slide armature
0.0005; these differ from the unchanged spatial plant (0.0075 and zero).
Those original results are not spatial solves and are not transferred claims.

Lift its planar joint rotations to ball quaternions; cart Y starts zero. Decode
the historical coordinate transform exactly: cart position scale 1.25m, absolute
angle scale 0.15rad, cart rate 0.5m/s, RELATIVE hinge rates 0.75rad/s. Verify
the lifted reference against its saved physical tape before simulation. Copy and
hash the original route, full controller, specification and plant config.

Retain its X feedback as a proposal, converted from normalized action to N and
projected from spatial tangent errors. Synthesize Y feedback by native finite
horizon Riccati on the lifted reference. All cart/joint freedoms remain present;
no equality, lock, restoring spring or extra actuator is allowed. Use the SAME
new spatial plant and world-coordinate stationary handoff. Budget <=500k design
physics steps, one complete nominal/proof; if nominal succeeds, 12 spatial noisy
development starts/proofs. A passing result requires a fresh qualification cohort.
Prediction: reusing a quieter arrival may help more than polishing the inherited
eight-second orbit. A planar exact pass alone is insufficient spatial evidence.

Planar-lift V1 stops before new-plant execution: original nominal reference and
original recorded physical tape differ by up to 2.04e-7 in mixed state components,
above the initial 1e-8 reconstruction guard. That is a diagnostic discrepancy,
not proven coordinate corruption or a model-parity certificate. V2 explicitly
parses the four scales from the frozen specification, retains the original
nominal reference, records the discrepancy, and uses a declared 2e-6 guard.
The original failed source/snapshot remain retained, not silently replaced.

## Amendment before native repair of lifted planar route

Planar lift V2 reconstructs its source within the declared guard, but the new
plant's actual tracking drifts: at 11.28s world tilts are 0.186/0.0067/0.108rad,
and stationary feedback asks for 438N. Nominal exits at 12.02s, 0.06s best hold;
no spatial cohort or passing claim. Changing passive damping/slide inertia and
using the foreign feedback has not preserved the quiet arrival.

Use the ACTUAL delivered force prefix (564 commands) from that failed new-plant
episode as a new feasible initializer. Save its actual states/actions/config and
the source tape hash. Resynthesize on the unchanged spatial plant with physical
terminal metric, world-coordinate handoff, weak pre-arrival running cost, 80
iterations, <=20M design physics steps. This changes initializer and duration to
11.28s; it is not a paired objective comparison against the eight-second route.
All route commands remain free to change. Test full nominal+12 spatial starts
and exact replay. Prediction: repairing a closer, quieter new-plant arrival may
be easier than discovering one from the longer uncontrolled hold extension.

## Amendment before lifted-route X-gain screen

The lifted route already approaches upright but develops an X tracking error on
the changed plant. Test a small offline bank changing ONLY its X tracking gain
scale, in fixed order 0.75,1.25,1.5,2,3. Retain the same full spatial Y feedback,
reference, force sequence, plant and stationary handoff. Each gets one nominal
full actual episode and proof. If nominal passes, run the shared 12 spatial
development starts; freeze the first 13/13 candidate. No online selection or
seed-dependent adaptation. Budget <=5*(13*1500) actual and equal proof control
transitions, generally much less through nominal-first rejection. No final
qualification seeds are used in this screen. Hypothesis: stronger tracking can
reduce passive-parameter-transfer error; saturation may instead worsen it.

## Amendment before circular-bound backward repair

Joined n3_003 completes 100 iterations, nominal exits at 13.76s with no hold;
native repaired planar n3_004 stops at 58 iterations, nominal exits at 11.76s,
best hold 0.04s. The five X-gain scales all fail nominal; no spatial cohorts
were selected. The accepted spatial frontier remains two.

Implementation diagnosis: at an axis force limit, centered native secants see
MuJoCo's per-axis actuator clamp on one side, halving the local interior force
response. Use an interior one-sided secant there. In the backward pass solve
the 2D SPD action quadratic subject to the actual circular force budget; use
the KKT tangent feedback derivative for active steps. Keep actual rollouts and
success gates unchanged. Unit checks verify native interior derivative agreement,
QP feasibility/stationarity/complementarity and its feedback sensitivity.

Warm-start n3_004/actions on its 11.28s horizon, physical terminal metric and
world-coordinate handoff. Run up to 120 iterations, <=25M design physics steps,
nominal+12 spatial starts/proofs. This is a combined optimizer correction, not
an isolated causal claim about which change helped. Use four persistent native
derivative workers; a random-state test gives byte-identical A/B matrices and
identical integration counts against one worker. No plant parameter changes.
Prediction: respecting saturated force geometry avoids the repeated failed-step
stall and improves the actual arrival/continuation. Preserve null results.

## Amendment before globally periodic running residuals

Bounded n3_005 stops after 14 iterations, 0/13 passes, 0.04s nominal hold. The
force-geometry correction alone did not recover the transferred arrival.

The old running cost adds principal relative quaternion logs to approximate
absolute angles. Across an individual joint's pi cut this can assign a finite
cost jump to nearly identical physical poses when downstream links are bent.
It is a poor global swing objective even though its local gradient checks pass.
Replace running orientation residuals with each WORLD rod direction minus +Z,
and rate residuals with the derivative of that unit direction. This is smooth,
periodic and twist-invariant. Near upright the orientation norm has the same
quadratic tilt scale. Keep the physical terminal Riccati residual.

Run two candidates from the same n3_004 force initializer / 11.28s horizon:
terminal metric multipliers 1 and 10. Both use periodic direction running cost,
bounded backward steps, four native workers, 80 iterations / <=25M physics steps
each, unchanged plant and force/plane/full-episode gates. Each receives nominal
and 12 spatial development starts/proofs; choose only by actual task success,
then a disjoint final qualification. No final seeds or post-hoc merit comparison.
Prediction: eliminating coordinate-cut barriers permits continued descent; the
stronger terminal multiplier may trade force effort for a more capturable state.

## Conditional three-link final qualification (before development completion)

If either periodic-running candidate passes nominal and all 12 spatial
development starts with exact replay, freeze it (choose 006 before 007 if both
pass). No further tuning before the final cohort. Test seeds 20261416..20261447,
32 fresh noisy starts under the unchanged distribution/gates, <=48000 actual
and equal independent proof transitions. Require 32/32 for a three-link release,
plus a full held-out video. Any failed final cohort remains evidence and cannot
be relabeled as fresh validation after controller changes.

## Amendment before conditioned three-link launch

Candidate 006 passes 10/13 development episodes; 007 passes 13/13 and is frozen
for the declared final cohort. That cohort returns 25/32: seven failures occur
between 4.72 and 11.76s, including out-of-plane and in-plane boundary exits.
This is a failed qualification. Preserve every result and the short failed first
episode video; do not publish it as a qualified new count.

Condition the launch using 12s of hanging-state LQR before the unchanged 11.28s
route. That leaves 6.72s of post-route continuation in the fixed 30s episode.
No resets or teleportation: apply only legal cart forces and observe the state.
Regenerate the hanging gain from native physics, excluding unactuated axial spin.

Passive axial frame angles need not decay at hanging. At the measured launch,
align the nominal route's per-body axial frames to the actual bodies. This is a
gauge transformation for axisymmetric capsules and isotropic ball armature /
damping: q_i' = G_parent^-1 q_i G_i, omega_i'=G_i^T omega_i. Rotate reference
states and tracking-gain columns consistently. Forces and physical rod motion
are unchanged by this coordinate transformation; no hidden actuator is added.
Tests verify one-step native spatial equivalence and exact feedback equivalence
for random full states at 1,3,6 links. The transformation uses observed launch
orientation only, never seed labels or future states. Save launch state and angles.

Test the nominal plus 44 DEVELOPMENT starts: original 12 and all 32 now-failed
qualification starts. Those 32 are explicitly no longer reserved. Budget <=67500
actual and equal proof transitions. If all pass, freeze the conditioned policy
and use a genuinely new cohort 20271416..20271447 for final qualification.
Require 32/32 full episodes/replay and a full held-out video before publication.
Prediction: active hanging contraction plus gauge-consistent launch reduces the
perturbations entering the sensitive swing while retaining all physical freedoms.

## Amendment before four-link transfer

Conditioned three qualifies 32/32 NEW seeds 20271416..20271447; every episode
holds 6.78s, completes 30s and replays exactly. This earns three-link publication.

Transfer only its 564-command XY force proposal to a NEW uniform four-link
spatial plant. Regenerate all native dynamics and gains, use periodic world
direction/rate running residuals, physical terminal Riccati multiplier 10,
bounded backward steps, four derivative workers, 100 iterations / <=25M design
physics steps. No inherited state or gain is treated as a four-link solution.
Nominal+12 spatial development starts/proofs still run after synthesis.

If a nominal route can be maintained, use the now-tested 12s hanging contraction
and measured axial-frame alignment architecture before its unchanged 11.28s
route. Its development cohort will be the shared 44 non-reserved seed indices;
four-link final qualification is reserved as 20261516..20261547 and remains
unused until a controller is frozen. All force, plane, sampling, reset-noise and
full-episode gates stay fixed. Prediction: count transfer supplies a useful
initializer, while the conditioned launch avoids repeating the raw-start failure.

## Amendment before four-link planar-source lift

The direct three-to-four force transfer stops at 17 iterations; its nominal
actual exits at 3.58s without upright hold. Its initial single-shooting trajectory
develops large boundary and terminal errors, so the next initializer should use
the original four-link motion rather than continue polishing this force trace.

Apply the already documented planar-source adapter to n4_route.json and
n4_full_fddp_controller.json, n4_uniform_exact.yaml, and the frozen historical
capture specification. Use the same 2e-6 nominal/actual reconstruction guard,
old X gain conversion and new native spatial Y gain synthesis. Keep all new
plant parameters unchanged. Budget <=500k design physics steps plus one nominal
episode/proof, then 12 spatial starts only if nominal succeeds. Retain any input
guard failure or new-plant execution failure. A lifted reference is a proposal,
never a claim of a new four-link spatial solve.

Four-link lift V1 stops on a 77.32 mixed-component nominal/actual discrepancy.
Inspection shows that the parent full artifact's stored actual episode failed
with tracking_gain_scale=0, whereas the published route library uses the curated
`n4_route_solver.json` with gain scale 1. The source hashes match; the failed
parent execution is not a valid parity comparator for that curated controller.
Its 23.04s nominal reference also differs fundamentally in timing from three.
The historical library adds 15s conditioning outside its configured 30s route
episode; do not import that as a 30s total spatial task or change our clock.

V2 uses the curated solver as a REFERENCE PROPOSAL ONLY. Require byte-identical
nominal coordinate arrays between parent and curated artifact, parse frozen
scales, and verify every lifted rod world direction against those coordinates
with native forward kinematics (1e-10 tolerance). Record the large old actual
discrepancy and reference-only scope; do not waive or relabel the V1 failure as
physical parity. Then synthesize new spatial Y feedback and execute the unchanged
new plant, same <=500k design / nominal-first budget. No four-link claim follows
from the historical gate, which used a different initialization/timing protocol.

## Amendment before four-link proposal defect measurement

The curated-reference lift validates coordinate directions, but the new plant's
nominal actual exits at 4.08s without upright hold. Measure independent native
one-command transitions at all 1152 foreign reference nodes on the unchanged
new plant, with its saved smooth forces. Budget 4608 physics steps, no controller
search or success cohort. Save full predicted states and tangent defects with
separate m/rad/m-per-s/rad-per-s summaries. Hypothesis: local discrepancies are
modest while long feasible shooting amplifies them, favoring defect-aware
multiple shooting or FDDP initialized by the geometric reference. If the local
reference is grossly inconsistent, prefer a new geometric initializer instead.
This diagnostic is not an actual episode, robustness result or four-link solve.

## Amendment before defect-aware four-link repair — 2026-10-16

Question: can retaining the curated four-link state path while closing its
measured native transition errors avoid the early divergence of force-only
shooting? Initial proposal is planar_lift_n4_002/route.npz (1152 commands,
23.04s). Keep the unchanged spatial plant, 80N force norm, +/-3m square,
50Hz commands, 200Hz RK4 and strict total 30s / 5s hold gate.

Implement an FDDP-inspired manifold forward pass: keep a fraction (1-alpha) of
each current dynamics gap during optimization, and set all gaps to zero at a
full step. Backward models include the affine native defect and the Jacobian
mapping successor tangents into reference-relative coordinates. Use the tested
circular action quadratic and physical periodic direction/rate running costs;
physical terminal Riccati multiplier 10. Acceptance uses actual scalar cost plus
a normalized gap penalty, initially 1000 and increased tenfold every ten
iterations if necessary. This differs from Crocoddyl's exact expected-improvement
acceptance rule; do not describe it as an exact Crocoddyl reproduction.

Sources consulted before implementation: Mastalli et al., Crocoddyl / FDDP,
https://arxiv.org/abs/1909.04947 and the official FDDP forward-pass / DDP backward
source at https://gepettoweb.laas.fr/doc/loco-3d/crocoddyl/devel/doxygen-html/.
Tests compare the affine backward pass against a separately condensed global
quadratic optimum, verify quaternion gap contraction at alpha=0/.5/1, verify
reference-relative Jacobians, and require a gap-free rebuilt native tape. The
test caught a zero-increment quaternion renormalization; full native steps now
retain their exact bytes rather than applying a zero retraction.

Budget: <=80 iterations / 25M design physics steps, four persistent native
workers. Preserve optimizer history and shooting state/action/gap checkpoints.
At the end, separately rebuild a native gap-free feedback route, then test a
full exact hanging-start actual episode with float32 forces and independent
replay. Only if nominal passes, test the shared 12 spatial development starts.
An inconsistent optimizer proposal is never executed as a physical state reset
or counted as a solve. Final four-link qualification seeds remain untouched.
Prediction: defects should contract while the useful arrival remains nearby.
If true rollout still fails, examine objective/feedback/handoff and defect
directions, not just scalar merit. No quiet tail or extra parking time is granted.

## Amendment before continuous three-to-four geometric split

The long historical reference leaves little time for launch contraction inside
our 30s episode. Build shorter four-link state proposals from the solved three's
11.28s base motion, not just its forces. For each source link index 0,1,2,
insert a new target ball with identity relative rotation and zero relative rate
in the REFERENCE ONLY. This continuously duplicates that body's world direction
and rate. The real four-link plant is fully unlocked, uniform, unchanged, with
total length/mass fixed; there is no equality or spring keeping the added joint
straight. The state proposal is explicitly allowed to be dynamically inconsistent.

Verify direction/rate preservation, then measure native local defects for all
three proposals (<=6768 physics steps). Select the lowest fixed normalized gap
sum, using the same scales as the repair solver. This is initializer selection,
not a success test. Freeze it, then use the SAME defect-aware repair recipe as
the historical-reference experiment: 80 iterations, terminal multiplier 10,
initial penalty 1000, four workers, <=25M design steps, nominal-first actual
validation/proof and the shared 12 development starts if nominal passes.
Prediction: this preserves a useful short arrival while giving the new joint
room to change, avoiding a discontinuous spatial interpolation or a force-only
rollout that immediately loses the predecessor's motion.

## Correction and restart of the first defect-aware experiments

The two initial repair runs are intentionally interrupted after identifying a
manifold linearization error. The implementation mapped native successor
perturbations using only the right-log Jacobian, while treating the optimized
successor's defect Jacobian as identity. Eliminating BOTH Jacobians yields
Ad(exp(c)) on each ball rotation block, c=log(q_ref^-1 q_F), with additive
cart/velocity blocks. This also matches the derivative of the actual
gap-preserving forward map at alpha=0. A new test compares the complete native
map (state AND force perturbations) against that transport, including nonzero
quaternion defects; it passes. The separate affine optimum, gap contraction and
gap-free native rebuild tests continue to pass.

Keep the original source, histories and checkpoints. Record interrupted design
accounting as a known lower bound plus a one-iteration upper bound; do not claim
an exact final step count while termination may interrupt a native batch. No
actual validation episode was reached by either old run.

Restart each from its last saved checkpoint using the corrected solver and the
penalty recorded at that checkpoint (the standard initial regularization is
restarted). These are new experiment identities, not continuations pretending
to use unchanged code. Each receives <=80 additional iterations / 25M new design
physics steps and the same nominal-first actual gates. This is a correctness
repair; any performance change remains empirical, not assumed.

## Amendment before finite four-link mechanical parity comparison

Compare independent reference-node transitions for three models: unchanged
benchmark, old total joint damping 0.015, and old damping plus slide armature
0.0005. All still have free XY slides and unrestricted ball joints. The last
two are ALTERED-PLANT diagnostics only. Save exact overridden arrays, successor
states and defects, <=13824 physics steps, no episode or success claim.
Prediction: restoring the historical mechanical values should reduce the
reference's one-command defect sharply if those values explain the mismatch.
If it does not, inspect reference construction/other parameters before attempting
parameter continuation. No new count is accepted until the unchanged benchmark
is restored and all original full-episode/noise/video gates pass.

The finite comparison gives a maximum joint-rate defect 0.2552rad/s on the
benchmark, 0.001448rad/s after restoring old damping, and 9.63e-8rad/s after
also restoring slide armature. This is finite local parity evidence, not a
complete replay. Add an explicit cart_armature config field, default zero;
the benchmark model is unchanged. Verify old qualified tape parity as a
regression. Both packaging and release publication now reject any config that
differs from the unchanged benchmark, preventing altered-plant promotion.

Before attempting continuation, replay the curated route with old damping
0.015 and slide armature 0.0005 on the fully spatial model. Rebuild its Y
tracking feedback on that model, retain transferred X feedback, and regenerate
its upright gain. Budget <=500k design physics steps plus one actual full
episode/proof. Save the positive armature in actual/proof configuration. A pass
is an altered-plant COMPONENT only; a failure must be diagnosed before assuming
parameter homotopy can start from it.

## Amendment before continuous restoration of four-link benchmark mechanics

The old-mechanics spatial replay completes 30s, holds 7.22s and cold-replays
exactly. This is an ALTERED-PLANT nominal component, not a four-link record.
Start from its actual delivered 1152-command prefix and actual states, retaining
the working tracking gain table. Increment a scalar fraction from 0 (damping
0.015, slide armature 0.0005) to 1 (benchmark 0.0075 and zero). Regenerate only
the stationary upright gain on each actual plant. A candidate receives one full
30s nominal episode and independent proof. Retain its new actual reference/forces
only if full gates pass; failed candidates do not replace the incumbent.

Initial/max fraction step 0.02, halve on failure, increase 1.5-fold after two
successes up to 0.02; stop below 0.0003125 or after 200 trials. Inclusive ceiling
<2.5M physics steps for design/actual/proof. Save every controller, actual tape,
configuration, pass/fail and parameter value. At fraction=1 use the exact default
benchmark Config, not a numerically near-restored altered plant. No parameter
changes occur during any one actual episode. Even a final nominal pass requires
subsequent spatial development and fresh qualification/video before publication.
Prediction: small actual-execution rebuilds can preserve the quiet arrival while
removing the old mechanics, instead of asking one long unstable rollout to absorb
the entire mismatch. If continuation stalls, record its boundary and improve
feedback/trajectory there; no altered result will be promoted.

## Amendment before state-trust repair of four links

Mechanical continuation stalls after restoring only fraction 0.0025; eight
trials do not reach the benchmark. Retaining the historical feedback is
insufficient. Corrected legacy defect repair also fails both actual nominal
episodes: its closed-gap proposals travel far outside the square. Preserve
these failures and their complete source/checkpoints.

Try the tested state/control increment regularization, centered on the current
proposal rather than on upright. Start afresh from the original historical
spatial reference and, independently, the short three-to-four geometric split.
Use the unchanged benchmark, initial penalty 1000, terminal scale 10, initial
trust multiplier 1, four workers, 80 iterations and <=25M design physics steps
each. Tiny accepted steps increase trust; only steps >=0.5 decrease it. Keep
the existing actual float32 nominal-first gate and 12 development starts.
This tests whether suppressing the large nonlinear departure allows defect
closure while preserving the useful reference geometry. Optimizer feasibility
or merit alone cannot promote a controller.

## Amendment before fixed-target tracking repair

Add a complementary repair objective that penalizes deviation from the known
old-mechanics actual motion and its delivered forces, rather than encouraging
an immediate new swing shape. The target is a proposal only. Every derivative
and actual episode uses the exact unchanged four-link benchmark. The affine
backward pass includes force-reference gradients; its optimum is checked
against independent condensed linear-quadratic optimization. A short native
test checks defect closure and continuous gap-free execution.

Freeze the historical old-mechanics nominal tape and its 1152-command prefix.
Use cart position weight 4, local rotation weight 40 (axial 0.1), velocity 0.05
(axial 0.005), force difference weight 0.001, terminal 10*upright P+Q, cart
soft penalty beyond 2.7m, initial gap penalty 1000, centered state/control trust
1. Four workers, 40 iterations, <=15M design physics steps. Preserve all gaps
in intermediate proposals and rebuild a separate native gap-free execution
candidate. Require actual float32 nominal, cold replay and the same 12 noisy
development starts before considering further qualification. Prediction: local
feedforward corrections can cancel mechanical mismatch while the target cost
preserves the quiet arrival. This does not promote the altered target plant.

The fixed-target repair closes its gaps on iteration zero and subsequently
reduces tracking cost. Before the full optimization finishes, freeze its FIRST
checkpoint (iteration 000, selected without actual outcomes) for a separate
full-episode float32 nominal-first probe plus 12 shared development starts if
nominal passes. Rebuild only the stationary upright design, <=244 design
physics steps. Actual/proof ceiling 156000 steps. Preserve this probe separately
from the final optimized policy; it receives no reserved qualification seeds.

## Four-link frozen qualification (before reserved execution)

The first-checkpoint probe passes nominal plus all 12 shared development starts,
with complete 30s episodes and exact cold replay. Select this already frozen
controller in checkpoint_n4_tracking_001/route.npz; do not substitute a later
optimizer result after observing qualification. Run exactly 32 reserved noisy
hanging starts, seeds 20261516..20261547, noise 0.02 under the unchanged model,
with no parking or timing extension. Require 32/32 full successes and replay
integrity; <=384000 actual+proof physics steps. On a complete pass render the
entire first qualification tape at 50fps, preserve failures and all alternate
experiments, and publish as explicitly 2D four-link evidence/video. If this gate
fails it becomes development, and a revised policy requires a new seed cohort.

## Amendment before five-link spatial transfer and native tracking repair

Four links is qualified and published. Advance to five using the curated
n5_route_solver.json and its declared source n5_route_solver_rail45_full_fddp.json
from the read-only historical checkout. Source selected morphology is uniform:
five 0.6m / 0.2kg links; old damping 0.015, slide armature 0.0005. The source
actual run used a 4.5m rail and reached 3.331747m, so it is NOT a pass of our
3m spatial boundary. Its short 421-command / 8.42s motion is only a proposal.
Freeze the original config/spec and source files; verify the coordinate lift
and shared nominal arrays using the reference-only importer. Budget <=150k
native design steps plus nominal-first actual gates and 12 shared development
starts if nominal passes. All new simulations use uniform free five-link
benchmark mechanics, two cart forces, and the unchanged strict 30s clock.

Then, irrespective of the direct transfer outcome, repair that frozen geometric
proposal with the SAME fixed-target tracking method and weights as four:
40 iterations, four workers, <=15M design physics steps; actual float32 nominal
and 12 shared development starts. The nominal target may have gaps and cross
our boundary; the separate actual execution must pass every original gate.
Prediction: bounded affine corrections plus the 2.7m cart soft penalty can
bring the inherited 3.33m excursion inside the square while preserving its
short arrival. This is empirical, not assumed. Reserved five-link seeds
20261616..20261647 remain untouched until a controller is frozen after passing
all development episodes.

Conditional five-link robustness intervention: if the repaired route passes
nominal but fails any noisy development starts, retain that failure and apply
the already tested hanging-feedback/axial-frame alignment wrapper. Use 12s
parking plus its 8.42s route, entirely within the fixed 30s episode. Rebuild
the hanging stationary LQR on the unchanged five-link model; test nominal plus
the same shared 44 development starts (20261116..27 and 20261416..47), with
<=1000 design steps and <=540000 actual+proof steps. No reserved seeds are used
for this intervention; a complete pass must still be frozen before fresh
qualification. Do not condition or alter the route if all raw development
episodes already pass.

## Five-link frozen qualification (before reserved execution)

tracking_repair_n5_001 passes nominal plus all 12 shared noisy development
starts, every episode complete with 21.70s hold and exact replay. Freeze its
final route.npz with no conditioning or additional modification. Run exactly
32 reserved hanging starts, seeds 20261616..20261647, noise 0.02, strict 30s
episode and unchanged model. Require 32/32 success and exact replay; <=384000
actual+proof physics steps. On full pass render the entire first reserved tape
and publish explicitly 2D five-link source, evidence and video. A failed gate
becomes development and any revised policy requires new qualification seeds.

## Amendment before six-link spatial transfer and native tracking repair

Five links is published at 2d-cartpole-5-link-v1. Advance to six using the
historical n6_route_solver.json and its declared source
n6_constraint_endpoint_gn_stage5_fddp.json. The selected source motion starts
hanging and uses uniform six 0.5m / 1/6kg links, damping 0.015 and slide
armature 0.0005. Its old execution used a 12m rail and reached 3.098808m;
that is not accepted under our unchanged 3m boundary. Its 209 commands / 4.18s
and corresponding nominal states are proposals only.

Freeze sources/config/spec, verify the reference-only coordinate lift and shared
nominal arrays, then test direct transfer on the unchanged spatial benchmark
(<=150k design physics steps plus nominal-first/12 shared development episodes).
Regardless of direct transfer performance, run the same native fixed-target
tracking repair as five, 40 iterations, four workers, <=15M design physics
steps. No initial conditioning in this first repair trial. Test final actual
float32 nominal, exact cold replay and 12 shared noisy development starts.
Prediction: the native affine correction and 2.7m soft cart penalty can adapt
the short motion while retaining its arrival. Reserved six-link seeds
20261716..20261747 stay unused until a selected frozen controller passes every
development episode. If raw nominal passes but noisy starts fail, a separately
frozen 12s hanging/gauge wrapper fits the same strict 30s clock and may be
tested on the 44 shared development starts before any reserved qualification.

## Six-link frozen qualification (before reserved execution)

The raw repaired policy passed nominal but failed four of 12 noisy development
starts. Its separate 12s hanging/gauge wrapper now passes nominal plus all 44
shared development starts, with 14.06s hold in every complete 30s episode and
exact cold replay. Freeze conditioned_n6_001/route.npz without modification.
Run exactly 32 reserved noisy hanging starts, seeds 20261716..20261747, noise
0.02, same unchanged plant and strict clock; <=384000 actual+proof steps.
Require 32/32 full success and replay. On pass render the full first reserved
tape, retain the raw failures, and publish explicitly 2D six-link controller,
source, evidence archive and video. Any failed qualification becomes development
and a revised policy must receive a new reserved cohort.

## Amendment before seven-link spatial transfer and native tracking repair

Six links is published. Use the frozen historical seven-link release controller
runs/swingup7_uniform/seven_link_release_controller.json and its declared
development_origin runs/swingup7_fddp_full_hanging_ilqr_terminal100k_deferred_lqr1.json,
from the read-only checkout. The nominal arrays match exactly; 228 commands /
4.56s, nominal cart maximum 2.370306m. This is still a proposal, not new spatial
evidence. Freeze both files, the source config and coordinate scales; verify
reference-only geometry/shared nominal arrays. New mechanics are the unchanged
uniform seven-link benchmark with free XY cart and unrestricted ball joints.

Test direct transfer (<=150k design physics steps, actual float32 nominal-first
and 12 shared noisy development starts if nominal passes). Then independently
apply the same fixed-target native tracking repair, 40 iterations, four workers,
<=15M design physics steps, nominal-first full actual gates plus 12 shared
development starts. If nominal passes but any noisy start fails, retain that
failure and test a separately frozen 12s hanging/gauge wrapper on nominal plus
44 shared development starts, <=1000 design and <=540000 actual+proof steps.
All phases must fit the unchanged 30s clock, with no state resets or locks.
Prediction: the native affine correction can adapt the old mechanics while
preserving a useful short swing. Seven-link reserved seeds 20261816..20261847
remain untouched until a frozen policy passes every development gate.

## Amendment before seven-link stronger initial feasibility penalty

Seven-link direct transfer fails nominal at 5.08s. First tracking repair stalls
after 19 iterations: normalized gap sum 0.0019955, maximum scaled gap 0.012289,
trust grows beyond 1e10 before the next scheduled penalty increase. Its separate
native gap-free rebuild fails actual nominal at 5.34s. These failures remain.

Restart from the ORIGINAL lifted seven-link proposal with the same tracking
weights and mechanics, but initial gap penalty 100000 (previously 1000).
Expose this as a validated configurable solver argument, default unchanged,
and retain every finite line-search trial's cost, gap and maximum cart position
in the optimizer journal. A native defect-closure test exercises both penalty
values. Budget 40 iterations / <=15M design steps, four workers; same actual
float32 nominal-first and 12 shared development starts, with no reserved seeds.
Prediction: stronger feasibility pressure allows larger gap-closing steps before
trust regularization stalls the solver. A closed-gap proposal remains insufficient;
the full native actual and spatial noise gates are still mandatory.

## Seven-link frozen qualification (before reserved execution)

The stronger-penalty repair closes the native gaps and passes nominal plus all
12 noisy development starts, full 30s episodes, 25.48s hold, exact cold replay.
Freeze tracking_repair_n7_002/route.npz unchanged. No hanging/gauge wrapper is
needed or applied. Run exactly 32 reserved noisy hanging starts, seeds
20261816..20261847, noise 0.02, unchanged spatial benchmark and strict clock;
<=384000 actual+proof physics steps. Require 32/32 success/replay, then render
the entire first held-out tape and publish explicitly 2D seven-link source,
controller, evidence archive and video. Retain the failed first repair. A failed
qualification becomes development and a revised controller needs new seeds.

## User-requested spatial physics audit before further publication

The user observes that the videos often follow a line and requests verification
of the simulator. Hold seven-link publication until this audit is resolved;
its already-running reserved evaluation may finish unchanged. Inspect compiled
canonical models at 1,3,7 links, serialized reference/actual tapes and native
responses. Verify two independent slide actuators, three DOFs per ball joint,
no equality/joint limits or joint motors, no axis clamping in state transitions.

Run matched X-only/Y-only force probes from hanging and compare physical cart
and rod responses after a 90-degree rotation. Perturb each joint separately
about both transverse axes, and verify passive axial spin evolves. Drive a
changing-direction XY force sequence for 3s and record native states, forces,
world rod directions and replay; require a rank-two cart path and rod motion.
These are diagnostic components, not swing-up successes. Budget <30000 native
physics steps. Quantify lateral cart/force/rod motion in each accepted count's
frozen reference and all qualification tapes. Save source, configuration and
all audit results. Explain any planar bias in the controller separately from
physical freedoms. If a constraint/axis bug exists, fix it and re-evaluate
affected results before publication. Otherwise strengthen documentation and
produce a visibly nonplanar controlled demonstration on the same plant.

The independent audit passes: compiled 1,3,7 models are free; persistent native
MuJoCo matches the environment bit-for-bit; every ball joint responds to both
transverse perturbations and passive spin. Changing-direction forces produce
rank-two cart/rod motion. References 4,5,6 are essentially planar, while the
two-link reference has substantial two-axis motion. Record this distinction.

For a controlled higher-count demonstration, keep the qualified six-link
controller unchanged until t=18s, then command a smooth XY orbit using its
existing world-coordinate upright feedback. Orbit radius ramps as
0.35*(1-exp(-tau/2))^2, angle 0.6*tau; desired position is
radius*[cos(angle)-1,sin(angle)]. Subtract desired cart position/velocity from
the physical feedback error and add total-mass*desired-acceleration feedforward.
All rods still target upright; no state resets, constraints or joint actuation.
Test nominal plus three fixed shared noisy development starts 20261116..18,
unchanged 30s/force/boundary/hold gates, <=48000 actual+proof physics steps.
Require post-18s cart trajectory minor/major ratio >0.1 and both XY spans >0.1m,
plus full success and exact replay before rendering a clearly labeled controlled
orbit demonstration. This is a separate audit task, not a new link-count record.
