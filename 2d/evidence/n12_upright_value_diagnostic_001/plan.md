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

## Amendment before seven-link conditioned robustness repair

The audit finds no axis-locking bug, and the controlled six-link XY-loop task
passes nominal plus three noisy starts. User explicitly requests publication
of seven and continued higher counts. Seven's raw first reserved cohort failed
at 31/32, so retain the failure and repair robustness before solved publication.

Apply 12s hanging LQR and causal axial-frame alignment to the unchanged
tracking_repair_n7_002 motion. Test nominal plus the shared 44 noisy development
starts AND all 32 seeds from failed qualify_n7_001, explicitly reused as
development: 76 distinct noisy starts, 77 total episodes. Freeze/hash the extra
development summary as an input. Budget <1000 design steps and <=924000
actual+proof steps. Strict unchanged 30s clock, force/boundary/hold criteria.
If all pass, freeze the controller and separately prerecord qualification on
NEW untouched seeds 20271816..20271847. Never reuse the failed cohort as final
qualification. Publish the physics-audit scope explanation alongside a successful
seven-link release; then continue toward higher counts.

## Seven-link corrected frozen qualification (before new reserved execution)

The conditioned policy passes nominal plus ALL 76 noisy development starts,
including the entire old failed reserved cohort: 77/77 full 30s episodes,
13.48s hold each, exact replay. Freeze conditioned_n7_001/route.npz unchanged.
Run exactly 32 NEW reserved noisy hanging starts, seeds 20271816..20271847,
noise 0.02, unchanged plant/clock/force/boundary/hold gates, <=384000 actual+proof
steps. Require 32/32 before publishing the explicitly 2D seven-link solved
release and full first held-out video. The original 31/32 failure remains
separate and publicly preserved in the spatial audit. Audit and controlled
six-link XY demonstration are now public, with original six-link release files
verified unchanged. If this new gate fails, retain it as development and
prerecord another untouched cohort only after a revised policy passes development.

## Amendment before eight-link spatial transfer and native tracking repair

Seven is qualified and publicly released with full video/evidence. Advance to
the historical n8_capture_fddp_feedback120.json, uniform eight 0.375m / 0.125kg
links, old damping 0.015 and slide armature 0.0005. Its 197-command / 3.94s
nominal route reaches 3.059733m on a wider rail; it is only a geometric proposal.
Freeze the full source as both route and parent plus original swingup8 config
and coordinate scales. Verify native reference-only world-direction lift.

Test direct transfer on the unchanged spatial benchmark, <=150k design steps,
nominal-first full float32/replay gates and 12 shared development starts if it
passes. Then repair the original lifted proposal with fixed-target native
tracking, initial gap penalty 100000 (the successful seven-link intervention),
same tracking weights, 60 iterations, four workers, <=15M design physics steps.
Require full actual nominal/replay and all 12 noisy development starts. If a
nominal pass fails noisy development, separately freeze/test 12s hanging/gauge
preparation on nominal plus the 44 shared development starts. No state resets,
locks, changed mechanics or extra episode time; the 3.94s route fits the 30s clock.
Eight-link reserved seeds 20261916..20261947 remain untouched until a controller
passes development and is frozen. Keep planar-reference and deliberate
nonplanar-motion claims explicit, as established by the spatial freedom audit.

## Eight-link frozen qualification (before reserved execution)

Native repair passes nominal but fails all 12 raw noisy starts. Its separately
frozen 12s hanging/gauge wrapper passes nominal plus all 44 noisy development
starts, 45/45 complete 30s episodes, 14.20s hold each and exact cold replay.
Freeze conditioned_n8_001/route.npz unchanged. Run exactly 32 NEW reserved
noisy hanging starts, seeds 20261916..20261947, noise 0.02, unchanged spatial
plant and clock/force/boundary/hold gates; <=384000 actual+proof physics steps.
Require 32/32 success and replay before rendering the full first reserved tape
and publishing explicitly 2D eight-link source/controller/evidence/video.
Preserve all raw failures. Any failed qualification becomes development, and
a revised policy requires a newly preregistered untouched qualification cohort.
Compiled eight-link freedoms, persistent native multiaxis replay, axis symmetry
and every joint's transverse motion/spin are now covered: 48 tests pass.

## Amendment before nine-link spatial transfer and native tracking repair

Eight is qualified and publicly released with its full video and evidence.
Use n9_fddp_terminal_angle20_widerail.json from the read-only checkout, uniform
nine links, original swingup9 config and coordinate scales. The 197-command /
3.94s nominal motion reaches 2.980098m and was synthesized on a wider rail;
it is only a proposal. Freeze source as route/parent, verify coordinate lift,
and test direct transfer on the unchanged spatial benchmark: <=150k design
steps, actual float32 nominal-first/replay and 12 shared noisy starts if nominal
passes. Then repair the original lifted reference using the same weights,
initial gap penalty 100000, 60 iterations, four workers, <=15M design steps.
Require full actual nominal/replay plus all 12 noisy development starts.

If nominal passes but any noisy start fails, test a separately frozen hanging
and axial-gauge wrapper on nominal plus 44 shared development starts. Choose
18s preparation for this higher-count robustness trial before observing nine's
outcomes: eight's raw controller failed all noisy starts, and longer native
hanging contraction may improve the smaller capture basin while leaving over
8s for upright hold inside the SAME 30s clock (18+3.94s route). This changes
the controller timing only; no state resets, locks, mechanics/force/boundary
changes or episode extension. Budget <=1000 design and <=540000 actual+proof
steps for this wrapper. Fresh nine-link reserved seeds 20262016..20262047 remain
untouched until a controller passes development and is frozen. Preserve the
documented distinction between near-planar references and free spatial physics.

## Ten-link continuation plan, recorded before any ten-link execution

After nine qualifies and its video/evidence are published, use the historical
`runs/generalized_solver/n10_fddp_refined_route_feedback100.json` as route and
parent, with `configs/swingup10_uniform.yaml` and the same coordinate-scale
specification. Its nominal arrays contain 401 states / 400 commands (8s),
maximum nominal cart 2.023276m. The historical ten-link release also used a
settled launch; none of its success is assumed on the new spatial plant.
Freeze/validate the coordinate lift and test direct transfer nominal first,
then 12 shared noisy starts only if nominal passes. Budget <=300k design steps.
Repair this proposal on the canonical spatial plant with fixed-target native
tracking, the same weights, initial gap penalty 100000, 80 iterations, four
workers, <=30M design physics steps. Require actual full nominal/replay and
all 12 shared noisy development starts. If nominal passes but noisy starts
fail, separately test a frozen 16s hanging/gauge wrapper on nominal plus the
44 shared development starts. Six seconds remain after the 8s route within
the unchanged 30s clock. The 16s duration is selected before any ten-link
outcome, informed by the historical route timing and present settled-launch
experience. Budget <=1200 wrapper-design and <=540k actual+proof steps.
No plant, force, boundary, noise or hold relaxation. Keep all failures.
Reserve NEW ten-link qualification seeds 20262116..20262147; do not use them
until the controller passes development and is frozen under a new memo.

## Nine-link frozen qualification (before reserved execution)

Native repair passes nominal with 26.18s hold but fails all 12 raw noisy
development starts. Its separately frozen 18s hanging/gauge wrapper passes
nominal plus all 44 shared noisy development starts, 45/45 full 30s episodes,
8.18s hold each and exact cold replay. Freeze conditioned_n9_001/route.npz
unchanged. Run exactly 32 NEW reserved noisy hanging starts, seeds
20262016..20262047, noise 0.02 and unchanged canonical physics, clock, force,
substep boundary and hold gates; <=384000 actual+proof physics steps.
Require 32/32 success/replay before rendering the full first held-out tape
and publishing explicitly 2D nine-link source/controller/evidence/video.
Retain every raw failure. Failed qualification becomes development; any
revised policy needs a new untouched qualification cohort. Nine-link compiled
freedoms, persistent native multiaxis replay, X/Y symmetry and each joint's
transverse motion/spin now pass within the 52-test suite.

## Eleven-link continuation plan, before any eleven-link spatial execution

After ten qualifies and its full video/evidence are publicly verified, use
`runs/swingup11_uniform/eleven_link_controller.json` from the read-only checkout
as route and parent, `configs/swingup11_uniform.yaml`, and the same scale spec.
The original nominal has 401 states / 400 commands (8s), cart maximum
2.071342m. Its continuous angle branches describe the same world directions;
the importer must verify native geometry before any acceptance test. All
historical results remain geometric proposals, not new spatial qualifications.
Test direct transfer nominal first, then 12 shared noisy starts only if it
passes, <=350k design physics steps. Repair the lifted proposal on the unchanged
canonical spatial plant using the same fixed-target weights, initial gap
penalty 100000, 80 iterations, four workers and <=35M design physics steps.
Gate actual full float32 nominal/replay plus the 12 shared noisy starts.
If nominal passes but any noisy development fails, separately freeze/test
16s hanging/gauge preparation, followed by the 8s route and upright feedback,
on nominal plus all 44 shared development starts. These phases fit in 30s;
the timing is chosen before any eleven-link outcome from the source route
and settled-launch recipe, without changing mechanics, noise, forces or gates.
Budget <=1300 wrapper design and <=540k actual+proof steps. Preserve failures.
Reserve new eleven-link qualification seeds 20262216..20262247 until a
development-passing controller is frozen under a separate qualification memo.

## Ten-link frozen qualification (before reserved execution)

The native fixed-target repair completes full nominal with 22.12s hold but
fails all 12 raw noisy starts. The preplanned 16s hanging/gauge wrapper passes
nominal plus all 44 shared noisy development starts, 45/45 full episodes,
6.12s hold each and exact cold replay. Freeze conditioned_n10_001/route.npz
unchanged. Run exactly 32 NEW noisy hanging starts, seeds 20262116..20262147,
noise 0.02, canonical unchanged physics and 30s clock/80N norm/3m substep
boundary/0.15rad five-second hold gates. Budget <=384000 actual+proof steps.
Require 32/32 full success/replay before rendering the first reserved full
video and publishing explicitly 2D ten-link source/controller/evidence/video.
Any failed qualification becomes development only, and a revised policy
needs a new untouched cohort. Ten-link independent native physical-freedom
checks are included in the current passing 56-test suite. Retain raw failures
and distinguish optimizer-trial instability from accepted physical episodes.

## Eleven-link numerical repair amendment, before second native repair

The first native repair accepts no step, stops after 11 iterations / 2,677,380
design steps, and fails actual nominal at 8.54s with no hold. Its original
proposal, rejected trials and source are retained. The stored upright value
has entries around 5e16 and a largest eigenvalue around 1.5e17; binary64 normal
value updates can lose order-one costs and acquire numerical negative modes.
The new optional residual-QR backward pass avoids normal-matrix downdates.
Independent condensed affine/trust optimization, a closed-form two-control
problem with 1e18 terminal value, an active-circle feedback derivative, and
native manifold defect closure all pass; the full suite now has 64 tests.

Run `tracking_repair_n11_002` from the ORIGINAL lifted proposal, unchanged
weights/initial gap penalty 100000, 80 iterations, four workers, <=35M native
design steps, with `--sqrt-backward`. The terminal optimizer cost is explicitly
evaluated as a residual norm: the rounded Riccati value's numerical negative
eigenvalues are projected to zero (material indefiniteness is rejected), and
the original diagonal Q is stacked separately. Record the smallest projected
eigenvalue each iteration. This is a disclosed approximation to the optimizer
terminal value, never a plant change or a change to the stationary runtime gain.
Full float32 nominal/replay and 12 shared noisy starts remain required. If
nominal passes but noisy starts fail, the previously chosen 16s hanging/gauge
wrapper and all 44 shared development starts still apply. Reserved eleven-link
seeds 20262216..20262247 remain untouched. No qualification relaxation.

## Eleven-link backward/virtual-gap diagnostic, before execution

The residual-QR run also stalls with no accepted step after 11 iterations;
do not attribute the entire failure to normal-matrix cancellation. Freeze
`n11_backward_diagnostic_001` from the original lifted route and current source.
Compute one native derivative batch and both backward implementations at
trust 1, 1e6, 1e9. For each, test virtual-gap alpha 0, 1e-8, .005, and 1.
Record gain/feed magnitudes, first reference deviation >1e-6, costs, and all
finite states/actions/errors. Alpha zero should recover the reference up to
roundoff; failure there locates a numerical/feedback issue before changing
weights or physics. This is solely a diagnostic, with <=500k native steps,
no noisy episodes, no reserved seeds and no success claim.

## Active force-circle curvature correction and third eleven-link repair

The virtual-gap diagnostic completes with 270980 native steps. Alpha zero
recovers the target within 5.1e-7 state units, while alpha .005 diverges under
both backward implementations. This rules out gross reference reconstruction
failure. An independent closed-form two-stage constrained control problem
then exposes a real optimizer defect: the active force circle's value Hessian
was missing lambda*K.T*K. For V(x)=.5||x+g||²-2||x+g||+4, g=(3,4), both
old implementations return the wrong preceding feedback derivative. Both
regressions fail before the correction and pass afterward. The normal and
residual-QR backends now retain the force constraint's Lagrange multiplier
and its curvature; the legacy bounded iLQR path receives the same correction.
This concerns synthesis, not native physics or saved controller execution.
The complete 66-test suite passes, including native freedom/replay tests.

Before further development, replay the frozen ten-link first reserved episode
with current source in `n11_optimizer_fix_regression_001`; require byte-identical
states/actions/tilts against its accepted tape. Then run `tracking_repair_n11_003`
from the ORIGINAL lifted proposal with the curvature-corrected residual-QR
backward, same weights/initial gap penalty 100000, 80 iterations, four workers,
<=35M native design steps. Use the same nominal-first/12-noisy development
gates and the same conditional 16s hanging/gauge wrapper and 44 development
starts. No reserved seeds or benchmark changes. Retain both prior failures.

## Eleven-link symmetry-preserving upright value, before fourth repair

The curvature-corrected third repair also stalls without an accepted step;
the correction remains independently required and covered by the closed-form
regression. The frozen ten-link tape remains byte-identical after the fix.
Further diagnosis finds the combined upright DARE introduces a cross-axis
value block as large as 2.688e15, although the native X/Y identifications agree
under rotation within 1.01e-12. A single horizontal-axis DARE after averaging
those native identifications yields a positive value (minimum eigenvalue
0.29048) and stable nominal linear feedback (radius 0.98774). Rotate this
gain/value to the other axis and keep cross-axis value blocks zero, as required
by upright rotational symmetry. This changes controller design, never native
XY freedom, joint motion, mechanics, action delivery or benchmark scoring.
Physical world-rotation feedback and positive-value checks at 3 and 11 links
pass; the full suite has 68 passing tests.

Run `tracking_repair_n11_004` from the original lifted proposal with
`--sqrt-backward --axis-design`, same weights, initial gap penalty 100000,
80 iterations, four workers, <=35M native design steps. Require actual full
nominal/replay plus the same 12 shared noisy starts, then the existing conditional
16s hanging/gauge wrapper and all 44 development starts if needed. Reserved
seeds 20262216..20262247 remain untouched. Preserve every failed variant.

## Eleven-link fine-step diagnostic, before cached-model execution

The fourth repair also stalls after 11 iterations and fails nominal. All four
failed repairs are terminal, with no reserved seeds used. Before another
repair, use `n11_linesearch_diagnostic_001` to inspect the untested small-step
region. Reuse the frozen native A/B/offset/gap arrays from
`n11_backward_diagnostic_001/linear_models.npz`; copy/hash this input through
the normal freezing mechanism. Rebuild the current corrected QR backward
at trust 1 and 1e6 with the axis-separated upright value, and test virtual-gap
alphas 0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, .005, 1. Record actual native
virtual-gap states/actions/errors, cost and defect, and whether the original
merit would accept at penalties 1e5, 1e8, 1e11. Budget <=120k native steps,
no actual noisy episodes, no reserved seeds and no accepted-solve claim.
This distinguishes inadequate minimum line-search size from an inaccurate
local model; do not repeat the same stalled repair without new evidence.

## Eleven-link finer line search and bounded trust growth, before fifth repair

The cached-model diagnostic completes in 29380 native steps. At trust 1,
alpha 1e-4 gives cost 5.96856e-6 and would improve the original 1e5-penalty
merit, while .001 and .005 diverge with costs over 1e18. Trust 1e6 shows the
same valid small-step region but a higher 1e-4 cost (0.00119371). Thus the old
minimum alpha .005 excluded valid descent steps. Add optional geometric
backtracking below .005, and an optional small-step trust growth factor.
Defaults retain earlier behavior. The full 68-test suite passes.

Run `tracking_repair_n11_005` from the ORIGINAL lifted proposal, corrected QR
and axis-separated design, same objective weights, initial gap penalty 100000,
80 iterations, four workers, <=35M native design steps, minimum alpha 1e-8,
and small-step trust factor 1. Hold trust on a successfully accepted small
step rather than grow it for that fact alone; failed steps still increase
trust, and accepted large steps reduce it. No mechanics or acceptance change.
Require full actual nominal/replay and the same 12 noisy starts. If nominal
passes but development fails, retain the existing conditional 16s preparation
and all 44 shared starts. Reserved seeds 20262216..20262247 remain untouched.

## Eleven-link derivative-order diagnostic, before native probes

The fine-step repair is live and accepting updates near alpha .00015625,
but dynamics defects contract slowly. Keep it running. Add an optional
fourth-order native derivative batch, with centered state stencils and
fourth-order inward force stencils near individual component caps. As in the
existing batched derivative code, this identifies the smooth unquantized
native force map; probes near a circular boundary may lie infinitesimally
outside the feasible set, while every optimizer rollout/actual command retains
the 80N norm bound. No physics or runtime action change. Independent native
spatial directional derivatives and inward Richardson force checks at both
saturation signs pass; the complete suite now has 72 passing tests.

Run `n11_derivative_order4_001` from the ORIGINAL lifted route, recomputing
native derivatives at order 4 / epsilon .001, current corrected QR, axis-
separated upright value, trust 1, and the same fine alpha diagnostic grid.
Budget <=600k native steps. Save models and all virtual-gap results, compare
against `n11_linesearch_diagnostic_001` before choosing another repair. No
noisy starts, reserved seeds or accepted-solve claim are part of this diagnostic.

## Eleven-link terminal-value conditioning diagnostic, before cached execution

The order-four diagnostic completes in 477380 native steps. It still needs
alpha around 1e-4; .001 and .005 diverge. Higher-order derivatives alone do not
resolve the problem, so do not start an expensive repeat with that change.
The live fifth repair remains intact. Next test whether the enormous terminal
Riccati value forces an excessively stiff first feasibility correction.
Run `n11_terminal_value_diagnostic_001` with the original cached native models,
current QR and axis-separated design, trust 1, and the same fine alpha grid,
but set only the optimizer terminal Riccati scale to ZERO (retain terminal Q).
The runtime upright controller design remains unchanged. This is a training
cost diagnostic, not an easier physical benchmark or an accepted result.
Budget <=30000 native steps. Save full costs/gaps/virtual rollouts before
choosing a staged feasibility/terminal-value continuation. No reserved seeds.

## Eleven-link rollout-feedback diagnostic, before cached-model execution

The terminal-value-zero probe still diverges at alpha .001, so do not start
the proposed terminal-value continuation solely on that evidence. Inspection
of the saved failing rollout locates in-plane divergence around steps 198–207,
before transverse error grows. At steps 195–196 and 203–204, the FULL bounded
feedforward update would saturate at 80N, so its constrained X feedback gain
is almost zero. A tiny alpha trial is actually far inside the force boundary
there, but inherits that clamped feedback; nearby steps have gains in the
millions. Test this mismatch directly before another repair.

Run `n11_rollout_feedback_diagnostic_001` using the cached original native
models, current corrected QR feedforward, axis-separated upright value, trust
1 and the same fine alpha grid. Replace ONLY forward feedback with the
unconstrained finite-horizon residual-QR gain (zero offsets/residuals, identical
Q/R/value/trust). Each trial still clips its actual force norm to 80N and
recomputes the true native merit. This is a candidate-generation variant,
not a claim of solving the same constrained local feedback problem. Budget
<=30000 native steps, no noisy starts or reserved seeds. Save all evidence.

## Eleven-link simultaneous shooting correction, before sparse solve

The free-feedback probe is worse (divergence already at alpha 1e-5), so retain
it only as a failed diagnostic. Do not change the running fifth repair's
feedback. Next diagnose a correction that updates every proposal node together,
without a causal intermediate feedback rollout. The new sparse defect Jacobian
uses BOTH SO(3) residual derivatives and full 70-dimensional spatial state
increments, two force inputs, fixed initial and terminal nodes. Its whole
native residual directional derivative and independent native interval parity
tests pass; the full suite now has 76 passing tests.

Run `n11_direct_shooting_diagnostic_001` from the cached ORIGINAL native models.
Undo their orthogonal gap transport before forming the true manifold residual
Jacobian. Solve one normalized sparse linear least-squares correction with
regularization .001, state-increment box +/- .1 normalized units, force-increment
box +/-20N intersected with +/-80N component bounds, at most 20 TRF iterations
and 300 LSMR iterations per inner solve. Full-dimensional intermediate proposal
states are never physical resets. Project every candidate force to the 80N
disk and recompute independent native interval defects at alphas 1,.5,.25,.1,
.05,.01. Save the Jacobian, linear solution, solver status, proposals, costs,
force norms and source. Budget <=15000 native steps. No actual episode, noisy
start, qualification seed, or accepted-solve claim is part of this diagnostic.

## Eleven-link iterative simultaneous shooting repair, before execution

The fifth feedback/gap repair completes 80 iterations, reduces the defect score
only from 1.59571e-4 to 1.55655e-4, and fails actual nominal at 4.22s. Preserve
it as a failed development attempt. The simultaneous shooting diagnostic
produces a much stronger measured correction: one full spatial update reduces
the native defect score to 6.48856e-6 (about 96%), maximum normalized defect
.000521687, cart maximum 2.071100m and force maximum 69.206165N. Its sparse
linear solve is approximate; measured nonlinear defects determine progress.

Run `n11_direct_repair_001` from the ORIGINAL lifted proposal, with full 70-
dimensional spatial node increments and two force inputs, fixed initial and
terminal nodes, 20 iterations, order-four native derivatives / epsilon .001,
four workers, <=12M design native steps. Each sparse correction uses the same
normalized .1 state-increment box and 20N force-increment box intersected with
component actuator bounds, then exact 80N disk projection. Initial increment
regularization .001; divide it by three after each fifth successful iteration
(floor 1e-7), multiply by ten after no accepted step. At most 20 TRF iterations
and 1000 LSMR iterations per linear solve. Backtrack alpha 1,.5,.25,.1,.05,.01,
.001 using measured native defects plus the increment trust penalty. Stop
early below maximum normalized defect 1e-11; otherwise retain every checkpoint.

After repair, build unconstrained finite-horizon QR tracking feedback with the
same local Q/R weights and the axis-separated upright value, then perform a
separate gap-free native rebuild. Save proposal and rebuilt route distinctly.
Only actual full float32 nominal/replay and, if nominal passes, the 12 shared
noisy starts count as execution gates. If nominal passes but noise fails,
retain the previously chosen 16s hanging/gauge wrapper and 44 development
starts. No extra clock time, resets, constraints, plant changes, or reserved
seeds; 20262216..20262247 remain untouched until a frozen controller passes.

## Eleven-link augmented sparse solve diagnostic, before execution

The simultaneous repair is live and accepting full updates, reaching defect
score 1.6e-10 after eight iterations, but its finite-iteration LSMR solve limits
accuracy. Add an augmented ridge system [lambda I, J.T; J, -lambda I], avoiding
normal-equation conditioning. Independent over- and underdetermined dense
residual least-squares comparisons pass; the full suite has 78 passing tests.
Run `n11_direct_augmented_diagnostic_001` from the same original cached models,
same .001 regularization and update/actuator bounds, one linear correction,
same six native candidate alphas and <=15000 native steps. Use the augmented
solution only if it lies within all increment bounds; otherwise record the
violations and fall back to the previous bounded LSMR procedure. This tests
linear-solve accuracy without relaxing a feasible set, physical plant or gate.

## Hybrid document transfer to 2D, before matched execution

User requests review of `D:/ChatGPT/cartpole/corben-marc-hybrid.md`, revision3.
Freeze its bytes/identity under research/sources/corben_marc_hybrid_2d_20261016.
It explicitly concerns 1D; its reported 17-link result is not evidence for this
spatial plant. Relevant sections 9–14 recommend reachable moving targets,
full affine QR values/defect offsets, native delivered-action correction and
complete-policy admission. Sparse shooting, QR, actual tapes, parking and fresh
qualification are already used here. Scalar state/action/angle conventions,
independent force clipping, published gains and orbit frequencies cannot be
copied as spatial controllers. Our force bound is a Euclidean disk and all
passive ball axes, including twist, remain physically free.

The direct sparse repair001 is terminal: 20 updates, 9744580 design native
steps, defect score3.73042264e-11, maximum normalized defect1.28568116e-6.
Despite better optimizer feasibility, rebuilt actual nominal exits at4.58s,
X3.027691m, no hold, exact replay. The rebuilt execution controller omitted
the proposal's remaining native defects and affine correction. This does not
establish the failure cause; test that intervention before more iterations.

Run `n11_hybrid_affine_001` from direct001's shooting_proposal.npz, keeping
the full spatial states, same Q/R, 10x axis-separated stationary terminal value,
8s reference, canonical plant/delivery, 30s episode and development seeds fixed.
Round reference forces with the canonical delivery function before fresh order4
native derivatives (epsilon .001, four workers). Use the actual logarithmic
reference-error Jacobian at each native successor, not virtual-gap transport.
Build full affine QR roots/residuals, offsets and gains; independent condensed
affine least-squares and predictor-isolation/affine-value native tests pass.
Three matched arms: same raw proposal/gains with zero offset; full affine
offset; full affine offset plus a new spatial native-value search. This is a
2D adaptation, not a reproduction of Marc's scalar B17 proposal algorithm.
Search every route command: delivered baseline, reference, midpoint, four XY
axis neighbors at1,.1,.01N around the current best, then one inward F32 neighbor;
deduplicate delivered bytes, cap16, stable strict-cost ties, full80N disk.
Retain every candidate/selected action, successor, cost, substep margin and
native count. Reject invalid/substep-outside-plane predictor candidates.
Predictors own independent state. After route, unchanged world balance.
No virtual-node reset or rebuilt-reference substitution in these actual arms.

Run nominal seed20261016 exact hanging for each arm; only nominal full pass
opens the12 shared noisy starts20261116..27 for that arm. Budget<=2.5M inclusive
native physics steps including design, candidates, actual episodes and proof.
Stop after these three arms; a lower value or smaller defect is not a solve.
Prediction: if omitted affine defects are material, affine improves full native
behavior over zero-offset; search additionally tests nonlinear action response.
If all fail early, retain that null result and consider earlier bridge/target
redesign instead of claiming numerical accuracy alone will suffice. Eleven
reserved seeds20262216..47 remain untouched. No release without full gates.

## Hybrid finite-tail comparison, before execution

Affine001 is terminal. Matched raw-proposal zero-offset feedback fails at5.08s
with no hold; preserving affine offsets reaches1.92s hold and exits at10.88s;
the 16-candidate native search reaches the same hold and exits10.66s, adding
25600 predictor steps. All three replay exactly. The affine controller tracks
the proposal to8s with tilt .0003054rad, then switches to stationary world
balance; tilt grows to .09035rad at8.40s and2.17068rad at8.80s. This identifies
the handoff as an actionable seam, not proof that finite feedback will fix it.

Test the document's finite continuation/guard recommendation next, retaining
the same original400-node repaired proposal, canonical plant and Q/R. Extend
the reference to the full1500 execution commands: preserve the original400
actions, use one zero-force transition from its final state to an exact upright
reference, then exact stationary upright/zero-force nodes. Preserve the initial
tail mismatch explicitly in the affine defect; do not reset any actual state.
All repeated upright nodes have identical derivative centers, so compute one
order4/.001 native model and reuse it exactly, plus a fresh bridge-state model.
Use zero finite terminal Riccati scale (terminal stage L_Q retained). This is a
stationary continuation experiment, not a periodic/moving target experiment.

Two preregistered design horizons:1500 (`n11_hybrid_finite_001`) and1550
(`n11_hybrid_guard_001`), both executing only1500 commands/30s. Each compares
zero-offset and affine feedback with the same per-run gains, no native search.
Nominal20261016 exact hanging first; a full nominal/replay pass opens only the
12 shared development starts20261116..27. Per-run inclusive budget<=2.5M
native steps. New roots/gains/offsets are controller changes, not a relabeling
of old evidence. Preserve all failures. No reserved seeds or public promotion.
Prediction: if stationary handoff rather than swing feasibility is decisive,
the affine finite design extends the physical hold; a guard comparison measures
terminal-boundary sensitivity. Failure leaves moving destinations/earlier entry
and accurate sparse correction as open alternatives, not more assumed success.

## Eleven-link finite-affine conditioned launch, before execution

Finite001's full nominal passes30s with23.50s hold and exact replay; all12
raw noisy starts fail. Zero-offset arm fails4.94s with no hold. Before guard001
outcomes are inspected, select the FIRST1500-design finite-affine controller
as the physical incumbent for preparation. Do not replace it with guard001
using held-out results. The document's affine/finite recommendations produced
a useful actual nominal pass, but reset reliability remains unestablished.

Run `conditioned_n11_001` from finite001/affine_controller.npz using the
previously selected16s hanging preparation, causal axial-frame alignment,
then the first700 commands of the full1500-command value design (30s total).
The input predates explicit swing/design markers: freeze CLI --swing-steps400,
retain original1500 roots/residuals/gains/offsets, preserve the old input bytes,
and save a new schema-marked payload rather than modifying the old checkpoint.
The reference's8s swing plus6s upright tracking executes after16s parking;
the value design extends beyond this prefix, as an explicit design guard.
Rotate feedback AND value-root tangent columns with the axial gauge; additive
force offsets and value residuals are unchanged. Actual physics states remain
continuous. No search is added. Native parking design <=1000 physics steps;
45 actual full episodes plus cold proofs <=540000 steps, inclusive<=550000.
Evaluate nominal20261016 and44 shared development seeds20261116..27 and
20261416..47. Only a45/45 pass can open the untouched32-seed qualification
20262216..47 after a separate freeze/memo. No new solve/public release yet.

## Frozen ten-link execution regression for affine adapter changes

Before using the new affine payload dispatcher for qualification, rerun the
first already-published ten-link held-out seed20262116 with its unchanged
frozen controller as `n11_hybrid_ten_regression_001`; compare states, delivered
actions and tilts byte-for-byte to qualify_n10_001/seed20262116.npz. This is an
execution-integrity regression, not new success evidence or controller tuning.
One full30s actual episode plus exact cold proof, <=12000 native physics steps.

## Eleven-link18s conditioned affine comparison, before execution

Conditioned001 is terminal at27/45 complete development successes, including
nominal7.50s hold. It does not open reserved qualification. Preserve all18
failures and full tapes. The ten-link regression states/actions/tilts remain
byte-identical;82 current tests pass. Guard001 also passes nominal23.50s but
0/12 raw noisy starts, so it supplies no demonstrated reset-reliability gain.

Run `conditioned_n11_002` with ONLY preparation duration changed from16 to18s,
the same finite001 frozen controller, alignment and44 shared noisy starts.
Actual execution then uses the first600 of its1500 design commands. Nominal
upright onset at route6.50s leaves a predicted5.50s hold, measured afresh.
All phases must complete inside30s; never extend or reset the episode.
Budget<=550000 inclusive physics steps, parking design<=1000, same nominal
and development seeds, no reserved seeds. Prediction: more contraction may
improve finite reset reliability at the cost of two seconds of hold; a full
45/45 is required to proceed. If it fails, retain the null outcome and redesign
parking costs, earlier entry or moving-target capture in separate bounded work.

## Eleven-link last finite preparation bracket, before execution

Conditioned002 is terminal40/45 development passes (39/44 noisy), nominal
5.50s hold, all exact replay. Failed seeds20261416,20261425,20261437,
20261445,20261447 remain development. More parking helps this finite cohort,
but does not yet pass the required gate. Before changing reference/gains,
test ONE final feasible timing point18.40s (`conditioned_n11_003`), same
finite001 controller/alignment and all44 shared starts. It executes580 commands
of the original1500-step design; predicted nominal hold5.10s. Budget<=550000
inclusive native steps. Nominal plus44 development, no reserved seeds. Do not
extend preparation beyond this bracket to consume the required hold window.
If it fails, retain it and pivot to launch-policy or reachable-target redesign,
rather than assume endlessly longer parking is available. No public new count.

## Online axial reference alignment pilot, before execution

Conditioned003 is terminal43/45, nominal5.10s; failures20261445(0.30s hold)
and20261447(1.86s) remain development. The timing bracket is exhausted.
Probe a different feedback law, not more parking: `n11_online_gauge_pilot_001`
uses18s preparation and the SAME finite001 arrays, but aligns the current
reference's body axial frames at EVERY feedback command using observed state.
Use closest axial polar angle atan2(R10-R01,R00+R11), rotate reference states
and gain tangent columns, retain measured angular velocities/spin and additive
force offsets. No state edit, joint lock, physical torque, future observation
or new offline synthesis is introduced. This is a changed nonlinear policy.
Independent1/3/11-link tests show force invariance under arbitrary changes of
observed passive axial frames, exact observed-state preservation and no native
physics advance. Earlier static launch alignment remains the baseline.

Pilot nominal20261016 plus the FIVE reused failed18s development seeds:
20261416,20261425,20261437,20261445,20261447, same resets and30s clock.
Budget<=73000 inclusive native physics steps (six actual episodes+cold proof,
parking design<=1000). If6/6, test all44 shared development starts in a new
frozen run before qualification; if any fail, retain their full physical tapes
and diagnose/pivot. Native search is disabled; values are not used for actions
in this gauge-feedback family. No reserved seeds or new qualified count.

## Online axial feedback full development gate, before execution

Pilot001 is terminal6/6 full30s passes,5.50s hold each, exact cold replay,
including all FIVE failures from static18s preparation.85 tests pass.
Run `conditioned_n11_004` from the frozen pilot001/route.npz with the identical
18s preparation,400 swing split, full1500-step affine design and online axial
reference alignment. Recomputing the same parking design must yield identical
controller arrays; compare every payload key against pilot001's frozen route.
No other policy/plant/delivery/timing change is permitted. All44 shared noisy
development starts20261116..27 and20261416..47 plus exact nominal20261016,
<=550000 inclusive native physics steps, all complete tapes/cold proof. Pilot
starts overlap development and are not independent evidence. Only a45/45 pass
opens fresh reserved qualification after a separate frozen memo. Reserved
20262216..47 remain untouched. Keep baseline timing and static-gauge failures.

## Eleven-link reserved qualification, before first held-out execution

Conditioned004 is terminal45/45: nominal plus44 shared noisy full30s episodes,
all5.50s hold and exact independent cold replay. Pilot/controller array parity
passes all28 payload keys.85 tests pass. Freeze EXACT conditioned004/route.npz
and CURRENT source into `qualify_n11_001`; no controller, parking, alignment,
arithmetic, plant or threshold changes after inspecting reserved outcomes.

Execute NEW untouched32 noisy seeds20262216..20262247, independent of all
development/pilot data. Canonical eleven-link spatial model, hanging reset with
independentXYZ ball noise .02 and velocity/cart noise from the declared reset;
18s hanging preparation plus the first600 commands of the full1500-command
finite-affine design, online axial reference alignment, no native search.
Each episode must finish1500 commands/30s inside the3m square at every physics
substep, force norm<=80N after F32 delivery, all rods world tilt<.15 for>=5s,
no native warning, exact independent cold native tape replay. All32 must pass.
Budget<=384000 physics steps (32 actual episodes+cold proofs; no setup design).
Record distinct initial-state count, maximum cart excursion/minimum substep
margin, minimum hold, maximum launch tilt and force. A failed held-out cohort
is retained and cannot be relabeled after tuning. Only a32/32 pass earns new
eleven-link qualification. Render the first reserved seed's full1501 frames,
verify published-source tests and byte-identical first tape, then normal GitHub
publication with explicit2D tag/full video/evidence and public hash verification.

## Eleven publication and twelve-link upright-design diagnostic

Eleven is public at source/tag84bd20a56dc15c7e9e470206a55f810c36c987eb,
2d-cartpole-11-link-v1.3335 staged/archive hashes pass; published source85
tests passes and reproduces first reserved states/actions/tilts byte-identically.
Public MP4 SHA6089f9b985ed9a4d39a6c6077a06d1283808b129f5525ff9b5457f61be1a410f,
ZIP1443984255 bytes SHA240edd10022dbea5e82660b02e8fad5d3bd4a5d4d68ee3e0a0de003febb1e71a.
Both public asset hashes/sizes and tag->commit verified. No eleven archive is
rebuilt after publication. Goal stays active; proceed to twelve.

Extending physical tests to12 passes native persistent multiaxis replay,
X/Y force symmetry, every ball joint's transverse/spin freedoms, and online
axial force invariance. The additional axis-separated STATIONARY design test
fails: existing binary64 DARE gives closed-loop radius1.5096127638. This is
a controller-design failure, not evidence of locked physics or impossibility.
Do not remove/relax that failing assertion to make the new count appear green.

Run `n12_upright_value_diagnostic_001` before importing/synthesizing a12-link
route. Freeze canonical config, native A/B/Q/R and existing DARE output; compare
identical native axis-averaged SISO26-state models with full residual-QR finite
backward designs at horizons500,1500,3000 (zero terminal value, same Q/R).
Retain gain/root, native matrix closed-loop radius, gain norms, value minimum
eigenvalue and measured design costs. No actual episode, noise or qualification;
budget<=2000 native physics steps, no target/plant change. These are numerical
diagnostics, not stationary physical robustness or a solved twelve-link count.
If finite QR produces a positive/stabilizing value, add it as an OPTIONAL
stationary design backend, preserving defaults and prior frozen controllers.
Check independent finite condensed/QR agreement and world rotation before
using it. If not, keep the limitation explicit and use a fully finite zero-
terminal trajectory design without claiming stationary stability.
