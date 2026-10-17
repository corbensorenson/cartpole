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

## Upright-value diagnostic correction, before rerun

Diagnostic001 is terminal but its finite-QR comparison is INVALID as a same-Q
comparison: it used elementwise sqrt(Q) on the coupled world-angle cost matrix.
Its finite costs therefore differ from the declared Q; retain the original run
as a corrected failed-method comparison, do not claim its radius0.9927 fixes
the original DARE problem. The stationary DARE failure remains independently
observed. Fix the residual factor to Cholesky(Q).T, verify L.T@L=Q, and run
`n12_upright_value_diagnostic_002`, same frozen native model/three horizons,
<=2000 native steps, no actual episodes/noise/reserved seeds. Save minimum
singular value of each QR root separately from rounded P eigenvalues: forming
S.T@S can lose positive definiteness in this ill-conditioned binary64 matrix.
Future design must retain the root as the authoritative nonnegative cost and
must not relabel a negative rounded eigenvalue as a physical instability.

## Twelve-link reference transfer and synthesis recipe, before execution

Corrected diagnostic002 (Cholesky root of SAME Q) has finite-QR native model
radii .98802591/.98802137/.98802142 at500/1500/3000 horizons.1500-step
root has minimum singular .537247; rounded P has numerical negative modes.
Existing DARE value has a MATERIAL negative mode about-1.497e19 and radius
1.5096. Add optional value_backend=finite_qr to upright axis-separated design,
retain authoritative value_factor and finite_value_steps, preserve all defaults.
Independent three-link native DARE/QR agreement, twelve world-rotation and
factor-based nonnegative cost tests pass;91 total tests pass. This is local
linear evidence, not native physical balance qualification. The published
eleven tag/archive/controller remain immutable and retain their85-test scope.

Run `planar_lift_n12_001` with reference-only importer and finite_qr upright
backend. Read-only historical geometric source and full-source:
`D:/ChatGPT/cartpole/runs/swingup12_uniform/twelve_link_controller.json`;
config `configs/swingup12_uniform.yaml`; scale spec
`benchmarks/p1_capture_envelope.yaml`. The file explicitly labels itself
not_solution/single-state diagnostic; do not promote it as spatial evidence.
It contains451 cumulative-world-angle coordinate states,450 actions/9s,
nominal old-model cart maximum1.913863756m. Freeze original files/hashes,
validate the material angle/rate map and all native world directions<1e-10,
rebuild canonical12-link mechanics and target feedback. Old planar damping
and slide/hinge mechanics remain distinct from the spatial model.
Evaluate full nominal/cold replay; only if nominal passes,12 shared noisy
starts20261116..27. Budget<=500000 inclusive native physics steps. Foreign
states/gains remain proposals regardless of native transfer outcome. No reserved
seeds or new qualified count. Twelve reserved cohort20262316..47 is untouched.

After transfer is terminal, use its saved geometric route in
`n12_direct_repair_001`:20 full76-dimensional spatial simultaneous state/two-
force shooting iterations, fixed initial/terminal nodes, order4 native stencil
epsilon .001,four workers, <=12M DESIGN steps. Use augmented sparse ridge when
its correction satisfies every registered state/control increment box; otherwise
fall back to the same bounded LSMR1000/TRF20 solver. Initial ridge .001, reduce
by3 after each fifth accepted update, floor1e-7; same native backtracking and
80N disk projection as eleven. Use finite_qr upright value FACTOR for the final
native QR/gap-free baseline; retain proposal separately even if baseline fails.
Nominal and conditional12 shared noisy actual/cold gates add<=156000 steps.
If appropriate, next test the same full-affine1500-step finite upright design
used successfully at eleven, with its own frozen driver/memo/output. Do not
assume the imported source or an improved defect score constitutes a solve.

## Optional-value-backend execution regression, before execution

Rerun already-published eleven's first held-out seed20262216 with its exact
frozen payload as `n12_value_backend_n11_regression_001` after the optional
finite_qr design-backend addition. One full actual30s plus cold proof<=12000
physics steps. Compare states/actions/tilts byte-for-byte with qualify_n11_001.
This is integrity regression, not new qualification or tuning data. The optional
synthesis backend must not affect execution of frozen gains/offsets/policies.

## Twelve-link raw-reference finite-affine comparison, before execution

Direct repair001 remains authoritatively LIVE (session33228, PID37724, first
bounded solve; no optimizer row yet). Do not restart it or infer termination
from an observation timeout. Its frozen numerical thread environment is empty.
While it runs, test whether native-defect-aware execution can use the original
geometric proposal without waiting for near-perfect shooting feasibility.

Run `n12_hybrid_raw_finite_001` from planar_lift_n12_001/route.npz, unchanged
canonical twelve-link model, full76D tangent state and two force components.
Fresh order4/.001 native derivatives (four native workers), same local Q/R as
eleven, zero finite terminal Riccati scale, exact upright continuation to1500
design/execution steps, no guard, no search, optional finite_qr upright backend.
Use full native log-chart defect offsets/QR roots and additive feed offsets.
Compare zero-offset and affine execution with shared gains/reference per design.
Explicit OPENBLAS_NUM_THREADS=1/OMP_NUM_THREADS=1 for this new experiment,
recorded in its runtime; this does not change the already-running repair.
Nominal seed20261016 exact hanging in each arm; only a full nominal/replay
pass opens12 shared noisy development seeds20261116..27 in that arm.
<=2.5M inclusive native steps, same30s/F32/80N-disk/substep-square gates,
complete tapes/cold replay. No twelve reserved seeds20262316..47 are used.
Prediction: if precise proposal feasibility is unnecessary for a useful affine
continuation, nominal may pass; failure is retained and does not invalidate the
ongoing repair. This is a separate controller comparison, not a restart or
claim of a newly qualified count. Do not choose using reserved outcomes.

## Twelve-link donor-coordinate correction and rerun, before execution

Raw finite001 fails both nominal arms (zero-offset2.34s, affine1.42s), but
these runs use an INCORRECT donor-coordinate interpretation and cannot assess
the intended twelve-link donor. Source12 explicitly declares identity26x26
coordinate_transform: physical relative q/v, initial root angle-PI and all
other relative angles0. The importer guessed the earlier dimensionless,
cumulative-angle convention instead, producing an initial geometric reference
with root-.47124rad and most world directions upright. Its reported1.11e-15
direction match compared against that same wrong assumption. The independent
physical-tape discrepancy3.13948 was the warning; reference-only incorrectly
allowed this mismatch without checking the initial hanging state.

Source11's declared transform really is the earlier cumulative/scaled map;
its published controller is unaffected. The simulation and native physical
tapes were not edited. Stop INVALID-INPUT direct repair001 (verified PID37724
command line matches; process termination and session exit1 confirmed). It has
one completed optimizer row565200 native steps, defect163.8035; in-flight work
after that row is not recovered, so total compute is only lower-bounded. Save
interruption.json and input_correction.json, never overwrite bad001 outputs.

Fix source decoding by inverting the donor's explicit physical-relative ->
planner coordinate matrix; project feedback through that same map. Require
finite/invertible correct dimensions, exact hanging initial error<=1e-6 and
native world-direction agreement with cumulative DECODED relative angles.
Retain scale spec only as provenance; never silently apply it over a declared
identity map. Independent identity/scaled-cumulative and tangent-feedback
tests pass, including all rods hanging down at identity-donor initialization.
Source12 decoded q/v matches its first450 physical tape rows EXACTLY offline.

Run `planar_lift_n12_002` with same original files/finite_qr backend and
WITHOUT reference-only: enforce independent foreign physical-tape parity<2e-6,
plus new initial/direction gates. <=500000 inclusive native steps; exact nominal
20261016, conditional12 shared noisy starts only on a full pass. No reserved
seeds. Then `n12_direct_repair_002` uses the corrected saved proposal, same
20/full76D/order4/.001/four-workers/12M-design recipe and bounds/augmented
inside-box solve plus bounded LSMR fallback. Explicit BLAS/OMP thread1 for
these corrected experiments. Preserve the old input/error evidence separately.
Only full actual episodes/cold proof and later fresh qualification can solve12.

## Corrected twelve-link finite-affine execution gate, before execution

Corrected import002 is terminal: foreign physical-tape error0, exact initial
hanging error0, decoded native-direction error1.776e-15; transferred policy
still fails nominal1.14s/Y3.006855m with no hold and exact replay. Repair002
is live and uses feasible augmented corrections (zero bound violations); after
seven updates its measured native defect score is3.703844e-14 and maximum
normalized defect1.978423e-8. Do not equate those scores with actual success.

Once repair002 is authoritatively TERMINAL, run `n12_hybrid_finite_001` from
its FINAL shooting_proposal.npz, regardless of its gap-free baseline's outcome.
Do not select an intermediate proposal using later reserved data. Same recipe
as eleven: full76D native order4/.001 logarithmic-chart models, Q cart4/
relative rotations40/velocities.05 with passive axial.1/.005, R=.001 I2,
full affine QR values/offsets, reference forces rounded through canonical F32
delivery, exact upright continuation from9s proposal through1500 design steps,
terminal Riccati scale0 (final local stage root retained), no guard/search,
finite_qr upright backend, BLAS/OMP1 and four native derivative workers.
Compare zero-offset and affine arms with identical gains/reference within run.
Full30s nominal20261016 and, only if a nominal/replay pass,12 shared noisy
starts20261116..27. Budget<=2.5M inclusive native steps. Keep all native tapes,
source/control arrays/roots/defects and replay. No reserved seeds. If nominal
passes but noise fails, select this first1500-step affine design for the same
18s hanging/per-command axial-reference alignment development gate used at
eleven, with its own pre-execution memo. No reset, physical joint constraint,
extra episode time or accepted-count claim follows from a proposal score.

## Twelve-link conditioned online-axial development gate, before execution

Repair002 is terminal:20 accepted augmented full corrections,11869828 design
native steps, final defect5.337252e-16/max normalized1.496971e-8. Its separate
gap-free baseline fails11.18s with4.24s hold and exact replay. Finite001 is
terminal: zero-offset arm fails10.72s/3.82s hold, full affine passes nominal
30s/23.50s hold, but all12 raw noisy starts fail. Keep these physical failures.

Select this FIRST corrected1500-step affine design for `conditioned_n12_001`.
Freeze finite001/affine_controller.npz,18s hanging preparation, causal static
launch alignment PLUS per-command closest-axial reference alignment, no search,
same canonical mechanics/delivery/full physical state. Use450 swing commands
(9s source) and execute only the first600 of the original1500 design commands
after parking; do not regenerate a shortened value recursion. Nominal upright
onset6.50s after launch predicts5.50s hold in the unchanged30s episode.
No live state change or locked joint; measured passive spin/rates are retained.

Evaluate exact nominal20261016 plus all44 shared noisy development starts
20261116..27 and20261416..47. Explicit BLAS/OMP1. Native parking design<=1000;
45 actual full episodes plus cold proofs<=540000; inclusive budget<=550000.
Only a45/45 complete pass opens a separately frozen NEW32-seed qualification
20262316..47. No reserved seeds used here, no public twelve count until gates.
If any fail, preserve the full tapes and test a declared controller change;
neither nominal success nor a tiny proposal defect substitutes for this gate.

## Twelve-link frozen reserved qualification, before first held-out execution

Conditioned001 is terminal45/45, all full1500 commands/30s,5.50s hold,
no native warning and exact cold replay; same fixed finite001 arrays and18s
online-axial preparation throughout.96 tests pass. Freeze EXACT
conditioned_n12_001/route.npz and current source/driver/runtime as
`qualify_n12_001`; no policy/synthesis/timing change after reserved inspection.

Run NEW untouched32 noisy hanging seeds20262316..20262347, disjoint from
all twelve development/pilot/invalid-input experiments. Canonical twelve-link
plant, full physical XYZ noisy reset and same fixed F32 delivery/80N disk,
all physics substeps inside3m square, every world rod tilt<.15 for>=5s,
complete30s/no warning, exact independent cold replay required for every seed.
18s preparation plus600 commands of the full1500-step finite-affine design;
9s geometric source and final3s upright tracking all fit inside30s. No search,
state overwrite, physical gauge edit or joint lock. Explicit BLAS/OMP1.
Budget<=384000 native steps including all actual episodes and proofs; no
online model design queries. All32 must pass to qualify twelve. Record all
initial-state hashes, longest streaks, maximum cart/force and minimum substep
margin. On success render first held-out seed's FULL1501 frames/50fps and
normal-speed preview, inspect video, verify packaged-source tests and first
tape byte parity, staged/archive hashes, normal fast-forward GitHub publication
with explicit2D tag/video/evidence and public asset/tag verification. Preserve
invalid-input/corrected-failure records and their typed coordinate correction.

## Twelve publication and thirteen-link transfer, before execution

Twelve is public at1ff88386cc1b563bc85b6938712f24894a36361d,
tag2d-cartpole-12-link-v1.3708 staged/archive hashes pass. Packaged source96
tests passes and reproduces first held-out states/actions/tilts byte-identically;
its11 source module hashes equal the qualification source. Public MP4 SHA256
2bc5c79efb073b417d3de42df7a7c9a38b14a680c5faa81c9e956863dd84c57f,
1142592 bytes; ZIP1753666834 bytes SHA256
b0774bac8ff0f3fc93ceb8747c3ee274f9ed4243a950f5edbec9b417afd49848.
Public API/tag/master checks match,1D content preserved. Never rebuild12's
published archive from later source changes. Goal stays active.

Extend native freedom/rotation and online axial-frame tests to13; use the
OPTIONAL finite_qr backend for the13 upright design test, retaining the same
stability/root/nonnegative-cost/world-rotation assertions. Report any failure
before attempting controller acceptance; do not infer a solved count from tests.

Run `planar_lift_n13_001` after those tests with strict explicit-coordinate
import. Donor/full-source read-only:
`D:/ChatGPT/cartpole/runs/swingup13_uniform/thirteen_link_controller.json`;
source config `configs/swingup13_uniform.yaml`, scale provenance
`benchmarks/p1_capture_envelope.yaml`. This file labels itself not_solution/
single-state diagnostic,450 actions/9s, identity28x28 physical-relative map,
old nominal cart max1.993426492m and old nominal hold23.50s. It is only a
geometric proposal. Require finite/invertible matrix, exact hanging initial
error<=1e-6, native world directions<1e-10 and independent stored physical-
tape parity<2e-6 (NO reference-only). Regenerate canonical13-link mechanics,
full82D tangent/two-force feedback with finite_qr upright factor. Same old/new
plant differences remain explicit. BLAS/OMP1, fresh native models. Nominal
20261016 plus conditional12 shared noisy starts only if nominal full pass;
<=550000 inclusive native physics steps, all full tapes/cold proof. No reserved
seeds. Thirteen reserves20262416..47, untouched.

After import is terminal, run `n13_direct_repair_001` on its saved proposal,
20 full82D simultaneous sparse shooting iterations, four workers, order4/.001,
<=14M DESIGN native steps (larger native dimension/horizon accounting), same
fixed initial/terminal nodes, .1 normalized state box/20N force increments,
80N disk, ridge .001 schedule and native backtracking as12. Accurate augmented
ridge only when within every bound; otherwise same bounded LSMR1000/TRF20
fallback. Explicit stage journals/BLAS/OMP1, finite_qr upright factor. Preserve
proposal and separate gap-free nominal/conditional12 noisy full actual/cold
baseline (<=156000 additional steps). Only physical gates qualify a controller;
if needed next test full-affine1500 continuation and18s online-gauge preparation
with separate preregistered outputs/gates. No published13 claim from a gap score.

## Thirteen-link finite-affine execution comparison, before execution

Direct repair001 remains live with feasible accepted augmented updates; do not
restart or use an intermediate proposal as if it were a final controller. After
its authoritative terminal result, test `n13_hybrid_finite_001` from its FINAL
shooting_proposal.npz, irrespective of the separate gap-free baseline outcome.
Same count12 recipe: canonical13-link/full82D tangent/two-force plant, fresh
order4/.001 native models/four workers, local Q cart4/relative rotations40/
rates.05 with passive axial.1/.005, R=.001 I2, exact logarithmic-chart native
defects, full affine QR roots/residuals and additive feed offsets. Quantize
reference forces through canonical F32 delivery before identification. Append
the exact upright continuation to1500 design/execution commands, terminal
Riccati scale0 (final local stage root retained), no guard/search, finite_qr
upright backend and explicit BLAS/OMP1. Same per-design reference/gains for
zero-offset versus affine arms. No live state reset or physical constraints.

Full30s exact nominal20261016 in each arm; a nominal/replay pass opens only
the12 shared noisy development starts20261116..27. Budget<=2.5M inclusive
native physics steps, complete actual/cold tapes. Thirteen reserved20262416..47
remain unused. Prediction: full affine finite continuation may avoid a failed
stationary handoff, but source feasibility/gain norms alone do not admit it.
If nominal passes but reset noise fails, select this FIRST1500-step finite
controller for the18s hanging/online-axial full development gate in a new memo.
If nominal fails, preserve it and diagnose its physical failure before redesign.

## Thirteen-link conditioned online-axial development, before execution

Repair001 is terminal20 updates/12777076 design native steps, defect score
1.308216e-14/max normalized6.583040e-8. Its separate gap-free baseline fails
8.00s/Y3.029320m with0.02s hold, exact replay. Finite001 is terminal: zero-
offset fails6.22s/0.02s hold; affine passes nominal30s/23.50s hold, but0/12
raw noisy starts. The proposal is nearly planar (cartY4.23e-13m, forceY2.04e-10N),
yet the synthesized transverse gain has large cross-axis coefficients. Treat
that as a design diagnostic, not a simulator-axis defect. All native freedoms
and gauge/rotation checks pass (101 tests). Preserve every failed actual tape.

Select this FIRST1500-step finite-affine design for `conditioned_n13_001`:
exact finite001/affine_controller.npz,18s hanging preparation, static launch
alignment and per-command closest-axial reference alignment,450 swing commands
(9s), then the remaining tracking prefix to600 commands/30s total. Preserve
the original full1500-step value design, all measured spin/rates and canonical
mechanics/force delivery; no search, state reset or joint constraint. Explicit
BLAS/OMP1. Nominal20261016 plus all44 shared noisy development seeds
20261116..27 and20261416..47, complete actual+cold tapes. Parking design<=1000
and45 full actual/proof episodes<=540000, inclusive budget<=550000 native steps.
Predicted nominal hold5.50s from6.50s upright onset after launch. Only45/45
opens separately frozen fresh qualification20262416..47; no reserved seeds here.
If it fails, retain results and investigate a declared policy/metric/timing
change using development only. No13 promotion from nominal or a gap score.

## Thirteen near-planar reference roundoff comparison, before execution

Conditioned001 is terminal30/45; fifteen failed development seeds are retained,
and no reserved13 seeds have been consumed. The reference is almost planar:
cartY4.2319e-13m, quaternion X/Z1.02645e-13, forceY2.03908e-10N, yet the
finite design has transverse-force gain from in-plane coordinates up to
51694946.064. At an exactly planar reference, physical Y-reflection symmetry
forbids those linear cross-plane terms. This suggests numerical sensitivity;
it does not prove the cause of physical cohort failures.

Add OPTIONAL --planar-reference cleanup of ONLY proposal arrays: cartY=0,
joint quaternion X/Z=0 with W/Y renormalized, transverse/spin velocity entries
zeroed, reference forceY=0. Require max normalized local-state and force/80N
change<=1e-8, reject substantial spatial references, preserve original input
arrays and live simulator state. This restricts the warm reference, never the
physical joints, real observed state, forceY feedback or noisy initialization.
New independent1/3/13 native reflection tests include genuinely spatial poses,
rates and both cart forces. Cleanup tests preserve live state/inputs, bound
changes and reject real spatial trajectories. Seven new tests pass; full108
suite is required before execution. Earlier controller outputs remain unchanged.

Run `n13_planar_reference_finite_001` from the SAME final direct001 proposal,
same native order4/.001/four-workers/Q/R/full1500-step continuation/zero-terminal
recipe, finite_qr backend, BLAS/OMP1 and --planar-reference. Fresh FULL native
models, no derivative symmetry projection or altered simulator. Keep native
defects, affine values/feed offsets exactly as computed. Save cleanup deltas and
cross-plane gain/feed diagnostics. Compare only the affine arm here against
the retained original affine controller; nominal20261016, conditional12 shared
raw noisy starts on a full pass, <=2.5M inclusive native physics steps. No search,
extra time or reserved seeds. Prediction: if near-planar reference roundoff is
material, forbidden cross-plane gain shrinks and prepared reliability improves.

After terminal design, if nominal passes, run `n13_planar_reference_pilot_001`
with the SAME18s online-gauge wrapper and nominal plus five explicitly reused
failed seeds20261120,20261417,20261418,20261419,20261421. <=73000 inclusive
native steps. Only6/6 opens a new complete44-start development gate. Pilot
overlaps development and is not independent qualification. If it fails,
retain null outcomes and consider a separately declared model/physical-cost
comparison. No derivative averaging or ignored physical-axis error is introduced
silently, and a lower surrogate/gain norm never replaces full native task gates.

## Thirteen physical-observable tracking comparison, before execution

Planar-reference finite001 is terminal: normalized state cleanup1.3644e-12,
force cleanup2.54884e-12, native cross-plane gain drops from5.16949e7 to
1.200565e5; nominal still23.50s hold and0/12 raw noisy passes. Its declared
18s pilot is terminal1/6 (nominal only); all five reused failures remain.
Do NOT open full development on that pilot. Cleanup changes the numerical
design but has no measured reliability benefit on this pilot.

Next isolate the tracking metric, using the ORIGINAL direct001 proposal (no
cleanup/model projection) in `n13_observable_finite_001`. Replace relative-
coordinate state penalties with local factors for each reference's physical
world rod directions and transverse direction rates, plus cart XY/velocity.
Weights remain cart4/directions40/cart velocity.05/rod direction rates.05;
all three vector components get equal weights. Factor L_t=sqrt(W)*J_observables
uses fresh centered kinematic tangent derivatives epsilon1e-5, targets the
reference's OWN observable vector (zero tracking residual), retains full82D
native dynamics and known affine defects, and does not penalize axial gauge
or instantaneous axial spin as if it were rod tilt. Spin remains a measured
state and may affect future native dynamics. No axis is dropped from the plant.
Identical upright centers share one freshly computed factor; every swing/bridge
center gets its own factor, and the final local factor is retained at deadline.

Native order4/.001/four workers/Q R=.001 I2/full1500-step upright continuation/
terminal Riccati scale0/finite_qr backend/BLAS OMP1 stay fixed. No search,
reference cleanup or added physical limits. New native-observable directional
checks at1/3/13, axial gauge-null and real-spin tests pass (5 new tests); full
113-test suite required before execution. Save factor arrays/weights as well
as roots/gains/offsets and complete native tapes. This is a cost intervention,
not an assumption of robustness or of a differentiated searched policy.

Nominal20261016 affine arm only, then conditional12 shared noisy starts on a
full pass; <=2.5M inclusive native physics steps (kinematic probes add no
physics steps). If nominal passes, run `n13_observable_pilot_001`, same18s
online-gauge wrapper, nominal plus reused failed seeds20261120,20261417,
20261418,20261419,20261421, <=73000 native steps. Only6/6 opens a separately
frozen44-start gate. These pilot labels overlap development; no reserved13
seeds are used. Prediction: a physical/gauge-invariant metric may improve
control authority/reliability; retain null or worse outcomes without promoting
a controller from an improved cost, factor rank or nominal-only pass.

## Thirteen online-axial hanging-preparation pilot, before execution

Observable finite001 is terminal: nominal23.50s hold,0/12 raw noisy passes.
Its18s pilot is terminal1/6 (nominal only); all five reused failures remain,
so do not open full development. This physical tracking-cost intervention has
no demonstrated pilot reliability gain. Keep factors/actual tapes/negative result.

Probe the preparation law itself in `n13_online_parking_pilot_001`, using the
ORIGINAL conditioned_n13_001/route.npz (best complete cohort30/45), same
18s duration, original relative tracking metric/gains/offsets/1500-step design,
same launch and per-command axial reference alignment. Add ONLY observed
axial-frame alignment during HANGING feedback, using the existing independently
tested axial_aligned_affine_force on the hanging reference/parking gain.
Retain measured spin/rates and all physical states; references/gain columns
are transformed causally, with no joint actuator, extra time or plant change.
No parking Riccati weight or gain is retuned. Optional online_parking defaults
FALSE, preserving prior frozen execution; save explicit payload/metadata flag.

Pilot nominal20261016 plus the same five reused failed seeds20261120,
20261417,20261418,20261419,20261421, full30s/no warning/cold replay required.
<=73000 inclusive native physics steps, BLAS/OMP1. Verify every source payload
array is unchanged except the added online_parking flag; save parity record.
Only6/6 opens a separately frozen full44-start gate. No reserved13 seeds used.
Prediction: if frame-dependent hanging feedback contributes to launch error,
online preparation may improve contraction/reliability at the same18s clock.
Null or worse results remain failures, not a reason to relax physical gates.

## Thirteen alternate material donor, before generation and repair

Online-parking pilot001 is terminal1/6 (nominal only), with all original
payload arrays verified unchanged except the added online_parking flag. None
of the three pilots admits a full cohort. Best complete13 development remains
30/45 with the ORIGINAL donor/relative metric/18s launch. All failures retained;
reserved20262416..47 remain untouched. Do not claim these probes exhaust13.

Change the reachable reference, rather than continue tuning the same failing
arrival. Generate `n13_material_from14_001` from read-only donor
`D:/ChatGPT/cartpole/runs/swingup14_uniform/fourteen_link_controller.json`,
config `configs/swingup14_uniform.yaml`. This is a marked not_solution/single-
state historical component, not target acceptance. It has450 commands/9s,
old nominal hold24.16s (earlier upright entry than donor13's23.50s), peak
normalized input.8078548312, old cart maximum1.913096751m, actual stored
14 uniform lengths3/14 and masses1/14. Decode its declared30x30 coordinate map
and require stored physical-tape parity<2e-6 before transfer. Interpolate
UNWRAPPED cumulative angles AND cumulative angular rates at normalized material
centers using stored source lengths and canonical target13 lengths, then
reconstruct relative q/v. Preserve cart position/velocity and original action
guesses/cadence. Rebuild canonical13 mechanics; require exact hanging error
<=1e-6 and all native target rod directions vs interpolated field<1e-10.
Freeze both original files/hashes, source/target physical arrays and geometry.
This generation uses zero native integration steps and no actual episode.
Five new tests cover rigid winding/rates/cart, analytic material shape/rates,
and invalid geometry; full118-test suite required first.

After generation is terminal, run `n13_direct_from14_001`: same20 full82D
simultaneous native shooting iterations/order4 .001/four workers/14M design
budget, fixed initial/terminal nodes, existing bounds/augmented inside-box solve
with LSMR fallback/ridge schedule/native backtracking, finite_qr upright factor,
BLAS OMP1. Preserve separate proposal and gap-free baseline/conditional12 raw
noisy actual episodes+cold replay (<=156000 additional native steps). A donor
with earlier entry may give a more usable arrival or later preparation margin;
this is an untested hypothesis, not evidence of success. No reserved data used.

Also run `n13_optional_policy_n12_regression_001` once with unchanged frozen
twelve controller and already-consumed seed20262316 after the optional parking
branch/kinematic helpers. <=12000 actual/proof steps, require byte-identical
states/actions/tilts versus qualify_n12_001. This is an execution regression,
not new qualification or tuning. Published twelve tag/ZIP stay immutable.

## Thirteen alternate-donor finite execution and timing rule, before execution

Material generation001 is terminal with source stored-tape parity0, target
hanging error0, native direction error1.11e-15 and zero physics integration.
Direct_from14_001 is LIVE, accepting bounded/augmented corrections within its
14M design ceiling. After its authoritative terminal result, use its FINAL
shooting_proposal.npz in `n13_from14_finite_001`, same original RELATIVE metric
(best full development family), no planar cleanup or online-parking addition.
Native full82D/order4/.001/four-worker/factored upright design and BLAS OMP1,
R=.001 I2, exact affine defects/QR,1500-step upright continuation/zero terminal
Riccati scale, no search or guard. Affine arm only, nominal20261016 then only
conditional12 shared raw noisy starts on a full nominal/replay pass. Inclusive
native budget<=2.5M, complete actual/cold tapes. No reserved13 seeds.

If nominal passes and raw noise fails, choose preparation using this declared
NOMINAL-only timing rule BEFORE noisy wrapper outcomes:19s if nominal's longest
physical hold>=24.10s (leaves>=5.10s by the episode deadline); otherwise18s if
hold>=23.10s; otherwise redesign timing/reference rather than add episode time.
Then `conditioned_n13_from14_001` uses this first finite design, existing static
launch/per-command axial alignment, original hanging feedback (online_parking
FALSE),450 swing commands, and only the prefix remaining inside1500 total
commands. Full1500-step backward design retained; no gain regeneration at the
shorter executed length. Nominal plus44 shared noisy starts20261116..27 and
20261416..47, <=550000 inclusive native steps. Every complete episode/no warning/
exact replay required; only45/45 opens a separately frozen new32-seed cohort.
This changes donor and potentially preparation duration together; do not claim
an isolated cause if it improves. Thirteen reserved20262416..47 remain untouched.

## Thirteen19.1s preparation pilot, before execution

Direct_from14_001 is terminal20 iterations/12777076 design steps, final defect
6.129852e-16/max normalized1.567505e-8; its gap-free baseline fails7.36s with
.26s hold, exact replay. From14 finite001 passes nominal24.16s hold but0/12
raw noisy starts. The declared timing rule selects19s; complete conditioned
gate is44/45 (all full/cold replay), nominal and successes5.16s hold. Only
seed20261437 fails, with.26s hold. This is improvement, not qualification.

Run `n13_from14_late_pilot_001`, same frozen controller/relative metric/online
launch+tracking gauge/original hanging law, change ONLY preparation from19.0
to19.10s (955 parking commands,545 tracking commands from original1500 design).
Full episode stays1500 commands/30s; nominal timing predicts5.06s hold. Exact
nominal20261016 plus reused failed development seed20261437, <=25000 inclusive
native physics steps. No online-parking/search/coordinate cleanup addition.
If2/2 full physical/no-warning/replay passes, evaluate a new frozen full44-start
gate in `conditioned_n13_from14_002` at the same19.10s before reserved data.
If either fails, retain result and redesign controller/arrival rather than add
unregistered preparation or relax5s. Thirteen reserved20262416..47 untouched.

## Thirteen observed cart-origin alignment pilot, before execution

Late pilot001 is terminal1/2: nominal5.06s hold, seed20261437 still fails.
Do not open a19.1s full cohort. Preserve this null outcome. New pilot
`n13_from14_cart_origin_pilot_001` returns to the stronger-margin19.0s controller
and adds ONLY causal translation of reference cart XY at launch to match the
observed cart origin. No real state or velocity is translated. All force,
world-square and30s limits are evaluated on the actual unchanged world state.
Pole references/gains/offsets are otherwise unchanged; cart translation is a
physical model symmetry, tested independently with full spatial rates at1/3/13.
Payload/live-state preservation is explicitly tested. Default flag FALSE
preserves existing frozen policies. Four new tests plus full122 suite required.

Use original conditioned_n13_from14_001/route.npz,19s preparation, same online
axial tracking/original hanging law and relative metric, nominal20261016 plus
reused failed seed20261437, <=25000 inclusive native steps. This experiment
chooses a reference origin, never recenters the physical boundary or edits the
plant. If2/2, run a separately frozen full44-start gate at19s before any reserved
data. If it fails, preserve it and consider a declared bounded observed-launch
profile forecast or a new reachable arrival. Reserved13 cohort untouched.

## Thirteen bounded observed-launch profile diagnosis, before execution

Cart-origin pilot001 is terminal1/2 (nominal only); seed20261437 still fails.
Keep19s/from14 as best full cohort44/45; no13 qualification opened. Test the
document's bounded native launch-continuation idea as DIAGNOSIS before deploying
any selector. `n13_launch_profiles_diagnostic_001` uses the recorded observed
state at command950 from conditioned_n13_from14_001/seed20261437.npz, with
exact cold-native reproduction of its entire950-command parking prefix.

Freeze route/tape/source and run nine complete remaining-policy forecasts in
independent native environments. Reference XY offsets in metres, fixed order:
(0,0),(+.0001,0),(-.0001,0),(0,+.0001),(0,-.0001),
(+.001,0),(-.001,0),(0,+.001),(0,-.001). No physical coordinate recentering;
real world3m boundaries, full19+11=30s clock, same F32/80N norm/tilt/hold gates.
Only reference positions shift; gains/roots/offsets/parking/online axial law stay
fixed. The zero-profile forecast MUST reproduce the retained actual failed
suffix states/actions byte-for-byte, or stop before interpreting candidates.
Save every candidate suffix, actions/raw forces/tilts/outcome and native count.
Maximum3800 prefix proof+9*2200 forecast native steps<=25000; no extra physics
queries, no live plant or real episode state restored from forecasts. This is
a changed reference family, not evidence that lower value guarantees success.
No new qualification/no seed-specific runtime lookup/no reserved labels. A
passing forecast could motivate a fixed-profile or observed-state selector
pilot, each requiring new full hanging-start development and qualification.

## Thirteen fixed transverse profile pilot, before full execution

Launch diagnostic001 is terminal: zero-profile forecast states/actions exactly
match the retained failure; profiles0..7 fail. Profile8, reference offset
(0,-.001)m, predicts a complete30s continuation with5.16s hold, positive
world-plane margin and no native warning. This is a forecast on a reused
development launch, not a full new hanging-start pass or qualification.

Run `n13_from14_yoffset_pilot_001` from the ORIGINAL19s conditioned_from14
payload, only add fixed cart_reference_offset=(0,-.001)m. No origin centering,
extra parking, online parking or search. Nominal20261016 plus reused failed
seed20261437, full actual30s+cold replay, <=25000 inclusive native steps.
Require the failed seed's actual suffix after command950 to match forecast
profile8 states/actions byte-for-byte; this tests predictor/live policy parity.
If2/2, test this fixed profile on the full44 shared development starts in a
new frozen gate; if fixed profile harms other starts, a separately specified
bounded observed-launch admission policy may be tested. Never select by seed
at runtime. Reserved13 seeds remain untouched; no qualified13 claim yet.

## Thirteen fixed Y-offset complete development gate, before execution

Yoffset pilot001 is terminal2/2 full30s passes/5.16s holds/exact cold replay.
Seed20261437's actual states/actions from command950 match diagnostic profile8
byte-for-byte. Freeze that SAME fixed-profile payload as
`conditioned_n13_yoffset_001`:19s preparation, original relative metric/online
launch+tracking gauge, offset(0,-.001)m only, no origin centering/online parking/
search. Verify payload arrays equal pilot. Nominal20261016 plus all44 shared
noisy development seeds, <=550000 inclusive native physics steps, complete
actual+cold tapes. Only45/45 opens new32 reserved qualification under a new
memo. If other starts fail, retain them and consider observed-state complete
native admission between zero and this fixed offset, with no seed lookup.
Such a selector would be a NEW controller and need its own full gates. No
13 reserved data is consumed here; no success from a favorable forecast alone.

## Thirteen causal two-profile admission, before implementation/execution

Fixed Y-offset full gate is terminal44/45; seed20261444 fails (.26s hold,
plane exit at command1308). The original zero-offset gate passes that start
but fails20261437. Retain both full gates; neither is qualified. Implement a
NEW causal controller with the ORIGINAL19s/from14 payload and fixed ordered
reference offsets [(0,0),(0,-.001)] metres. At command950, use only the current
observed state, global command clock and accumulated task counters/margins.
Forecast BOTH complete remaining550-command policies in separate native
environments using the exact live policy/delivery/substep gates. Select the
first full30s/no-warning/positive-margin/5s-hold forecast; if neither passes,
fall back to profile0 and record admission failure. No seed input/lookup,
future actual state, actual reset, physical translation or changed boundary.
All candidate states/actions/raw actions/tilts/outcomes and query counts saved
in actual episode tapes/metadata; predictors never edit/advance the live env.
Maximum4400 predictor physics steps per episode, separately counted from
actual+cold replay. New meaningful tests require live isolation, rejection,
deterministic order and exact chosen forecast/live suffix parity. Full suite
required before pilot. Existing twelve heldout byte parity must remain exact.

Pilot `n13_launch_admission_pilot_001`: nominal plus reused development
20261437 and20261444, full actual30s/cold replay, <=50000 inclusive native
steps including predictions. If3/3, freeze SAME payload in
`conditioned_n13_admission_001`, nominal plus all44 shared noisy development
starts, <=750000 inclusive native steps (including parking design and all
forecasts). Require45/45 and every selected suffix byte-equal to its saved
forecast before a new memo opens reserved13 seeds20262416..47. No qualification
from the two complementary previous gates alone. Selector is simulated
full-state predictive feedback; no claim of 50Hz hardware execution.

## Thirteen frozen reserved qualification, before first held-out execution

Admission full gate001 is terminal45/45, all full30s/5.16s hold/no warning/
exact cold replay. Chosen forecast states/actions/raw actions/tilts equal actual
suffixes byte-for-byte for every start. Profile0 selected44 times/profile1 once;
196504 predictor and737180 total inclusive native steps, within750000 budget.
Payload35 arrays match pilot.131 tests and twelve heldout byte parity pass.

Freeze EXACT `conditioned_n13_admission_001/route.npz` as `qualify_n13_001`:
19s hanging preparation,550 executed commands of1500-step full affine design,
original relative metric/online axial feedback, ordered reference offsets
[(0,0),(0,-.001)]m. At launch forecast BOTH complete remaining policies in owned
native environments; first full passing candidate wins, zero fallback if none.
No synthesis, policy, arithmetic, timing or profile change after outcomes.

Run NEW untouched32 noisy hanging seeds20262416..20262447, disjoint from every
thirteen development/diagnostic start. Canonical spatial13/fullXYZ noise,
F32 force norm<=80N, all physics substeps within actual world3m square,
every world rod tilt<.15rad for>=5 consecutive seconds, full30s/no warning/
independent exact cold replay required. Both full candidate forecasts and all
query counters retained in each actual tape; chosen forecast must match actual
suffix byte-for-byte. Inclusive budget<=524800 physics steps:384000 actual+
cold proofs and<=140800 predictor. Explicit BLAS/OMP1. No hardware time claim.
All32 must pass to qualify13; failures are preserved and require a NEW untouched
qualification cohort after any redesign. If all pass, render first heldout full
video, verify packaged source/tests/tape parity, full staged/archive hashes and
publish explicit2D13 source/video/evidence. Historical published ZIPs unchanged.

## Thirteen qualification outcome and publication checks

Frozen qualification001 is terminal32/32, all30s/5.16s hold/no warning/exact
cold replay. Independently rescored32 distinct initial states, command-state
cart max1.883801165m, minimum substep margin1.115844461m, delivered F32 force
norm max65.984695451N. All32 select profile0, both forecasts always evaluated;
chosen forecast states/actions/raw forces/tilts byte-equal actual suffixes.
524800 inclusive native steps =384000 actual+proof+140800 predictors, exactcap.
No controller change. Full first-heldout video/GIF/contact sheets inspected:
1501frames50fps/30.02encoded, GIF300frames30s normal speed. MP4 SHA256
c87797b5a375c9bac212287e5e2817a5a7a4c727f6088cad2e0521539199976a,
1144425 bytes. Package complete current source and all new/changed evidence;
bind prior immutable12 bundle for unchanged history. All full inventory blobs,
recursive ZIP coverage, published-source tests and first heldout byte parity
required before push/release. Never overwrite earlier archives/tags/assets.

## Fourteen native preflight and repair, before execution

Thirteen qualification is terminal32/32; full public source/archive gates pass:
4589 staged hashes,1102 direct+3487 inherited evidence files, prior12 ZIP exact
hash/manifest coverage,131 packaged tests, all12 source hashes/controller match
qualification, first heldout states/actions/raw actions/tilts/forecast tapes
byte-identical. Source commit79861ded77cd7c0ca8203bab6621398430c0b2de is pushing;
do not claim public release until upload/public digest/tag checks finish.

After public13 verification, advance to14. Extend native persistent full-spatial
motion, unrestricted ball/twist, F32 disk, upright rotation/factored design and
online axial force checks to14;136 total tests required. Thirteen frozen public
source retains131-test scope and its immutable archive must not be rebuilt.

`planar_lift_n14_001` uses read-only historical
D:/ChatGPT/cartpole/runs/swingup14_uniform/fourteen_link_controller.json as BOTH
route/full source, configs/swingup14_uniform.yaml, scale provenance
benchmarks/p1_capture_envelope.yaml. This marked not_solution/single-state9s
component is ONLY a proposal. Invert its explicit coordinate matrix, require
stored physical-tape parity<=2e-6, exact hanging tangent error<=1e-6, native
world directions<=1e-10, unchanged canonical target14 and450 force guesses.
No reference-only bypass. Finite_qr axis-separated upright value backend,
original donor X gains plus fresh native spatial Y gains for this baseline.
Full nominal30s/cold replay; only if it passes open12 shared noisy development
starts20261116..27. Inclusive native budget<=750000, BLAS/OMP1. No qualification.

If strict import succeeds, after its authoritative terminal result run
`n14_direct_repair_001` from its route, regardless of baseline failure. Same
full88D simultaneous native shooting,20 updates, design ceiling15000000 native
steps, order4/eps.001/four workers, regularization.001, augmented ridge plus
bounded LSMR fallback(max1000inner/20outer), same physical residual scaling/
force disk/worldbox/rotation bounds and line search. Preserve all iterates,
solver counters and failed actual rebuild. Final native factored design and
gap-free baseline actual+cold proof separate from planning; no intermediate
node reset. A subsequent finite-affine controller requires its own memo after
repair terminal. Fourteen reserved20262516..47 remains untouched.

## Fourteen upright preflight failure and bounded diagnosis, before execution

New136-test preflight is135 pass/1 fail: physical freedoms/native parity and
online gauge checks pass at14, but the1500-step finite_qr stationary upright
gain reports closed-loop radius1.036626804, failing the existing strict<1
assertion. Do not weaken it or claim a stable fourteen balance design. No14
import/repair/qualification has run. Thirteen's frozen131-test source passes
and public archive is unchanged. Its upload is still running.

Run `n14_upright_value_diagnostic_001`, same native upright model/Q/R as
balance_design, compare DARE (retain failure if exception) and finite residual
QR horizons500/1500/3000, cost factorsCholesky(Q).T, terminalzero. Save model,
gains/factors/spectra/query counts. <=2000 native steps, no real episodes or
qualification. A different stable horizon, conditioning or precision approach
must be frozen explicitly before replacing the failed preflight design. No
physical damping, armature, force or dimension changes are authorized here.

## Fourteen same-model high-precision spectrum oracle, before execution

Diagnostic001 completes with one native identification (<2000 steps); DARE
raises LinAlgError, finite QR SISO horizons500/1500/3000 report float64 radii
1.02638/1.02419/1.01832 despite positive factor singular minima~.53899 and
gain magnitudes~2.4035e10. These spectra differ from full-spatial radius1.03663;
high sensitivity makes ordinary eigensolver error a question, not an established
cause. Do not claim more horizon fixes stability or that fourteen is solved.

Install/pin mpmath1.3.0 for a separate diagnostic (official versioned PyPI
metadata checked); published13 runtime/lock/ZIP remains immutable. Freeze
`n14_upright_mp_spectrum_001` with diagnostic001 exact native A/B and stored
binary64 K. Convert the EXACT stored float values to MP; form A-BK in MP rather
than promote an already rounded product. Compare SISO500/1500/3000 at80 decimal
digits and each actual native spatial XY block with1500 K at80 digits. Use
separate blocks ONLY if all native cross-axis A/B entries are exactlyzero;
otherwise use the full observable matrix. Repeat1500 SISO/native at120 digits
and require radius agreement<=1e-30. Keep eigenvalues and all inputs/hashes.
No new model queries (zero native steps), no gain synthesis/replacement or
actual physical episode, wall budget<=900s. Stability of a rounded linear model
does not establish delivered-F32 native nonlinear capture or qualification.

## Fourteen spectrum-oracle result and explicit diagnostic backend

MP oracle001 reports stable SAME rounded gains/models: averaged1500 radius
.9885085409791293, actual nativeX .9885011835562239/Y .9885158929193543;
native cross-axis entries are exactlyzero.80/120-digit agreement passes1e-30.
Thus the failed float64<1 diagnostic does not establish linear instability.
No controller or native model was changed by this oracle; no physical14 claim.

Add optional eigenvalue_digits=80 to balance_design, using exact input-float
conversion BEFORE A-BK products and MP spectra. Default0 preserves all prior
controller generation/metadata. Retain ordinary float64 radius alongside MP
result; never overwrite its discrepancy. Split independent blocks only from
exact structural zeros in A and B/K factors. Native14 preflight explicitly
requests80 digits and requires<1, positive QR factor, world rotation, exact
gain/Q/R/P parity with default calculation, plus a rational-input product test
that fails if an already-rounded F64 A-BK product is promoted. Pin MP runtime.
Importer/direct repair expose --eigenvalue-digits default0;14 registered runs
use80. This changes a diagnostic, not gains/physics/episode acceptance. After
the corrected preflight passes, run the already-declared strict import/repair.

## Fourteen corrected preflight outcome and frozen13 regression

139 tests pass. Terminal strict14 import has stored physical-tape parity0,
hanging0, native directions2.0331e-15,326524 design steps; baseline fails3.66s,
Y3.066347m/no hold, exact cold replay/no warning. Total327988 inclusive native
steps within750000 cap. Registered n14_direct_repair_001 is now running with
explicit --eigenvalue-digits80; do not restart. Thirteen is public and its
assets/tag independently verify at79861ded77cd7c0ca8203bab6621398430c0b2de.

Before relying on current optional spectrum source for future controllers,
run `n14_optional_spectrum_n13_regression_001`: original frozen13 controller,
first heldout seed20262416 only, full30s+cold replay+both causal forecasts,
<=16400 inclusive native steps. Require states/actions/raw actions/tilts and
both forecast tapes byte-identical to the original qualification. This is a
regression on reused data, not new qualification. Published13 remains frozen.

## Fourteen finite-affine comparison, before final proposal execution

Previous goal turn made progress: thirteen was independently verified public,
fourteen physical preflight/import completed, and native repair started. Current
repair001 remains live with accepted full-state updates; do not restart or use
intermediate nodes as physical resets. Optional-spectrum thirteen regression
is terminal full30s/5.16s hold, states/actions/raw/tilts and BOTH forecast tapes
byte-identical,16400 inclusive native steps. Source13 ZIP remains immutable.

After repair001's authoritative terminal result, use its FINAL
shooting_proposal.npz for `n14_hybrid_finite_001`, regardless of the separately
recorded gap-free baseline outcome. Canonical14/full88D, fresh native
order4/eps.001/four workers/BLAS OMP1, F32-quantized reference forces, true native
defects with logarithmic-chart derivative transport. Original relative tracking
cost: cart4, relative rotations40 with passive axial.1, rates.05 with axial.005,
R=.001I2. Append exact upright continuation to1500 design commands; terminal
Riccati scale0 (last local stage factor retained), no guard/search/cleanup.
Factored finite_qr upright backend with explicit80-digit spectrum diagnostic;
default gain/value arithmetic unchanged. Freeze full affine QR roots/residuals/
offsets and compare matched zero-offset versus affine arms on SAME reference,
gains, Q/R, reset and arithmetic. No preparation/gauge wrapper in this comparison.

Each arm full30s exact nominal20261016+cold replay; only if it passes open12
shared noisy development starts20261116..27. Inclusive native budget<=2500000,
all source/runtime/config/driver/document identities, tapes and failures retained.
Only actual unchanged native physical gate/replay counts as success. No reserved
fourteen data. After results, separately register timing/launch/continuation
changes if needed; no inference of noisy capture from tiny proposal defects or
MP spectra. Fourteen qualification requires a separately frozen new cohort.

## Fourteen two bounded nominal corrections, before execution

Direct repair001 is terminal20 accepted updates/13684324 design steps, final
defect1.194934e-16/max5.256265e-9; gap-free baseline fails6.58s/Y3.008552m,
.26s hold, exact cold replay/no warning. Finite001 is terminal: zero-offset
fails6.74s/.26hold, full affine improves to1.84s hold but fails8.84s/Y3.119497m;
all exact cold replay.653412 design/659644 inclusive native steps. No noisy or
reserved14 data opened. Preserve finite001 incumbent and all failed tapes.

Recorded affine tape has Y~4.5e-7m at6s, -1.6e-6m at7s, then1.192m at8s.
Reference Ymax3.8503e-13m/forceYmax1.9524e-9N, but transverse feedmax.00010161N
and transverse gain from in-plane coordinates1.363207e8. This motivates a
reference-only roundoff probe, not a claim that cleanup is the cause or the
plant is planar. No live state/force projection, no joint lock/symmetry projection.

Two independent bounded probes from SAME final shooting_proposal and finite001
design recipe (1500 continuation, terminal0, relative metric, full88D native
order4/.001/four workers/finite_qr/eigenvalue_digits80/BLAS OMP1):

- `n14_planar_reference_finite_001`: change ONLY existing tolerance-guarded
  --planar-reference flag; keep full affine feedback/no native search. Require
  normalized reference state/force change<=1e-8 and save reflection gain/feed
  diagnostics. This changes only the almost-planar proposal, never actual physics.
- `n14_native_search_finite_001`: ORIGINAL uncleaned reference and same gains/
  roots/residuals/offsets as finite001; use existing<=16-candidate delivered-F32
  native one-step value search. Explicitly verify frozen controller array parity
  with finite001. Full affine residual retained, owned native predictor, all
  candidates/selected actions/query counts saved; no live reset/seed input.

Both full exact nominal20261016/30s+cold replay, and only a full nominal pass
opens12 shared noisy starts20261116..27. Each inclusive budget<=2500000 native
steps, including synthesis, every forecast and cold proofs. No dynamic profile
selection/parking/gauge addition. These may run concurrently in independent
environments. If both nominal pass, prefer cleanup for next preparation pilot
because it uses no online native action queries, BEFORE inspecting noisy cohort
outcomes; otherwise select the sole nominal passer. No passing incumbent is
replaced by lower planning merit. After terminal outcomes register any new
preparation/launch strategy. Reserved14 seeds20262516..47 remain untouched.

## Fourteen reflection-averaged feedback, before implementation/execution

Both probes are terminal failures: cleanup2.00s hold/9.82s exit (657340 total
native steps), search1.96s/9.70s exit (688332 total incl31040 predictors), exact
cold replay/no warnings. Search's26 stored controller arrays equal finite001
byte-for-byte. No noisy/reserved data opened. Cleanup changes normalized state
5.7933e-12/force2.4405e-11 and reduces transverse-from-in-plane gain1.3632e8
to2.8560e6, but its transverse feed remains1.4528e-4N. Preserve all outcomes.

Test a NEW feedback family on the cleanup controller: reflection-average the
fitted affine FORCE law. For tangent reflection T and actuator J=diag(1,-1),
K_new=(K+J K T)/2, feed_new=(feed+J feed)/2; same planar reference/actions.
This is the exact average of u(x) and J u(reflect(x)) around a reflection-fixed
reference, not a modification of native dynamics. All full tangent coordinates
remain; genuine odd spatial errors still command Y. This removes fitted
symmetry-inconsistent linear cross-terms and constant Y feed. It does NOT prove
those terms caused failure or that the averaged law is optimal/robust.

No value search in this family: original QR roots/residuals retained only as
design provenance, not a value for the changed feedback. Factory must reject
native_search=True on an averaged payload. No live state projection, boundary
translation, twist reset, gain-schedule tuning or new force/hold/time limit.
Tests at1/3/14 must compare actual full quaternion force calculations with the
two-branch group average, test reflected-force equivariance, preserved inputs/
live state and nonzero transverse response. Native14 reflection test added.
All tests required before actual nominal execution; old13 heldout byte parity
remains a required regression after any core policy guard change.

`n14_reflection_feedback_001` freezes cleaned finite001 controller and new
driver/source, projects only gains/feed, saves all changes, full exact nominal
20261016/30s+cold replay and only on success12 shared noisy starts20261116..27.
<=160000 inclusive native steps, zero online predictor queries, no new native
model synthesis. Keep original reference exactly; no search or preparation.
If nominal still fails, retain the result; physical-observable metric or another
material arrival may be tested only under a new memo. No14 reserved seeds.

Reflection feedback tests complete145/145. The force law matches the independent
full-quaternion two-branch average at1/3/14, reflected actions transform properly,
actual transverse response remains nonzero, all input/live state preserved and
native14 reflection dynamics pass. Native search is rejected for this payload.
Run `n14_reflection_n13_regression_001` with the original frozen13 controller,
first recorded heldout20262416, full30s/cold proof/both forecasts, <=16400 total
native steps. Require all original state/action/raw/tilt and both forecast tape
arrays byte-identical. This is old-data regression only;13 public stays frozen.

## Fourteen prepared reflection-feedback gate, before execution

Reflection feedback001 is terminal: unchanged canonical spatial plant/full
nominal30s passes24.16s hold/exact cold replay; all12 raw noisy development
starts fail. The averaged law is a new tested controller, not an optimum or
physics restriction. Gains/feed change together under exact group averaging;
do not attribute success to only one removed coefficient. No reserved14 data.

Use the same nominal-only timing rule already measured for this9s proposal:
19s preparation if nominal hold>=24.10s (otherwise18s if>=23.10s; otherwise
redesign arrival). Measured24.16 selects19s, before prepared noisy outcomes.
Freeze `conditioned_n14_001` from reflection_feedback001/route.npz:950 hanging
feedback commands, observed launch alignment plus per-command axial reference
alignment,550 executed commands from the ORIGINAL1500 finite design. Nine-second
swing plus2s upright tracking all fit inside the SAME30s clock; no regenerated
shortened value design. Original relative metric, reflection-averaged gains/feed
and reference kept exactly. No online parking/origin translation/reference
offset/search/admission selector. Spin and actual3D joint states remain physical.

Full exact nominal20261016 plus44 shared noisy hanging starts20261116..27 and
20261416..47, <=550000 inclusive native steps including parking design and all
actual/cold proofs, BLAS OMP1. Save full source/driver/runtime/payload/tapes.
Only45/45 full physical/no-warning/exact replay passes opens a NEW separately
frozen32-start reserved qualification20262516..47. If any fail, preserve all
and register next bounded controller change; no qualified14 claim from nominal.

## Fourteen physical observable cost comparison, before execution

Prepared reflection gate001 remains live; do not restart or cancel. Its first
16 completed episodes pass only3 (including nominal), so it cannot qualify14;
still retain the entire declared45-start gate. No reserved data used. New145
tests and original13 heldout/forecast byte parity pass. Registration here is
independent bounded work while that complete gate finishes.

Probe gauge-invariant physical tracking cost, whose factor routine already has
independent native/gauge-null/spin tests. `n14_observable_finite_001` uses SAME
final direct_repair001 proposal, tolerance-cleaned planar reference and1500
continuation/terminal0/native88D/order4/.001/four workers/finite_qr/spectrum80/
BLAS OMP1 as cleanup finite001. Change ONLY tracking metric to `directions`:
physical WORLD rod directions/transverse rates with existing weights, local
Jacobian eps1e-5, full state/spin/native dynamics retained.452 distinct kinematic
centers (swing+bridge+upright),79552 centered observable queries, no extra
physics integration for cost factors. Save all factors and original Q weights.

Full exact nominal20261016+cold replay, only on pass12 shared raw noisy starts,
<=2500000 inclusive native integration steps (no online predictor), no search/
preparation/gauge. After authoritative terminal result, REGARDLESS of its
nominal outcome, `n14_observable_reflection_001` applies the SAME registered
reflection-average to its frozen gains/feed and leaves its reference/cost/root
arrays intact. Original roots/residuals remain unused design provenance.
Full nominal and conditional12 shared starts, <=160000 actual+cold native
steps. Comparing averaged relative versus averaged physical controllers changes
the tracking metric while using the same averaging rule; it is a new law and
needs its own actual tests. No physics/force/noise/time/hold changes. Only after
terminal results register any new preparation gate. Reserved14 cohort untouched.

## Fourteen observable prepared pilot and relative launch diagnosis

Relative prepared gate001 is terminal4/45 (nominal and noisy20261119/20261124/
20261435),464932 inclusive native steps; all success holds5.16s/full30s/replay.
Retain every failure, no qualification. Physical metric plain finite001 fails
nominal8.12s/1.80s hold; its reflection-averaged law passes nominalfull30s/
24.16s hold but0/12 raw noisy starts. It remains unqualified.

Before committing another complete cohort, `n14_observable_prepared_pilot_001`
uses the SAME physical-metric averaged payload,19s preparation selected by the
same nominal-only timing rule, observed launch+online axial reference alignment,
no search/offset/origin centering/online parking. Nominal20261016 plus five
REUSED relative-gate failures20261116/17/18/20/21, full30s/cold replay/no warning,
<=75000 inclusive native steps including hanging design. If6/6 then freeze
SAME pilot payload in a new full45-start gate before reserved14 data. Otherwise
preserve the failed pilot and register a different arrival/feedback strategy.

Independent `n14_launch_profiles_diagnostic_001` tests the previously registered
nine reference-offset family on relative-gate failed20261116's OBSERVED launch
at command950. Original conditioned_n14_001 payload, exact cold-native parking
prefix proof, independent native full suffixes from that observed state only.
Offsets [(0,0),(+.0001,0),(-.0001,0),(0,+.0001),(0,-.0001),(+.001,0),
(-.001,0),(0,+.001),(0,-.001)]m. Actualworld3m/F32disk80N/30s/5s tilt gate,
no physical translation or state edit. Baseline forecast must reproduce retained
failed actual suffix states/actions byte-for-byte before interpreting candidates.
Save all forecast tapes/raw forces/tilts/outcomes/counters, <=25000 native steps
including prefix proof. Forecast success is not a new hanging-start pass. Any
deployed selector/fixed profile needs separately frozen actual/cohort gates.
No reserved14 data or seed-specific runtime action rule.

## Fourteen observed-state launch-time diagnosis, before execution

Physical-metric prepared pilot001 is terminal1/6 (nominal only); do not open
its full cohort. Relative launch-offset diagnostic001 is terminal0/9 forecasts,
12900 native steps with exact baseline suffix parity. Neither smaller physical
cost nor the previous nine cart offsets resolves the first failed relative
launch. Preserve all null results and unchanged14 task limits.

Test the document's launch-window idea as DIAGNOSIS: from relative-gate failed
seed20261116's actually observed state at command900 (18s), prove its entire
parking prefix bitwise with independent native integration. Freeze
`n14_launch_timing_diagnostic_001`, ORIGINAL conditioned_n14_001 reference/
averaged gains/feed/original hanging law/online axial alignment, no offsets,
no physical recentering. Candidate preparation endpoints in fixed order
900,905,910,915,920,925,930,935,940,945,950,955 commands (18.0..19.1s),
followed by the corresponding prefix of SAME1500-command finite design through
globalcommand1500/30s. Each owned predictor starts at the same observed18s
state and executes any extra hanging commands before its proposed launch.
Native world bounds/F32force disk/5s hold/no warning unchanged. Candidate950
MUST reproduce retained actual failed suffix states/actions byte-for-byte.

Save every candidate complete suffix/actions/raw/tilts/outcome/query count;
zero new actual episodes and no seed-specific runtime rule. Maximum3600 prefix
proof+12*2400 forecast native steps=32400, cap33000, BLAS/OMP1. Full continuation
success is conditional model evidence only. If none pass, retain the result and
consider hanging-gain or arrival redesign; if any pass, a separately registered
causal timing selector/pilot/full gate/fresh qualification remains necessary.
Do not add physical episode time; latest19.1s leaves5.06s nominal hold.

## Fourteen hanging-feedback R ladder, before implementation/execution

Timing diagnostic001 is terminal0/12, original950 candidate suffix byte-exact;
neither timing nor nine cart offsets gives a complete conditional forecast for
reused20261116. Launch actual tilt~9.85e-7rad and cart~1e-5m are small, but
its first2s raw Y demandmax.0572N versus.0036-.0051N in three passing starts.
This is evidence about feedback demand, not proof that scalar launch tilt ranks
capture or that a particular mode causes failure. All earlier failures retained.

Change ONLY the hanging LQR action cost R (existing balance_design r parameter),
keeping Q/cart/angle/rate weights,19s preparation, full relative-metric reflection-
averaged swing law, online axial reference alignment and all physical limits.
Add --parking-r default.01 to condition_route; defaults preserve old payload
coefficients. Candidate values in fixed descending order .003,.001,.0003,.0001.
Each independently freezes source/driver/runtime/route and native hanging design.

`n14_parking_r003_pilot_001`, `n14_parking_r001_pilot_001`,
`n14_parking_r0003_pilot_001`, `n14_parking_r0001_pilot_001`: nominal20261016
plus five REUSED relative-gate failures20261116/17/18/20/21. Full actual30s/
cold replay/no warning, unchanged F32 force disk80N/world3m/5s hold. No offsets,
origin shift/online parking/native search/admission or extra physical time.
Each <=75000 inclusive native steps incl parking design; all four <=300000.
May run independent pilots concurrently. Prefer FIRST (largest R)6/6 full
passer for a separately frozen45-start gate; no qualification from pilot.
If none pass, preserve all and redesign hanging response or arrival. No14
reserved data. Driver-only option wiring uses the already-tested native LQR;
actual full-policy pilots provide its required behavioral check.

## Fourteen reflection-consistent factored synthesis, before implementation

All four hanging-R pilots are terminal1/6 (nominal only); none opens a full
cohort. Retain all coefficients/tapes. Best complete14 cohort remains4/45.
Timing and offset native diagnoses also fail, so do not deploy a selector from
those banks. No14 qualification data consumed;13 remains public/frozen.

Test reflection symmetry DURING planning, rather than only projecting the final
fitted force law. On the tolerance-cleaned reflection-fixed reference, native
analytic tangent parity is T, actuator J=diag(1,-1). Project freshly identified
chart-transported models A'=(A+TAT)/2, B'=(B+TBJ)/2 and d'=(d+Td)/2. This is
explicit numerical denoising of a local planning approximation, NOT a projection
of the actual native flow/state/force or a physical joint lock. Save ORIGINAL
native A/B/d and all projection deltas. Guard max removed normalized entries
<=1e-8 using existing shooting scales (cart3/angle.15/cartvel4/rates15 and80N
input scale); reject larger corrections before interpreting a controller.

With original relative diagonal Q/R and terminal0+local stage factor, the
projected LQ problem separates EXACTLY into even30D/X and odd58D/Y branches.
Compute each full affine residual-QR recursion independently in F64, retaining
all88 tangent coordinates, including odd physical tilt/spin modes. Re-embed
both gains/feeds/value roots/residuals; full roots must represent their summed
quadratic value. No mode deletion, mechanical change or reduced physical plant.
This changes feedback design, so no optimality claim for the unprojected native
approximation. Check guards, independent small dense finite-horizon optimum,
full-vs-block projected QR agreement, value differences, zero forbidden gains/
odd feed, input preservation and rejection of substantial model changes.
Existing native reflection tests cover1/3/13/14 with unconstrained spatial flow.

After tests pass, `n14_reflection_model_finite_001` uses same FINAL direct repair
proposal/clean reference/relative Q/R/1500 continuation/terminal0/native order4/
.001/four workers/finite_qr/spectrum80/BLAS OMP1. New --reflection-model flag
requires --planar-reference and relative metric; false default leaves all old
designs unchanged. Affine-only full exact nominal20261016/30s+cold proof, only
on full pass12 shared raw noisy starts20261116..27. <=2500000 inclusive native
steps, no native search/parking/gauge addition. Save raw native models and
guarded projected model provenance. After terminal, separately register any
19s prepared pilot/full gate. No14 reserved seeds or new solved-count claim.

## Fourteen reflection-model prepared pilot, before execution

148 tests pass, including independent condensed optimum, full/block QR value
and gain agreement, true Y control response, guarded projection and unchanged
inputs. Reflection-model finite001 is terminal nominalfull30s/24.16s hold/
exact cold replay but0/12 raw noisy starts. Fresh native projection normalized
maxA5.20695e-11/B2.93631e-12/d1.33303e-14, all below1e-8; all88 state coordinates
retained and raw native models saved. No physical state/dynamics projection.

`n14_reflection_model_prepared_pilot_001`: SAME new frozen controller,19s
preparation by the same nominal-only rule, observed launch+online axial reference
alignment, no cost/search/offset/origin/online-parking change. Original hanging
R=.01; nominal20261016 plus five REUSED relative-gate failures20261116/17/18/
20/21. Full30s/no warning/cold proof, <=75000 inclusive native steps. Only6/6
opens a separately frozen full45-start gate before any reserved14 data. If
failed, preserve the result and change the arrival or feedback family rather
than infer that smaller planning residual/consistent QR guarantees capture.

## Fourteen alternative moving-target material proposal, before generation

Reflection-model prepared pilot is terminal1/6 (nominal only). Full88D
reflection-consistent QR changes allowed gains bymax26205.7 and feed3.95e-6
relative to postfit averaging, but does not pass the five reused difficult
starts. Preserve all earlier results; best completed14 prepared gate4/45.
Do not continue ranking this arrival by tiny defects or gain norms alone.

Read-only inspection found the hybrid document's pinned Marc17 candidate02
public source locally at D:/ChatGPT/cartpole/research/external/neuralyogi_20261016/
arena_update_45326b0/benchmark/canonical17/candidate02. Its701x36 physical
q18/v18 reference and700 normalized F32 actions exactly equal the full value
arrays in its combined payload (all5 common fields). This is a14s design toward
a6.25Hz bent target, with450 bridge commands and250 additional moving-target
commands. It is foreign 1D geometric evidence, not a spatial controller or our
independent17-link reproduction. No foreign policy module will be executed.

Generate `n14_material_from_marc17_001` with a new explicit adapter. Freeze the
MIT license/credit, candidate README, frozen-manifest, original identity, source
model.xml/configuration/controller NPZ, reference NPZ and design metadata.
Verify selected original hashes/sizes against its manifest, payload SHA against
configuration/identity and XML hash against identity. Decode states ONLY as
physical relative hinge q/v, validate source nq/nv18 and17 Y hinges plus X slide
in native XML. Original source uses9-digit lengths.176470588 and masses
.0588235294; retain these actual material centers (source sums differ by~4e-9/
2e-10), do not pretend source/target mechanics match.

Independently compare original source native body directions with unwrapped
cumulative angles (<=1e-10) and its700 saved F_native-ref-next defects with
fresh four-substep native maps (<=2e-6). These are reference-coordinate/one-step
checks only, not actual source-policy/task qualification. <=4000 SOURCE native
steps, no target integration. Use normalized source controls*80N for this check;
target guesses use canonical physical-F32 delivery, with quantization change
recorded. Interpolate unwrapped cumulative angle AND rate fields at material
centers into canonical14. Source initial cart is-.050000001924m: translate the
ENTIRE PROPOSAL cart path by exactly+ .050000001924m to begin at0, preserving
velocities. This is a frozen reference translation, never a live state/boundary
shift. Require target initial hanging error<=1e-6 and native world direction
error<=1e-10. Keep full700 controls/701 poses and450 bridge-step provenance;
no source gains, cost, search, locks, rail/force or runtime are inherited.

After valid terminal generation, `n14_direct_from_marc17_001` repairs that FINAL
700-command proposal on full88D canonical native spatial dynamics:20 accepted
updates maximum,24000000 design native-step ceiling, order4/eps.001/four
workers/BLAS OMP1, augmented ridge with bounded fallback1000inner/20outer,
regularization.001, same physical residual scales/control disk/worldbox/rotation
bounds/line search, finite_qr/spectrum80. Separate actual gap-free baseline+
cold proof <=160000 additional native steps, no intermediate state reset.
After terminal, any finite moving-target execution needs a new memo preserving
design/execution clocks; do not append a stationary target by assumption.
Fourteen reserved20262516..47 remains untouched; original D checkout read-only.

Material external17 generation001 is terminal: original full/payload5 arrays
equal, all selected pins match, source native direction2.165e-15 and700
one-step defects agree1.26843e-12 (2800 source native steps, zero target steps).
Target hanging4.04810e-9/directions1.33227e-15, translated reference maxcart
1.890318897m, target F32 action quantization change3.81470e-6N. This validates
the proposal adapter only, not a source-policy or target task pass.
Registered direct_from_marc17_001 is live, session93100; first model1010800
native steps/.03922754 defect/.01247891 normalized max. Augmented update has
69 bound violations and uses declared bounded LSMR fallback. Do not restart
or relax limits. No controller/qualification/public14 artifact exists yet.

## Fourteen alternate moving-target finite execution, before repair outcomes

Previous goal turn made progress: new148-test controller/model symmetry checks,
full actual14 nominal passes, retained noisy/pilot failures, a hash-bound external
material adapter and the live alternate native repair. Current session93100 is
confirmed live in its bounded solver; do not restart or relax the69 violated
limits. No14 reserved data consumed. This registration governs ONLY work after
its authoritative terminal result and FINAL shooting_proposal.npz exists.

Preserve the moving destination and the separate design/execution clocks:
700-command14s repaired reference/value design;17s hanging preparation plus
650-command13s literal prefix inside1500commands/30s.450 commands identify the
bridge, followed by200 executed commands of moving-target tracking. Remaining
50 design commands influence feedback but are never added to episode time.
No shortened650-step recursion or assumed stationary appendage.

After final repair, measure reflection cleanup using existing guard<=1e-8 and
save its diagnostics (no physics integration). If it passes, use cleaned reference
and guarded reflection-model30D/58D full-state QR; if it fails, retain the failed
guard and use full unprojected88D QR/reference. Selection uses input symmetry,
not actual episode outcomes. `n14_from_marc17_finite_001` computes fresh native
order4/eps.001/four-worker/BLAS OMP1 models on the actual700-step proposal,
true affine defects/chart transport, original relative Q/R, terminal Riccati
scale0+last local stage factor, finite_qr/spectrum80. NO --extend-to-episode,
no guard tail/search/new destination. Store native/projection provenance when
used. Full exact nominal20261016/30s+cold replay under this raw law, only on
full pass12 shared raw noisy starts. <=2500000 inclusive native steps. Its
post-design stationary handoff is an explicitly separate diagnostic; that
branch will NOT execute in the proposed17s/650-prefix controller.

After authoritative finite-design terminal result, regardless of its raw30s
handoff outcome, `n14_from_marc17_prepared_pilot_001` freezes EXACT700-step
affine payload,17s hanging R=.01/original Q, launch+per-command axial reference
alignment, --swing-steps450 to preserve physical bridge/target split. Same
actual target plant/F32 force disk80N/world3m/tilt<.15 for>=5s/full30s/no warning/
cold proof. No native search/profile/offset/origin shift/online parking.
Nominal20261016 plus five reused14 failures20261116/17/18/20/21, <=75000
inclusive native steps incl hanging design. Only6/6 opens a separately frozen
full45-start gate before fresh reserved qualification. A failed pilot remains
failed even if briefly upright; no spatial17 claim or new qualified14 record.

## Fourteen raw moving-reference incumbent, before independent execution

Alternate repair remains confirmed live in its first bounded LSMR solve (no
accepted update yet). Preserve the RAW geometric proposal as a separate candidate
instead of requiring sparse feasibility repair to win first. Full affine native
defects can compensate for a nonzero local reference gap; only actual episodes
can establish whether that is sufficient. Do not relabel raw as repaired.

Independent `n14_raw_marc17_finite_001` uses FINAL generation001/route.npz,
not an intermediate repair checkpoint. Full700-step moving reference, original
relative Q/R, terminal Riccati scale0+local stage factor, fresh full88D native
order4/eps.001/four workers/BLAS OMP1, finite_qr/spectrum80, guarded planar
cleanup/reflection-model denoising/separate30D+58D QR as registered. No stationary
tail appended; raw full30s nominal+cold proof, conditional12 shared raw noisy
starts, <=2500000 inclusive native steps. Preserve all reference defects and
actual failures. This is another reference/controller family, not a source
reproduction or result inferred from the ongoing solve.

After its authoritative terminal result, regardless of raw post14s stationary
handoff outcome, `n14_raw_marc17_prepared_pilot_001` freezes EXACT700-step payload,
17s hanging R=.01, --swing-steps450, launch+online axial reference alignment,
650-command literal prefix through global1500. Full nominal plus reused failures
20261116/17/18/20/21, <=75000 inclusive actual/proof/parking native steps, all
original gates unchanged. Only6/6 opens a separately frozen45-start gate. The
parallel repair continues under its own registered bounds; never cancel/restart
it because the raw candidate looks better/worse. Reserved14 cohort untouched.

## Fourteen free-terminal proposal repair, before implementation/execution

Raw moving finite001 fails nominal3.58s/no hold; its17s prefix pilot is terminal
0/6, including nominal. Preserve both; no new gate opens. Fixed-terminal
repair remains live, first2 accepted updates reduce .03922754 to1.99126e-5;
both bounded solves hit20 outer iterations (status0), not convergence. Bounds
and line-search merit still hold. Third model has24 augmented bound violations.

Source700's exact final moving pose is not a physical task requirement. Test
a separate FREE final proposal node while keeping initial hanging fixed and
all actual force/world/hold/time gates intact. Existing sparse Jacobian and
retraction already support fix_terminal=False and their native directional
derivative tests cover BOTH modes. Add --free-terminal defaultFalse to the
driver, consistently change state-variable count/control-bound offset and
regularization merit to include the last node. Save this choice in summary/
proposal. Do not change any limit or the running fixed-terminal process.

After targeted existing native shooting/Jacobian tests pass, run independent
`n14_free_terminal_from_marc17_001` from the SAME raw generation001/route.npz,
20 maximum updates/24000000 design-step ceiling, original .001 regularization/
order4/eps.001/four workers/augmented plus bounded1000inner/20outer fallback/
finite_qr/spectrum80/BLAS OMP1. Every node correction keeps the same bounds;
only last-node inclusion changes the proposal family. Same true native defect
line search, physical geometry and delivered-control actual baseline/cold proof,
<=160000 additional native steps. A free last planning node is not a reset of
the plant or an easier acceptance definition. Keep its final world tilt/velocity
and all changed references visible; actual full30s/noisy gates decide success.
No14 reserved data. After terminal, assess with a separately recorded finite
moving-target controller; fixed-terminal branch continues to its own endpoint.

Endpoint-variant existing native Jacobian/retraction tests pass6/6. Both repair
handles are confirmed live: fixed93100 has3 accepted updates/.03922754->9.48118e-6
defect/3040800 native steps (third bounded solve converges13/status1); free25982
is in first bounded fallback after47 augmented bound violations/1010800 steps.
Do not restart either from a stale checkpoint. Raw moving finite001 and17s
prefix pilot are terminal failures (nominal3.58s raw/20.58s prepared,0/6 pilot,
all cold replay exact); no noisy/reserved qualification is opened. Future final
moving designs must retain700 design and650 executed commands, no stationary
appendage or extra physical time. Public13 artifacts remain unchanged.

## Fourteen axial-phase arithmetic audit, before offline execution

Both bounded repairs remain live; preserve them. While waiting, measure a
specific numerical question on the old4/45 incumbent: its huge fitted gains may
amplify rounding in observed axial-frame phase calculations. Do NOT label the
native simulator or existing law buggy from this hypothesis.

`n14_axial_phase_diagnostic_001` freezes conditioned_n14_001/route.npz and its
retained failed20261116/passing20261119 tapes. Reconstruct the unchanged launch
wrapper from each actually recorded command950 state. At steps950,1000,1050,
1100,1150,1200, first require the matrix-phase raw force to equal the recorded
raw force byte-for-byte. In independent scratch environments, compare the same
law with ONLY equivalent phase arithmetic changed: relative world quaternion
r=conj(q_ref_world)*q_obs_world, phase=atan2(2*r_w*r_z,r_w²-r_z²), versus existing
polar phase of R_ref_world.T R_obs_world. Same reference/gain axial transport,
native observed q/v and tangent error, offsets, caps and physical-F32 delivery.
No MP, coefficient change, new projection or candidate selection. Save both
raw/delivered forces, exact baseline parity, per-component F32 differences and
all source/tape identities. Zero native integration steps/zero actual episodes.
This is offline numerical evidence only; if it changes delivered bits enough
to merit testing, register a NEW arithmetic-family controller and independent
actual pilot/cohort gates. No14 reserved seeds or task-success inference.

## Fourteen feedback dot-product oracle, before offline execution

Phase audit001 is terminal: all12 sampled original matrix raw forces match
retained tapes byte-for-byte, quaternion phase changes raw force at most7.15e-7N
but delivered force only1.455e-11N (two F32 components). This small sampled
effect does not establish a useful controller or rule out all arithmetic effects.
Do not deploy a phase variant from this diagnostic alone.

`n14_dot_product_diagnostic_001` extends the same offline driver with an optional
--dot-oracle branch. Same two tapes/12 states/route; reconstruct ORIGINAL
matrix-phase aligned gains and error vector, require exact original raw force
again. Save all error/gain/action/feed operands. Compare ordinary dot product,
math.fsum of rounded F64 products, MP sum of EXACT stored F64 operands, and the
whole affine force evaluated before final rounding in MP. Compute at80/120
decimal digits and require identical final F64 forces, report mpmath1.3.0.
Same delivered-force cap/F32 conversion for every arithmetic result, record
both changed bit patterns and actual changed component values. <=5000 MP scalar
products/zero native integration/zero actual episodes. This audits cancellation
in one control calculation; it does not establish task stability or change a
production law. Only measured differences can motivate separately registered
full-policy experiments. Both repair handles stay running;14 reserves untouched.

## Fourteen material continuation from the qualified spatial13 reference

Dot audit001 is terminal: all12 original forces byte-exact,80/120-digit full
affine sums agree after F64 rounding; maximum raw difference7.15e-7N but ZERO
changed delivered F32 values or bits. No precision-family controller is admitted
from these samples. Keep both existing moving-target repair branches alive.

Test the natural +1 material continuation from our OWN qualified spatial13
controller as another reference family. `n14_material_from_spatial13_001` freezes
qualify_n13_001/frozen_route.npz and the public13 manifest, verifies its SHA256
against that manifest's controller inventory and canonical Config13. Use ONLY
its first450-command9s geometric reference; do not copy preparation, profile
selector, gains, value roots or qualification outcomes into14. Reference data
comes from a qualified policy but is not an actual episode tape.

Apply existing planar proposal cleanup guard<=1e-8 (record changes), then decode
native relative Y quaternions via2*atan2(q_y,q_w), unwrap continuously in time
and anchor the first hanging branch at [pi,0,...]. Preserve cart X and native
relative Y angular velocities; independently re-encode every source pose/rate
and require tangent parity<=1e-10. Reject genuine spatial input or angular-rate
sampling above pi/2 per command (observed source max.10955rad/command), rather
than guess a winding. New decoder tests at1/3/13 include4pi temporal winding,
arbitrary quaternion sign flips, input/live-state preservation and rejection
of real spatial/aliased-rate input. All tests required before generation.

Interpolate UNWRAPPED cumulative angles and rates at actual canonical13 material
centers into canonical14, preserve force guesses/cadence with target F32 delivery,
require exact hanging error<=1e-6 and native14 world directions<=1e-10. Zero
source/target physics integration; geometric kinematic checks only. Source13
remains immutable and its results are not14 evidence.

After valid generation, `n14_direct_from_spatial13_001` uses SAME standard full88D
shooting:20 updates/15000000 native design-step ceiling, fixed initial and final
proposal nodes, order4/eps.001/four workers/BLAS OMP1, augmented ridge plus
bounded1000inner/20outer fallback, regularization.001, finite_qr/spectrum80.
No model/control/physical-limit change. Separate gap-free actual baseline+cold
proof <=160000 further steps. After terminal register a finite controller on
the FINAL reference under full actual/cohort gates. No14 reserved data. This
independent shorter reference branch does not cancel either live700-step repair.

## Fourteen free moving-target final execution protocol

New spatial-reference decoder tests pass153/153, including winding/sign/native
roundtrip and rejection guards. Generation from qualified spatial13 is terminal:
public controller SHA796dfefd9adb83efe4390ee4b31fc85fea4666bc1b0d58b0e409c906e78f638a,
manifest37785c8233720b5bab7733bfc2770514cb6c86254ec756de6948ead90c107c95,
roundtrip8.882e-16, target hanging1.225e-16/directions1.332e-15, zero integration.
Independent direct_from_spatial13_001 is live, session98793,20/15M bounds.
Original fixed93100 and free25982 moving-reference repairs continue; free has
9 accepted updates to1.592e-13 defect at9122400 native steps, whereas fixed
has5 accepted updates to6.642e-6. No physical task success follows from this.

After free25982 is authoritatively terminal, use ONLY its FINAL proposal in
`n14_free_marc17_finite_001`. Measure planar cleanup input guard<=1e-8; if it
passes use guarded reflection-model QR, otherwise retain rejection and use
full unprojected native88D reference/QR. Same700-step moving design, relative
Q/R, terminal Riccati scale0+local factor, no stationary appendage/shortening,
native order4/eps.001/four workers/finite_qr/spectrum80/BLAS OMP1. Raw full30s
nominal+cold proof and conditional12 shared starts, <=2500000 inclusive native
steps. No inference about the17s prefix from a later raw stationary handoff.

After its authoritative terminal design, `n14_free_marc17_prepared_pilot_001`
freezes EXACT700-step payload,17s hanging R=.01, --swing-steps450, observed
launch+online axial reference alignment,650-step literal prefix inside30s.
No search/offset/origin/online-parking/dynamic timing selection. Nominal20261016
plus five reused failures20261116/17/18/20/21, <=75000 inclusive native steps.
All physical force/boundary/tilt/full-episode/no-warning/cold-replay gates apply.
Only6/6 opens a separately frozen45-start development gate.14 reserved cohort
stays untouched until a full gate passes and a new memo freezes qualification.
If multiple branches become eligible, the first complete45/45 physical/cold
gate opens reserved qualification; preserve every other branch independently.

## Fourteen free moving-target longer preparation, before execution

Free repair is terminal after13 accepted updates/15201924 design steps, final
defect3.74894e-22/max1.78922e-12; actual gap-free baseline holds8.40s but exits
14.98s, exact replay. Final cleanup normalizedstate6.6612e-10/force3.1139e-9
passes1e-8. Finite700 design is terminal, symmetry projection within guard,
raw nominal holds8.40s but fails15.08s at its later stationary handoff. This
does not establish full30s success or invalidate an earlier literal prefix.

The17s/650-prefix pilot is terminal4/6 full30s/cold replay: nominal and noisy
20261117/18/21 pass7.36s hold;20261116/20 fail before holding. This improves
three previously failing development cases, not qualification. Independent
saved-tape rescore of NOMINAL's first540 tracking commands gives258 upright
post-states/5.16s. Thus19.2s hanging +540-command literal prefix fits the SAME
30s deadline with5.16s nominal hold. Source17's parking duration is not treated
as a universal spatial prescription; no shorter recursion is regenerated.

`n14_free_marc17_longpark_pilot_001` changes ONLY preparation17->19.2s (960
commands) on the SAME finite700 controller/relative metric/online axial law,
--swing-steps450, original hanging R=.01. Nominal plus five REUSED failures
20261116/17/18/20/21, full30s/no warning/force80/world3m/5s hold/exact cold replay,
<=75000 inclusive native steps. Require nominal states/actions/raw/tilts from
newcommand960 onward to match original17s nominal tracking prefix850..1390
byte-for-byte (540 commands/541 states), since exact hanging is stationary.
No profile/offset/search/physical recentering or state reset. Only6/6 opens a
new frozen full45-start gate; otherwise retain and register next controller
change.14 reserves20262516..47 untouched. Other repairs continue independently.

## Fourteen moving-target fine offset diagnosis, before execution

19.2s moving pilot is terminal4/6, same successes as17s: nominal+20261117/18/21
pass full30s/5.16s hold;20261116/20 fail with no hold. Do not open full cohort.
The NOMINAL 19.2s tracking states/actions/raw/tilts must match17s prefix exactly
per prior registration. These are reused development data; no14 reserve used.

Independently diagnose BOTH remaining failures with the NEW moving-target law,
not the old static arrival's failed offset bank. Freeze
`n14_free_marc17_offsets_116_001` and `n14_free_marc17_offsets_120_001` from
longpark_pilot001/route.npz and each actual tape's observed command960 state.
Existing cold-native prefix and zero-profile suffix parity requirements apply.
Fixed order: zero, then +/-X/+/-Y for magnitudes1e-6,1e-5,1e-4,1e-3m (17 total).
Only reference cart coordinates shift; actual states/bounds/clock stay fixed.
All full remaining540-command native forecasts use the same observed state,
original spin/full88D law, real F32/80N disk,3m world square/5s/full30s gates.
Save all forecasts/outcomes/raw actions and query counts. Per diagnosis bound
3840 prefix proof+17*2160=40560 native steps, cap41000; zero new actual episodes.

Add optional --scales/default[.0001,.001] and --physics-budget/default25000 to
the diagnostic driver only, preserving old9-profile order when omitted. Reject
nonpositive/nonfinite/duplicate scales. If forecasts pass, this only motivates
a separately frozen state-only selector/fixed-profile actual pilot and full
cohort; no per-seed runtime lookup or qualification from forecast scores.

## Fourteen spatial13-derived final finite test, before execution

Both fine moving-reference offset diagnoses are terminal0/17, with exact cold
prefix/zero-profile suffix parity; no selector is admitted from them. Longer
moving pilot remains4/6, with the same two failed seeds at17s and19.2s. Do not
consume14 qualification. Source13-derived native repair is now terminal20
accepted updates/13684324 design steps, defect1.43973e-15/max2.09629e-8; actual
baseline fails7.34s/.26s hold/Y3.071622m, exact cold replay. Final proposal
cleanup normalizedstate1.81664e-11/force8.85364e-11 passes1e-8, zero integration.

`n14_from_spatial13_finite_001` uses ONLY its FINAL shooting_proposal.npz, same
canonical14/full88D native identification/order4/eps.001/four workers/BLAS OMP1,
guarded clean planar reference and reflection-model30D+58D residual QR, original
relative Q/R, true native defects/chart transport, exact upright continuation
to1500 design commands, terminal Riccati scale0+last local factor, finite_qr/
spectrum80. No search/preparation/origin/offset. Full30s nominal/cold proof,
only on full pass12 shared raw noisy starts. <=2500000 inclusive native steps.

After terminal, choose preparation using ONLY its retained NOMINAL trace:
fixed descending candidates19.2,19.0,18.0,17.0s. For each, count longest upright
post-state streak in first(1500-park_steps) tracking actions. Require>=258 ticks/
5.16s AND prefix ends strictly before a failed episode's terminal action (or
is within an actual full pass). Choose first satisfying candidate. Keep the
same1500-step design/gains; no short-horizon regeneration. If none qualify,
retain the result and do not open preparation under this memo.

Eligible `n14_from_spatial13_prepared_pilot_001` freezes that exact payload,
selected hanging duration/R=.01, observed launch+online axial reference alignment,
450 bridge commands, same full30s/80N/world3m/5s/no-warning/cold-proof gates.
Nominal20261016 plus reused failures20261116/17/18/20/21, <=75000 native steps.
Only6/6 opens separately frozen full45-start development before14 reserves.
This is a new per-count reference, not evidence inherited from source13.

## Fourteen aligned native action-value family, before implementation

The moving-reference19.2s pilot is4/6, with remaining failures20261116/20;
both17-profile offset diagnoses are0/17. Sampling arithmetic alone produces
negligible delivered changes, so do not promote an arithmetic-only law. The
fixed700 repair remains live; source13-derived final finite design passes
nominal24.16s but0/12 raw starts; its chosen19s prepared pilot is running.

Test the document's inner native action improvement WITH observed axial-frame
feedback, a combination the old wrapper deliberately rejected. Implement a
separate aligned_value module and explicit payload marker, leaving old affine
classes/functions unchanged (the live repair can import them later). Existing
experiment module is already loaded by that repair; the new factory branch is
only for explicitly marked future policies. Do not change the running job.

At each active command: compute the unchanged aligned affine baseline, deliver
F32 under the80N Euclidean cap, then predict its ONE interval in an owned native
environment. Align the NEXT reference/value-root columns to that baseline's
predicted axial frame. Hold that frame fixed while comparing all candidate
successors. Full affine residual r remains in the value. Evaluate differences
relative to baseline as (2*w_base+S*delta_e)'(S*delta_e) plus the exact quadratic
input-cost difference, avoiding subtraction of two large absolute costs. This
is a new nonlinear local-value law, not an optimality or disturbance guarantee.

Maximum16 distinct delivered candidates, deterministic order: baseline/reference/
half; coordinate +/- trials around current best at radii max(|baseline-ref|,
8 F32 ULPs,1e-4N), then half those radii; four adjacent F32 coordinate neighbors;
final half of best/baseline. Deduplicate/cap every query. All queries use the
FULL unchanged3D native model and real-world bounds; invalid forecasts cannot
win. If none valid, delivered baseline fallback is recorded. Predictor never
advances/sets the real environment. Save every proposal/delivered action/native
successor/relative cost/validity/margin/anchor frame/selected action/query count.

Unit/native tests require stable cost-difference agreement with independent
absolute value/input differences, nonzero affine-residual response, candidate
dedup/cap/budget, exact candidate native reproduction, real-state/clock/counter
isolation and native trace accounting. Original13 heldout/forecast byte parity
must remain exact after adding the opt-in experiment factory. No native search
is allowed for postfit-reflection-averaged payloads with unused value roots.

After all tests pass, `n14_aligned_value_pilot_001` freezes EXACT
n14_free_marc17_longpark_pilot_001/route.npz with only the explicit aligned-native
search flags added.19.2s/960 preparation,540 executed commands of original700
design,450 bridge commands, original Q/R/reference/gains/feed/roots/residuals,
no offsets/online parking/origin/admission/physics changes. Nominal plus five
REUSED failures20261116/17/18/20/21, full30s/no warning/5s/3m/80N/cold replay.
Per episode<=34560 predictor+12000 actual/proof steps;6-case pilot incl hanging
design<=285000 native steps. Only6/6 opens a new frozen45-start gate. No14
reserved data or hardware20ms claim; all online model cost must be reported.

## Fourteen aligned native-value terminal pilot and integrity audit

New opt-in aligned_value module is separate from the old running repair's
affine_value implementation.158 tests pass; original13 first-heldout states,
actions, raw forces, tilts and BOTH full forecast tapes remain byte-identical.
Own13-derived prepared pilot is terminal1/6, not admitted. Aligned-value pilot
is terminal4/6, the SAME pass/fail labels as the paired plain moving controller:
nominal,117,118,121 pass full30s/5.16s hold;116 and120 fail without hold. Every
actual tape cold-replays exactly.225008 inclusive native steps includes158008
predictor,33500 actual and33500 proof. No new full cohort or reserves opened.

Before final result audit: compare ALL selected inner successors to actual next
states, including last command, and validate delivered candidate uniqueness,
disk cap, <=16 count, selected index/cost/fallback and counter accounting.
Native independently reproduce ALL candidates at each command960,1060,1160,
1260,1360,1460 that exists in each tape (<=2304 native steps). Reexecute complete
nominal with new policy/search in a separately saved nominal_reexecution.npz,
retaining actual/cold steps and all inner traces; require byte equality of all
numeric tape arrays with original nominal.<=44560 additional native steps,
combined pilot+audit remains <=285000. Save identities, terminal receipts,
candidate/cold reexecution parity and inclusive audit accounting. These are
integrity checks on REUSED starts, not extra independent successes.

## Fourteen physical moving reference, before construction/redesign

The aligned native search pilot adds cost without improving paired4/6 outcome;
its integrity audit is running. Test document12.1's physical-reference idea as
a separate branch, preserving every old policy and failed outcome. Use ONLY
first700 actual delivered commands and701 states from the raw nominal tape
n14_free_marc17_finite_001/affine/nominal.npz. Its later stationary handoff fails;
do not label the source episode a full pass. The prefix reaches the moving tail
with0.04825rad terminal tilt, cartX1.865605m/Y5.93e-13m. Input-only symmetry
guard measured6.224e-12 normalizedstate/3.287e-12 force (<1e-8), zero native
steps. This informs the declared model branch, not a new outcome selection.

`n14_physical_moving_reference_001` freezes source tape plus source controller,
exact first700 delivered F32 actions/701 fullnative states/config. Independently
native reproduce all700 maps byte-for-byte, including finaltransition;2800
native steps. Raw physical tape has EXACT zero native reference defects; no
virtual nodes/recenter or actual-state edits. Save physical reference unchanged.

`n14_physical_moving_finite_001` separately freezes that reference, uses existing
guarded planar cleanup (record that this CAN introduce small defects), original
relative Q/R, native full88D order4/eps.001/four workers/BLAS OMP1,
reflection-consistent30D+58D QR, terminal Riccati0+lastlocalfactor,
finite_qr/spectrum80.700 design commands, no stationary append/search. Full raw
30s nominal and cold proof; conditional12 noisy raw starts only on fullpass,
<=2500000 inclusive native steps. Preserve all repaired-vs-physical provenance.

After finaldesign, regardless of its separate post14s handoff result,
`n14_physical_moving_prepared_pilot_001` freezes exact700 design,19.2s/960 causal
hanging preparation,R=.01,450 bridge,540 literal executedprefix,launch+online
axial feedback. No search, offsets, admission, origin, online parking or
shortened value redesign. Nominal plus reused116/117/118/120/121,<=75000 native
steps incl hanging design. Original full30s/5s/.15rad/3m/80N/no-warning/exactcold
gates unchanged.6/6 alone opens separately frozen45-start gate; no14reserves.

## Fourteen terminal physical-reference and aligned-value outcomes

Physicalization700/701 native maps are byte-exact;2800 steps. After explicitly
recorded guard6.224e-12/3.287e-12 cleanup, fresh planning A/B/d parity changes
4.676e-10/4.077e-10/1.333e-14 pass1e-8. Physical finite design uses1011524
native steps; raw actual8.40s hold/fails15.08s exactly as prior controller,
inclusive1017556. Prepared actual pilot is4/6, with the SAME failed116/120 as
plain/aligned-value moving pilots; all cold exact.67780 inclusive steps incl
724 hanging design. No gain in complete paired outcomes from physicalization.

Aligned-value audit passes ALL selected successor comparisons including final
commands, distinct F32 cap/count/validity/mincost accounting,1812 independent
native candidate spot steps, and complete nominal reexecution44548 native
steps. Every actual/inner trace numeric array is byte-identical on reexecution.
Total pilot+audit271368<285000. Source/candidate tapes/audit driver/plan retained.
The local cost law remains a failed reliability intervention despite exact
prediction integrity.14 remains unqualified; reserved20262516..47 untouched.

## Fourteen moving-reference preparation authority pilot, before execution

The repeated remaining failures occur before hold on both original and physical
moving references. Test ONE stronger hanging control penalty choice on ORIGINAL
incumbent moving reference, document8/12's preparation/correction-authority idea.
Earlier R-family probes were on the DIFFERENT stationary14 reference; do not
claim they tested this new moving family. No full Cartesian sweep is authorized
by this memo. `n14_moving_strongpark_pilot_001` freezes EXACT source
n14_free_marc17_finite_001/affine_controller.npz, only hanging design R=.001
(incumbent .01),19.2s/960,450 bridge,540 literal prefix of700 original design,
launch+online axial alignment. Native canonical mechanics/80N/world3m/30s/5s/
.15/no-warning/cold proof unchanged. No search/offset/origin/online parking/
admission/new tracking design. Nominal plus same reused116/117/118/120/121,
<=75000 inclusive native steps incl hanging design.6/6 opens separately frozen
45-start gate; otherwise retain failed arm with no reserves or wider grid.

## Fourteen stronger moving preparation terminal outcome

Stronger R=.001 moving pilot is terminal4/6, SAME failed116/120; no new full
cohort, wider R grid or reserved tests opened. Nominal/117/118/121 pass full30s/
5.16s hold;116 fails24.00s,120 fails24.50s without hold. All cold replay exact.
Retain plain R=.01 incumbent; this intervention did not improve paired endpoint.
The fixed-end repair remains live in its original bounded budget; no restart.

## Fourteen axial-quotient planning consistency, before implementation

Prior goal turn made concrete progress: opt-in native-value implementation,
158 passing tests, exact13 heldout parity, completed paired pilots/integrity
checks and committed lab9d0c648. No14 qualification earned. Fixed-end repair
handle93100/PID34108 confirmed LIVE, now iteration15 bounded solver; never
restart merely for an observation timeout or small planning-gap reduction.

Current online law changes reference axial frames on each observation, yet its
full local value design penalizes those arbitrary cyclic ORIENTATION phases.
Test consistency of the planning model with that causal frame alignment. This
is a controller/model reduction, NOT removal/locking of native physical DOFs.
All14 measured angular SPIN VELOCITIES remain, as do all cart/tilt/rate modes.

New isolated quotient module: analytic closest-axial alignment derivative C
maps full tangent errors to14 body axial-frame angles. D is the14-column exact
infinitesimal generator of axial_gauge_state (relative-quaternion local tangent
and rotating local reference angular-velocity components). G=I-D*C is the
observed alignment error derivative, C*D=I, G^2=G. Use orthonormal basis E for
ker(C), F=E'*G; F*E=I, E*F=G. Full88D becomes74 planning coordinates, preserving
all44 velocity coordinates, including twist. At reflection-fixed references,
preserve even30 dimensions as identity and odd44 via a deterministic complete
QR of C_odd'; no even/odd contamination by a dense arbitrary basis.

Tests before design: independent centered native quaternion/alignment finite
differences for C,D,G at1/3/14 links with generic orientations and spin; generator
physical directions/rates invariance; G/dual identities, spin-coordinate
retention; planar even/odd basis parity; native finite gauge-flow equivalence;
small reduced finite affine QR vs independent dense condensed optimum retaining
nonzero defects/residuals, and embedded full-error force/value parity. No
modification of affine_value.py needed while live repair may import it late.

`n14_quotient_moving_finite_001` freezes EXACT controller plus saved native-model
cache n14_free_marc17_finite_001, its source repaired proposal, inputs/cleanup/
projection provenance. Verify source input hashes and independently regenerate
clean reference to exact stored states/actions before cache reuse. Reapply SAME
guarded reflection denoising; no foreign/stale-center cache. A_y=F_next*A*E,
B_y=F_next*B,d_y=F_next*d; L_y=L*E; original relative L/R, terminal Riccati0
plus LAST local L*E. Retain affine residuals and all700 design times. Save all
C/D/G/E/F, reduced A/B/d, original and lifted gains/roots, model identities and
max normalized A*D-D_next gauge discrepancy. Abort if normalized discrepancy
>1e-5; this is a planning consistency guard, not a stability certificate.
Separate even30/odd44 affine QR. Lift K=K_y*F, S=S_y*F; residual/feed unchanged.
No value/gain postfit averaging, actual-state projection, force change, new
reference, moving frequency, physics, seed-dependent branch or state reset.

Raw nominal full30s/cold proof under lifted affine law; conditional12 reused
raw noisy starts only on fullpass. Cache/design native0, <=180000 actual/proof
steps total. Raw stationary handoff after700 remains a separate diagnostic.
After design terminal regardless of handoff result,
`n14_quotient_moving_prepared_pilot_001` freezes exact700 payload,19.2s/960
parking R=.01,450 bridge,540 literal prefix, launch+online axial alignment.
Nominal plus SAME reused116/117/118/120/121,<=75000 native steps incl parking
design. Original30s/5s/.15rad/world3m/F32disk80N/no-warning/exactcold gates.
Only6/6 opens separately frozen45-start development; no14 reserved data.

## Fourteen axial-quotient pilot success and full development activation

165 tests pass, including independent native finite differences of C/D/G,
full velocity retention, native gauge-flow equivalence at1/3/14, full/reduced
force/value parity and independent condensed finite optimum with affine defects.
Cache source/input/driver/derivative-provider hashes and exact cleaned arrays
verify. Quotient A*D-D_next normalized residual1.74752e-8<1e-5; all native
mechanics/88 physical state coordinates remain unchanged. Raw nominal holds
8.40s, then fails15.08s after its explicitly separate post700 handoff, exactcold.

Prepared quotient pilot is6/6, ALL full30s/5.16s/no-warning/worldbounds/exactcold,
including both formerly failing116/120. This earns ONLY full development,
not a qualified14 claim. Freeze EXACT
n14_quotient_moving_prepared_pilot_001/route.npz, no parameter/array changes.
`conditioned_n14_quotient_001` executes nominal20261016 noise0 plus old canonical
44 development starts20261116..27 and20261416..47, independent complete30s
actual+cold tapes,19.2s/960 parking,540 literaltrackingprefix/700 design/450
bridge, unchanged80N disk/world3m/.15rad/5s/no-warning. <=540000 inclusive
actual+proof native steps, design0. Freeze source/driver/route/config/runtime,
no seed lookup, no admission/profiles/native search. No early successful-only
filtering; save all45 outcomes even after a failure. Only45/45 opens32 NEW
reserved14 seeds20262516..47 under separately frozen unchanged source/payload.

## Fourteen quotient full gate terminal and observed-offset diagnosis

Full45 quotient development is terminal43/45 (replayexact all), not admitted.
Failures20261431 (25.46s/no hold/X3.115332) and20261437 (24.82s/no hold/
Y3.010038) are retained. Both old moving failures116/120 now pass, and every
success has full30s/5.16s hold. No reserved14 data opened. Fixing planning-frame
consistency helps the paired pilot but does not by itself qualify fourteen.

Before any new actual selector: run existing diagnose_launch_profiles on BOTH
failed quotient tapes, exact frozen fullgate controller, fixed ordered profiles
zero then +/-X,+/-Y at.0001m then.001m (9 per launch), full540-command remaining
policy. Every profile evaluated; preserve all raw/action/state/tilt forecasts,
world bounds/counters and zero-profile suffix byte parity. Independent actual
hanging-prefix cold proof3840 native steps each, <=25000 inclusive native steps
per diagnosis. No parameter sweep/refit/native search/plant edits. Select the
FIRST nonzero profile in fixed order passing BOTH full remaining task forecasts;
if none, retain result and preregister another action rather than deploy selector.
Any eligible two-profile admission variant needs separately frozen actual pilot
and full45 gate before reserved14. Profile changes reference error only; cart
never moves/reset at selection. No seed is an input to the deployed policy.

## Fourteen quotient offsets null and original-law complement check

Both9-profile diagnoses are terminal0/9; zero suffixes byte-exact, no selector
admitted. Rather than expand offsets, test whether the preserved ORIGINAL full
88D-value moving controller covers the two failures of the74D-value quotient
law. Before execution: `n14_moving_complement_431_001` and437_001 each freeze
EXACT n14_free_marc17_longpark_pilot_001/route.npz, run ONE fullactual episode
with reused seed20261431 or20261437/.02 plus independent cold proof. No arrays,
parking, branch, offset, reference, arithmetic or gains changed. <=12000 native
steps each. Compare first960 actual states/actions/raw/tilts with corresponding
failed quotient tape; require exact shared preparation before comparing suffixes.
This tests complementary controllers only. No bank/admission implemented or
qualified from these two outcomes; any later controller bank needs separate
source/ownership tests, full actual pilot and full45 gate before reserves.

## Fourteen quotient preparation pilot, before execution

Both original-law complement trials fail431/437 without hold; actual first960
commands/961states/rawforces/tilts are byte-identical to corresponding quotient
preparation, and full cold replay exact. No controller bank is admitted.

Test ONE stronger hanging design R=.001 with the newly consistent QUOTIENT
tracking law; the prior stronger-moving arm used unreduced88D value feedback,
not this43/45 law. `n14_quotient_strongpark_pilot_001` freezes EXACT
n14_quotient_moving_finite_001/affine_controller.npz,19.2s/960 preparation,
R=.001 (old .01),450 bridge/540 literalprefix of700 design. Launch+online
axial reference alignment; no offset/admission/search/origin/onlineparking/
tracking redesign/reference/physics changes. Nominal + reused116/117/118/120/
121 AND fullgate failures431/437;8 actual/cold episodes,<=100000 inclusive native
steps incl724 hanging design. All unchanged30s/5s/.15/3m/80N/no-warning gates.
Only8/8 opens separately frozen full45 gate; otherwise retainfailedarm and
return to separately registered fixed-end final repair or new synthesis, with
no wider preparation grid or reserved14 data under this memo.

## Fourteen quotient strong preparation terminal / online preparation pilot

Stronger hanging R=.001 quotient pilot is terminal6/8, same431/437 fail;437
fails13.46s DURING preparation, before the quotient route exists. Do not tune
tracking against a prelaunch failure. Original R=.01 quotient remains43/45
incumbent. All actual tapes replay exactly, no new qualification.

`n14_quotient_onlinepark_pilot_001` tests ONE other explicit preparation law:
return to original R=.01/19.2s and use existing online_parking=True closest
observed axial reference/gain alignment EACH hanging command, consistent with
online tracking. This is a new nonlinear hanging controller, not a numerical
rewrite; previous online-parking probes used different count/reference laws.
Freeze exact quotient finite payload, same450/540/700 clocks, originalparkR,
launch+online tracking, no offsets/admission/search/origin/physics changes.
Nominal plus reused116/117/118/120/121/431/437 actual/cold8 cases,<=100000 native
steps incl724 hanging design. Full unchanged gates. Only8/8 opens separately
frozen45-start development; no further R/parking combination grid or reserves.

## Fourteen online preparation terminal / thirteen source regression

Online preparation quotient pilot is terminal6/8, same431/437 failures without
hold, all cold replay exact. Original R=.01/non-online preparation43/45 controller
is retained. No wider parking family, selector or reserved14 activation follows.
Return to separately registered final fixed-end reference after its live handle
terminates, or preregister another synthesis hypothesis.

Before further source edits/publication: `n14_quotient_n13_regression_001` runs
the already consumed first13 heldout20262416 full30s/.02 using its unchanged
qualified payload/current source, exact cold proof and BOTH forecast tapes.
Require states/actions/rawforces/tilts/fullforecasts byte-identical to immutable
qualification;<=16400 inclusive native steps. Reused integrity only, not new
qualification or a replacement for14's unconsumed reserved gate.
