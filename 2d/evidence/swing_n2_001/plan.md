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
