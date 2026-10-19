# 2D cartpole / spatial ball-joint chain

A cart moves over an XY plane, driven by two forces. Every link has a passive,
unlimited ball joint: it can tilt in both directions and twist about its axis.
This is a separate benchmark from the one-dimensional cart and hinge chain.

**One- through fourteen-link hanging-start swing-ups are qualified:** 32/32 fresh
noisy 3D starts at each count, in addition to nominal and development checks.
Episodes run 30 seconds; the latest fourteen-link controller holds upright for
5.16 seconds. All delivered-action
tapes replay exactly. Both cart axes are used. Full-state simulation feedback,
not a learned policy or a hardware demonstration.

The benchmark permits arbitrary XY cart motion and unrestricted ball joints,
but does not require a nonplanar nominal swing. The four- through fourteen-link
reference swings are essentially planar; Y feedback handles spatial start
perturbations. This is a controller/reference bias, not a locked simulator axis.
The independent [spatial freedom audit](https://github.com/corbensorenson/cartpole/tree/master/2d/audits/spatial_freedom_001)
checks native two-axis motion and includes a controlled six-link XY loop.

[Full one-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-1-link-v1/2d-cartpole-1-link-swingup.mp4)
· [GitHub release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-1-link-v1)

[Full two-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-2-link-v1/2d-cartpole-2-link-swingup.mp4)
· [Two-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-2-link-v1)

[Full three-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-3-link-v1/2d-cartpole-3-link-swingup.mp4)
· [Three-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-3-link-v1)

[Full four-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-4-link-v1/2d-cartpole-4-link-swingup.mp4)
· [Four-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-4-link-v1)

[Full five-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-5-link-v1/2d-cartpole-5-link-swingup.mp4)
· [Five-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-5-link-v1)

[Full six-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-6-link-v1/2d-cartpole-6-link-swingup.mp4)
· [Six-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-6-link-v1)

[Full seven-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-7-link-v1/2d-cartpole-7-link-swingup.mp4)
· [Seven-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-7-link-v1)

[Full eight-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-8-link-v1/2d-cartpole-8-link-swingup.mp4)
· [Eight-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-8-link-v1)

[Full nine-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-9-link-v1/2d-cartpole-9-link-swingup.mp4)
· [Nine-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-9-link-v1)

[Full ten-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-10-link-v1/2d-cartpole-10-link-swingup.mp4)
· [Ten-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-10-link-v1)

[Full eleven-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-11-link-v1/2d-cartpole-11-link-swingup.mp4)
· [Eleven-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-11-link-v1)

[Full twelve-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-12-link-v1/2d-cartpole-12-link-swingup.mp4)
· [Twelve-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-12-link-v1)

[Full thirteen-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-13-link-v1/2d-cartpole-13-link-swingup.mp4)
· [Thirteen-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-13-link-v1)

[Full fourteen-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-14-link-v1/2d-cartpole-14-link-swingup.mp4)
· [Fourteen-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-14-link-v1)

| Milestone | Current evidence | Upright hold |
|---|---|---|
| One-link hanging-start swing-up | 32/32 final qualification; full video | 21.74s |
| Two-link hanging-start swing-up | 32/32 final qualification; full video | 21.68s |
| Three-link hanging-start swing-up | 32/32 fresh final qualification; full video | 6.78s |
| Four-link hanging-start swing-up | 32/32 fresh final qualification; full video | 7.22s |
| Five-link hanging-start swing-up | 32/32 fresh final qualification; full video | 21.70s |
| Six-link hanging-start swing-up | 32/32 fresh final qualification; full video | 14.06s |
| Seven-link hanging-start swing-up | 32/32 fresh final qualification; full video | 13.48s |
| Eight-link hanging-start swing-up | 32/32 fresh final qualification; full video | 14.20s |
| Nine-link hanging-start swing-up | 32/32 fresh final qualification; full video | 8.18s |
| Ten-link hanging-start swing-up | 32/32 fresh final qualification; full video | 6.12s |
| Eleven-link hanging-start swing-up | 32/32 fresh final qualification; full video | 5.50s |
| Twelve-link hanging-start swing-up | 32/32 fresh final qualification; full video | 5.50s |
| Thirteen-link hanging-start swing-up | 32/32 fresh final qualification; full video | 5.16s |
| Fourteen-link hanging-start swing-up | 32/32 NEW fresh qualification; full video | 5.16s |
| Fifteen-link hanging-start swing-up | Next development count | — |
| Near-upright balance, 1–3 links | 12/12 local perturbed starts at each count | — |
| Near-upright balance, 4 links | 9/12; incomplete, not accepted | — |

## Run on Windows

From this directory (the `2d` directory in the published parent repository):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q
```

Reproduce the frozen fourteen-link controller on its recorded qualification starts:

```powershell
.\.venv\Scripts\python.exe -m spatialpole.experiment evaluate --links 14 --route evidence/qualify_n14_002/frozen_route.npz --episodes 32 --seed 20262616 --out output/reproduce_fourteen
.\.venv\Scripts\python.exe -m spatialpole.experiment video --tape output/reproduce_fourteen/seed20262616.npz --out output/2d-cartpole-fourteen-link.mp4
```

Local development uses `output/qualify_n14_002/frozen_route.npz` before packaging.
Output directories must be new; failed runs are preserved. These are previously
tested starts, so reproduction does not add independent qualification evidence.

From count13 onward, each ZIP contains complete current source/runtime and all
new or changed evidence. Unchanged historical raw evidence is retained in
earlier immutable bundles, linked by URL, size, SHA256 and exact prior manifest
in `manifest.json`; all historical files remain in the Git checkout. The current
count reproduces from its current archives. Fourteen retains all current evidence
across the primary ZIP and hash-bound supplements; runtime and qualification are
in the primary. Large Git checkpoints use Git LFS: run `git lfs pull` after
cloning. Release ZIPs contain the full original bytes, without LFS pointers. Older examples below require the matching
count's release bundle or the complete checkout. `verify_release.py` also checks
historical coverage recursively when prior ZIPs are beside the current one.

For the qualified two-link controller use
`evidence/n2_world_handoff_001/controller.npz` with `evaluate`; its saved
`world_balance=true` selects the physical world tilt/rate coordinate map for
the stationary handoff. The original two-link local-quaternion handoff failed.
The frozen route and trajectory gains are identical between these two tests.

For three links use `evidence/qualify_n3_002/frozen_route.npz` with `evaluate`.
Its policy applies **12 seconds of hanging LQR**, aligns the reference's passive
axial coordinate frames from the observed launch state, tracks an 11.28-second
swing, and then applies upright feedback. Every physical state evolves through
legal cart forces; there is no state reset or joint lock at either switch.
The new final seeds are 20271416..20271447. The earlier raw-route qualification
failed at 25/32 and is retained separately as `qualify_n3_001`.

For four links use `evidence/qualify_n4_001/frozen_route.npz`. It tracks a
23.04-second repaired motion immediately from the noisy hanging state and then
uses world-coordinate upright feedback. No preliminary conditioning or axial
alignment is required. Its fresh qualification seeds are 20261516..20261547.

For five links use `evidence/qualify_n5_001/frozen_route.npz`. The repaired
motion lasts 8.42 seconds, starts immediately, and then uses world-coordinate
upright feedback. Qualification seeds are 20261616..20261647. No preliminary
conditioning or gauge alignment is required.

For six links use `evidence/qualify_n6_001/frozen_route.npz`. Its policy applies
12 seconds of hanging feedback and measured axial-frame alignment before a
4.18-second repaired swing, followed by world-coordinate upright feedback.
All phases fit inside 30 seconds. Fresh seeds are 20261716..20261747.

For seven links use `evidence/qualify_n7_002/frozen_route.npz`. It applies
12 seconds of hanging feedback and axial-frame alignment, a 4.56-second
repaired swing and world-coordinate upright feedback. Fresh final seeds are
20271816..20271847. The raw first qualification failed 31/32 and is retained
separately; those seeds were reused as development only.

For eight links use `evidence/qualify_n8_001/frozen_route.npz`. It applies
12 seconds of hanging feedback and measured axial-frame alignment, a 3.94-second
repaired swing and world-coordinate upright feedback. All phases fit inside
30 seconds. Fresh qualification seeds are 20261916..20261947.

Start a count ladder or optimize a swing route:

```powershell
.\.venv\Scripts\python.exe -m spatialpole.experiment balance --max-links 8 --out output/balance_new
.\.venv\Scripts\python.exe -m spatialpole.experiment swing --links 2 --iterations 100 --route evidence/swing_n1_001/route.npz --out output/two_link_new
```

## Physics and scoring

The chain is 3m long and 1kg in total, with uniform links and a 1kg cart.
RK4 integrates at 200Hz; forces change at 50Hz. The **total XY force norm is
limited to 80N**, including after float32 rounding. The cart stays inside a
6m × 6m square. The plane is a visual reference without contacts; links can
swing below it. There are no hidden supports, locked joints or joint motors.
Armature is 0.0005 per rotational DOF; total joint damping is 0.0075 per axis
distributed across links. Capsule radius is 0.025m. These are explicit benchmark
choices, not an equivalence claim to the old hinge model.
Both cart slides have zero armature. Altered-mechanics diagnostics are retained
as development components; the packaging/publishing guards reject altered
configurations as benchmark results.

Success requires a complete valid 30-second episode, no boundary exit even at
physics substeps, and every rod's **world** tilt below 0.15rad for at least five
consecutive seconds. Twist is free and is not scored as tilt. Quaternions are
used for joint orientations; tangent rotations are used for feedback differences.

Starts are hanging (root rotated pi about Y, remaining joints identity), with
independent local rotation perturbations up to ±0.02rad on all three joint axes,
velocities up to ±0.04, and cart position noise ±0.05m. Local joint noise does
not mean every cumulative world tilt is independently bounded by 0.02rad.

## Method and evidence

Native MuJoCo finite differences feed iLQR trajectory optimization. A finite
feedback table tracks the swing, then switches to an upright discrete LQR
controller. The first two counts switch at eight seconds; three uses the
conditioning and timing described above. Cyclic unactuated twist coordinates are excluded from
the upright Riccati design; this does not add a twist actuator or constraint.
The optimizer uses smooth force derivatives. Qualification uses actual float32
forces and complete nonlinear episodes, followed by independent open-loop replay.

The released one-link controller was synthesized with the initial approximate
cost gradient. The current optimizer includes a tested SO(3) cost-gradient
correction for later experiments. Frozen synthesis source and gains are retained
in its evidence directory; qualification does not regenerate its gains.
Two-link synthesis uses the corrected cost gradient and the world-coordinate
stationary handoff. The failed baseline and failed local-coordinate handoff
remain in the evidence alongside the successful continuation.
Three-link synthesis starts from an earlier planar route proposal, repairs it
on the new spatial model, and uses smooth world-direction/rate running costs
plus a circular force-constrained backward pass. Direct policy transfer failed.
The conditioned launch was developed on 44 starts, then passed a new 32-start
cohort. Axial-frame alignment rotates reference quaternions, velocities and gains
consistently; native physics and feedback equivalence are tested at 1,3,6 links.

Four-link synthesis identifies the historical damping and slide-armature
differences, reproduces a nominal motion on that altered plant, and uses its
actual motion as a geometric target. Native affine tracking repair on the exact
benchmark cancels the local dynamics defects, with bounded XY commands and
state/control increment regularization. The first checkpoint was selected
before actual outcomes, passed nominal plus 12 development starts, then passed
all 32 reserved starts. A later lower tracking-cost policy also passes the
development cohort but is not substituted into this release. Direct transfer,
mechanical continuation and unsuccessful defect-aware alternatives are retained.

Five links uses the same fixed-target native repair, directly from a lifted
historical nominal proposal. Its old planar reference reaches 3.33m and was
only valid on a wider rail. The new optimizer repairs both the dynamics and
cart excursion under the spatial benchmark's unchanged 3m boundary. The final
31-iteration controller passes nominal, 12 development starts, and 32 reserved
starts. All 32 initial states are distinct. Maximum cart excursion over final
qualification is 2.731121m; no wider rail or extended episode is used.

Six-link native repair passes nominal but fails four of 12 raw noisy starts.
The frozen hanging/gauge wrapper passes nominal plus 44 development starts,
then all 32 fresh qualification starts. Maximum final-cohort cart coordinate
is 2.729448m and all holds are 14.06s. The original raw failures are retained.

Seven's first repair stalls; a stronger initial gap penalty closes its dynamics
gaps. That raw policy passes all 12 development starts but fails its first
reserved cohort at 31/32. The conditioned policy passes nominal plus 76 noisy
development starts, including that old cohort, then all 32 NEW final starts.
All final holds are 13.48s; maximum cart coordinate is 2.366483m. The spatial
freedom audit adds independent native dynamics and each-joint regression checks,
and its controlled six-link XY-loop demonstration deliberately exercises both
cart axes while balancing. That release had 44 passing tests.

Eight-link native repair passes nominal but fails all 12 raw noisy starts.
Its conditioned controller passes nominal plus all 44 development starts,
then all 32 fresh qualification starts, holding upright for 14.20s each.
Maximum final-cohort cart coordinate is 2.726041m. The physical-freedom checks
now include the compiled eight-link model, persistent native multiaxis replay,
X/Y symmetry and each joint's transverse motion/spin: 48 tests pass.

Nine-link native tracking repair also passes nominal but fails all 12 raw
noisy starts. Its preplanned 18s hanging/gauge wrapper passes nominal plus
all 44 development starts, then all 32 fresh qualification starts. Preparation,
the 3.94s swing and upright feedback fit inside the same 30-second episode;
every final hold is 8.18s. Maximum cart coordinate is 2.716248m, with a minimum
substep boundary margin of 0.283506m. All 32 initial states are distinct.
The native physical-freedom checks now include nine links: 52 tests pass.

Ten-link native repair closes the historical proposal's dynamics gaps and
passes nominal with 22.12s hold, but all 12 raw noisy starts fail. Its preplanned
16s hanging/gauge wrapper passes nominal plus all 44 development starts and
then all 32 fresh qualification starts. The 8s route and upright feedback fit
inside the same 30s episode; every final hold is 6.12s. Maximum final-cohort
cart coordinate is 2.022729m, minimum substep boundary margin 0.977271m, and
all 32 initial states are distinct. Ten-link native freedoms are covered in
the passing 56-test suite. Native instability warnings arose in rejected
optimizer search trials; accepted actual episodes report no physics warning.

Eleven-link development exposed a synthesis bug: the active XY force circle's
value Hessian omitted its constraint-curvature term. The correction is covered
by an independent closed-form control regression. Development also adds an
optional residual-QR backward calculation and an upright axis-separated
Riccati design to reduce high-count numerical errors. The current suite has
85 passing tests in the eleven-link release. The frozen ten-link controller
still reproduces its first held-out state/action/tilt tape byte-for-byte after
these synthesis changes; the native simulation and acceptance gates are intact.

Eleven links combines full spatial sparse shooting, full affine QR offsets and
a finite upright continuation, adapting ideas reviewed in `corben-marc-hybrid.md`.
The raw finite controller passes nominal with23.50s hold but fails all12 noisy
starts. Static launch alignment with16s/18s/18.4s preparation passes only
27/45,40/45,43/45 development episodes. The selected18s policy additionally
aligns reference axial frames at every observed feedback step. It passes nominal
plus all44 noisy development starts, then32/32 fresh reserved starts
(20262216..20262247), each full30s with5.50s hold and exact cold replay.
It executes600 commands of a1500-command value design; that design horizon adds
no episode time. Maximum final-cohort cart coordinate2.072389m, minimum
substep boundary margin0.927587m,32 distinct initial states. Physical ball-joint
twist/rates remain free. See [the transfer review](research/hybrid_2d_review.md)
for matched results and the moving-target/native-search ideas still unproven.

Reproduce eleven using `evidence/qualify_n11_001/frozen_route.npz` with `evaluate`.
Its saved affine/online-alignment flags select the complete frozen execution law.

The twelve-link release has96 passing tests, adding twelve-link physical freedoms and
an optional finite-QR upright value backend after the stationary DARE design
became numerically indefinite. Historical donor coordinates now decode through
their explicit matrix, with hanging-start/native-direction/physical-tape checks;
the twelve-link donor uses different coordinates than eleven. An invalid earlier
import/repair is retained with correction notes. Corrected twelve transfer matches
its stored source tape exactly but fails direct control. Native repair002 closes
the proposal gaps; finite-affine execution passes nominal23.50s hold but fails
raw noisy starts. The18s prepared online-gauge policy then passes45/45
development episodes and32/32 NEW reserved starts
20262316..20262347, every full30s with5.50s hold and exact replay. Maximum
cart coordinate1.914864m, minimum substep margin1.085024m, maximum delivered
force norm62.186687N. All32 initial states differ. Reproduce using
`evidence/qualify_n12_001/frozen_route.npz`:18s preparation, a9s swing and3s
upright tracking are a600-command prefix of the full1500-command design.
Measured physical spin and all ball axes remain free. The frozen eleven-link
first-heldout tape still reproduces byte-for-byte after the synthesis changes.

The thirteen-link release has131 passing tests, including thirteen-link native freedoms,
world rotation, gauge/observable tracking, material transfer and causal native
launch admission. The first thirteen donor reaches30/45 development passes;
an earlier-arrival material donor reaches44/45 at19s preparation. A fixed-1mm
Y-reference offset also reaches44/45, with a different failed start. A new
controller forecasts both full continuations from the observed launch and
selects the first complete passing forecast. Its pilot passes nominal and both
reused failures, then all45 development episodes, with byte-identical
forecast/actual suffixes. Frozen qualification passes32/32 NEW noisy starts
20262416..20262447, each full30s/5.16s hold/no warning/exact cold replay.
Every chosen forecast states/actions/raw forces/tilts equals the actual suffix.
All32 choose profile0; development also exercises profile1. The full law always
forecasts both candidates. Inclusive524800 native steps comprise384000 actual+
cold proof and140800 predictor steps. Maximum recorded-command cart coordinate
1.883801m, minimum substep boundary margin1.115844m, maximum delivered force
norm65.984695N;32 distinct initial states. Prediction cost is recorded, with no
hardware real-time guarantee. The full video has1501 frames/50fps and includes
19s preparation,9s swing and2s upright tracking, a550-command prefix of the
original1500-command design. Fourteen is next. The frozen twelve first-heldout
tape still reproduces exactly. See the hybrid-document review for the useful
finite-affine, material-transfer and native-admission ideas. Thirteen is public
at source79861ded77cd7c0ca8203bab6621398430c0b2de; all4589 staged hashes,
direct/prior ZIP coverage and public asset digests/tag verify.

Current lab212 tests include fourteen-link native freedoms and an optional
high-precision spectrum diagnostic. Ordinary float64 eigenvalues misleadingly
report an unstable radius for the saved fourteen upright linear design;
80/120-digit checks of exact input A/B/K agree on stable nativeX/Y radii
.988501/.988516. Both diagnostic results are retained, and gains/physics stay
unchanged. This does not establish nonlinear fourteen-link swing-up. Strict
fourteen transfer passes stored-tape/geometry checks but fails direct control
at3.66s. Full spatial repair completes20 updates, but its actual baseline fails
at6.58s with.26s hold. Full affine feedback reaches1.84s, while reflection-
averaged feedback passes nominalfull30s/24.16s hold. Its prepared full noisy
gate passes only4/45, so fourteen is unqualified. Physical-cost, timing, offset,
stronger-hanging and reflection-consistent-model probes retain their failures.
The tests include independent dense optima, full/block QR agreement and
unrestricted native reflection motion; no plant axis or joint is projected.
A moving-target material proposal from the document's pinned external17 source
has separate fixed/free final-node native repairs. The free-end proposal has
tiny native gaps; its prepared moving controller passes4/6 reused starts.
Bounded native action search and redesign around an actual physical reference
both remain4/6. All selected one-step predictions/reexecutions match native actual
successors exactly, including final commands. Only geometry/force guesses
transfer; source gains/physics/results do not. Finite design and execution
clocks remain separate.

A new axial-quotient planning design matches the per-command reference-frame
alignment. It removes14 arbitrary axial ORIENTATION phases from the value
model, retaining all44 native velocities including spin; the actual plant
still has all88 tangent coordinates and unrestricted ball joints. Analytic
frame derivatives, native gauge-flow equivalence and an independent condensed
affine optimum are tested. On the SAME moving reference its pilot improves to
6/6, followed by43/45 full development passes, every success30s/5.16s hold and
all cold replays exact. Two noisy starts still fail, so fourteen is unqualified.
Nine reference offsets per failed launch rescue neither; the original full-state
law also fails those two starts. Stronger preparation is6/8, including one
prelaunch failure. Online preparation and physical-reference quotient redesign
also pass6/8, with the same two remaining failures; neither advances the gate.
One-step native correction using quotient values, four-command native lookahead
and full-spin world-coordinate feedback each remain6/8 on the expanded pilot.
Lookahead retains every candidate path/cost/guard: selected FIRST successors,
sampled complete native paths and a full nominal reexecution match exactly.
Its later predicted plain-feedback actions are not claims about later actual
search actions. A zero-integration diagnostic finds large local prediction
remainders before the failed captures; it cannot uniquely separate derivative,
nonlinear/chart and arithmetic effects. All null results and compute are retained.
Reserved20262516..47 is untouched; no14 release or record is claimed.
Published thirteen retains its original131-test source and immutable archive/runtime.

A rereview of `corben-marc-hybrid.md` prioritizes feedback-aware repair of the
entry and short upright tail. A new matched world-tilt/body-rate tracking cost
on the original moving quotient reference regresses to5/8; the original pilot
is6/8. A different source16-derived moving reference gives4/8 after native repair;
an eight-command launch delay and refined input derivatives each remain6/8.
All actual cold replays are exact. These candidates are retained as failed
development interventions; the43/45 incumbent stays unchanged. The detailed
recommendations and boundaries are in `research/hybrid_2d_review.md`.

The next hybrid intervention produces useful new candidates: full spatial
suffix repairs from two recorded actual entries each rescue their own failed
start, with every preceding22.2s physical prefix byte-identical. A causal bank
of the original controller and those two repairs forecasts complete continuations
from the observed launch and selects the first full pass. It passes8/8 pilot
starts; selected states/raw forces/delivered forces/tilts match execution through
the final command, and independent full-policy reruns match all numeric bytes.
That three-controller bank passes45/45 development but fails its first reserved
cohort31/32, so fourteen remains unqualified. The failed negative-twist start is
retained. A new five-profile family adds physical Y-reflections of the two
spatial suffix laws; it passes9/9 development pilots including that explicitly
reused failure. Its new full45 gate is in progress. A different untouched32-start
cohort is required before acceptance; the old cohort cannot become fresh again.

Each experiment freezes source, arguments, runtime and the pre-experiment memo,
then records optimizer history, states, raw/delivered forces, tilts, counts,
timing, failures and replay checks. See `research/plan.md` and `PROGRESS.md`.
The final 32-seed cohort at each count was disjoint from its development data. This establishes performance
under that reset distribution; arbitrary disturbances and indefinite balance
remain untested. MuJoCo 3.3.5 / Python 3.11 / Windows was used for this release.

Physics reference: [MuJoCo computation documentation](https://mujoco.readthedocs.io/en/3.3.5/computation/).


## Fourteen-link qualified result

The fixed five-profile controller completes46/46 distinct development starts,
then32/32 NEW untouched reserved starts20262616..47. Every actual episode lasts
30s with5.16s strict upright hold, no physics warning and exact cold replay.
Minimum200Hz plane margin1.134046m; maximum recorded cart coordinate1.865605m.
The full video is the first heldout episode20262616, not a selected development
success. The prior three-profile qualification31/32 remains a failed result.

The original moving quotient controller supplies most launches. Two full-spatial
suffix repairs and their reflected force/value laws supply alternatives from
route150 onward; all profiles preserve the original first22.2s physical behavior.
At the observed19.2s launch, owned complete forecasts in fixed order choose the
first full passing policy. There is no seed lookup, state reset or joint lock.
All selected states, raw/delivered forces and tilts match actual execution
byte-for-byte, including command1499. All44 velocity coordinates, including
physical spin, remain in74D value planning and the full88D native plant.

The9s bridge and finite upright tail execute540 commands of the original700
command design. Qualification uses462916 inclusive native steps, including all
forecasts and cold proofs;28 starts select original, one patch431 and three its
reflection. Separate development exercises the patch437 profile. Current212
tests cover native spatial freedoms, independent affine/quotient optima,
reflection, owned forecasts/final transitions and multipart/LFS evidence hashes.
This is empirical reset-noise performance; no arbitrary-disturbance, indefinite
balance or hardware20ms-deadline claim is made. The next count to solve is15.

## Ten-link immediate swing-up

The immediate variant passes46/46 development starts, then32/32 new reserved
starts20263416..47. All30-second episodes hold upright for22.12 seconds with
zero hanging preparation. A fixed affine-first/native-corrected-second feedback
bank chooses by complete owned forecasts from the observed state and verifies
actual execution through the final transition. All32 physical velocity
coordinates remain. The physical plant and task limits are unchanged.

The prior affine family failed31/32 on20263216..47; that failed cohort stays
consumed development data. The new range is separate. All forecasts cold-replay
exactly; full frozen execution and candidate audits pass, with294 regression
tests. The first new qualification episode is the full50fps video.

[Release, video and complete source/evidence](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-10-link-immediate-v1) ·
[Study and reproduction](research/immediate_10_link_20261017/README.md).
Fourteen remains the separate prepared record.

## Eleven-link immediate noisy swing-up

The 2D immediate variant now passes **32/32 fresh noisy starts**, after 45/45
development cases, with zero hanging preparation and at least
23.50 seconds upright in every full 30-second episode.
The full-state controller plans offline before applying feedback during motion.
The fourteen-link conditioned comparison and ten-link immediate result remain preserved.

[Full video and release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-11-link-immediate-v1) · [Method, source and reproduction](research/immediate_11_link_20261018/README.md).

## 12-link immediate noisy swing-up

[12-link 2D immediate noisy swing-up](research/immediate_12_link_20261018/README.md): 32/32 fresh
starts pass, with zero hanging preparation and at least
23.50 seconds upright in full 30-second episodes.
[Video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-12-link-immediate-v1/2d-cartpole-12-link-immediate.mp4) · [Release and reproducible evidence](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-12-link-immediate-v1). Planning is
offline from the observed initial state; the physical motion begins immediately.
