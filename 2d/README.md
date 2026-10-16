# 2D cartpole / spatial ball-joint chain

A cart moves over an XY plane, driven by two forces. Every link has a passive,
unlimited ball joint: it can tilt in both directions and twist about its axis.
This is a separate benchmark from the one-dimensional cart and hinge chain.

**One- through nine-link hanging-start swing-ups are qualified:** 32/32 fresh
noisy 3D starts at each count, in addition to nominal and development checks.
Episodes run 30 seconds; the latest nine-link controller holds upright for
8.18 seconds. All delivered-action
tapes replay exactly. Both cart axes are used. Full-state simulation feedback,
not a learned policy or a hardware demonstration.

The benchmark permits arbitrary XY cart motion and unrestricted ball joints,
but does not require a nonplanar nominal swing. The four- through nine-link
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
| Ten-link hanging-start swing-up | Next development count | — |
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

Re-evaluate the frozen one-link controller on a new cohort:

```powershell
.\.venv\Scripts\python.exe -m spatialpole.experiment evaluate --route evidence/swing_n1_001/route.npz --episodes 32 --seed 20261216 --out output/reproduce_one_link
.\.venv\Scripts\python.exe -m spatialpole.experiment video --tape output/reproduce_one_link/seed20261216.npz --out output/2d-cartpole-one-link.mp4
```

Local development uses `output/swing_n1_001/route.npz` before publication packaging.
Output directories must be new; failed runs are preserved.

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

Each experiment freezes source, arguments, runtime and the pre-experiment memo,
then records optimizer history, states, raw/delivered forces, tilts, counts,
timing, failures and replay checks. See `research/plan.md` and `PROGRESS.md`.
The final 32-seed cohort at each count was disjoint from its development data. This establishes performance
under that reset distribution; arbitrary disturbances and indefinite balance
remain untested. MuJoCo 3.3.5 / Python 3.11 / Windows was used for this release.

Physics reference: [MuJoCo computation documentation](https://mujoco.readthedocs.io/en/3.3.5/computation/).
