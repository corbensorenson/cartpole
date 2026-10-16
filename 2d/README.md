# 2D cartpole / spatial ball-joint chain

A cart moves over an XY plane, driven by two forces. Every link has a passive,
unlimited ball joint: it can tilt in both directions and twist about its axis.
This is a separate benchmark from the one-dimensional cart and hinge chain.

**One- and two-link hanging-start swing-ups are qualified:** 32/32 fresh noisy
3D starts at each count, in addition to nominal and development checks. Episodes
run 30 seconds; one link holds for 21.74s and two for 21.68s. All delivered-action
tapes replay exactly. Both cart axes are used. Full-state simulation feedback,
not a learned policy or a hardware demonstration.

[Full one-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-1-link-v1/2d-cartpole-1-link-swingup.mp4)
· [GitHub release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-1-link-v1)

[Full two-link video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-2-link-v1/2d-cartpole-2-link-swingup.mp4)
· [Two-link release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-2-link-v1)

| Milestone | Current evidence |
|---|---|
| One-link hanging-start swing-up | 32/32 final qualification; full video |
| Two-link hanging-start swing-up | 32/32 final qualification; full video |
| Three-link hanging-start swing-up | In development |
| Near-upright balance, 1–3 links | 12/12 local perturbed starts at each count |
| Near-upright balance, 4 links | 9/12; incomplete, not accepted |

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
feedback table tracks the swing, then switches at eight seconds to an upright
discrete LQR controller. Cyclic unactuated twist coordinates are excluded from
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

Each experiment freezes source, arguments, runtime and the pre-experiment memo,
then records optimizer history, states, raw/delivered forces, tilts, counts,
timing, failures and replay checks. See `research/plan.md` and `PROGRESS.md`.
The 32 final seeds were disjoint from development. This establishes performance
under that reset distribution; arbitrary disturbances and indefinite balance
remain untested. MuJoCo 3.3.5 / Python 3.11 / Windows was used for this release.

Physics reference: [MuJoCo computation documentation](https://mujoco.readthedocs.io/en/3.3.5/computation/).
