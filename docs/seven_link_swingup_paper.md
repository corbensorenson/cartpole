# Seven-Link Cart-Pole Swing-Up

## A Settled-Launch Two-Expert Controller That Passes the Canonical Gates

**Status:** released canonical 20/100-gate result; public fresh-clone audit passed
**Date:** 2026-09-12
**Target:** uniform seven-link MuJoCo cart-pole, hanging-start swing-up and hold

## Abstract

This report documents a reset-free seven-link swing-up and hold controller
that passes the canonical noisy hanging-start gates in this repository. The
method uses a deterministic hanging-equilibrium conditioning phase to bring
the cart back to the center and dissipate link motion, a finite-horizon
Box-FDDP feedback route for swing-up, and a terminal LQR for capture and
maintenance. The same frozen chain reaches 20/20 success and 100/100 success
from the declared noisy initial distribution. A held-out noisy-start video
completes for 30.00 s with zero resets and full seven-link framing.

The result is a released canonical benchmark result. The evidence JSON records
the exact clean tracked commit, runtime, configuration, MuJoCo XML, controller,
and manifest hashes. A fresh-clone audit runs all 78 tests, the artifact
verifier, and the published checksums. External-rule comparison remains open.
The method intentionally spends the first 10 seconds conditioning the initial
state; this is part of the policy, not a reset or an omitted warm-up.

## 1. Benchmark and admissibility

The canonical target is the uniform seven-link plant in
`configs/swingup7_uniform.yaml`:

| Quantity | Value |
|---|---:|
| Links | 7 |
| Total link length | 3.0 m |
| Total link mass | 1.0 kg |
| Cart mass | 1.0 kg |
| Rail | +/-3.0 m |
| Force limit | +/-80 N |
| MuJoCo timestep | 0.005 s |
| Frame skip | 4 |
| Policy rate | 50 Hz |
| Episode length | 30 s |
| Upright threshold | max absolute angle < 0.15 rad |
| Required hold | 5 s |

The canonical initial distribution is a hanging start with relative angles
`[pi, 0, 0, 0, 0, 0, 0]` plus independent Gaussian angle and velocity noise
of scale 0.05. The final evaluation uses this distribution directly. An exact
noiseless hanging state remains in the repository as a development diagnostic,
but it is not used for the 20/100 gate or the final public video.

The repository's completion contract requires all of the following before a
record or competition-standard claim is made:

1. At least 80% success over 20 held-out canonical noisy episodes. **Passed:
   20/20.**
2. At least 90% success over 100 held-out canonical noisy episodes. **Passed:
   100/100.**
3. A reset-free 30-second video from a held-out hanging start.
4. Published controller artifacts, hashes, metrics, and fresh-clone replay.

## 2. Controller architecture

The runtime chain has a deterministic conditioning prelude followed by two
experts. It has no simulator-state handoff:

1. **Hanging conditioning.** For the first 10 seconds, an LQR linearized at
   the hanging equilibrium drives cart position and velocity toward zero and
   damps the link perturbations. It does not reset or replace qpos/qvel. The
   measured settled cart position is then used only to translate the route's
   nominal cart-position coordinate, exploiting the cart's translational
   symmetry.
2. **Swing expert.** A finite-horizon Box-FDDP trajectory is executed with
   time-indexed state feedback. The nominal route has 228 controls, or 4.56 s
   at the 50 Hz policy rate. The feedback correction is computed in the
   dimensionless absolute-coordinate state used by the capture evaluator, with
   tracking correction scale 2.0.
3. **Capture expert.** After the route horizon, the controller switches to a
   terminal LQR around the upright equilibrium. The successful replay uses
   LQR scale 1.0. The switch happens inside the same MuJoCo episode: qpos,
   qvel, simulator time, and hidden simulator state are not reset or replaced.

The conditioning prelude is the key robustness change. Passive zero-force
waiting reduced link motion but allowed cart drift and did not pass. Active
hanging LQR plus stronger route tracking made the same route tolerate the
canonical initial perturbations. The capture switch was deliberately deferred
until the route horizon because earlier LQR handoffs were unstable while the
chain still carried swing energy.

## 3. What actually worked

### 3.1 Rebuilding the warm-start states in the correct coordinates

The first FDDP attempts consumed iLQR artifacts whose nominal states encoded
cumulative absolute angular rates, while the FDDP evaluator expected relative
hinge-rate coordinates. The controller looked numerically plausible but was
not tracking the intended state trajectory. Rebuilding the initial states from
qpos/qvel with `--rebuild-initial-states` removed that schema mismatch.

This was a contract bug, not a reward-tuning detail. For this system, a
trajectory optimizer cannot be trusted until the state convention is checked
at the artifact boundary.

### 3.2 Making the endpoint a quiet equilibrium, not merely an upright crossing

The successful route used a terminal state weight of `100000` and a rail soft
limit of `2.80 m` inside the optimization. The optimized terminal state had a
small state norm and nearly zero relative hinge velocity. This gave the LQR a
real capture state instead of asking it to arrest an energetic chain after the
angle threshold had already been crossed.

The useful objective was therefore not "reach upright quickly." It was "reach
an upright, centered, low-velocity terminal state that the local capture law
can actually hold." The low-momentum measurement is a ranking and training
signal here, not a universal gate that would reject every valid nonlinear
capture.

### 3.3 Using LQR scale 1.0 only after the route is quiet

The terminal replay with LQR scale 0.5 failed after handoff. Replaying the same
optimized route with LQR scale 1.0 produced the successful hold. This isolates
an important interaction: lowering a local stabilizer gain is not necessarily
more robust when the plant has seven coupled links and the terminal state is
only approximately exact. The correct gain must be measured from the actual
linearization and tested in the full nonlinear MuJoCo plant.

### 3.4 Verifying the result with the actual plant and a reset counter

The evidence replay uses `NLinkCartPoleEnv`, the canonical force and rail, and
records every transition. The video metadata records the generated XML hash,
config hash, controller hash, runtime versions, frame count, reset count, and
final termination information. A separate OpenGL renderer could not initialize
CoreGraphics in the current headless execution context, so the delivered video
is a state-faithful 2D rendering of the exact MuJoCo trajectory rather than a
claim that a rendered approximation was the physics source.

### 3.5 Waiting for a normalized hanging state

The decisive robustness experiment was to make the wait an explicit controller
phase. A finite-difference LQR around the hanging equilibrium uses the gain
stored in `seven_link_swingup_manifest.json` to damp link motion and return the
cart to its expected center. After 10 seconds, the route launches from the
measured state rather than overwriting it. The route's nominal cart coordinate
is translated by the settled cart position, and its feedback correction is
scaled by 2.0. This combination passed the 20- and 100-episode gates.

The important negative controls were also preserved. Zero-force waiting did
not pass, and the original tracking scale 1.0 remained phase-sensitive after
settling. Tracking scale 2.0 passed the full cohort; scale 4.0 and 8.0
destabilized the same representative seed. The successful setting is thus a
measured narrow operating point, not an arbitrary claim that more gain is
always better.

## 4. Canonical noisy-start result

The frozen controller route is
`runs/swingup7_uniform/seven_link_release_controller.json`; its historical
optimization trace is retained separately for research provenance. The
complete chain definition is
`runs/swingup7_uniform/seven_link_swingup_manifest.json`.

| Gate or metric | Result |
|---|---:|
| 20-episode success gate | 20/20 = 100% |
| 100-episode success gate | 100/100 = 100% |
| First upright, all evaluated episodes | 14.54 s |
| Conditioning prelude | 10.00 s |
| Swing route | 4.56 s / 228 policy steps |
| Upright hold remaining after first upright | 15.48 s |
| Maximum cart excursion across 100 episodes | 2.3708 m |
| Episode terminations | 100 time limits, 0 rail failures |
| Resets inside episodes | 0 |

The public held-out video is
`runs/swingup7_uniform/seven_link_swingup_success.mp4`. It uses seed 50732,
outside the disjoint 20- and 100-episode cohorts beginning at 30732 and 40732.
Its metadata is
`runs/swingup7_uniform/seven_link_swingup_success.video.json` and reports 1,500
frames, 50 fps, 1280x720 resolution, zero resets, a canonical noisy hanging
start, and successful time-limit completion. The renderer dynamically fits
the chain to the frame so all seven links remain visible during the hold.

### 4.1 Stress characterization outside the claim

`runs/swingup7_uniform/robustness_sweep.json` applies the same 20 paired seeds
to 15 non-canonical perturbations without retuning the route or LQR gains. The
controller retained `20/20` success when the initial angle/velocity noise
standard deviation was increased from `0.05` to `0.10` and `0.20`, and with an
8-second conditioning phase; it scored `19/20` with 6 seconds. It scored
`0/20` under each tested 40 N or 20 N force limit, +/-5% total mass or length,
halved or doubled joint damping, 0.01-standard-deviation measurement noise, and
20 ms or 40 ms control delay. These failures do not change the frozen
canonical gate, but they sharply limit the appropriate interpretation: the
released controller is not robust to plant mismatch or latency.

## 5. What did not work

The following branches were negative controls or were superseded before the
settled-launch configuration:

- Terminal DARE value variants did not improve the route.
- FDDP with the unconverted iLQR nominal states used the wrong rate
  coordinates.
- Early or weak LQR handoff failed to capture the moving chain.
- Static local LQR, online MPC, global CEM, tail CEM, passive waiting, and
  phase-adaptive route tracking did not recover noisy hanging starts.
- Rail extensions and morphology gradients involving link length, mass,
  damping, and friction were useful training-wheel diagnostics but did not
  transfer to a uniform canonical hold.
- PPO curriculum probes and handoff-state capture probes did not pass their
  repeated evaluation gates.

The most informative negative control is
`runs/eval_swingup7_fddp_two_expert_canonical20.json`, which uses the route
without the settled-launch conditioning and scores 0/20. Zero-force waiting
also remained at 0/20. The final positive controls are
`runs/swingup7_uniform/eval_swingup7_20.json` and
`runs/swingup7_uniform/eval_swingup7_100.json`; both use the same frozen
settled-launch manifest and score 100%.

## 6. Reproduction

From a fresh clone, create the planner environment and regenerate the complete
release bundle:

```bash
make setup-aligator
make release-swingup7
```

The release target refuses a dirty tracked tree, then generates the disjoint
20- and 100-episode cohorts, the independently seeded state-faithful video,
15 paired-seed non-canonical stress scenarios, `SHA256SUMS`, and the verifier
report. Individual artifacts can be regenerated with:

```bash
make eval-swingup7-20
make eval-swingup7-100
make render-swingup7
make verify-swingup7
```

## 7. Conclusion and next experiment

The project has now crossed the canonical control threshold: the seven-link
chain swings up from the declared noisy hanging distribution and holds in
100/100 evaluated episodes. The decisive changes were correct state-coordinate
conversion, a terminal quieting objective, an explicit active hanging-state
conditioning phase, cart-centered nominal translation, doubled route
feedback, delayed LQR handoff, and a sufficiently strong terminal LQR.

The seven-link trajectory and release work are complete. The remaining work is
independent third-party reproduction and exact external-rule comparison. The
defensible public description is therefore “released canonical seven-link
20/100-gate result with a hybrid settled-launch controller,” not an unsupported
universal world-record claim. The active control frontier is now eight links.
