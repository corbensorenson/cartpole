# Immediate swing-up and disturbance recovery â€” 2026-10-17

**Five spatial links recover during swing-up without hanging preparation.**
In the fixed eight-start development test, feedback handles a previously
unobserved cart-force bias of (-8,+8) N during 3.0â€“3.2 seconds. Every episode
completes 30 seconds and holds every rod within 0.15 rad of upright for 21.70
seconds. The uninterrupted video starts at the original noisy hanging reset.
This is a development disturbance result. Fourteen remains the highest qualified
and published count; no fifteen-link qualification is claimed.

[Video and evidence release](https://github.com/corbensorenson/cartpole/releases/tag/2d-mid-swing-recovery-v1)

## What was tested

The original spatial plant stays fixed: two cart forces, unlimited passive ball
joints including axial twist, 3 m / 1 kg chain, 1 kg cart, 80 N total force,
3 m square boundary, 50 Hz control and 200 Hz RK4. No joint torque, support,
state reset, shortened episode or hidden preparation was introduced.

The disturbance is added to the controller's raw XY request before the common
80 N disk delivery. This is actuator-force disturbance and physical reset noise;
there is **no observation noise**. Exact native cart/joint state is available.
It is one specified disturbance, not arbitrary disturbance robustness.

The fixed development cohort is nominal seed 20261016 with zero noise, then
20261116, 20261117, 20261118, 20261120, 20261121, 20261431 and 20261437 at 0.02
joint orientation noise, original cart-position and full native velocity noise.
These are reused development starts, not eight fresh qualification samples.

| Count / controller | Feedback, no gust | Feedback, gust | Open loop, no gust | Open loop, gust |
|---|---:|---:|---:|---:|
| 1, original immediate motion | 8/8 | 8/8 | 8/8 | 0/8 |
| 5, original immediate repaired motion | 8/8 | 8/8 | 0/8 | 0/8 |
| 6, original immediate repaired motion | 8/8 | 0/8 | 1/8 | 0/8 |
| 6, online axial reference alignment | 8/8 | 0/8 | 1/8 | 0/8 |

The one-link no-gust open-loop arm was added after the first three arms finished;
the five- and six-link four-arm protocols were registered before their execution.
At one link the factorial comparison isolates the effect of the gust. At five
links open-loop replay also fails without the gust, so that ablation alone does
not isolate gust sensitivity. Feedback succeeds with the gust at both counts.

Six-link alignment changes reference coordinates, never physical joint spin.
It does not improve this disturbance gate. Its ordinary eight-start pass also
does not replace the larger historical development cohort, which had failures.

## Corrections that did not solve the harder cases

At six links a native 64-command correction from the measured state at 3.20 s
reduces its exact tracking objective from 14912.30 to 969.88, retaining 0.369 m
forecast rail margin. All 64 selected transitions execute exactly, but the
continuation fails at 5.28 s with only 0.06 s upright. A valid local forecast
and a lower objective do not imply complete recovery.

A second six-link method estimates uncommanded force from consecutive measured
cart/joint states and the previous **requested** force. It never receives the
actual delivered force or the disturbance schedule. The exact native model
inverts the measured successor cart velocities, then compensates the inferred
bias on the next command. All full joint velocities enter prediction.

This observer identifies the bias within 0.001 N on every disturbed start after
one 20 ms control interval, and detects its removal one interval later. Its
inactive no-gust episodes match the original controller byte for byte. Despite
correct inference, all eight disturbed episodes still fail. Prompt force
estimation alone cannot undo the chain motion caused before compensation.

At fifteen links, removing the 19.2 s preparation gives 0/8 full-task passes.
The exact noiseless start swings up and holds 8.38 s, then fails after the
700-command motion ends; all seven noisy starts fail much earlier. Two matched
native correction methods at 3.0 s also fail the full episode. The prepared
fifteen-link incumbent remains 2/8 development passes and is unqualified.

The fifteen-link comparison's original helper missed an original-controller
cache switch when resuming after an overridden segment. Both completed searches
were preserved and independently audited; saved plans were reexecuted with the
cache bookkeeping fixed. An original inline claim of 64 observed selected
commands was corrected to the 63 actually executed before boundary termination.
The other arm executed all 64. Neither arm passes; the original frozen failure
and the explicit correction remain in the evidence bundle.

## Where video and joint sensors could help

[V-JEPA 2](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) combines
video pretraining with action-conditioned robot training and predictive planning.
Its [official implementation](https://github.com/facebookresearch/vjepa2) is a
useful architectural reference; its robot checkpoint has not been shown to
model this spatial chain. No JEPA checkpoint was downloaded or trained here.

For this experiment, a useful model would consume recent video, joint
orientations/rates, cart state and past commands, then predict the different
futures caused by candidate XY forces. Joint rates matter: visual motion alone
does not fully reveal axial spin in these axisymmetric rods. Exact simulator
state currently supplies that information, so a video representation must show
measured benefit over the state-based control baseline.

The hybrid document's section 20 recommends action-sensitive branches, whole
episode splits, disabled/shuffled-action controls and complete native outcomes.
It also records a failed small 17-link JEPA probe; that is a narrow negative,
not proof that learned proposals cannot help. Those evaluation rules transfer.
Our retained joint/cart/action records and native candidate branches provide a
starting dataset, but are not a trained world model or an independent test set.

The most useful next alternatives are:

1. **Robust motion design from reset.** Optimize the swing and feedback together
   over noisy starts and mid-swing disturbances, with explicit recovery and rail
   margin, instead of assuming a quiet launch and repairing a late snapshot.
2. **Observed-state, phase-adaptive continuation.** Select a reachable moving
   target from the current chain motion rather than returning to an absolute
   reference time after a disturbance. Test full continuation, not local cost.
3. **Action-conditioned video plus joint prediction.** Use it to propose/rank
   corrective forces or approximate a residual model. Test held-out whole
   episodes/phases, action-disabled/shuffled controls, native validation and
   total collection/training/inference cost. Prediction loss alone is insufficient.

The six-link observer result makes the priority clear: better recovery motion
and feedback are needed alongside estimation. A learned predictor adds no joint
actuation and cannot itself create recovery margin.

## Evidence and reproduction

The development release contains the full five-link video and a separate evidence
ZIP with all positive/negative runs, raw and delivered forces, state/tilt tapes,
50 Hz joint/cart observations, native query ledgers, frozen drivers/source,
registered plans, tests and per-file SHA256 inventory. Every actual run has an
independent cold replay. Sensor/action timing and native spots are audited.
Every force-observer forecast is independently reproduced byte for byte.

From the extracted bundle root, create the environment from requirements-lock.txt
and install this package with `pip install -e .`. For example:

```powershell
python scripts/probe_mid_swing_gust.py --route output/n5_mid_swing_gust_001/input_route.npz --include-openloop-control --out output/reproduce_five
python scripts/audit_swing_recovery.py --run output/reproduce_five
python scripts/probe_force_observer.py --route output/n6_force_observer_001/input_route.npz --out output/reproduce_observer
```

Use OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1 and MKL_NUM_THREADS=1. The frozen
per-run source and driver are authoritative for earlier variants. Current source
adds optional extensions while preserving their defaults. The full lab suite
passes 277 tests, with the two existing Gym observation-bound warnings.
