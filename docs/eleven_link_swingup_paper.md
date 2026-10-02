# Eleven-link canonical swing-up and hold: experiment appendix

October 1, 2026. This is an internal repository benchmark result. The source
bundle, evaluation evidence and full video are included in this repository
and the eleven-link GitHub release. Packaging JSONs retain the provenance
and publication status at the time they were created. The ongoing campaign
targets every count through twenty. Twelve
is the next frontier, not an established result.

## Benchmark and result

One cart actuator swings up eleven passive uniform links from the canonical
noisy hanging distribution and holds them upright without a state reset.
Total link length is 3 m, total link mass 1 kg, cart mass 1 kg, force is bounded
by ±80 N, and cart position by ±3 m. MuJoCo uses 0.005 s RK4 physics and a
50 Hz control period. The unchanged 30 s episode requires every absolute
link angle to remain within 0.15 rad for at least five continuous seconds.
Relative-joint angle and velocity noise have standard deviation 0.05;
cart velocity noise has standard deviation 0.05 m/s and cart position starts
at zero. Damping, armature and capsule geometry follow the canonical config.

| Frozen evaluation | Seeds | Success | Minimum continuous hold | Maximum cart excursion |
| --- | --- | ---: | ---: | ---: |
| Noisy 20 | 211000–211019 | 20/20 | 7.50 s | 2.121357 m |
| Disjoint noisy 100 | 211100–211199 | 100/100 | 7.50 s | 2.121369 m |
| Exact replay 20 | 211200–211219 | 20/20 | 7.50 s | 2.121342 m |
| Held-out video | 211500 | 1/1 | 7.50 s | 2.121345 m |

Every accepted episode completes all 1,500 control steps and terminates at
the time limit. The video first reaches upright at 20.74 s; the uninterrupted
final hold begins at approximately 22.52 s. Upright LQR starts at 24.00 s.
The five-second success threshold is reached at 27.50 s. These are distinct
events. The video renders the actual recorded post-step state at every
control sample, uses no interpolation or second simulated controller, and
shows the full eleven-link chain. Its metadata reports zero resets after
the initial episode reset.

## Shared synthesis and executed policy

The new discovery family starts with a two-link inverse-dynamics spline
calibration and transfers its material-coordinate curve one count at a time.
It does not reuse the released seven-through-ten swing trajectories.
Missing-joint acceleration residuals guide the inverse fit; sparse least
squares repairs discrete target-plant dynamics; exact-MuJoCo Box-FDDP
refines the route and its time-varying feedback. The release follows this
family's successful ten-link route, then adds directional terminal-value
shaping for eleven. Commands, failed branches, sources and stage budgets are
in `runs/frontier_campaign_20261001/inverse_increment_experiments.json`.

The executed policy has three phases. It parks the hanging chain for 16 s
around cart target −0.05 m, executes the 8 s saved Box-FDDP route with gain
scale one, and then applies upright LQR around the same target. The hanging
regulator retains the physical absolute-angle/rate cost but uses cart-position
cost 10, cart-velocity cost 5 and normalized-action cost 1000. Capture uses
the saved canonical upright cost and normalized-action cost 1000.

At launch, the nominal cart trajectory is translated to the actual launch
position. A constant integer multiple of 2π aligns the continuous nominal
joint-angle branch with the measured branch. Neither operation changes the
physical state. The release uses a fixed route clock with no phase adaptation
and no early state-gated handoff. The frozen policy contains both equilibrium
gains, their hashes, all phase parameters and preprocessing rules; the route
artifact contains controls, nominal states and feedback gains.

The final nominal optimizer run uses 100 iterations after an offline rebuild
of the useful predecessor's feedback trajectory. The rebuild saves both
the controls actually applied and the states physically rolled out. Its
initial and final checked dynamics defects are exactly zero. It holds
upright for 23.50 s in an unparked exact-start full episode and stays within
2.071342 m. Its cost is 4866.176 and measured optimizer time 425.69 s under
concurrent CPU load. It does not meet the native convergence criterion;
acceptance rests on independently checked dynamics and complete physical
replay. Earlier inverse, restoration and refinement compute is additional
and remains disclosed in the ledger.

## Comparisons explaining the former eleven-link failure

| Development comparison | Result |
| --- | --- |
| Tiny capture-value weight with the original pre-restored initializer | FDDP closes gaps into a high-cost route with zero hold |
| More restoration, more initial regularization, or accurate physical-curve transfer | Each tested variant still fails actual capture |
| Same useful 200-iteration source; 100 more iterations, regularization 10, forced 8 s handoff; coefficient 0.01 on the capture-value matrix | 3.70 s hold, then rail failure |
| Matched trial changing that coefficient to 1 | Full 30 s success, 23.50 s hold |
| Same rebuilt swing route and twenty development seeds; default hanging cart weights 0.1/0.1, parking 17.5 s | 16/20 |
| Same seeds and default weights, parking 18 s | 19/20 |
| Same seeds, cart weights 10/5, parking 16 s | 20/20 |
| Separate hundred development seeds, cart weights 10/5, parking 16 s | 100/100 |

The terminal value is the saved local closed-loop Lyapunov state-value matrix
in dimensionless coordinates. A componentwise quiet endpoint alone does not
adequately weight sensitive arrival directions. The matched coefficient
comparison holds initializer, iteration budget, regularization, plant and
executed handoff fixed. The exact rebuild is a further consistency check,
not part of that one-variable comparison.

The launch comparison changes only the hanging regulator's cart-state
weights and duration. Reducing its action penalty from 1000 to 100 or 10
does not recover the same failing four-seed case. Increasing cart-state
weights does: the failing seed's launch tilt falls from about 7.7e−4 to
6.8e−7 rad at 17.5 s and its cart speed from 0.0093 to 8.3e−6 m/s. This
explains why “more aggressive control” and “longer parking” were weaker
proxies than measuring contraction of the launch state itself.

Eleven's improvement requires no increase in control frequency. Separate
50/100/200 Hz local component studies show modest benefits in selected
eleven/twelve cohorts and no rescue in the tested twenty-link cohorts.
Those studies do not establish that higher cadence is useless, or define
a physical count boundary. They are separate from this 50 Hz release.

## Evidence and reproduction

The manifest is `runs/swingup11_uniform/eleven_link_swingup_manifest.json`.
The clean frozen source commit is
`1920582683b2fd500386a986b276951469966c45`, distributed as the local
`eleven_link_source.bundle`. This isolated Git snapshot preserves the shared
working tree. All reserved evaluations run from a fresh clone of it and
record a clean source state. Ninety-nine focused source tests pass in that
clone, and all 302 root tests pass in the shared checkout. The existing
ten-link release verifier and a fresh 2/2 noisy regression still pass.

From the original repository:

```bash
PYTHONPATH=src:scripts .conda-aligator/bin/python scripts/verify_frontier_release.py \
  --manifest runs/swingup11_uniform/eleven_link_swingup_manifest.json

git clone runs/swingup11_uniform/eleven_link_source.bundle /tmp/cartpole-eleven-repro
```

Use the recorded dependency versions in `eleven_link_source.json` with
`environment-aligator.yml`. Inside the fresh clone, the following reproduces
the noisy twenty cohort; output stays outside the clone to retain a clean
source state:

```bash
PYTHONPATH=src:scripts OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python \
  scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup11_uniform.yaml \
  --controller runs/swingup11_uniform/eleven_link_controller.json \
  --episodes 20 --seed 211000 --park-seconds 16 --cart-target -0.05 \
  --settle-control-cost 1000 --settle-cart-position-cost 10 \
  --settle-cart-velocity-cost 5 --tracking-gain-scale 1 \
  --phase-window 12 --release-evidence --out /tmp/eleven-repro-20.json
```

Change episodes/seed to 100/211100 for the second noisy cohort, or to
20/211200 with `--zero-noise` for exact replay. For the video use one episode
at seed 211500 with `--include-traces`, then run
`render_parked_evaluation_trace.py` on that saved evaluation. Full commands
are retained in `reserved_commands.json`.

## Claim limits and next scientific gate

These results concern deterministic exact-model simulation with full-state
feedback and the stated initial distribution. They do not validate sensor
noise, delay, parameter uncertainty, hardware transfer or a nonlinear
capture certificate. The high sensitivity of the capture value is itself
a measured numerical limitation: a separate 100-digit audit changes the
computed value on identical binary64 inputs. The saved eleven policy and
metrics are not retroactively changed by that diagnostic.

Development tuning preceded the reserved gates. The successful counts
demonstrate a shared synthesis family, but the eleven recovery was developed
through multiple explicitly recorded interventions. A frozen automatic
re-synthesis recipe and independently held-out morphologies require their
own evaluations. The next frontier is twelve, where default binary64 DARE
already fails numerical validity checks; high-precision component designs
exist, but hanging-start success has not been demonstrated.
