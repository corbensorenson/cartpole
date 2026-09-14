# Reset-Free Swing-Up and Hold of Seven Through Ten Serial Links with One Cart Actuator

## Funnel Composition, Target-Plant Trajectory Optimization, and Evidence-Gated Scaling

**Corben Sorenson**<br>
**Technical report, revision 2 - 2026-09-14**<br>
**Project:** [github.com/corbensorenson/cartpole](https://github.com/corbensorenson/cartpole)

**Status:** Seven-, eight-, nine-, and ten-link internal benchmark releases. This
revision supersedes the original seven-link-only report. The public-record claim
is scoped in Section 8.

## Abstract

We report reset-free swing-up and sustained upright stabilization of uniform
serial cart-poles with seven, eight, nine, and ten passive links, driven by one
bounded horizontal cart force. Every plant starts from a noisy hanging state and
uses the same three-phase control architecture in one uninterrupted MuJoCo
episode: a hanging-equilibrium LQR contracts the launch distribution, a
target-plant Box-FDDP trajectory with saved time-varying feedback performs the
swing, and an upright LQR captures and maintains the chain. The plant has total
link length 3 m, total link mass 1 kg, cart mass 1 kg, force limit +/-80 N,
rail limit +/-3 m, and a 50 Hz control rate. Success requires every absolute
link angle to remain within 0.15 rad of upright continuously for at least 5 s.

For each link count, disjoint noisy 20-episode and 100-episode cohorts passed
without a rail failure or state reset: 480/480 noisy trials in total. The eight-,
nine-, and ten-link releases also passed 60/60 exact-start checks. Four held-out
30 s videos use seeds outside the noisy cohorts. The ten-link run reaches upright
at 23.90 s and holds for the remaining 6.12 s. The sequence exposes a reusable
control principle: do not ask a narrow global swing controller to absorb all
uncertainty. First use an inexpensive stable-equilibrium controller to contract
the initial distribution, then optimize the nonlinear route to enter a verified
local capture funnel quietly.

A structured public search performed on 2026-09-14 found no directly comparable
seven-or-more-link cart-actuated swing-up-and-hold result with a frozen plant,
bounded rail and force, noisy hanging starts, sustained all-link upright gate,
public controller, and multi-seed evidence. We therefore claim a **provisional
public link-count record** for this precisely declared simulated benchmark. It is
not a universal world record across unmatched plants or rules, and independent
reproduction remains open.

## 1. Contributions

This report makes five contributions.

1. It consolidates four separately frozen results into one benchmark and evidence
   contract, spanning seven through ten serial links.
2. It identifies settled-launch funnel composition as the common mechanism that
   made narrow target-plant trajectories robust to the declared hanging-start
   distribution.
3. It isolates the count-dependent changes that were actually required: cart
   parking, terminal angle/rate shaping, longer arrival shaping, and a longer
   ten-link route. The controller architecture did not change.
4. It publishes disjoint noisy gates, exact checks, held-out videos, SHA-256
   manifests, negative controls, and replay commands instead of relying on one
   hand-selected trajectory.
5. It states a falsifiable record claim tied to a precise benchmark, while
   separating that claim from ongoing arbitrary-morphology solver research.

## 2. Problem definition

### 2.1 Dynamics and underactuation

Let the generalized coordinate be

```text
q = [x, theta_1, ..., theta_n]^T,
```

where `x` is cart position and the `theta_i` are relative revolute-joint angles.
The serial-chain dynamics have the standard manipulator form

```text
M(q) q_ddot + C(q, q_dot) q_dot + g(q) + d(q_dot) = B F,
B = [1, 0, ..., 0]^T,        |F| <= 80 N.
```

There are `n+1` generalized coordinates but only one actuator. The absolute
orientation of link `i` is

```text
alpha_i = sum(theta_j, j=1..i).
```

The success gate is evaluated on the absolute angles, not on the relative joint
coordinates:

```text
max_i |wrap(alpha_i)| < 0.15 rad continuously for at least 5.0 s.
```

This matters increasingly with link count. Small relative-angle errors can
accumulate into a large distal-link error, and low relative hinge-rate RMS does
not guarantee that every physical link has low absolute angular velocity.

### 2.2 Frozen benchmark

All four released plants use the same aggregate contract.

| Quantity | Frozen value |
|---|---:|
| Passive serial links | 7, 8, 9, or 10 |
| Total chain length | 3.0 m |
| Total chain mass | 1.0 kg |
| Cart mass | 1.0 kg |
| Cart-center rail limit | +/-3.0 m |
| Cart force | continuous, +/-80 N |
| MuJoCo integration step | 0.005 s |
| Control frame skip | 4 |
| Policy rate | 50 Hz |
| Episode duration | 30.0 s |
| Start | hanging, with Gaussian perturbations |
| Angle and generalized-velocity noise | standard deviation 0.05 |
| Upright set | all absolute angles within 0.15 rad |
| Required continuous hold | 5.0 s |
| State resets after episode start | prohibited |

The links are uniform within each plant: length `3/n` m and mass `1/n` kg. The
initial relative-angle vector is `[pi, 0, ..., 0]` plus noise. An episode receives
one seeded reset at its beginning. Phase transitions carry forward position,
velocity, simulation time, and all hidden simulator state unchanged.

### 2.3 Dimensionless view

For total chain length `L`, total moving mass `M`, and gravity `g`, define the
natural time `t0 = sqrt(L/g)`. The main scale-free groups are force authority
`Fmax/(M g)`, rail ratio `R/L`, control period `dt/t0`, cart-to-chain mass ratio,
link length fractions, link mass fractions, and dimensionless damping and
armature. These groups support analytic transfer between physically similar
plants, but they do not make one finite-horizon force trace universal across link
counts. Internal modal structure still changes with `n`; every released route is
therefore refined and gated on its exact target plant.

## 3. One controller architecture for all four plants

The runtime policy is a composition of three feedback funnels:

```text
noisy hanging distribution
        |
        v
hanging LQR contraction and cart parking
        |
        v
Box-FDDP swing route with time-varying feedback
        |
        v
quiet terminal state inside the upright LQR basin
        |
        v
sustained all-link upright hold
```

### 3.1 Phase A: contract the launch distribution

The hanging configuration is stable and inexpensive to regulate. A finite-
difference linearization at that equilibrium produces a discrete LQR that damps
the reset perturbations and steers the cart to a chosen park target. If `x_p` is
that target and `x_h` is the measured state, the phase uses

```text
u_h = clip(-K_h (x_h - x_h*), -1, 1).
```

This is active control, not waiting and not a simulator reset. Its purpose is to
map a relatively broad initial distribution into the narrow launch basin of the
nonlinear swing route. The seven-link release centers the cart. Links 8-10 use a
small negative park target to reserve rail room in the direction required by the
selected swing branch.

### 3.2 Phase B: track a target-plant nonlinear swing route

Box-FDDP optimizes bounded normalized actions on the exact MuJoCo target plant.
The controller stores nominal actions `u_bar[k]`, nominal states `z_bar[k]`, and
time-varying gains `K[k]`. Execution uses

```text
u[k] = clip(u_bar[k] + kappa K[k] e_periodic(z[k], z_bar[k]), -1, 1),
```

where `z` contains dimensionless cart states and absolute link orientations.
`e_periodic` evaluates revolute-joint errors on the nearest angular branch. The
nominal cart channel is translated to the measured parked cart position, an
exact symmetry of the horizontal plant away from rail contact.

The route objective does more than cross upright. It must deliver low absolute
angle error, low internal rate, modest cart velocity, and feasible rail position
at the route boundary. A visually upright crossing with unresolved internal
momentum is not a valid handoff.

### 3.3 Phase C: capture and maintain

At the route horizon, an upright linearization supplies a saturated LQR around
the same cart target. The switch is delayed until the planned route boundary.
The local expert is intentionally not responsible for creating swing energy or
arresting a high-energy chain. The nonlinear route must enter its practical
capture basin first.

### 3.4 Why the composition works

Let `D0` be the noisy hanging distribution, `Ds` the contracted launch set, and
`Fu` the practical upright capture funnel. The design objective is the set map

```text
D0 --pi_settle--> Ds --pi_swing--> Fu --pi_LQR--> x_upright.
```

The principal insight is organizational. Robustness is easier when assigned to
the controller phase that can obtain it cheaply. The stable hanging LQR reduces
launch uncertainty; trajectory feedback corrects along the global maneuver; the
upright LQR handles only local maintenance. Trying to make any one phase do all
three jobs produced the strongest negative controls in this project.

## 4. How the solution scaled from seven to ten links

The same architecture survived every promotion. What changed was the target-
plant route and a small number of measured phase parameters.

| Links | Hanging phase | Park target | Swing route | Route feedback | Terminal shaping |
|---:|---:|---:|---:|---:|---|
| 7 | 10.0 s | 0.00 m | 228 steps / 4.56 s | 2.00 | quiet terminal state |
| 8 | 14.0 s | -0.15 m | 197 steps / 3.94 s | 0.75 | quiet terminal state |
| 9 | 14.0 s | -0.05 m | 197 steps / 3.94 s | 1.00 | angle 20x, hinge rate 4x |
| 10 | 16.0 s | -0.05 m | 400 steps / 8.00 s | 1.00 | angle 20x, hinge rate 4x |

### 4.1 Seven links: correct coordinates and a settled launch

The original exact-state route was excellent but failed the noisy gate 0/20.
The decisive repair had two parts. First, optimizer warm-start states were
rebuilt from MuJoCo position and velocity because an earlier artifact confused
cumulative absolute rates with relative hinge rates. Second, a 10 s hanging LQR
contracted the reset distribution before launching the 4.56 s route. Translating
the nominal cart coordinate to the measured settled location and using route
feedback scale 2.0 produced 20/20 and 100/100.

### 4.2 Eight links: park the cart to use the rail asymmetrically

Direct seven-to-eight transfer failed. The exact eight-link route retained the
three phases, but its centered swing consumed slightly too much positive rail.
A 14 s park at -0.15 m used translational symmetry to shift the route without
changing link dynamics. Open-loop replay could pass an exact start but scored
0/20 with noise; saved route feedback was essential. A shorter 12 s park at
-0.20 m reached 95/100 and was rejected in favor of the frozen 100/100 setting.

### 4.3 Nine links: optimize for the capture state, not the crossing

The first nine-link route arrived near upright with maximum angle about 0.314
rad, outside a useful local basin. Increasing the terminal absolute-angle block
by 20x and the hinge-rate block by 4x produced a route-boundary state with
maximum absolute angle about 0.00004 rad, hinge-rate RMS about 0.00068 rad/s,
and cart velocity about 0.00041 m/s. A 14 s park at -0.05 m then left enough
canonical rail margin for the 3.94 s feedback route.

### 4.4 Ten links: shape an arrival segment, then re-optimize feedback

The ten-link campaign exposed the difference between one quiet sample and a
robust arrival. Several routes ended almost exactly upright yet required
impossible saturated feedback immediately afterward. The successful warm start
first shaped the final two seconds of arrival. Box-FDDP then re-optimized an 8 s
target-plant route and its feedback with a 2.85 m soft rail, inside the actual
3.0 m benchmark rail. A 16 s park reduced launch residuals. This route enters
the local basin with enough time to complete a 6.12 s hold before the 30 s limit.

## 5. Evaluation protocol

### 5.1 Promotion gates

Each release had to satisfy all of the following:

1. A 20-episode noisy hanging-start gate.
2. A disjoint 100-episode noisy hanging-start gate.
3. Zero state resets after the initial seeded reset.
4. No cart-center rail violations.
5. At least 5 continuous seconds with every absolute link angle inside 0.15 rad.
6. A 30 s held-out video with a seed outside both noisy cohorts.
7. A manifest binding the config, generated XML, controller, gates, video, and
   metadata by SHA-256.

Links 8-10 additionally have 20-episode zero-noise checks. Exact checks are
diagnostics, not substitutes for noisy trials.

### 5.2 Statistical interpretation

Every link count passed 100/100 in its larger noisy cohort. A two-sided 95%
Clopper-Pearson interval for 100 successes in 100 Bernoulli trials has a lower
bound of approximately 0.964. This does not prove certainty or robustness outside
the declared distribution. Pooling all four counts into 400/400 would also be
misleading because the plants and controllers differ. The correct statement is
four separate 100/100 results under four frozen, closely related benchmarks.

### 5.3 Independent evidence layers

The evidence stack deliberately separates planning, execution, and presentation:

- planner artifacts contain the target trajectory and local feedback policy;
- evaluation artifacts execute that policy in the exact target MuJoCo plant;
- release verifiers check hashes, cohorts, dimensions, outcomes, and metadata;
- videos are state-faithful renders of held-out simulated trajectories;
- negative controls remain public and are not relabeled as successful runs.

## 6. Results

### 6.1 Main outcomes

| Links | Noisy 20 | Noisy 100 | Exact 20 | First upright in video | Hold through video end | Max cart, noisy 100 |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 20/20 | 100/100 | not a release gate | 14.54 s | 15.48 s | 2.3708 m |
| 8 | 20/20 | 100/100 | 20/20 | 17.80 s | 12.22 s | 2.9300 m |
| 9 | 20/20 | 100/100 | 20/20 | 17.84 s | 12.18 s | 2.9800 m |
| 10 | 20/20 | 100/100 | 20/20 | 23.90 s | 6.12 s | 1.9976 m |

All 400 episodes in the four 100-episode noisy gates reached the 30 s time
limit. None ended at the rail. Across the separate noisy 20-episode cohorts,
another 80/80 passed. The 60 exact checks for links 8-10 also passed.

### 6.2 Rail use is a route property

The nine-link result approaches the 3 m cart-center boundary most closely, at
2.9800 m. Ten links uses substantially less rail despite having more internal
modes, because its longer route and constrained arrival were optimized as a
different branch. Link count alone therefore does not determine rail demand.
For a trajectory with cart positions `x_k`, cart half-length `r_c`, and desired
clearance `c`, the body-aware requirement is

```text
rho_required = (max_k |x_k| + r_c + c) / L.
```

This quantity should be reported alongside force authority, timing, damping,
and route construction. A single empirical line from link count to rail length
would hide the dominant controller dependence visible in the table above.

### 6.3 What the results establish

The evidence establishes that one phase architecture can swing up and hold four
increasingly underactuated uniform serial chains under the declared simulation
contract. It also establishes robustness to the declared hanging-state reset
noise. It does not establish model-error robustness, hardware transfer, minimum
time, minimum force, or minimum rail.

## 7. Negative controls and lessons

The failed branches were scientifically useful because they localized each
bottleneck.

- The unconditioned seven-link route scored 0/20. Exact trajectory quality did
  not imply a useful noisy launch basin.
- Passive waiting did not replace active hanging-state contraction.
- The eight-link open-loop route scored 0/20 with noise even after an exact
  replay succeeded.
- The 12 s, -0.20 m eight-link park scored 95/100 and was not promoted.
- Direct eight-to-nine transfer failed its initial probe.
- The balanced nine-link terminal objective left too much absolute-angle error
  for saturated LQR capture.
- The strongest early ten-link arrivals could be nearly upright at one sample
  while remaining far outside the five-second actuator-feasible capture set.
- Wider-rail, ghost-link, spring-support, CEM-tail, online-MPC, and static-gain
  probes were retained as development evidence but excluded from the releases.
- The seven-link controller failed tested changes in force authority, link mass
  or length, damping, sensor noise, and delay. The released route is robust to
  its declared initial distribution, not arbitrary plant mismatch.

Three general lessons follow. State-coordinate schemas are part of the control
contract. Terminal objectives must represent a dynamically quiet capture state.
And empirical promotion must evaluate the complete switched controller, not an
optimizer trace or a visually convincing crossing.

## 8. Public-record claim and literature boundary

### 8.1 Claim

As of 2026-09-14, we claim a **provisional public link-count record for
reset-free simulated cart-pole swing-up and sustained all-link upright hold under
the benchmark frozen in Section 2**: ten uniform passive serial links, one
horizontal cart actuator, bounded force and rail, noisy hanging starts, and
public multi-seed evidence.

The word `provisional` is essential. There is no recognized registry for this
problem, and records across different plants are not automatically comparable.
The claim is falsified by a prior public result meeting or exceeding ten links
under the same or stricter declared dimensions, force convention, rail geometry,
start distribution, action timing, sustained-upright rule, and reset policy.

### 8.2 Search performed

The public search used combinations of `n-link cart-pole`, `multi-link cart-pole
swing-up`, `seven-link`, `ten-link`, `serial pendulum on cart`, `world record`,
and `swing-up and stabilization`. The results were dominated by conventional
single-pole benchmarks, double cart-poles, and n-link robots with actuators at
multiple joints or only one passive joint. For example, Hirano et al. analyze an
n-link planar robot with one passive revolute joint and demonstrate four links;
that is a different actuation topology from an all-passive chain driven only by
cart translation [6]. Standard continuous-control suites also focus on the
single-pole cart-pole or double-pendulum variants [5,7]. No search result supplied
a directly comparable seven-to-ten-link evidence bundle.

Search-engine absence is not proof of priority. Until independent comparison or
peer review, the safest short public description is:

> To our knowledge, this is the highest publicly evidenced link count for the
> precisely declared reset-free simulated cart-pole swing-up-and-hold benchmark.

### 8.3 What is not claimed

We do not claim a hardware record, a universal record across arbitrary force or
rail limits, a minimum-time solution, an optimal controller, or a proof that no
older unpublished solution exists. We also do not claim that one unchanged
force trace solves all four plants. The shared contribution is the architecture
and evidence discipline; the routes are target-plant solutions.

## 9. Reproduction and artifact map

### 9.1 Seven links

```bash
make setup-aligator
make release-swingup7
```

Primary manifest: `runs/swingup7_uniform/seven_link_swingup_manifest.json`

Video: `runs/swingup7_uniform/seven_link_swingup_success.mp4`

### 9.2 Eight links

```bash
PY=.conda-aligator/bin/python
$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup8_uniform.yaml \
  --controller runs/generalized_solver/n8_capture_fddp_feedback120.json \
  --episodes 100 --seed 80801 --park-seconds 14 --cart-target -0.15 \
  --tracking-gain-scale 0.75 \
  --out /tmp/n8_noisy100.json
```

Primary manifest: `runs/generalized_solver/eight_link_swingup_manifest.json`

Video: `runs/generalized_solver/eight_link_swingup_success.mp4`

### 9.3 Nine links

```bash
PY=.conda-aligator/bin/python
$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup9_uniform.yaml \
  --controller runs/generalized_solver/n9_fddp_terminal_angle20_widerail.json \
  --episodes 100 --seed 91041 --park-seconds 14 --cart-target -0.05 \
  --tracking-gain-scale 1.0 \
  --out /tmp/n9_noisy100.json
```

Primary manifest: `runs/generalized_solver/nine_link_swingup_manifest.json`

Video: `runs/generalized_solver/nine_link_swingup_success.mp4`

### 9.4 Ten links

```bash
PY=.conda-aligator/bin/python
$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup10_uniform.yaml \
  --controller runs/generalized_solver/n10_fddp_refined_route_feedback100.json \
  --episodes 100 --seed 102101 --park-seconds 16 --cart-target -0.05 \
  --tracking-gain-scale 1.0 --release-evidence \
  --out /tmp/n10_noisy100.json

PYTHONPATH=src $PY scripts/verify_ten_link_release.py
sha256sum -c runs/generalized_solver/n10_release_SHA256SUMS
```

Primary manifest: `runs/generalized_solver/ten_link_swingup_manifest.json`

Video: `runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005.mp4`

The individual eight-, nine-, and ten-link notes remain in `docs/` as detailed
experiment appendices. The release manifests are authoritative if prose and a
machine-readable field ever disagree.

## 10. Limitations and next work

The four results are simulated, exact-model, route-based controllers. The
declared initial-state noise does not cover sensor noise during execution,
actuator delay, unmodeled friction, parameter drift, or link manufacturing
tolerances. No independent laboratory has yet reproduced the release. The
state-faithful public videos are 2D renders of MuJoCo trajectories, not footage
of a physical apparatus.

The conditioning phase is legitimate feedback control but consumes 10-16 s of
the 30 s horizon. Faster controllers may exist. The routes are local solutions
of a nonconvex trajectory problem and are not certificates of global optimality.
The 100-episode gates characterize one frozen reset distribution; they are not
formal reachability proofs.

The generalized-solver track should remain separate from the record release. It
already expresses length, mass, force, damping, armature, timing, body geometry,
and rail in dimensionless form; transfers route states and feedback by normalized
arc length; and uses bounded exact-target continuation plus a thin actuator
adapter. It has not yet automatically regenerated every 7-10 route from arbitrary
unequal morphologies. That is the next methodological target.

## 11. Conclusion

A single bounded cart force can swing up and hold a ten-link passive serial chain
from a noisy hanging start on a finite rail. The result was reached not by one
monolithic learned policy, but by composing feedback funnels: contract the launch
distribution at the stable equilibrium, track a target-plant nonlinear route,
and enter the local upright controller only after the chain is dynamically quiet.

The progression from seven through ten links strengthens the claim because every
promotion preserved the benchmark, control interface, reset rule, and evidence
gate. It also clarifies what generalized: the architecture and experimental
logic. The optimized swing route did not. The public artifacts support a
provisional record claim under the exact frozen benchmark and invite the two
tests that matter next: independent reproduction and a strictly matched prior-art
comparison.

## References

1. A. G. Barto, R. S. Sutton, and C. W. Anderson. "Neuronlike adaptive elements
   that can solve difficult learning control problems." IEEE Transactions on
   Systems, Man, and Cybernetics, 1983.
   [DOI 10.1109/TSMC.1983.6313077](https://doi.org/10.1109/TSMC.1983.6313077).
2. E. Todorov, T. Erez, and Y. Tassa. "MuJoCo: A physics engine for model-based
   control." IEEE/RSJ International Conference on Intelligent Robots and Systems,
   2012. [DOI 10.1109/IROS.2012.6386109](https://doi.org/10.1109/IROS.2012.6386109).
3. C. Mastalli et al. "Crocoddyl: An Efficient and Versatile Framework for
   Multi-Contact Optimal Control." IEEE International Conference on Robotics and
   Automation, 2020.
   [DOI 10.1109/ICRA40945.2020.9196673](https://doi.org/10.1109/ICRA40945.2020.9196673).
4. Y. Tassa, N. Mansard, and E. Todorov. "Control-Limited Differential Dynamic
   Programming." IEEE International Conference on Robotics and Automation, 2014.
   [DOI 10.1109/ICRA.2014.6907001](https://doi.org/10.1109/ICRA.2014.6907001).
5. Y. Duan et al. "Benchmarking Deep Reinforcement Learning for Continuous
   Control." Proceedings of Machine Learning Research 48, 2016.
   [PMLR](https://proceedings.mlr.press/v48/duan16.html).
6. Y. Hirano, K. Yoshida, and M. Sampei. "Swing-up control for n-link planar
   robot with single passive joint using the notion of virtual composite links."
   IEEE Conference on Decision and Control, 2008.
   [DOI 10.1109/CDC.2008.4738922](https://doi.org/10.1109/CDC.2008.4738922).
7. Y. Tassa et al. "DeepMind Control Suite."
   [arXiv:1801.00690](https://arxiv.org/abs/1801.00690), 2018.
8. W. Yu, J. Tan, C. K. Liu, and G. Turk. "Preparing for the Unknown: Learning
   a Universal Policy with Online System Identification." Robotics: Science and
   Systems, 2017. [arXiv:1702.02453](https://arxiv.org/abs/1702.02453).
