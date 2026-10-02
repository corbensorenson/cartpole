# Eleven link frontier and general solver review

Review date: October 1, 2026. The numerical results below were executed on the current working tree, using the existing Python environment. Existing controller code, configurations, and release artifacts were preserved.

The evidence supports continuing toward eleven links. It does not establish an impossibility boundary at eleven. The main limitation is that the current method must deliver the chain into an extremely narrow, directional capture region, while its trajectory discovery and transfer methods repeatedly produce states outside that region. Several concrete implementation defects further weaken the interpretation of past failures. Beyond eleven, the current numerical Riccati implementation itself breaks before we have established a physical limit.

The successful seven through ten link releases show that the three phase architecture works on carefully synthesized routes: contract the hanging launch distribution, execute a nonlinear trajectory with feedback, and enter upright maintenance quietly. They do not establish that route discovery scales automatically, or that a conventional upright LQR provides a useful capture basin in arbitrary state directions.

This review covers the main dynamics, morphology, optimization, controller, evaluation, and generalized pipeline code; the release papers and manifests; the development ledger; earlier frontier, harmonic co-design, hybrid training, and research handoff studies; and an automated inventory of all 445 top-level eleven link JSON artifacts in `runs/generalized_solver`. It does not independently regenerate every historical experiment or validate every nested training checkpoint. The historical studies use different plants and must retain their original claim boundaries.

## Follow-up: corrected synthesis and control cadence

The defect descriptions in the original audit below describe the code at
audit time. Subsequent work repairs feedback action/state bookkeeping,
independent dynamics-feasibility checks, continuous angle references,
simulator warning/time guards, and checked upright Riccati designs. The
new shared discovery procedure transfers an inverse-dynamics spline to
the adjacent count, optimizes missing-joint acceleration residuals, repairs
discrete dynamics with sparse least squares, refines target-plant feedback
with Box-FDDP, and validates the complete episode. It now has exact-start
calibrations from two through eleven, independently of the old released
seven-to-ten swing trajectories. Eleven now also passes its canonical noisy
release bundle, as documented in the promotion follow-up below. Commands,
source/input hashes, stage budgets, failures, and policy comparisons are in
[the synthesis ledger](/Users/corbensorenson/Documents/cartpole/runs/frontier_campaign_20261001/inverse_increment_experiments.json).

The most informative new comparisons are concrete failures of the old
proxies. At eight, a nominally feasible route with tiny endpoint angle/rate
errors fails after a forced eight-second capture. Removing only the clock
restriction passes the same controls and gains, with a 6.54-second handoff
and 23.48-second hold. Two endpoint-value refinements also pass while a
matched extra-iteration control with the old objective fails. At nine,
changing only the sparse inner budget from 300 to 3000 recovers a failed
initializer. At ten, a threshold-five state gate never opens (minimum
5.20); thresholds ten and twenty-five pass the same route for all 30
seconds, while thresholds one hundred and eighteen hundred switch too
early and fail. A scalar local Lyapunov sublevel is an empirical policy
selection parameter here, not a nonlinear capture certificate.

The parked evaluator now honors the saved gate, centers it at the actual
declared cart target, checks the reconstructed metric hash, and latches
capture without overwriting state. Eight and nine pass four noisy
development starts each after 16-second parking. New ten passes only
three of four at that duration; the same four seeds pass with 17.5-second
parking, leaving 5.98 seconds upright. A matched fresh twenty gives 15/20
at sixteen seconds and 20/20 at seventeen-and-a-half; a separate hundred
at the latter duration gives 93/100. These screens do not replace reserved release gates.
The released ten remains unchanged and passes fresh 2/2 regression;
all 302 root tests pass. New development video renders recorded evaluator
states, avoiding the legacy video's different parking regulator.

Higher control frequency is a useful measured lever, but the current
evidence does not support a link-count-proportional correction rate as
the main explanation. With identical 0.005-second RK4 physics and gains
redesigned for 50, 100, and 200 Hz, a selected eleven-link local cohort at
amplitude 1e-8 improves from 1/4 to 2/4 successes. A selected twelve-link
cohort at 1e-10 improves from 3/4 to 4/4. Twenty still fails all tested
directions. The fastest upright growth rate changes only from roughly
27 to 28.5 per second between ten and twenty, whereas required static-LQR
gains and numerical sensitivity grow enormously. More weakly controlled
directions through one cart actuator are distinct from simply faster
motion. The [rate figure](/Users/corbensorenson/Documents/cartpole/runs/frontier_campaign_20261001/control_frequency_figure/control_frequency.pdf)
and [controlled local comparisons](/Users/corbensorenson/Documents/cartpole/runs/frontier_campaign_20261001/control_frequency_local_boundary/result.json)
retain the sampled directions, parameters, and limits.

High-precision Riccati computation recovers stabilizing designs for the
supplied linear matrices through twenty, so failed default DARE is not
a physical impossibility result. Actual bounded nonlinear twenty-link
replays still fail even at tested errors down to 1e-19. A matched 360-case
[action/arithmetic audit](/Users/corbensorenson/Documents/cartpole/runs/frontier_campaign_20261001/action_precision_rate_local/result.json)
finds the same outcomes for float32 versus float64 actions and extended
precision gain evaluation. Smaller identification steps also do not repair
the tested fourteen/twenty outcomes. These findings keep capture geometry,
conditioning, nonlinear arrival shaping, and robustness central; they do
not eliminate a different controller or prove a hard count limit.

The paper's strongest path is a shared, explicitly budgeted synthesis
algorithm producing a separately verified controller at each count.
Count-specific controllers and recorded capture/launch parameters are
compatible with that contribution. An arbitrary-morphology universal
solver is a broader claim requiring separate evidence. The current
adjacent eleven baseline closes gaps into a high-cost, noncapturing route.
Accurate physical-curve transfer, extra restoration iterations, and stronger
initial regularization do not recover this tested branch. Directional
terminal-value shaping now does: from the same useful 200-iteration source,
two additional 100-iteration trials differ only in the coefficient on the
saved capture value. Coefficient 0.01 reaches 3.70 seconds upright before
rail loss; coefficient 1 passes all 30 seconds, holds for 23.50 seconds,
and stays within 2.0717 m. Rebuilding the warm start using the actual applied
feedback controls also passes, with exactly zero independently checked
dynamics gaps both before and after refinement.

This nominal success does not settle noisy launch robustness. Initial
eleven screens pass 2/4 with sixteen-second parking and 3/4 at seventeen
and a half. Reducing the hanging regulator's control penalty from 1000
to 100 or 10 does not recover the failing seed. Increasing the cart-state
weights instead substantially contracts the remaining launch error and
passes all four at sixteen seconds. Matched twenty and disjoint hundred
development cohorts then pass 20/20 and 100/100. After freezing that policy
and clean source snapshot, reserved cohorts pass noisy 20/20, disjoint noisy
100/100, exact 20/20, and the full held-out video. Every episode completes
30 s with 7.50 s hold; the noisy hundred's maximum cart position is 2.121369 m.
The source-clone tests and release verifier pass. Eleven is now an internal
canonical result, and twelve is the active frontier. The bundle remains
available with its full video in the eleven-link GitHub release. Packaging
records retain their original pre-publication status. See the
[eleven-link appendix](/Users/corbensorenson/Documents/cartpole/docs/eleven_link_swingup_paper.md)
and [frozen manifest](/Users/corbensorenson/Documents/cartpole/runs/swingup11_uniform/eleven_link_swingup_manifest.json).

## Fresh evidence and its limits

The reproducible audit is [audit_link_count_frontier.py](/Users/corbensorenson/Documents/cartpole/scripts/audit_link_count_frontier.py). Its complete measurements, source hashes, runtime metadata, component rollouts, artifact inventory, and fresh ten link trace are in [diagnostics_complete.json](/Users/corbensorenson/Documents/cartpole/runs/frontier_audit_20261001/diagnostics_complete.json).

The existing ten link release verifier passed with no errors. A fresh noisy replay with seed `20261001`, outside the published release cohorts, also passed: first upright at 23.90 seconds, 6.12 seconds maximum upright hold, 6.02 seconds low momentum hold, and 1.9657 meters maximum cart excursion. The root test suite passed all 239 tests. These checks confirm that the current environment still executes the successful predecessor. They do not validate the untested numerical assumptions identified below.

### Upright feedback becomes much more sensitive with count

I rebuilt the upright linearization and LQR with the existing design: normalized cart action, control cost 1000, the existing physical state weights, the canonical plant parameters, and 50 Hz actions. For counts beyond eleven, I changed only link count in the uniform eleven link configuration. These are synthesized diagnostic plants, not released benchmarks.

| Links | Physical coordinate gain norm | Computed closed loop spectral radius | Interpretation |
| --- | ---: | ---: | --- |
| 7 | 1,974.29 | 0.995091 | Numerically stable local design |
| 8 | 9,183.37 | 0.995250 | Gain is 4.65 times the seven link value |
| 9 | 47,126.14 | 0.995386 | Gain is 5.13 times the eight link value |
| 10 | 265,134.93 | 0.995505 | Gain is 5.63 times the nine link value |
| 11 | 1,611,819.18 | 0.995610 | Gain is 6.08 times the ten link value |
| 12 | 111,154,473.69 | 11.198142 | Returned gain fails the stability check |
| 13 | 6,594,362.12 | 1.661352 | Returned gain fails the stability check |
| 14 and 15 | No default solution | Not available | Riccati routine raises an error |

Gain norm here mixes the declared physical state units. It is a comparison within this fixed benchmark family, not a universal dimensionless measure of difficulty. A spectral radius below one describes the linear model with unsaturated feedback. It says nothing by itself about nonlinear recovery under bounded force.

The hanging spectrum does not undergo a sudden discontinuity at eleven: its lowest frequency stays near 0.346 Hz, and its highest frequency changes from 4.421 Hz at ten to 4.498 Hz at eleven. The weakest normalized hanging cart-acceleration coupling remains nonzero, changing from 0.06155 to 0.05582. These facts do not prove global reachability, but they provide no evidence of an abrupt loss of all actuation at eleven.

The difference between ten and eleven is therefore much larger in capture sensitivity than in the apparent change in small-oscillation frequencies. Linear feedback must coordinate more unstable directions through the same input. Stable poles coexist with large transient responses and a very restrictive nonlinear region.

![Measured feedback sensitivity and bounded capture outcomes](/Users/corbensorenson/Documents/cartpole/runs/frontier_audit_20261001/capture_scaling.png)

The figure is also available as an [exportable PDF](/Users/corbensorenson/Documents/cartpole/runs/frontier_audit_20261001/capture_scaling.pdf).

### Direct nonlinear capture tests expose the missing robustness

For each count, I drew eight fixed directions of relative-angle and hinge-rate perturbation about upright. Each component was uniform in `[-a,a]`, with angle amplitudes in radians and rate amplitudes in radians per second. Cart position and velocity began at zero. Directions were reused across amplitudes within each count. Each case ran the exact clipped upright LQR for eight seconds, with simulator warning and time continuity checks.

| Links | Passes at amplitude 0.0001 | Passes at amplitude 0.00001 | Passes at amplitude 0.0000001 |
| --- | ---: | ---: | ---: |
| 7 | 8/8 | 8/8 | 8/8 |
| 8 | 2/8 | 8/8 | 8/8 |
| 9 | 0/8 | 1/8 | 8/8 |
| 10 | 0/8 | 0/8 | 2/8 |
| 11 | 0/8 | 0/8 | 0/8 |
| 12 | 0/8 | 0/8 | 0/8 |

At still smaller amplitudes, ten links passed 8/8 at `1e-8`. Eleven passed 2/8 at `1e-8` and 8/8 at `1e-9`. Twelve failed all tested directions even at `1e-12`, consistent with the unstable numerical gain returned by the current design routine.

These are exploratory component tests with a small direction set, not statistical estimates of basin volume. Initial states are deliberately reset for these component experiments; they are not hanging-start swing-up evidence. The finding is nevertheless decisive for interpreting the architecture: a successful route must land in favorable state directions. A small maximum angle or modest hinge-rate RMS does not imply that capture will succeed.

The successful ten link route can coexist with failures from tiny arbitrary upright perturbations because its feedback guides the state along a structured trajectory into a favorable terminal region. Independent hanging noise is first contracted by the settling phase; it does not arrive as an arbitrary upright perturbation. This also predicts fragility to new sensing noise, model error, and delay, which the released reset-noise benchmark does not test.

### The positive eleven link bank is much easier than a swing handoff

I reproduced 24/24 eight-second capture successes from the saved assisted eleven link state bank, on the canonical free uniform target, with warning and time continuity checks.

However, inspecting the actual bank changes its scientific interpretation. Its states were sampled at 16.22 through 29.92 seconds of the assisted source episode, after at least 8.34 seconds of uninterrupted upright hold had already accumulated. Across the bank, maximum initial absolute angle is only `5.60e-6` radians and maximum initial hinge-rate RMS is `7.05e-6` radians per second.

These are real visited states, but they are already settled upright states with favorable correlations. The 24/24 result proves that the target can maintain these states. It does not establish a broad nonlinear capture region or tell us that the bank lies on the reachable boundary of a free hanging-start swing.

Optimizing distance to this bank is essentially optimizing distance to an extremely quiet upright endpoint. The bank does not provide a sequence of executable nonlinear bridges from the observed swinging states into that endpoint. Calling it a measured capture manifold is stronger than the evidence warrants unless its surrounding recoverable region is also measured.

### A useful eleven link crossing remains far outside that region

The latest settled-launch cart-target route has a recorded crossing with maximum angle `0.11297` radians, cart position `0.42282` meters, cart velocity `0.07922` meters per second, hinge-rate RMS `3.06605`, and absolute-rate RMS `3.63750` radians per second.

At that state, the current canonical eleven link LQR asks for normalized action `-39,652.48`, against an available magnitude of one. This is the controller's request, not a proof that every controller would require that force. In a fresh exact canonical capture replay, it saturated on all 18 steps, accumulated only 0.02 seconds upright, and exited the rail after 0.36 seconds. The trajectory remained numerically clean.

The swing has solved much of the geometric problem at this crossing: it is upright, reasonably centered, and the cart is slow. It has not solved the distribution of chain momentum or the weak directions of capture. The gap is not adequately described by a scalar energy deficit.

## What the successful frontier actually required

The clean common mechanism is a composition of three controllers, each solving a different problem.

The hanging LQR removes much of the initial uncertainty before the nonlinear swing. The route and its time-varying feedback control the trajectory on the target plant. The upright expert then maintains the endpoint that the route was specifically shaped to reach.

The ten link release added a 16-second park, an eight-second target route, late arrival shaping, stronger terminal angle/rate shaping, and target-plant Box-FDDP feedback. The ten link paper explicitly reports that an isolated quiet sample, gain sweeps, wider rails, open-loop tail search, MPC, and ghost support had previously failed. The successful release is evidence for coordinated arrival shaping and launch contraction, not for those earlier methods being sufficient alone.

At eleven, many attempts returned to a seed whose target-plant rollout had already departed badly from the useful route. They then asked a local optimizer, low-dimensional residual, or local learned expert to recover the entire missing nonlinear maneuver. That is a much harder task than the successful ten link refinement.

There is also little spare time under the frozen 30-second contract. A 16-second park and five-second hold leave at most nine seconds for swing and capture entry. A 32-second windup can investigate a separate physical question, but cannot solve the existing 30-second benchmark. Settling, swing duration, and capture margin should be optimized together within the actual time budget.

## Concrete implementation defects

These defects are reproducible on the current tree. They should be repaired before interpreting another failed campaign as evidence of a physical limit. Their existence does not establish that fixing them alone will solve eleven links.

### Feedback warm starts retain the wrong action sequence

In [search_fddp_capture.py](/Users/corbensorenson/Documents/cartpole/scripts/search_fddp_capture.py), `rebuild_feedback_warm_start` propagates the target plant under feedback-corrected actions, but returns only the resulting states. The caller supplies the original action list to FDDP and marks the pair feasible whenever feedback rebuilding is enabled.

The resulting state/action pairs generally do not satisfy the dynamics. The audit demonstrates this with a one-dimensional transition: the rebuilt states are `[1,0,0]`, the supplied original actions are `[0,0]`, and the first dynamics defect is one. This is an exact bookkeeping error, independent of MuJoCo or chaotic dynamics.

The repair is to return the executed actions with the rebuilt states. Before setting `is_feasible`, evaluate all dynamics defects under exactly those actions, the same transition map, and the same coordinate convention. This must also reject simulator warnings or resets.

### Some transferred nominal paths are explicitly marked feasible despite defects

The one-shot generalized pipeline unconditionally passes `--initial-feasible`, including its option to preserve transferred nominal states rather than reconstructing the target rollout. Transfer preserves a reference for feedback; it does not make that reference dynamically feasible on the new plant.

The actual warm start used by `n11_from_n10_probe_fddp.json` has maximum target-plant coordinate defect `2.45793`, median defect `0.30905`, and first defect `0.79371`, over 400 warning-free one-step checks in the artifact's correct coordinates. The subsequent solve declares `initial_feasible=true`, while both rebuild flags are false.

Box-FDDP is designed to handle infeasible trajectories. The method should receive the actual feasibility status. Falsely declaring feasibility throws away the distinction that the solver needs to close the defects.

### Periodic angles are handled inconsistently

[ilqr.py](/Users/corbensorenson/Documents/cartpole/src/gcartpole/ilqr.py) wraps physical relative angles when producing transformed states and uses a periodic difference for finite-difference dynamics. But [fddp.py](/Users/corbensorenson/Documents/cartpole/src/gcartpole/fddp.py) constructs Crocoddyl `StateVector`, whose state difference and integration are Euclidean. Several route executors and the feedback warm-start function also subtract the transformed states directly.

At the hanging branch cut, two states separated physically by `2e-8` radians produce a Euclidean transformed error of `138.926` in the audit's declared scales, whereas the periodic transformed error is `4.42e-7`. The local derivative convention and the optimizer/executor error convention therefore disagree.

Some generalized execution paths already provide `periodic_coordinate_error`; the issue is incomplete propagation of that convention. FDDP needs a consistent state manifold or a continuous angle lift along the trajectory, and feedback errors must use the same convention. Periodic errors should not simply be inserted into a quadratic objective without checking its derivatives and the chosen reference branch.

This is particularly relevant to swinging trajectories that perform several rotations. It is a proven representational inconsistency, but I have not isolated its contribution to every eleven link failure.

### The legacy modal coupling diagnostic uses the wrong eigenvectors

In [linear.py](/Users/corbensorenson/Documents/cartpole/src/gcartpole/linear.py), `modal_input_coupling` projects the input onto right eigenvectors. For a nonnormal state matrix, modal input coefficients require left eigenvectors, or the corresponding rows of an inverse eigenvector basis. The current projection can report a zero coupling for a controllable mode.

The audit reproduces this on `A=[[1,10],[0,2]]`, `B=[[0],[1]]`: the controllability matrix has rank two, while the current routine reports zero weakest unstable coupling.

That file also uses a simplified point-mass plant and a raw powers-of-A controllability matrix. It is a ranking diagnostic, not an exact test of the MuJoCo target's physical controllability. Scaling and finite-horizon singular values, together with exact-model left-mode or PBH checks, are better diagnostics. The newer mass-normalized mechanical mode calculation is a different implementation and does not share this specific right-eigenvector error.

### Numerical integrity checks remain incomplete and inconsistent

The ledger correctly retracts the apparent long holds under strong split support: MuJoCo warnings and zeroed state rows had been counted as upright. The repair in `search_ilqr_capture.py` catches several collapse patterns, but the environment itself still accepts `mj_step` outputs without rejecting warning counters or simulator time resets. The parked evaluator, chain search, and state-bank evaluator do not uniformly apply a simulator-level guard.

The audit constructs a trace that passes the current heuristic despite collapsing to zero after a brief upright row. The heuristic allows this because it sees an earlier upright state. It has no warning or simulator-time evidence from which to distinguish a physical trajectory from a simulator reset.

The guard should run at the simulator boundary on each internal step, checking state finiteness, relevant warnings, and expected time advancement. Optimizer transitions need the same guard: otherwise they can differentiate through a simulator reset and treat its zero state as a cheap endpoint. Invalid dynamics must stop the candidate rather than yield an attractive cost.

The existing success function also uses the longest upright streak anywhere in an episode. For future robustness experiments, final sustained hold should be recorded separately. This is a metric distinction, not a retraction of the ten link release, whose fresh replay remains upright through the end.

### Successful Riccati return is not a valid capture certificate

The upright gain routine returns the DARE result without independently checking its residual and the resulting closed-loop poles. The audit finds returned but unstable designs at twelve and thirteen links.

Equivalent coordinate transformations also produce very different numerical results: at eleven, the default physical-coordinate balanced solve gives radius `0.995610`, while the algebraically equivalent scaled-coordinate balanced solve gives `2.986233`. At twelve neither of the tested physical/scaled balanced/unbalanced designs was stable. Changing coordinates alone is therefore not an established repair.

The implementation should verify residuals, symmetry, positivity, closed-loop stability, and representative exact nonlinear rollouts. A scaled residual should be assessed against all equation terms, not just a nominal absolute tolerance. The current audit records a residual normalized by the Riccati matrix norm as an additional diagnostic; it is not a sufficient acceptance criterion by itself.

I also cross-checked the upright stepping derivatives by constructing the continuous linear system from MuJoCo mass, passive gravity stiffness, and damping, then applying the RK4 polynomial across the same four physics steps. Across ten through fifteen links, the resulting state/input matrices agree with the existing transition finite differences to relative errors below `7e-13`. The mechanical construction still uses finite differences for gravity stiffness, so it is not an independent validation of all MuJoCo mechanics. It does rule against a gross error in the stepping derivative as the immediate explanation: the rebuilt twelve link design remains unstable, with radius about `13.115`, and fourteen/fifteen still fail the default Riccati solve. The eleven link capture results are unchanged on the sampled `1e-7`, `1e-8`, and `1e-9` directions. The [supplemental audit](/Users/corbensorenson/Documents/cartpole/scripts/audit_upright_linearization.py) and [measurements](/Users/corbensorenson/Documents/cartpole/runs/frontier_audit_20261001/mechanical_linearization_reproduced.json) retain this check.

This behavior is consistent with the numerical sensitivity described in the official [SciPy DARE documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html): isolating the stable subspace can fail, and balancing can amplify numerical noise. The unstable result is a numerical limit of this implementation, not a theorem about twelve link physics.

## Support curricula were sometimes numerically harder than the target

The near-locked split path exactly preserves the predecessor on a constrained manifold. It does not initially solve the added degree of freedom. Lifted source feedback is `K_source P`; any new state direction in the kernel of the projection receives no source correction. Releasing that direction requires a new controller and a new reachable route.

The rigid split implementation also starts with inserted-body masses near a `1e-8` fraction and zero inserted armature, then redistributes mass and inertia during release. This creates a singularly perturbed problem. Tiny changes in the homotopy coordinate can have large dynamical consequences. Reaching a near-locked progress of approximately `2e-5` or `4e-5` is not meaningful progress toward the free uniform endpoint without measuring the actual physical quantities and modes.

The separate free-split stiffness/damping branch has a clearer numerical failure. The XML fixes RK4 at a 0.005-second physics step. I computed the local frozen damping operator `-M^-1 D` and evaluated the RK4 stability polynomial on its poles.

| Added support configuration | Fastest damping pole per second | Maximum frozen damping RK4 amplification per physics step | Full hanging discrete spectral radius per policy step |
| --- | ---: | ---: | ---: |
| Canonical eleven link plant | -1.970 | approximately 1 | 1.000 |
| Spring 40 and damper 0.8 | -868.614 | 7.259 | 880 |
| Spring 50 and damper 1.0 | -1,085.571 | 19.817 | 63,244 |
| Spring 100 and damper 2.0 | -2,170.355 | 413.862 | about 19.4 billion |

The frozen damping calculation is a local diagnostic, not a nonlinear theorem. The full hanging transition provides a second check: these support configurations have enormous numerical growth even near the supposedly easy equilibrium. Their failures cannot be used to rule out a physically integrated support-removal path.

For the damper-one configuration, a necessary bound suggested by the isolated negative-real damping pole is a physics step below roughly 0.00257 seconds; spring and nonlinear effects still require stricter accuracy checks. A concrete retry would use 0.001 seconds and 20 internal steps per unchanged 0.02-second command, then compare against 0.0005 seconds and 40 steps. This preserves action timing while testing integration convergence. An implicit integrator is another diagnostic option, but changes the discrete plant and must be labeled accordingly.

The official [MuJoCo integrator documentation](https://mujoco.readthedocs.io/en/3.3.5/computation/#integrators) explains the distinction between fixed-step RK4 and implicit treatment of damping. Improving numerical integration in temporary supported plants should precede judging whether their continuation is physically useful. The canonical release plant and its frozen timestep should remain separate from those discovery tests.

## Naive research assumptions that repeatedly consumed budget

### Geometric proximity was treated as capture proximity

Several route objectives reward ever becoming upright, the best angle at one time, or peak potential height. In the current chain score, merely having an upright event gives a 120-point reward before the actual capture succeeds. The scalar quality penalty helps, but does not measure distance to the bounded controller's recoverable region.

Hinge-rate RMS also misses collective motion: absolute link rates are cumulative sums of relative hinge rates. The generalized transformed velocity block retains relative rates. Neither a relative-rate threshold nor an isotropic terminal identity represents the weak capture directions of the full chain.

The repository has already added actuator-aware terminal factors and exact LQR-tail evaluation in some tools. The problem is that these remain optional and are not consistently built into whole-route synthesis, terminal constraints, and promotion. A heavy quadratic penalty at one endpoint is not equivalent to a finite-horizon bounded capture constraint.

### More energy or more rail was expected to cure a phase problem

The ledger contains near-full potential-height crossings with high cart and internal velocities, followed by failed linear and nonlinear capture. It also records wide-rail failures that simply moved outward with the boundary. Energy injection and cart centering are useful, but they do not specify how eleven modes should arrive together.

Longer horizons and wider rails may expose otherwise hidden routes; they are legitimate labeled discovery coordinates. They should only be retained when the candidate gains measurable capture margin and can be contracted back toward the target constraints. Boundary chasing is a failed controller symptom.

### Linear modes were extrapolated too far

The analytic minimum-energy modal seed is derived at hanging equilibrium and asks that local linear model to make an order-pi transition. It can solve its linear equations accurately while its exact nonlinear rollout never approaches capture. The energy pump uses only a collective power direction and an aggregate internal damping direction, with a fixed small-oscillation modal basis.

That is an interpretable proposal family, not a universal large-angle synthesis law. At large motion, mode shapes, phase relationships, nonlinear couplings, and effective controllability change along the route. The earlier harmonic co-design study itself explicitly separates changed frequencies and local recovery from a demonstrated nonlinear harmonic swing-up mechanism.

### Learning was asked to discover an absent bridge

PPO runs initialized at the hanging target, route imitation from assisted dynamics, residual learning around a failing transferred route, and scaled near-upright curricula all face a missing executable bridge. A handful of already settled bank states provides almost no information about actions along that bridge.

Imitation error on recorded source states does not certify target-plant closed-loop execution. Reset curricula that scale angles or velocities create convenient initial states but do not establish that a hanging-start policy can reach them. Teacher trajectories need to be dynamically valid on the target plant, and curriculum expansion should follow states connected by executable capture controllers.

The short direct PPO failures reject those specific initialization/objective/budget combinations. They do not show that policy learning is inherently incapable of solving eleven links. Simply increasing the same training duration is nevertheless a low-information next experiment.

### A bounded failed run was sometimes treated as closing a method family

The experiment ledger is unusually careful about separating claims and retracting false positives. Its repeated statements that a path is closed should be read as closing the tested implementation and budget. Invalid integration, false feasibility declarations, weak representations, and a poor seed can invalidate a broad conclusion that a whole algorithmic family is unsuitable.

This distinction matters for the paper: negative evidence should identify the tested parameterization, initialization, compute, dynamics, and failure class. It should not imply a general impossibility result from one local optimizer returning a failure.

## Why the generalist also stops

The current generalist is a reusable execution, transfer, refinement, selection, adaptation, and evidence pipeline. It is not yet an independently demonstrated route generator for every count or morphology.

The published uniform ladder documents that the seven link rung reuses the frozen record route; eight through ten use separate frontier releases. Generalized physical-scale transfer has meaningful successes, but matching dimensionless groups within a fixed topology does not preserve dynamics when the number and location of free joints change. Adding a link changes the state dimension and creates additional directions that the old controller never controlled.

The pipeline therefore inherits the frontier's dependence on a useful seed and target-plant local repair. Analytic mirror selection chooses between existing routes; it cannot create a missing route. The force gain/bias adapter addresses compensable actuator mismatch; it cannot correct structural modal differences or regain authority after saturation.

Fixed total length and mass also do not make increasing count an exact discretization of one unchanged mechanism. The benchmark retains a fixed armature per hinge and a fixed capsule radius while segment dimensions and damping per joint change. The initial angle noise is applied to relative joints, so distal absolute-angle variance grows with count. These are legitimate declared benchmark choices, but a paper should expose them when interpreting a count limit or comparing to a continuum chain.

The smaller unequal-morphology frontier is a useful control experiment. Even a three link target remains only partially reached by adaptive morphology continuation, and the two-to-three supported split path still rejects its unsupported endpoint. This demonstrates a route-branch and continuation limitation independent of large link count. Eleven exposes the same limitation more severely because capture sensitivity has grown substantially.

The older consolidated research handoff also explicitly lists nonlinear modal continuation, executable broad capture regions, reverse curricula, and full global learning integration as unfinished tasks. Those are plans and foundations, not features already delivered by the current generalist.

Consequently, count-agnostic code is an engineering property. Reliable arbitrary-count discovery is a scientific capability that still needs its own benchmark, budgets, independent synthesis runs, and evidence.

## Recommended experiments in priority order

### Establish a trustworthy optimizer and capture baseline

First repair the state/action warm-start mismatch, measure feasibility rather than assert it, use a consistent angular state convention, and reject simulator warning/reset transitions at their source. Add focused regression tests for each failure demonstrated by the audit. Preserve the existing release artifacts and rerun their evaluations after any change.

Then treat capture design as its own numerical project. Check the upright DARE result before use, sweep solver tolerances and derivative scales, compare independent derivative constructions, and examine coordinate/precision sensitivity. Do not accept a gain merely because a routine returned it. At eleven, measure recoverable directional radii, raw and clipped action trajectories, peak transient motion, and sensitivity to perturbations around the *actual* arrival states. At twelve and above, numerical validity comes before interpreting a failed nonlinear rollout.

This work has clear acceptance criteria: dynamically consistent warm starts; warning-free transitions; matching derivative predictions over shrinking perturbations away from branch cuts; valid checked capture gains; and retained seven through ten release behavior. It can reveal whether the present numerical capture limit is unnecessarily restrictive before spending another global-search budget.

### Build nonlinear capture bridges from measured reachable states

Use the settled bank as a terminal anchor, not as the complete capture curriculum. Solve bounded nonlinear capture trajectories from progressively harder initial states into that anchor, with explicit dynamics, force, rail, and final-hold constraints. Retain only states for which the exact capture controller actually succeeds.

Expand along directions suggested by free-plant swing crossings and capture sensitivities. A backward reachable-set construction through optimization is appropriate; literally reversing a damped downfall trajectory is not. Add time-varying feedback and bounded receding correction around the accepted bridges. Verify that adjacent stages overlap under perturbation.

The scientific test is whether this procedure reaches a clean state visited by a free hanging-start swing, or whether a swing optimizer can reach one of its verified entry states. The two experts must have intersecting reachable/recoverable sets. Independent component scores cannot substitute for that intersection.

### Synthesize a new whole trajectory branch with explicit states

Use sparse direct transcription or multiple shooting with state variables, explicit dynamics defects, a consistent angle lift/manifold, and hard force/rail constraints. Include terminal capture dynamics or constraints derived from a verified capture bridge. Allow timing to vary within the frozen episode budget. Use several materially different initial paths rather than only the transferred ten link waveform.

The repository already contains multiple shooting and a function named direct collocation. The latter enforces defects by rolling through each shooting segment; it is a discrete multiple-shooting transcription, not a new Hermite-Simpson or implicit continuous-mechanics formulation. Historical attempts around the same failing seed do not cover every transcription design. The relevant difference is better conditioning, coherent state branches, validated defects, and an executable terminal region.

The standard [MIT trajectory optimization notes](https://underactuated.mit.edu/trajopt.html) describe the distinction between shooting and state-based transcription and their conditioning. My recommendation is an inference from that numerical structure and this repository's failed local repairs; it is not a claim that collocation guarantees a solution.

Judge progress by validated dynamics defect, actual capture success, required rail, force margin, and robustness of the arrival interval. Once a nominal exact route exists, use target-plant feedback refinement and the successful settled-launch architecture to recover the full noisy benchmark.

### Revisit support continuation only with converged integration

If support continuation is retained, first demonstrate that the supported starting plant is accurately integrated at the unchanged action period. Give the new mode finite physical inertia during discovery, with any departures labeled, and explicitly build feedback for it. Separate geometry, inertia, stiffness, damping, and equality release as measured coordinates where possible.

A successful revised path would explain both why the old branch failed and why the new route survives support removal. Another sequence of microscopic homotopy steps ending in instability would provide little new information.

### Benchmark the generalist independently of the saved frontier routes

Freeze a recipe, hyperparameter policy, and compute budget. Evaluate independent synthesis from declared generic seeds over counts, scale families, and unequal morphologies. Record whether each solution required a previously solved frontier route, manual arrival shaping, or a new branch. Use leave-one-morphology-out tests and report unsuccessful cases alongside successes.

This separates portable execution from portable discovery. It also gives a paper a stronger contribution than a sequence of count-specific successes: which parts of the recipe transfer, how much synthesis work is required, and where the actual algorithm ceases to work.

## What breaking point should mean for the paper

A link-count stopping point alone confounds several limits. Measure at least nominal reachable count, noisy-start success after launch contraction, arrival/capture perturbation tolerance, model and sensor uncertainty, actuator authority, rail requirement, episode duration, and end-to-end synthesis compute. Holding an accurately simulated nominal route may remain possible after practical robustness has already become very poor.

For the existing benchmark, the most defensible claim remains the demonstrated reset-free seven through ten link result. The fresh evidence identifies an eleven link discovery/capture bottleneck and a twelve-plus numerical LQR bottleneck. Neither is a demonstrated physical impossibility boundary.

The promising paper concept is composition of launch contraction, nonlinear route synthesis, and verified capture, with explicit measurements of how the overlap between those stages degrades. A broad claim about a general solver or a harmonic mechanism needs independent route generation and a causal ablation, respectively. The earlier co-design studies caution that changing mass and length distributions also changes lifting work, inertia, and saturation, so an apparent resonance benefit needs matched comparisons.

My judgment is that another link is plausible, especially given the clean centered eleven link crossings already recorded. The strongest next step is to enlarge or deliberately enter a verified bounded capture region using corrected, numerically trustworthy synthesis. Repeating angle-ranked search, widening the rail, or training from the same assisted teacher would continue to optimize proxies whose connection to success has already failed.

## Reproduction and retained evidence

From the repository root:

```bash
PYTHONPATH=src:scripts .conda-aligator/bin/python scripts/audit_link_count_frontier.py \
  --out runs/frontier_audit_20261001/reproduction.json

PYTHONPATH=src:scripts .conda-aligator/bin/python scripts/verify_ten_link_release.py

PYTHONPATH=src:scripts .conda-aligator/bin/python scripts/audit_upright_linearization.py \
  --out runs/frontier_audit_20261001/mechanical_reproduction.json

PYTHONPATH=.:src:scripts .conda-aligator/bin/python -m pytest -q
```

The audit refuses to overwrite an existing output. It uses no training, paid services, or parameter changes to the saved releases. Its supported-plant checks use the existing temporary configurations when present. Their results remain development diagnostics. The current working tree contains pre-existing uncommitted changes; hashes and runtime metadata in the audit identify the files actually reviewed.

Key local sources are the [consolidated method paper](/Users/corbensorenson/Documents/cartpole/docs/seven_link_swingup_paper.md), [ten link appendix](/Users/corbensorenson/Documents/cartpole/docs/ten_link_swingup_paper.md), [generalized solver design](/Users/corbensorenson/Documents/cartpole/docs/generalized_solver.md), [experiment ledger](/Users/corbensorenson/Documents/cartpole/docs/levers_and_pitfalls.md), and [harmonic co-design results](/Users/corbensorenson/Documents/cartpole/harmonic_codesign_study/RESULTS.md).
