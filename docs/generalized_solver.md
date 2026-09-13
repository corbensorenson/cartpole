# Generalized n-link swing-up and hold

## Objective and claim boundary

The objective is one solver **recipe** that accepts an arbitrary serial-chain
cart-pole morphology and returns a swing-up-and-hold controller. The same code
path should handle different link counts, link lengths, masses, damping, cart
mass, force authority, control rate, and rail length. It does not mean that the
same recorded force samples can be replayed unchanged on every plant.

The existing seven-link release remains the only public record claim. The
bottom-up ladder below is development evidence for a reusable solver recipe;
it does not retroactively change the record controller. Files produced by
`generalized_swingup_solver.py transfer` are warm starts, explicitly labelled
`not_solution`, and are not evidence of success.

## The dimensionless plant

Let

- \(L=\sum_i l_i\) be total chain length;
- \(M=m_c+\sum_i m_i\) be total moving mass;
- \(t_0=\sqrt{L/g}\) be the gravitational time scale; and
- \(v_0=\sqrt{gL}\) be the translational velocity scale.

The solver describes a setup using the following dimensionless quantities:

| Group | Definition | Meaning |
|---|---:|---|
| Link geometry | \(l_i/L\) | where each joint lies along the chain |
| Link mass | \(m_i/\sum_jm_j\) | mass distribution along the chain |
| Cart ratio | \(m_c/\sum_i m_i\) | cart inertia relative to the chain |
| Force authority | \(\phi=F_{max}/(Mg)\) | maximum horizontal force relative to weight |
| Rail ratio | \(\rho=R/L\) | rail half-length relative to chain length |
| Usable rail | \((R-r_c)/L\) | rail after allowing for cart half-length |
| Control period | \(\Delta t/t_0\) | controller rate on the gravitational clock |
| Cart damping | \(c_xt_0/M\) | dimensionless translational damping |
| Joint damping | \(d_it_0/(ML^2)\) | dimensionless rotational damping |
| Armature | \(J_a/(ML^2)\) | dimensionless joint inertia |
| Link radius | \(r_l/L\) | finite-body geometry scale |

Plants with matching groups are dynamically similar. Length scaling therefore
changes trajectory duration by \(\sqrt{L}\), cart position by \(L\), cart speed
by \(\sqrt{L}\), angular speed by \(1/\sqrt{L}\), and physical force by mass.

## Solver architecture

```text
measured morphology
        |
        v
dimensionless plant + upright controllability audit
        |
        v
nearest solved anchor -- arc-length state/feedback transfer
        |                         |
        |                         +-- natural-time resampling
        |                         +-- force-authority scaling
        v
exact MuJoCo feasible rollout (warm start only)
        |
        v
bounded Box-FDDP / direct trajectory optimization
        |
        +-- rail continuation: measure max |x|, solve for minimum rho
        |
        v
terminal Riccati funnel + sustained-upright gate
        |
        v
bounded online force calibration; replan if mismatch is structural
```

The promotion order is deliberately `1 -> 2 -> 3 -> ...`, never a search that
starts at six or seven. At every count the process is:

1. solve the simplest deterministic energy/phase problem available;
2. compute an exact target-plant trajectory and time-varying feedback;
3. add its exact planar mirror rather than searching both directions;
4. select a route by exact forward simulation from the measured settled state;
5. require a full noisy swing-and-five-second-hold gate;
6. back-check all previously accepted link counts before promoting the new one.

The exact route selector is intentionally narrow. It evaluates a small,
symmetry-derived library with the plant model; it is not an RL policy. The
bounded actuator RLS adapter is narrower still: it can estimate force gain and
bias, but structural mismatch must trigger replanning.

Absolute link orientation is treated as a field over normalized chain arc
length. Transfers convert relative joints to that field, interpolate at the
target link centers, and convert back. This lets unequal-length chains transfer
without pretending that “link 4” has the same physical meaning everywhere.

### Why total energy stops being enough

For one link, normalized energy error and angular phase describe the swing
well enough for exact partial feedback linearization followed by LQR. With two
or more links, the same total energy can be distributed among internal modes
that cancel in aggregate momentum. The two-link ladder therefore adds an
explicit deterministic phase term. At three links, an exact modal objective
(absolute-angle coherence plus absolute-rate damping) is needed before Box-FDDP
can arrest the chain. At four links, one aggregate phase variable reaches the
upright neighborhood but leaves too much energy in the distal modes. Exact
morphology-derived normal modes resolve that ambiguity: ranking measured PFL
states by collective and internal modal energy identifies a better handoff,
and full-horizon Box-FDDP then closes the route.

Five links exposed a second relationship: the three-metre rail used at lower
counts was too short for the deterministic approach that Box-FDDP could
capture. Continuing the rail to 4.5 m let the same fixed-size 13-parameter PFL
proposal produce a handoff that exact tail optimization could arrest. A final
full-horizon Box-FDDP pass from hanging supplied stabilizing feedback for the
entire route. Allowing phase-adaptive skips reduced robustness; strict
time-order replay with local feedback passed. This promotes five links without
adding per-link learned parameters. Six links is the next unsolved stage.

The online adapter estimates only actuator effectiveness and bias. It projects
observed one-step model error onto the exact model's action Jacobian, fits
`equivalent_action = gain * command + bias` with bounded recursive least
squares, and compensates conservatively. Mass, length, timing, or contact errors
that cannot be explained along the action direction require short-horizon
re-optimization; they must not be hidden by an unbounded adaptive controller.

## Rail-length relationship

There is no morphology-only constant guaranteeing swing-up: rail demand also
depends on force authority, control rate, damping, and the chosen route. For a
specific successful trajectory, the exact requirement is

\[
\rho_{required}=\frac{\max_t |x(t)|+r_c+c}{L},
\]

where \(r_c\) is cart half-length and \(c\) is desired clearance. The solver
records this value for every route. A rail continuation then tightens \(R\),
re-optimizes, and brackets the smallest passing \(\rho\). Reporting this curve
against link count and the other dimensionless groups is meaningful; fitting a
single rail/link-count line before those controls are fixed is not.

The MuJoCo slide limit constrains the **cart center**. The body-aware ratio
reported here additionally includes the 0.18 m cart half-length, so a successful
run can have center travel below the configured `3.0 m` limit while reporting
`rho_required` slightly above `1.0` for a 3 m chain. This distinction is
intentional and must accompany any physical-rail comparison.

## Verified bottom-up checkpoint

All rows use uniform 3 m, 1 kg chains, a 1 kg cart, 80 N authority, 50 Hz
control, the declared noisy hanging start, no state resets, and a five-second
upright requirement.

| Links | Result | Deterministic backbone | Required physical rail ratio |
|---:|---:|---|---:|
| 1 | 20/20 | energy PFL -> exact LQR | 0.9594 |
| 2 | 20/20 | phase-aware PFL -> Box-FDDP -> exact LQR; mirror selection | 1.0127 |
| 3 | 20/20 | modal route -> full-horizon Box-FDDP -> exact LQR; mirror selection | 1.0008 |
| 4 | 20/20 | modal-ranked PFL handoff -> full-horizon Box-FDDP -> exact LQR; mirror selection | 0.9953 |
| 5 | 20/20 | PFL handoff -> rail continuation -> tail/full-horizon Box-FDDP -> exact LQR; mirror selection | 1.1706 |

Evidence: [n=1 gate](../runs/generalized_solver/energy_n1_noisy20_v2.json),
[n=2 gate](../runs/generalized_solver/n2_gate.json), and
[n=3 gate](../runs/generalized_solver/n3_gate.json), and
[n=4 gate](../runs/generalized_solver/n4_gate_20.json), and
[n=5 gate](../runs/generalized_solver/n5_gate_20.json).
Prediction and uninterrupted execution agree on every accepted n=2 through
n=5 episode. The [n=5 frontier artifact](../runs/generalized_solver/frontier_n5.json)
records the three-metre failure boundary, rail continuation, deterministic
repair, and accepted gate.

## Current reproducible commands

```bash
make generalized-gates

PYTHONPATH=src python scripts/generalized_swingup_solver.py analyze \
  --config configs/swingup7_uniform.yaml --min-links 1 --max-links 20 \
  --out runs/generalized_solver/morphology_n1_n20.json

PYTHONPATH=src python scripts/evaluate_generalized_energy.py \
  --config configs/swingup7_uniform.yaml --n-links 1 --episodes 20 \
  --out runs/generalized_solver/energy_n1_noisy20.json

PYTHONPATH=src python scripts/generalized_swingup_solver.py transfer \
  --source-config runs/generalized_solver/n3_uniform_exact.yaml \
  --source-links 3 --target-links 4 \
  --controller runs/generalized_solver/n3_route.json \
  --out runs/generalized_solver/n3_to_n4_warm_start.json

PYTHONPATH=src python scripts/evaluate_generalized_route_library.py \
  --config configs/swingup7_uniform.yaml --n-links 5 \
  --override env.rail_limit=4.5 \
  --controller runs/generalized_solver/n5_route_solver.json \
  --controller runs/generalized_solver/n5_route_solver_mirror.json \
  --episodes 20 --seed 78001 --conditioning-seconds 15 \
  --tracking-gain-scale 1 --phase-window 0 \
  --out runs/generalized_solver/n5_gate_20.json
```

The morphology table and transfer output are analysis/warm starts, not success
claims. Promotion requires exact optimization followed by independent noisy
evaluation with rail checks and a five-second upright hold.

## Current six-link boundary

Six links has not been promoted. Arc-length transfer from both accepted n=5
and the frozen n=7 release produced exact, rail-feasible target-plant warm
starts, but neither direct replay nor full-horizon Box-FDDP reached capture.
Increasing the diagnostic rail from 6 m to 12 m did not fix the n=7-to-n=6
route, which rules out rail clipping as that branch's primary failure.

The fixed-size 13-parameter PFL search reached 0.936 rad from upright at its
best exact-modal handoff, but retained too much absolute-rate and internal-mode
energy for the tested tail optimizer. A three-scalar extension adds exact
collective-mode pumping, gated internal-mode damping, and a modal acceleration
limit while keeping parameter count independent of links. It reduced unwanted
motion but did not improve the complete handoff score.

The next deterministic layer is now implemented. It builds the exact hanging
small-oscillation model from the measured mass, stiffness, damping, and cart
coupling matrices, discretizes it on the plant's natural clock, and solves a
regularized finite-horizon controllability-Gramian problem. The resulting
acceleration schedule has no learned or per-link tuning parameters. At n=6, a
3.9 s schedule placed the *linear* model within `5.36e-4` of its complete target
and the exact nonlinear plant passed within `0.1783 rad` of upright at 2.56 s.
That is the best six-link approach on this generalized track, but its
`3.9028 rad/s` absolute angular-rate RMS and body-aware required rail ratio
`1.9464` are far outside the capture envelope. It is a warm start, not a solve.

This experiment also exposed an action-precision trap: tail CEM had evaluated
float64 actions while `env.step` accepts float32 actions. The chaotic n=6
trajectory diverged under real replay. Tail search now quantizes every batched
candidate at the public action boundary and emits a mandatory independent
serial verification. The invalid earlier tail proposal is not evidence and is
not published as a controller. A fresh 100-iteration search under the corrected
precision contract produced no feasible handoff; its best proposal crossed the
12 m diagnostic rail at 12.046 m and was still 0.699 rad from upright.

Reproduce the analytic warm start with:

```bash
PYTHONPATH=src python scripts/materialize_modal_phase_seed.py \
  --config configs/swingup7_uniform.yaml --n-links 6 --seconds 3.9 \
  --override env.rail_limit=12.0 \
  --out runs/generalized_solver/n6_analytic_modal_seed_h3p9.json
```

The next step is deterministic nonlinear continuation that jointly penalizes
cart travel and terminal modal velocity before Box-FDDP, followed by the same
noisy gate and lower-count regressions. See the
[compact frontier record](../runs/generalized_solver/frontier_n6.json) and
[analytic checkpoint](../runs/generalized_solver/frontier_n6_analytic_phase.json).

### Whole-route nonlinear continuation

The analytic schedule is now preserved at full policy resolution while a
serial exact-MuJoCo CEM searches only bounded, low-dimensional additive
residual knots. This is the intended thin optimization layer: it cannot replace
the deterministic controller, and every candidate is evaluated through the
same float32 `env.step` path used by replay.

A conventional weighted sum reduced route cost from `5815.25` to `3709.99`
and cart-center travel from `5.659 m` to `5.165 m`, but it exposed an important
failure mode: the optimizer could buy lower terminal cost by moving away from
upright. The search therefore supports a capture-envelope barrier that chooses
the phase with the smallest explicit violation of angle, hinge-rate,
absolute-rate, cart-position, and cart-velocity limits before considering the
smooth quality score.

On n=6, 100 barrier iterations reduced violation from `8.7709` to `4.1654`.
The best serial state at 4.18 s had `0.1772 rad` maximum angle, `1.0144 rad/s`
hinge RMS, `2.2515 rad/s` absolute-rate RMS, cart position `0.3435 m`, and cart
velocity `-0.4601 m/s`. Maximum cart-center travel fell to `4.6148 m`, a
body-aware rail ratio of `1.5983`. This is meaningful progress but still not a
valid handoff.

Exact upright modal decomposition localizes the remaining error: `98.05%` of
the measured modal energy is in the first collective mode and only `1.95%` in
all internal modes combined. The core solver now also exposes a finite-horizon
minimum-energy modal transition from any measured state, not only the hanging
equilibrium. Direct linear capture-tail probes were not nonlinear-feasible at
this energy, and Box-FDDP became unstable, so neither is promoted. The next
deterministic continuation should shape a phase-paired collective-mode braking
segment inside the full route rather than adding a larger learned controller.
