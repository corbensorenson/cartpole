# Nine-Link Cart-Pole Swing-Up

## A Parked-Launch Feedback Route With Angle-Weighted Capture

**Status:** canonical internal 20/100-gate result; external competition match remains open

**Date:** 2026-09-13

**Target:** uniform nine-link MuJoCo cart-pole, hanging-start swing-up and hold

## Abstract

This report documents a reset-free nine-link swing-up and hold controller on
the canonical uniform MuJoCo plant. The controller inherits the seven- and
eight-link architecture: hanging-equilibrium conditioning, a saved finite-
horizon Box-FDDP swing route with time-varying feedback, and a terminal
upright LQR. The nine-link route required one link-count-specific optimizer
change: the terminal objective weighted the absolute-angle block by `20x` and
the terminal hinge-rate block by `4x`. A 14-second hanging park at `-0.05 m`
then reserved enough rail margin for the feedback route.

The frozen controller passes `20/20` and `100/100` noisy hanging-start
episodes, plus an exact `20/20` check. All noisy episodes reach the 30-second
time limit without rail failure or reset. A held-out reset-free video shows
all nine links and holds upright for `12.18 s` after first upright. This is an
internal canonical benchmark result, not a claim that the route matches an
external competition's undisclosed plant or record rules.

## 1. Benchmark And Claim Boundary

The target is `configs/swingup9_uniform.yaml`:

| Quantity | Value |
|---|---:|
| Links | 9 |
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

The hanging initial state is relative angles `[pi, 0, ..., 0]` with
independent angle and generalized-velocity noise of scale `0.05`. Parking the
cart is an action sequence inside the episode. It does not overwrite qpos,
qvel, simulator time, or hidden state.

The evidence bundle satisfies the repository's internal promotion gate:

1. `20/20` noisy episodes pass.
2. `100/100` disjoint noisy episodes pass.
3. An exact `20/20` replay passes.
4. A held-out 30-second video completes with zero resets and full framing.
5. The route, feedback gains, config, XML, evaluations, and video metadata are
   hashed in the nine-link manifest.

It is not sufficient evidence for an external world-record claim until the
external plant, rail geometry, force convention, timing, noise distribution,
and judging rule are matched and independently checked.

## 2. Controller Architecture

The runtime controller has three deterministic phases in one uninterrupted
MuJoCo episode:

1. **Hanging park.** For `14.0 s`, a hanging-equilibrium LQR damps the chain
   and drives the cart toward `-0.05 m`. The measured state is carried forward
   directly.
2. **Swing expert.** A 197-step (`3.94 s`) Box-FDDP route is replayed with
   saved time-varying feedback in dimensionless absolute-coordinate state. The
   nominal cart channel is translated by the measured parked cart position.
   Feedback scale is `1.0`.
3. **Capture and maintenance expert.** After the route horizon, the terminal
   upright LQR runs at scale `1.0`, targeting the same `-0.05 m` cart reference.

The route is stored in
`runs/generalized_solver/n9_fddp_terminal_angle20_widerail.json`. Its optimizer
used a diagnostic soft rail setting of `6.0 m` so terminal angle quality could
be explored without rail cost dominating the local solve. That setting is not
the benchmark rail. The exact controller was replayed independently on the
canonical `+/-3 m` rail; its deterministic replay peaked at `2.9801 m`, and
the noisy 100-episode gate peaked at `2.9800 m`.

## 3. What Actually Worked

### 3.1 The eight-link phase structure generalized

Direct 8-to-9 route transfer was not sufficient. The successful controller
kept the settled hanging launch, saved route, state feedback, and delayed LQR
capture from the eight-link result. The link-count-specific work stayed inside
the existing Box-FDDP route optimization rather than introducing a new policy
family.

### 3.2 The park target was part of the solution

The first five-seed probe at `-0.15 m` produced `4/5` success and one rail
failure. Moving the settled target to `-0.05 m` produced `20/20` in the next
cohort and kept the maximum cart excursion below the canonical rail. The
target is therefore frozen as a controller parameter, not an undocumented
presentation adjustment.

### 3.3 Terminal angle quality was the missing route objective

The balanced terminal norm produced a near-upright nine-link endpoint with
maximum angle about `0.314 rad`. A terminal LQR saturated from that state even
when the route had low hinge RMS. A strict open-loop tail and a wide-rail tail
both retained too much internal momentum.

The decisive route refinement increased the terminal absolute-angle block by
`20x` and the hinge-rate block by `4x`. The resulting canonical exact replay
arrived at the route boundary with:

| Handoff quantity | Result |
|---|---:|
| Maximum absolute angle | `0.00004 rad` |
| Hinge-rate RMS | `0.00068 rad/s` |
| Cart velocity | `0.00041 m/s` |
| First route handoff time | `3.94 s` |

This is why the terminal LQR can remain a simple maintenance expert: the swing
expert delivers a genuinely quiet state instead of asking capture to arrest
unresolved internal modes.

### 3.4 Feedback is required during the swing

The inherited route is not accepted as an open-loop force trace. The noisy
evaluation replays the saved time-varying feedback gains on the exact target
MuJoCo plant. This preserves the route's terminal approach while correcting
the hanging-start perturbations. The noisy result is therefore a property of
the full two-expert route, not of a hand-picked deterministic trajectory.

## 4. Results

| Gate or metric | Result |
|---|---:|
| 20-episode noisy gate | 20/20 = 100% |
| 100-episode noisy gate | 100/100 = 100% |
| Exact 20-episode check | 20/20 = 100% |
| Park phase | 14.00 s to cart target -0.05 m |
| Swing route | 3.94 s / 197 policy steps |
| First upright in held-out video | 17.84 s |
| Upright hold through video end | 12.18 s |
| Maximum cart excursion, noisy 100 | 2.9800 m |
| Episode terminations, noisy 100 | 100 time limits, 0 rail failures |
| Resets inside episodes | 0 |

The primary evidence files are:

- `runs/generalized_solver/n9_fddp_terminal_angle20_parked_target005_20.json`
- `runs/generalized_solver/n9_fddp_terminal_angle20_parked_target005_100.json`
- `runs/generalized_solver/n9_fddp_terminal_angle20_parked_target005_exact20.json`
- `runs/generalized_solver/nine_link_swingup_success.mp4`
- `runs/generalized_solver/nine_link_swingup_success.video.json`
- `runs/generalized_solver/nine_link_swingup_manifest.json`

The noisy 20-episode cohort uses seeds `91021--91040`; the noisy 100-episode
cohort uses `91041--91140`. The exact cohort uses zero noise with seeds
`91061--91080`. The video uses held-out seed `91141`, outside both noisy
cohorts.

## 5. Negative Controls And Limits

The following results remain in the repository and are not silently replaced:

- Direct generalized 8-to-9 transfer: `0/5` parked noisy success.
- Balanced-terminal-objective nine-link route: no sustained hold from the
  measured high-angle endpoint.
- Strict 3 m tail CEM from the improved endpoint: no valid independent replay.
- Wide-rail tail CEM: lower angle but persistent high hinge and absolute-rate
  momentum.
- The repaired locked-link morphology continuation: unresolved and excluded
  from the canonical result.

The result is specialized to the frozen uniform nine-link plant and exact
action timing. It does not establish robustness to link-length or mass
variation, friction changes, actuator delay, sensor noise, a different rail,
or a different force limit. It also does not establish an external record
without a matched benchmark comparison.

## 6. Reproduction

From the repository root, use the configured planner environment:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTHONPATH=.conda-aligator/lib/python3.12/site-packages:src:scripts
PY=.conda-aligator/bin/python

$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup9_uniform.yaml \
  --controller runs/generalized_solver/n9_fddp_terminal_angle20_widerail.json \
  --episodes 20 --seed 91021 --park-seconds 14 --cart-target -0.05 \
  --tracking-gain-scale 1.0 \
  --out runs/generalized_solver/n9_fddp_terminal_angle20_parked_target005_20.json

$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup9_uniform.yaml \
  --controller runs/generalized_solver/n9_fddp_terminal_angle20_widerail.json \
  --episodes 100 --seed 91041 --park-seconds 14 --cart-target -0.05 \
  --tracking-gain-scale 1.0 \
  --out runs/generalized_solver/n9_fddp_terminal_angle20_parked_target005_100.json

$PY scripts/render_fddp_parked_route_video.py \
  --config configs/swingup9_uniform.yaml \
  --controller runs/generalized_solver/n9_fddp_terminal_angle20_widerail.json \
  --seed 91141 --park-seconds 14 --cart-target -0.05 \
  --tracking-gain-scale 1.0 --seconds 30 --fail-on-failure \
  --out runs/generalized_solver/nine_link_swingup_success.mp4 \
  --metadata-out runs/generalized_solver/nine_link_swingup_success.video.json
```

The manifest records the exact hashes and execution settings. A fresh-clone
audit must run these commands against a clean checkout and verify the hashes
before any external claim is made.

## 7. Conclusion And Next Experiment

Nine links is now an internal canonical solve under the same architecture that
solved seven and eight: hanging-state conditioning, a saved state-consistent
swing route, time-varying feedback, a measured parked cart reference, and
delayed terminal LQR capture. The next frontier is ten links. It should start
from this nine-link controller and preserve the same launch, feedback, and
capture contract while re-optimizing only the link-count-dependent route.
