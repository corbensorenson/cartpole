# Eight-Link Cart-Pole Swing-Up

## A Parked-Cart Launch, Feedback Swing Route, And LQR Capture

**Status:** canonical internal 20/100-gate result; external competition match remains open

**Date:** 2026-09-13

**Target:** uniform eight-link MuJoCo cart-pole, hanging-start swing-up and hold

## Abstract

This report documents a reset-free eight-link swing-up and hold controller on
the repository's canonical uniform MuJoCo plant. It inherits the seven-link
architecture: active conditioning at the hanging equilibrium, a saved
finite-horizon feedback swing route, and a terminal upright LQR. The eight-link
route adds one practical change: during the hanging phase it parks the cart at
`-0.15 m`, using translational symmetry to reserve rail room for the swing.

The frozen controller passes `20/20` and `100/100` noisy hanging-start
episodes, plus an exact `20/20` no-noise check. All episodes complete the
30-second horizon with zero rail failures and zero resets. A held-out video
shows the complete eight-link chain from hanging start through capture and
holds upright for `12.22 s` before the episode ends. This is an internal
canonical benchmark result, not a claim that the route matches an external
competition's undisclosed plant or record rules.

## 1. Benchmark And Claim Boundary

The target is `configs/swingup8_uniform.yaml`:

| Quantity | Value |
|---|---:|
| Links | 8 |
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

The hanging initial state is relative angles `[pi, 0, 0, 0, 0, 0, 0, 0]`
with independent angle and generalized-velocity noise of scale `0.05`. The
final noisy evaluations use this distribution directly. Parking the cart is
part of the policy; it does not overwrite qpos, qvel, simulator time, or any
hidden state.

The evidence is sufficient for this repository's internal promotion gate:

1. `20/20` noisy episodes pass.
2. `100/100` noisy episodes pass.
3. An exact `20/20` replay passes.
4. A held-out 30-second video completes with zero resets and full framing.
5. The route, feedback gains, config, XML, evaluation outputs, and video
   metadata are hashed in the release manifest.

It is not sufficient evidence for an external world-record claim until the
external plant, rail geometry, force convention, timing, noise distribution,
and judging rule are matched and independently checked.

## 2. Controller Architecture

The runtime controller has three deterministic phases in one uninterrupted
MuJoCo episode:

1. **Hanging park.** For `14.0 s`, an LQR linearized at the hanging
   equilibrium damps the chain and drives the cart toward `-0.15 m`. The
   measured state is carried forward directly.
2. **Swing expert.** A 197-step (`3.94 s`) Box-FDDP route is replayed with
   saved time-varying feedback in dimensionless absolute-coordinate state.
   The nominal cart channel is translated by the measured parked cart position.
   The feedback scale is `0.75`.
3. **Capture and maintenance expert.** After the route horizon, the terminal
   upright LQR runs at scale `1.0`, with its cart-position target set to the
   same `-0.15 m` parked reference.

The route artifact is
`runs/generalized_solver/n8_capture_fddp_feedback120.json`. Its exact
trajectory was generated with a diagnostic wide rail during optimization, but
the final replay uses the canonical +/-3 m rail. The route is therefore only
accepted because the complete parked controller, not the optimizer's wide-rail
trace, passes the canonical noisy gates.

## 3. What Actually Worked

### 3.1 Reusing the seven-link structure

The first eight-link attempts applied the seven-link hanging LQR, saved swing
route, nominal-cart shift, and terminal LQR directly. They failed because the
eighth internal mode left the nonlinear capture basin. The successful route
keeps the same phase structure but replaces the open-loop swing waveform with
the saved Box-FDDP state feedback produced from the exact eight-link endpoint.

### 3.2 Parking exploits rail geometry without changing link dynamics

The chain dynamics are invariant to a uniform cart translation. A quiet
hanging LQR park therefore shifts the route's cart excursion without changing
the link trajectory. A centered route peaked just beyond the canonical rail;
parking at `-0.15 m` moved the final noisy maximum to `2.9300 m` while keeping
the same feedback route and force limit.

The shorter `12.0 s` park at `-0.20 m` looked promising in a five-seed probe,
but failed the larger check at `95/100`. The released choice is the longer,
more conservative `14.0 s` park at `-0.15 m`.

### 3.3 Feedback is required during the swing

The saved open-loop route passed an exact no-noise parked replay but scored
`0/20` on the noisy parked replay. The feedback route recovered the reset
noise while preserving the route's low-momentum terminal approach. This is the
same lesson as seven links: exact endpoint quality is not enough; the full
trajectory must be tracked in the true nonlinear plant.

### 3.4 Capture remains a separate expert

The upright LQR is deliberately delayed until the route horizon. It is not
used to create the swing energy. The final handoff is stateful and deterministic
inside one episode, and the capture law targets the parked cart reference so
the cart does not immediately spend the saved rail margin correcting back to
zero.

## 4. Results

| Gate or metric | Result |
|---|---:|
| 20-episode noisy gate | 20/20 = 100% |
| 100-episode noisy gate | 100/100 = 100% |
| Exact 20-episode check | 20/20 = 100% |
| Park phase | 14.00 s to cart target -0.15 m |
| Swing route | 3.94 s / 197 policy steps |
| First upright in held-out video | 17.80 s |
| Upright hold through video end | 12.22 s |
| Maximum cart excursion, noisy 100 | 2.9300 m |
| Episode terminations, noisy 100 | 100 time limits, 0 rail failures |
| Resets inside episodes | 0 |

The primary evidence files are:

- `runs/generalized_solver/n8_fddp_parked_target015_14s_noisy20.json`
- `runs/generalized_solver/n8_fddp_parked_target015_14s_noisy100.json`
- `runs/generalized_solver/n8_fddp_parked_target015_14s_exact20.json`
- `runs/generalized_solver/eight_link_swingup_success.mp4`
- `runs/generalized_solver/eight_link_swingup_success.video.json`
- `runs/generalized_solver/eight_link_swingup_manifest.json`

The noisy 20-episode cohort uses seeds `81801--81820`; the noisy 100-episode
cohort uses `80801--80900`. The cohorts are disjoint. The video uses held-out
seed `90901`, outside both cohorts. The video is a
state-faithful 2D rendering of the exact MuJoCo trajectory because the
headless execution environment does not provide the separate CoreGraphics
OpenGL renderer.

## 5. Negative Controls And Limits

The following results remain in the repository and are not silently replaced:

- Seven-link route transfer to eight links with a 10-second hanging prelude:
  `0/5` holds and rail failure.
- The saved open-loop eight-link route after cart parking: exact success but
  `0/20` noisy success.
- Direct local-LQR and online-MPC capture from the earlier near-upright
  endpoint: no sustained hold.
- The shorter `12.0 s / -0.20 m` parking variant: `95/100` noisy success.
- Wider-rail, spring-assisted, morphology-gradient, and ghost-link runs:
  useful discovery diagnostics only, not canonical evidence.

The result is specialized to the frozen uniform eight-link plant and exact
action timing. It does not establish robustness to link-length or mass
variation, friction changes, actuator delay, sensor noise, a different rail,
or a different force limit. It also does not establish an external record
without a matched benchmark comparison.

## 6. Reproduction

From the repository root, use the configured planner environment:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTHONPATH=gradient_cartpole_handoff/.conda-aligator/lib/python3.12/site-packages:src:scripts
PY=gradient_cartpole_handoff/.conda-aligator/bin/python

$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup8_uniform.yaml \
  --controller runs/generalized_solver/n8_capture_fddp_feedback120.json \
  --episodes 20 --seed 80801 --park-seconds 14 --cart-target -0.15 \
  --tracking-gain-scale 0.75 \
  --out runs/generalized_solver/n8_fddp_parked_target015_14s_noisy20.json

$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup8_uniform.yaml \
  --controller runs/generalized_solver/n8_capture_fddp_feedback120.json \
  --episodes 100 --seed 80801 --park-seconds 14 --cart-target -0.15 \
  --tracking-gain-scale 0.75 \
  --out runs/generalized_solver/n8_fddp_parked_target015_14s_noisy100.json

$PY scripts/render_fddp_parked_route_video.py \
  --config configs/swingup8_uniform.yaml \
  --controller runs/generalized_solver/n8_capture_fddp_feedback120.json \
  --seed 90901 --park-seconds 14 --cart-target -0.15 \
  --tracking-gain-scale 0.75 --seconds 30 --fail-on-failure \
  --out runs/generalized_solver/eight_link_swingup_success.mp4 \
  --metadata-out runs/generalized_solver/eight_link_swingup_success.video.json
```

The manifest records the exact hashes and the exact 20/100 artifacts used for
this result. The fresh-clone audit must run these commands against a clean
checkout and verify the manifest before any external claim is made.

## 7. Conclusion And Next Experiment

Eight links is now a genuine canonical internal solve under the same general
method that solved seven: active hanging-state conditioning, saved
state-consistent swing controls, time-varying feedback, cart-reference
translation, and delayed terminal LQR capture. The next frontier is nine links.
It should begin with this exact phase structure and only change the link-count
dependent route and capture artifacts. Broader controller-family searches are
not justified until the inherited eight-link stack has been tested and logged.
