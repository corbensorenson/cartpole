# Ten-Link Cart-Pole Swing-Up: A Settled-Launch Two-Expert Controller

**Status:** released internal canonical ten-link benchmark; external competition matching remains open

## Abstract

This paper documents the project's first verified ten-link result on the
uniform MuJoCo cart-pole plant. A hanging-equilibrium conditioning expert parks
the cart for 16 seconds, a target-plant Box-FDDP route with saved time-varying
feedback swings the chain up for 8 seconds, and an upright LQR captures the
measured state without a reset. The controller passes disjoint noisy `20/20`
and `100/100` gates and an exact `20/20` check on the canonical `+/-3 m` rail.
The held-out video is a 30-second, 1,500-frame replay with zero resets.

This is an internal repository benchmark. It is not an external world-record
claim until the competing implementation's plant, force convention, timing,
rail geometry, initial distribution, and judging rules are independently
matched.

## 1. Frozen benchmark

The acceptance plant is `configs/swingup10_uniform.yaml`:

- ten equal links with total length `3.0 m` and total mass `1.0 kg`;
- cart mass `1.0 kg`, force limit `80 N`, and canonical rail `+/-3.0 m`;
- MuJoCo timestep `0.005 s`, four internal steps, and 50 Hz policy actions;
- hanging start with independent `0.05` angle and velocity noise;
- upright threshold `0.15 rad` and five seconds of sustained success;
- no angle termination and no state reset at phase boundaries.

The declared controller contract parks at `-0.05 m` for `16.0 s`, executes a
400-step (`8.0 s`) swing route, then applies the terminal LQR around `-0.05 m`.
The final route was refined on the uniform ten-link target plant, not on a
temporarily supported or altered-morphology plant.

## 2. Controller

### 2.1 Conditioning expert

The first expert is a hanging-equilibrium LQR with control cost `1000`. It
damps the initial angle and velocity perturbations while steering the cart to
the parked reference. The phase is a continuous controller phase; it does not
reinitialize the simulator.

### 2.2 Swing expert

The route is a target-plant Crocoddyl Box-FDDP solution with 400 normalized
force actions, nominal states, and time-varying feedback gains. It was warm
started from an exact endpoint-shaped route produced by a longer tail
refinement. The final Box-FDDP settings used a `2.85 m` soft rail inside the
canonical `3.0 m` rail, terminal angle factor `20`, terminal hinge-rate factor
`4`, control cost `0.01`, and terminal state weight `100000`.

The important lesson was that an isolated quiet endpoint was insufficient. The
single-sample route could hold in exact replay but its feedback demand was too
large for noisy capture. Re-optimizing the route with feedback and then
conditioning longer produced a bounded capture basin.

### 2.3 Capture expert

After the route horizon, the measured MuJoCo state is handed directly to the
upright LQR with control cost `1000` and scale `1.0`. The controller tracks the
same parked cart target. The handoff is time-indexed and reset-free.

## 3. Results

| Gate | Seeds | Result | Maximum cart excursion |
|---|---:|---:|---:|
| Noisy 20 | `102001--102020` | **20/20** | `1.9942 m` |
| Noisy 100 | `102101--102200` | **100/100** | `1.9976 m` |
| Exact 20 | `102201--102220` | **20/20** | `1.9714 m` |

All gate episodes terminated by the 30-second time limit, not by rail
violation. Every noisy episode reached upright. The held-out video uses seed
`102221`, outside all gate cohorts, reaches upright at `23.90 s`, and remains
upright through the end of the video for `6.12 s`.

The video is deliberately zoomed out to keep all ten links visible. It is
encoded at `1280x720`, 50 fps, for 1,500 frames and records a maximum cart
excursion of `1.9717 m`.

## 4. Reproduction

Use the bundled Python 3.12 environment:

```bash
PY=./.conda-aligator/bin/python

$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup10_uniform.yaml \
  --controller runs/generalized_solver/n10_fddp_refined_route_feedback100.json \
  --episodes 20 --seed 102001 --park-seconds 16 --cart-target -0.05 \
  --tracking-gain-scale 1 --release-evidence \
  --out runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005_20.json

$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup10_uniform.yaml \
  --controller runs/generalized_solver/n10_fddp_refined_route_feedback100.json \
  --episodes 100 --seed 102101 --park-seconds 16 --cart-target -0.05 \
  --tracking-gain-scale 1 --release-evidence \
  --out runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005_100.json

$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup10_uniform.yaml \
  --controller runs/generalized_solver/n10_fddp_refined_route_feedback100.json \
  --episodes 20 --seed 102201 --zero-noise --park-seconds 16 \
  --cart-target -0.05 --tracking-gain-scale 1 --release-evidence \
  --out runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005_exact20.json

$PY scripts/render_fddp_parked_route_video.py \
  --config configs/swingup10_uniform.yaml \
  --controller runs/generalized_solver/n10_fddp_refined_route_feedback100.json \
  --out runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005.mp4 \
  --metadata-out runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005.video.json \
  --seed 102221 --park-seconds 16 --cart-target -0.05 \
  --tracking-gain-scale 1 --phase-window 0 --seconds 30 \
  --fps 50 --width 1280 --height 720 --fail-on-failure

PYTHONPATH=src $PY scripts/verify_ten_link_release.py
sha256sum -c runs/generalized_solver/n10_release_SHA256SUMS
```

The [ten-link manifest](../runs/generalized_solver/ten_link_swingup_manifest.json)
binds the controller, configuration, generated XML, gate outputs, video, and
video metadata by SHA-256.

## 5. What changed from the failed attempts

The earlier n10 campaign demonstrated that exact terminal angle minimization,
static LQR gain sweeps, wider rails, open-loop tail CEM, nonlinear MPC, and
temporary ghost-link support were not sufficient. The strongest failed route
had a nearly zero final angle but immediately left the nonlinear capture basin
under bounded feedback.

The successful route combines three changes:

1. A longer 16-second park removes the initial noisy hanging-state residual.
2. The final two seconds of the swing arrival are shaped before Box-FDDP so
   the route is not just quiet at one isolated sample.
3. Box-FDDP re-optimizes the ten-link target route and its time-varying
   feedback, rather than treating the nine-link route as a controller.

## 6. Limitations

This result is narrow by design. It does not establish robustness to model
error, sensor noise beyond the declared reset distribution, actuator delay,
different control rates, altered damping, unequal link lengths or masses, or a
different competition definition. It also does not establish that ten links
is an external record. The next project frontier is eleven links using this
same settled-launch, target-plant route-refinement, and terminal-capture
architecture.
