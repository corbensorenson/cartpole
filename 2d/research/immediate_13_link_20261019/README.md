# 13-link 2D cartpole: immediate noisy swing-up

**32/32 fresh original noisy starts pass after 45/45 development states**
(44 noisy and one nominal). Every qualification episode completes the full
30 seconds with at least **24.16 seconds upright**.
Hanging preparation is **zero**. [Full 50 fps video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-13-link-immediate-v1/2d-cartpole-13-link-immediate.mp4) · [Release and artifacts](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-13-link-immediate-v1).

The cart moves in XY and all joints are passive spatial ball joints. All
41 native velocities, including spin, remain active.
The original 3 m / 1 kg uniform chain, 1 kg cart, .02 initial perturbations,
80 N radial force limit, +/-3 m square, 50 Hz controls and 200 Hz MuJoCo RK4
physics are unchanged. Qualification minimum substep boundary margin is
1.084459 m; maximum delivered force is
67.137817 N.

The frozen classical controller plans from the observed initial state, then
uses feedback throughout the physical motion. Planning is offline and takes
3.43–15.22 minutes per reserved start on
the recorded runtime. These are recorded execution segments: the administratively
resumed case excludes earlier interrupted and cached preparation. The manifest
separately bounds uncheckpointed compute. Simulation does not advance during planning. Hardware
real-time operation, a learned model and additional mid-swing force-gust
robustness are unqualified. The 14-link conditioned comparison is separate.

Fresh starts 20263716–20263747 were reserved before evaluation and consumed
once by this fixed candidate. Each passes independent original-reset, native
model, full-rate, feedback, substep-boundary and cold-replay audits. The first
reserved case also repeats its full synthesis and actual/cold episode from the
extracted runtime. Replays, internal forecasts and inherited pilot cases do not
add fresh qualification samples.

Download the video, runtime ZIP, manifest and every evidence ZIP part from the
release. Extract the evidence parts together. The case index locates all 76
distinct noisy development/qualification actual tapes and their audits. Failed
forecasts, accepted corrections and certificates remain available. The first
reserved case includes all model caches; other regenerable model/value caches
have exact SHA256/size receipts and the bootstrap requires identical rebuilt
digests. Original raw caches remain in the lab.

Extract the standalone runtime, install its pinned Windows/Python 3.11
requirements, set BLAS/OMP/MKL threads to one and follow its README. Use
`scripts/reproduce_spatial_observed_synthesis.py` with `--expected-case` pointing to the already recorded
first reserved case (20263716). This reproduces existing evidence and is not
another fresh sample. Full-state source and frozen hashes are in source/;
original metadata paths are provenance and need not exist after relocation.

Controller identity: `d797b2fc96ec05f68afe78c5fb648be2219f541b1084ae2a8d31b283459ab4c5`

Video SHA256: `2b85d3c1530b656ffb620ea68c4ceaa3f903c8bdc38b97a5e1eee69d8914c365`
