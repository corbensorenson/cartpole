# 2D spatial cartpole progress — 2026-10-16

Goal: develop the XY cart / unrestricted ball-joint chain environment, solve
hanging-start swing-up sequentially, push the validated link count, and publish
each new qualified count with its full video to corbensorenson/cartpole.
The one-dimensional benchmark at D:/ChatGPT/cartpole remains separate.

## Completed

- Reproducible Python 3.11 environment, MuJoCo 3.3.5, exact dependency lock.
- Gymnasium interface, quaternion state, two force inputs with an 80N norm cap,
  passive ball joints, 50Hz control, 200Hz RK4, explicit 30s/5s upright gate.
- Physics/interface, replay, derivative parity, and quaternion gradient tests.
- Balance ladder `output/balance_001`: 12/12 at 1,2,3 links; 9/12 at four.
  World-coordinate feedback comparison `output/balance_world_001` yields the
  same pass counts. This does not prove twist is irrelevant generally.
- `output/swing_n1_001`: 60-iteration iLQR, nominal + 12/12 spatial noisy starts;
  all hold 21.74s and replay exactly. Native design: 2,768,100 physics steps.
  The optimized XY action maxima are about 77.43N and 5.84N. No planar lock.
- `output/qualify_n1_001`: frozen controller, disjoint seeds 20261216..20261247,
  32/32 complete 30s successes, exact replay. No retuning on this cohort.
- One-link release published to GitHub at tag `2d-cartpole-1-link-v1`, commit
  `6698d08a157be47df43e421ede8f453ae54c81ec`, with full held-out video and ZIP.
- All 32 energy-shaping initializer candidates failed; preserve their tapes at
  `output/energy_n1_001`. Its input snapshot was added retrospectively, not
  before execution; do not describe it as a fully frozen experiment.

## Two-link development

- `output/swing_n2_001`: unchanged one-link force initializer; 82 baseline
  iterations before regularization stop. Nominal fails at 8.84s, rail exit,
  zero upright hold. Complete optimizer trace and actual tapes retained.
- `output/swing_n2_002`: same initializer with corrected quaternion cost
  gradient / Gauss-Newton Hessian; completed 100 iterations, 0/13 passes.
  Nominal obtains 0.22s upright, then exits at 8.76s. Lower merit (308.08 vs
  442.54) did not qualify the local-coordinate handoff.
- `output/n2_world_handoff_001`: same route and gains, only stationary feedback
  switched to world tilt/rate coordinates. Passes 12/12 noisy development starts.
- `output/qualify_n2_001`: frozen variant passes 32/32 fresh seeds
  20261316..20261347; all hold 21.68s, complete 30s, exact replay. Maximum
  per-axis cart excursion 0.982710m, minimum physics-substep boundary margin
  2.017052m. This earns a two-link 2D release with its full held-out video.
- Two-link release published to GitHub at tag `2d-cartpole-2-link-v1`, commit
  `3693d3094a88ec128bae09f1547b11fc75ffb26a`. Public asset SHA256 digests match
  local video and ZIP. All 207 ZIP manifest entries verify. The published source
  passes all 11 tests and reproduces a qualifying held-out two-link episode.
- `output/swing_n3_001`: three-link transfer from two-link XY forces, 120-iteration
  ceiling, world-coordinate handoff; completed 0/13 passes, no upright hold.
  Actual nominal exits at 8.32s. The endpoint has 0.26–0.34rad world tilt and
  substantial transverse motion, outside the successful local capture regime.
- `output/swing_n3_002`: warm-start n3_001, change terminal residual to physical
  world tilt/rate coordinates consistent with stationary feedback; stops at 106
  iterations, 0/13 passes. Nominal rail exit at 9.26s with zero upright hold.
- `output/swing_n3_003`: optimize a joined 13s route with a five-second physical
  upright target segment from 8..13s, warm-start n3_002 plus zero-force extension.
  Completed 100 iterations, 0/13 passes. Nominal exits at 13.76s, no upright hold.
- `output/planar_lift_n3_001`: input guard rejects a 2.04e-7 mixed-component
  discrepancy between historical nominal and recorded actual states before
  new-plant simulation. V2 explicitly parses source scales and records that
  discrepancy under a declared 2e-6 guard; original failed snapshot remains.
- `output/planar_lift_n3_002`: old planar X feedback plus newly synthesized
  spatial Y feedback. Nominal exits at 12.02s; best hold 0.06s. Its actual
  delivered prefix becomes the native repair initializer, with a source hash.
- `output/planar_lift_n3_gainbank_001`: five declared X-gain scales all fail
  nominal. No final or spatial validation seeds consumed by that screen.
- `output/swing_n3_004`: native repair of the lifted actual force prefix,
  11.28s horizon; stops at 58 iterations, 0/13 passes, nominal hold 0.04s.
- `output/swing_n3_005`: force-limit derivative/backward-step corrections;
  stops at 14 iterations, 0/13 passes, nominal hold 0.04s.
- `output/swing_n3_006` and `007`: smooth periodic WORLD direction/rate running
  residuals, circular force-bound backward steps, four native workers; terminal
  multipliers 1 and 10. Completed 80 iterations each. 006 passes 10/13;
  007 passes nominal+12/12 with 18.78s hold. Frozen raw-route qualification
  `output/qualify_n3_001` fails at 25/32; all failures and tapes retained.
- `output/conditioned_n3_001`: add 12s native hanging LQR and causal axial-frame
  alignment to the SAME 007 route/gains. Passes nominal+44/44 development starts,
  including all failed raw-route qualification seeds, with 6.78s hold. Maximum
  launch tilt from down is 8.37e-6rad. Those reused seeds are development now.
- `output/qualify_n3_002`: frozen conditioned policy on NEW seeds
  20271416..20271447 passes 32/32 full 30s episodes and exact replay, all with
  6.78s upright. Maximum per-axis cart excursion 2.097708m; minimum substep
  boundary margin 0.902058m. Maximum launch tilt from down 7.43e-6rad. This earns
  a three-link 2D release; keep the earlier failed 25/32 qualification distinct.
- `output/swing_n4_001`: transfer only three's XY force proposal to four links;
  new native dynamics/gains, periodic direction residuals, terminal multiplier
  10, bounded backward pass, four workers, 100 iterations. Currently running.
- Next steps: inspect final actual execution, qualify if successful, otherwise
  vary route duration/initialization with a declared budget. Three has now been
  started after passing two. Keep balance-only and full swing-up counts separate.

## Publication policy

Each new count needs a frozen controller, full nominal/development episode
tapes, fresh qualification, independent replay, source/runtime/config hashes,
and a full held-out video. Publish under the explicitly labeled `2d/` subtree,
using an isolated checkout (`publication/`), plus a `2d-cartpole-N-link-v1`
GitHub release with video and evidence. Never publish balance-only passes as
full hanging-start solves. Use a fast-forward push and preserve concurrent work.

The persistent goal remains active; no upper frontier or impossibility has been
established. Thirty-two reset-noise trials do not certify arbitrary disturbances.

## Resume checkpoint

The currently active experiment is swing_n4_001 (execution session 23065,
if retained by the execution service; otherwise inspect the live command).
Inspect `completed.json` and the actual execution handle before
calling them finished. Checkpoints every ten iterations preserve states/actions as warm
starts; they are not qualified policies. After completion inspect full nominal
and all spatial development outcomes, then either freeze/qualify/publish three
or prerecord the next duration/initializer intervention. Never publish a new
count from optimizer merit alone. All future successful count releases must
include an explicitly 2D label, full video, state/action evidence and hashes.

Source snapshots remain specific to each run. New experiments additionally
freeze their driver, input route, package metadata, dependency lock and thread
settings. Early auxiliary importer/gain-bank drivers were not copied before
execution; do not call those complete frozen synthesis reproductions. Native
state/action tapes and frozen final controllers remain authoritative evidence.
