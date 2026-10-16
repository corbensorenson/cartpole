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
- Three-link release published at `2d-cartpole-3-link-v1`, commit
  `b119c020b43852012c307a2a82bb1932a601a4ed`. Full video and 90.86MB evidence ZIP
  uploaded; public SHA256 digests match. All 644 staged/archive manifest entries
  verify; published source passes 23 tests and reproduces a held-out episode.
  Concurrent overview PR #6 was preserved by rebasing and updating its 2D card,
  badge, table and method paragraphs. The 1D docs and other overview edits remain.
- `output/swing_n4_001`: transfer only three's XY force proposal to four links;
  new native dynamics/gains, periodic direction residuals, terminal multiplier
  10, bounded backward pass, four workers. Stops at 17 iterations, nominal exits
  at 3.58s, no hold. No four-link success.
- `output/planar_lift_n4_001`: input guard failure before new-plant simulation.
  The parent stored actual run failed with zero tracking; it cannot validate the
  curated gain-one reference. Original failure and its source snapshot remain.
- `output/planar_lift_n4_002`: explicit reference-only lift of the curated
  four-link solver; native world-direction reconstruction passes. Actual new
  plant exits at 4.08s, no hold. Its route has 1152 commands / 23.04s, so twelve
  extra seconds of conditioning would violate our fixed 30s clock. The older
  library added conditioning outside its configured route episode; never import
  its 20/20 directly as a solve of this new spatial benchmark.
- `output/n4_foreign_defects_001`: 4608 independent native physics steps quantify
  the lifted reference's local defects: maximum joint rotation 0.002314rad,
  joint rate 0.255217rad/s, cart position 7.73e-5m, cart rate 0.00804m/s.
  This is a trajectory-proposal diagnostic, not a physical episode or solve.
- `output/fddp_n4_001` and `002`: historical and geometric-split starts,
  interrupted after a quaternion defect-transport correction was identified.
  Source, histories, checkpoints and bounded interrupted accounting are retained.
- `output/fddp_n4_003` and `004`: corrected restarts close their defects but
  fail actual nominal execution, with cart excursions far outside the square
  in their optimizer proposals. Zero-gap proposals are not accepted episodes.
- `output/n4_plant_parity_001`: finite local diagnostic identifies the old
  damping and slide armature; restoring both reduces maximum joint-rate defect
  from 0.2552rad/s to 9.63e-8rad/s. The altered models are components only.
- `output/n4_old_mechanics_replay_001`: altered damping 0.015 and slide
  armature 0.0005 give one nominal complete spatial episode with 7.22s hold.
  This does not earn a benchmark count. Native design costs 281332 steps.
- `output/n4_mechanics_continuation_001`: eight trials restore only fraction
  0.0025 toward benchmark mechanics; fails to reach the benchmark. Keeping the
  old trajectory gain table is insufficient. Every trial/actual tape is retained.
- `output/fddp_n4_005` and `006`: fresh state/control trust variants. The
  geometric split (006) fails actual nominal at 1.92s after 80 iterations.
  Historical-reference 005 passes nominal with 7.36s hold but fails all 12 noisy
  development starts; 23,583,988 design physics steps. It is not accepted.
- `output/tracking_repair_n4_001`: fixed-target affine repair of the
  old-mechanics actual motion on the UNCHANGED benchmark. The first iteration
  closes all gaps with cart maximum 2.728592m; further optimization lowers
  tracking cost. The final 25-iteration policy passes nominal+12/12 development
  starts (7.22s hold), but is not substituted for the selected first checkpoint.
  Total final synthesis cost 7,561,972 native physics steps.
- `output/checkpoint_n4_tracking_001`: FIRST repair checkpoint, selected before
  actual outcomes, passes nominal+12/12 noisy 3D development starts and exact
  replay. No hanging preparation or gauge alignment; the 23.04s motion begins
  immediately and switches to world-coordinate upright feedback.
- `output/qualify_n4_001`: frozen first-checkpoint controller passes NEW seeds
  20261516..20261547, 32/32 complete 30s episodes and exact replay, every one
  with 7.22s hold. Maximum cart coordinate 2.728591608m; minimum substep
  boundary margin 0.270903698m. This earns the four-link 2D release.
- Full first held-out video: `output/2d-cartpole-4-link-swingup.mp4`, 1501
  frames / 50fps / 30.02s encoding, SHA256
  `a11909aa3970ca2c44462ad704c04fabf76046ec0a161edbf4ab59e9576d462a`.
  Full-duration 30s GIF and six-frame contact sheet were generated and inspected.
- Explicit slide armature defaults to zero. A published three-link episode
  remains byte-identical in states, forces and tilts in
  `output/regression_n3_cart_armature_001`. Packaging and publication reject
  any altered final Config. Current source passes 33 tests.

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

All four-link experiments are terminal. The GOAL stays active at a qualified
frontier of FOUR. Complete the isolated
four-link GitHub publication, including the full video and evidence ZIP, then
advance to five. Historical planar route proposals are available in the original
read-only checkout; inspect their geometry, dynamics and timing before transfer.
The successful new tool is native fixed-target affine trajectory repair,
not force-only transfer. Preserve the strict 30s clock and all physical freedoms.
Record the chosen method, warm start, seeds and compute limit BEFORE execution.
Do not restart an experiment while its process is alive or its completed marker
exists. Checkpoints are proposals until full actual float32 gates pass. Five-link
reserved qualification seeds will be 20261616..20261647, untouched at this entry.
All future successful count releases require an explicitly 2D label, full video,
state/action evidence and hashes. No upper frontier has been established.

Source snapshots remain specific to each run. New experiments additionally
freeze their driver, input route, package metadata, dependency lock and thread
settings. Early auxiliary importer/gain-bank drivers were not copied before
execution; do not call those complete frozen synthesis reproductions. Native
state/action tapes and frozen final controllers remain authoritative evidence.
