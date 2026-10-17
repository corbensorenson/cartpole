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
- Four-link release is public at `2d-cartpole-4-link-v1`, commit
  `cac969fdaf37d0279fbaa2d547e9418328c2567f`. Full video and 188.57MB ZIP
  uploaded. Both public asset SHA256 digests match; 1141 staged/archive files
  verify. Published source passes 33 tests and reproduces the first held-out
  episode byte-for-byte. Root overview now features four; all 1D content remains.
- `output/planar_lift_n5_001`: reference-only lift from the old uniform five-link
  motion passes direction reconstruction (6.66e-16); direct new-plant nominal
  fails at 3.46s. Old source used a wider rail and reached 3.331747m, so its old
  success cannot promote this count. Source config, coordinates and hashes remain.
- `output/tracking_repair_n5_001`: same fixed-target affine repair on the
  unchanged five-link plant, 31 iterations / 4,043,576 native design steps.
  Nominal plus all 12 noisy development starts complete with 21.70s hold and
  exact replay. No parking or gauge wrapper is needed or used.
- `output/qualify_n5_001`: frozen final repaired controller on NEW seeds
  20261616..20261647 passes 32/32 full 30s episodes and exact replay, all with
  21.70s hold. All initial states are distinct. Maximum cart coordinate
  2.731120998m; minimum substep margin 0.267718018m. This earns the five-link
  2D release, with its own full held-out video and evidence archive.
- Full five-link held-out video has 1501 frames / 50fps / 30.02s encoding,
  SHA256 `f413c17a880d6cf5537509886af213c22caa8820690864b48ffb1b600ef3aade`.
  A normal-speed full-duration GIF and contact sheet were generated and inspected.
- Five-link release is public at `2d-cartpole-5-link-v1`, commit
  `127d16017a00a80a4e8f0a104f1075619a046b6e`. Full video and 217.02MB
  evidence ZIP uploaded; both public asset SHA256 digests match. All 1258
  staged/archive files verify, published source passes 33 tests and reproduces
  the first held-out tape byte-for-byte. Root 2D overview now features five.
- `output/planar_lift_n6_001`: direct lifted-policy nominal fails at 1.56s,
  exact replay. The historical uniform six-link motion reached 3.098808m on a
  wider rail and is only a proposal. Native coordinate reconstruction passes.
- `output/tracking_repair_n6_001`: same native fixed-target repair, 40 iterations
  / 2,892,064 design physics steps. Nominal holds 26.06s but four of 12 noisy
  development starts fail; nominal plus development is 9/13. Not accepted.
- `output/conditioned_n6_001`: preplanned 12s hanging LQR plus causal axial-frame
  alignment on the unchanged repaired 4.18s route. Nominal passes the complete
  30s episode with 14.06s hold and exact replay. All 44 noisy development starts
  also pass. Maximum launch tilt from down is 1.42504e-5rad. Design: 340 steps.
- `output/qualify_n6_001`: frozen conditioned controller on NEW seeds
  20261716..20261747 passes 32/32 full 30s episodes and exact replay, all with
  14.06s hold. All initial states are distinct. Maximum cart coordinate
  2.729448095m; minimum substep boundary margin 0.270551905m; maximum launch
  tilt from down 1.23419e-5rad. This earns the six-link 2D release.
- Six-link full held-out video: 1501 frames / 50fps / 30.02s encoding, SHA256
  `6dd2771d6b791293b93d84336924d63aace5e8bed1070bcc40847f0b556c520b`.
  A full-duration 30s preview and phase-labeled contact sheet were inspected.
- Six-link release is public at `2d-cartpole-6-link-v1`, commit
  `0f15e3979e469b378bea2d8928baa56d8fa42e30`. Full video and 273.22MB
  evidence ZIP uploaded, with both public asset SHA256 digests matching.
  All 1438 staged/archive files verify; published source passes 33 tests and
  reproduces the first held-out state/action/tilt tape byte-for-byte.
- `output/planar_lift_n7_001`: frozen historical seven-link route/parent nominal
  arrays match and coordinate reconstruction passes; direct spatial transfer
  fails at 5.08s with exact replay. Source motion is a proposal only.
- `output/tracking_repair_n7_001`: first fixed-target repair stalls after 19
  iterations with unresolved gaps and fails actual nominal at 5.34s. Native
  design costs 1,782,436 steps. No seven-link success from this policy.
- `output/tracking_repair_n7_002`: initial feasibility penalty raised from
  1000 to 100000, otherwise same weights/model/target; detailed line-search
  diagnostics added. Forty iterations / 3,667,540 design steps. Nominal plus
  all 12 noisy development starts pass full 30s with 25.48s hold and exact replay.
  No parking/gauge wrapper is required or used. Source now passes 34 tests.
- `output/qualify_n7_001`: frozen second repaired policy, 32 NEW seeds
  20261816..20261847, finishes at 31/32. Seed 20261831 exits in Y at 4.04s.
  This policy is NOT accepted or published. These seeds are development now.
- User-requested `output/spatial_audit_001`: compiled 1,3,7-link models have
  independent X/Y slides, free ball joints, no equality/limit/joint-actuator
  constraints. All per-joint transverse perturbations and passive spin respond.
  Matched X/Y force probes agree under 90-degree rotation within 2.56e-15.
  Persistent native MuJoCo matches environment transition states bit-for-bit.
  Changing-direction forces yield cart path minor/major ratios 0.60..0.69 and
  nonplanar rod motion. 4420 physics steps; all recorded cohort kinematics checked.
  Counts 4..6 have essentially planar nominal references; Y feedback is active
  under spatial noise. Count two already has substantial nominal two-axis motion.
- `output/nonplanar_demo_n6_001`: new controlled XY loop after t=18s, same
  physical plant and six-link swing/parking prefix. Nominal+3 noisy starts all
  pass full 30s with exact replay, 14.06s hold; post-capture cart spans at least
  1.11556m X and 0.911999m Y, minor/major ratio >=0.7779748. Maximum world rod
  tilt throughout the orbit is 0.0233697rad. This is an audit demonstration,
  not an additional count qualification. Full phase-labeled video rendered and
  inspected; SHA256 0aa0e1d7900caf25f3dceb79a0346961e17b41761661957b901e53c86537e296.
  New spatial-freedom regression checks bring the suite to 44 passing tests.
- Spatial audit and controlled XY-loop bundle is public under
  `2d/audits/spatial_freedom_001`, commit
  `788b876d6d1f9bb61e7deca7e586889058ca223a`. Its 114 files verify, bundled
  source passes 44 tests, audit metrics and all four controlled tapes reproduce
  exactly. The original six-link release's 1438 staged/archive hashes remain
  unchanged. Root overview now explains the planar-reference scope.
- `output/conditioned_n7_001`: 12s hanging feedback plus axial-frame alignment;
  nominal and all 76 noisy development starts pass (77/77), including the entire
  failed first reserved cohort. All hold 13.48s and replay exactly. No state
  reset, lock or clock extension. Extra development summary is frozen/hashed.
- `output/qualify_n7_002`: frozen conditioned controller on NEW untouched seeds
  20271816..20271847 passes 32/32 full 30s episodes and exact replay, all holding
  13.48s. All initial states are distinct. Maximum cart coordinate 2.366482402m;
  minimum substep margin 0.631862733m; maximum launch tilt from down 1.43734e-5rad.
  This earns the explicitly 2D seven-link release.
- Full first final-cohort video: 1501 frames / 50fps / 30.02s encoding,
  SHA256 `5a5a18fe16e68436fd46df0391f86ea1d3a24069bd0f2a0320835c9ba54054e8`.
  Full-duration preview and phase-labeled contact sheet were inspected.
- Seven-link release is public at `2d-cartpole-7-link-v1`, commit
  `a36bbeb6fb1802fa29ee48d67ddebd1acbed08ca`. Full video and 412.70MB ZIP
  uploaded with both public SHA256 digests matching; tag resolves to the pushed
  source commit. All 1868 staged/archive files verify. Published source passes
  44 tests and reproduces the first final held-out tape byte-for-byte. The
  physics audit and controlled XY demonstration are included and linked.
- `output/planar_lift_n8_001`: source world-direction reconstruction passes;
  direct transfer fails nominal at 2.74s with exact replay. The historical
  wider-rail source is only a proposal for the unchanged eight-link model.
- `output/tracking_repair_n8_001`: 60 iterations / 5,332,832 native design steps,
  initial gap penalty 100000. Native gap-free rebuild passes full nominal with
  26.20s hold and exact replay, but all 12 noisy development starts fail. This
  raw controller is not accepted.
- `output/conditioned_n8_001`: preplanned 12s hanging feedback and causal axial
  alignment; nominal plus all 44 noisy development starts pass complete 30s
  episodes with 14.20s hold and exact replay. Design cost 436 native steps;
  maximum development launch tilt from down 1.94236e-5rad.
- `output/qualify_n8_001`: frozen conditioned controller on NEW reserved seeds
  20261916..20261947 passes 32/32 complete 30s episodes and exact replay, every
  one with 14.20s hold. All initial states are distinct. Maximum cart coordinate
  2.726040999m; minimum substep margin 0.272202528m; maximum launch tilt from
  down 1.71736e-5rad. This earns the eight-link 2D release. Full compiled-model
  freedom and native multiaxis regression checks now include eight; 48 tests pass.
- Full first eight-link reserved video: 1501 frames / 50fps / 30.02s encoding,
  SHA256 `665abc541ae53675c5a9c51e03b0cef9c07aa5c1227b3c82567e3fb62acfe0a1`.
  A normal-speed full-duration preview and contact sheet were inspected.
- Eight-link release is public at `2d-cartpole-8-link-v1`, commit
  `29095220cd47f2fd52aa45a8668c4b9e1ad78f38`. Full video and 481.91MB ZIP
  uploaded; public digests match and the tag resolves to the pushed commit.
  All 2049 staged/archive files verify; published source passes 48 tests and
  reproduces the first final held-out state/action/tilt tape byte-for-byte.

Nine-link qualification is complete: the lifted historical route fails nominal
at 2.86s. Native tracking repair (60 iterations, 5,901,028 design steps)
completes nominal with 26.18s hold and exact replay, but all 12 raw noisy
development starts fail. The preplanned 18s hanging/gauge wrapper passes
nominal plus all 44 noisy development starts in `output/conditioned_n9_001`.
`output/qualify_n9_001` passes all 32 NEW reserved starts, 20262016..20262047,
with full 30s episodes, exact replay, 8.18s hold each and 32 distinct initial
states. Maximum cart coordinate 2.716247807m; minimum substep boundary margin
0.283506480m; maximum launch tilt from down 5.53955e-7rad. This earns the
nine-link release. Native freedom checks bring the passing suite to 52 tests.
The full first reserved video has 1501 frames at 50fps (30.02s encoding),
SHA256 `59887f487fa8a35b5a941811b59a0f45feef5e53cfde3123b0448e7f279c55e2`.
Its 300-frame normal-speed full-duration preview and both general/swing-phase
contact sheets have been inspected.
The nine-link release is public at `2d-cartpole-9-link-v1`, commit
`f875179a15742b854aab9c59627d3d7c03b16730`. The public full video and
558,116,227-byte ZIP digests match, and the tag resolves to the pushed commit.
All 2230 staged/archive hashes verify; published source passes 52 tests and
reproduces the first final held-out state/action/tilt arrays byte-for-byte.

Ten-link direct transfer fails nominal at 3.36s. Fixed-target native repair
`output/tracking_repair_n10_001` completes 80 iterations and 17,501,332 design
physics steps. Native proposal gaps close to roundoff; the actual full nominal
passes with 22.12s hold and exact replay, but all 12 raw noisy starts fail.
The optimizer emitted a native instability warning during rejected search
trials; no actual nominal/development episode reports a physics warning.
The preplanned 16s hanging/gauge wrapper in `output/conditioned_n10_001`
passes nominal plus all 44 shared noisy development starts, 45/45 full
30s episodes with 6.12s hold and exact replay. Wrapper design cost 532 native
steps; maximum development launch tilt from down 6.42840e-6rad.
The frozen controller passes fresh reserved qualification in
`output/qualify_n10_001`, seeds 20262116..20262147: 32/32 complete 30s episodes,
6.12s hold each, exact replay and 32 distinct initial states. Maximum cart
coordinate 2.022728709m; minimum substep boundary margin 0.977271291m;
maximum launch tilt from down 5.73604e-6rad. This earns the ten-link release.
The full first held-out video has 1501 frames / 50fps / 30.02s encoding,
SHA256 `adafc1b7d48cd4e2dd033243b5d954ffd3e1f98e3a886ad792f653356f584054`.
The 300-frame normal-speed full-duration preview and general/swing-phase
contact sheets have been inspected.

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

Nine-link public video/evidence release is complete. Ten-link experiments and
qualification are complete, qualified frontier TEN. Render/package/publish
its separate full video/evidence release, then advance to eleven under the
recorded plan. Historical proposal:
`runs/swingup11_uniform/eleven_link_controller.json`, 400 commands / 8s,
uniform eleven links, nominal cart maximum 2.071342m; source config
`configs/swingup11_uniform.yaml`. It is only a geometric proposal for unchanged
spatial mechanics. No simulation is still running; ten's video is rendering.
Ten-link physical-freedom checks pass: 56 tests total.
Keep physical freedom and near-planar-reference scope explicit. User authorizes
continued higher counts. Check live processes/terminal markers before resuming.
Historical planar route proposals are available in the original read-only
checkout; inspect their geometry, dynamics and timing before transfer.
The successful new tool is native fixed-target affine trajectory repair,
not force-only transfer. Preserve the strict 30s clock and all physical freedoms.
Record the chosen method, warm start, seeds and compute limit BEFORE execution.
Do not restart an experiment while its process is alive or its completed marker
exists. Checkpoints are proposals until full actual float32 gates pass. Eleven-link
reserved qualification seeds will be 20262216..20262247, untouched at this entry.
All future successful count releases require an explicitly 2D label, full video,
state/action evidence and hashes. No upper frontier has been established.

Source snapshots remain specific to each run. New experiments additionally
freeze their driver, input route, package metadata, dependency lock and thread
settings. Early auxiliary importer/gain-bank drivers were not copied before
execution; do not call those complete frozen synthesis reproductions. Native
state/action tapes and frozen final controllers remain authoritative evidence.
