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
The ten-link release is public at `2d-cartpole-10-link-v1`, commit
`c55ebd6eeb8850c65686d78e3880db71eb6b779f`. The full public video and
650,526,266-byte ZIP digests match; the tag and master point to this commit.
All 2413 staged/archive hashes verify; published source passes 56 tests and
reproduces the first held-out state/action/tilt tape byte-for-byte.

## Publication policy

Eleven-link direct transfer fails nominal at 4.64s. Native repairs
`tracking_repair_n11_001`, `_002` (residual QR) and `_003` (QR plus circle
curvature correction) each stall after 11 iterations without an accepted step;
none passes nominal or uses reserved seeds. Their design costs are 2,677,380,
2,677,380 and 2,677,364 native steps. All failed source/optimizer/actual tapes
are retained. `n11_backward_diagnostic_001` uses 270980 native steps and shows
zero-alpha virtual-gap recovery within 5.1e-7 state units, with divergence at
alpha .005 under either backend.

The independent closed-form active-circle value regression exposed a genuine
synthesis defect: the value Hessian omitted lambda*K.T*K. Both normal and
residual-QR backwards and bounded legacy iLQR now retain that curvature. The
frozen ten-link first held-out episode remains byte-identical in
`n11_optimizer_fix_regression_001`. The combined upright DARE also introduces
large spurious cross-axis value terms at eleven. The optional axis-separated
design solves an averaged native horizontal system, rotates its gain/value,
and passes physical world-rotation checks. That stage's suite had 68 passing tests.
`tracking_repair_n11_004` also stalls after 11 iterations and fails nominal.
The cached fine-step diagnostic `n11_linesearch_diagnostic_001` uses 29380
native steps and finds valid merit descent at alpha 1e-4, below the old .005
minimum; .001 and .005 diverge. The optional finer geometric backtracking and
small-step trust-growth setting retain the old defaults for prior recipes.
`tracking_repair_n11_005` completes 80 iterations / 20097532 design steps:
small accepted steps reduce defect1.59571e-4 to1.55655e-4, but nominal fails
4.22s, no hold. Order4 derivatives, zero optimizer terminal-value and free
forward-feedback diagnostics also fail to enable large causal repair steps.
Full spatial simultaneous sparse shooting is more effective at proposal repair:
`n11_direct_repair_001` completes20 iterations /9744580 design steps, reducing
defect score to3.73042264e-11 and maximum normalized defect1.28568116e-6,
but its rebuilt actual controller fails4.58s. Accurate augmented sparse ridge
solves improve a one-update diagnostic without establishing an actual solve.

User-requested hybrid-document review is recorded in research/hybrid_2d_review.md.
`n11_hybrid_affine_001` retains native reference defects and the full affine QR
value/offsets on the raw repaired proposal. Zero-offset fails5.08s/no hold;
affine reaches1.92s hold and fails10.88s after stationary balance handoff.
Capped2D native search also holds1.92s and fails10.66s, adding25600 queries'
physics steps. `n11_hybrid_finite_001` adds an upright continuation to1500
design steps and removes the stationary terminal Riccati value: affine passes
nominal30s with23.50s hold, but all12 raw noisy starts fail; zero-offset fails.
`n11_hybrid_guard_001` with1550 design/1500 execution has the same nominal
hold and0/12 raw noisy passes. These are separate controllers and finite tests.
The FIRST1500-step design was selected before guard results for preparation.
`conditioned_n11_001` uses16s hanging/gauge alignment and700 commands from
the1500-step design: nominal holds7.50s, but only27/45 development episodes
pass, all exact replay. `conditioned_n11_002` tests18s preparation and600
tracking commands, passing40/45 including nominal5.50s hold. It remains
unqualified. `conditioned_n11_003` tests18.4s/580 tracking commands and passes
43/45, still unqualified. The timing bracket is exhausted.
`n11_online_gauge_pilot_001` uses18s and per-command observed axial reference
alignment, passing nominal plus all FIVE failed18s starts (6/6). The complete
`conditioned_n11_004` gate passes45/45 with5.50s hold and exact replay; its28
payload arrays are identical to the pilot. Inclusive native cost540580 steps.
`qualify_n11_001` passes all32 NEW reserved starts20262216..20262247,
full30s,5.50s hold each, exact cold replay and no native warning. All32 initial
states differ. Maximum cart coordinate2.072389423m, minimum substep boundary
margin0.927586999m, maximum force69.039894N, maximum launch tilt1.65013e-6.
This qualifies ELEVEN. All phases remain inside30s; physical twist stays free.
The suite now has85 passing tests, including independent condensed affine
optima, predictor isolation/candidate replay, and affine value/gauge transport.
`n11_hybrid_ten_regression_001` reproduces ten's first published held-out tape
with states/actions/tilts byte-identical after the new payload adapter changes.
Full held-out eleven video/preview rendered and inspected:1501 frames/50fps,
30.02s encoding; normal-speed300-frame30s GIF. Video SHA256
6089f9b985ed9a4d39a6c6077a06d1283808b129f5525ff9b5457f61be1a410f.
Published source passes85 tests and reproduces the first held-out state/action/
tilt tape byte-for-byte (`release11_reproduction_001/parity.json`). All3335
staged/archive hashes verify. Eleven is public at tag2d-cartpole-11-link-v1,
source84bd20a56dc15c7e9e470206a55f810c36c987eb. Public MP4 digest/size match;
evidence ZIP1443984255 bytes SHA256
240edd10022dbea5e82660b02e8fad5d3bd4a5d4d68ee3e0a0de003febb1e71a.
Public API confirms both assets and exact tag/source commit; master also matches.
No upper frontier has been established.

Twelve-link native motion/joint/gauge checks pass. Extending the stationary
axis-DARE test exposes a real numerical design failure: radius1.5096 and a
materially indefinite value. `n12_upright_value_diagnostic_001` mistakenly
uses elementwise sqrt on coupled Q and is invalid as a same-cost comparison;
the corrected002 retains Cholesky(Q).T and shows finite-QR model radius.9880214
at1500 steps. The factored value remains nonnegative while rounded P loses
small eigenmodes. New OPTIONAL finite_qr backend preserves its authoritative
factor and defaults; three-link DARE agreement, twelve world rotation and
root-based value checks pass. Current suite91 tests passes. This is local
linear design evidence, not twelve-link physical balance acceptance.
`n12_value_backend_n11_regression_001` repeats eleven's first published held-out
tape with states/actions/tilts byte-identical after these optional synthesis changes.
`planar_lift_n12_001` has an INPUT DECODING ERROR: the donor declares identity
coordinates (physical relative q/v), but the importer applies an earlier
normalized cumulative convention. Its direction check was against the same
wrong interpretation, and its3.13948 physical-tape mismatch was incorrectly
allowed by reference-only. Initial geometric reference is not hanging. Raw
finite001 fails both arms; these are genuine failed physical tapes but not
valid tests of the intended donor. Correction notes preserve the full evidence.
Invalid-input direct repair001 is deliberately stopped after one completed
row565200 native steps/defect163.8035; in-flight work is not recovered, so this
is only a lower bound on total compute. Session33228 terminal exit1 confirmed.

Importer now inverts the explicit donor coordinate_transform, uses that same
map for feedback, and checks exact hanging initialization, decoded native
directions and independent source physical-tape parity. Identity and legacy
cumulative/scaled tests pass;96 total tests pass. Eleven's frozen physical
controller/tapes are unchanged. `planar_lift_n12_002` WITHOUT reference-only
matches old source physical states EXACTLY (error0), initial hanging error0,
native world-direction error1.776e-15. Its direct policy still fails1.14s,
Y3.006855m/no hold, exact replay; design283228 native steps. The source remains
a proposal for the different canonical spatial mechanics, never accepted evidence.
`n12_direct_repair_002` completes20 full76D augmented corrections, each inside
registered bounds,11869828 native design steps. Defect1.45213169e-4 becomes
5.33725194e-16; maximum normalized1.49697125e-8. The separate gap-free baseline
fails11.18s with4.24s hold and exact replay. `n12_hybrid_finite_001` on the final
proposal has zero-offset failure10.72s/3.82s hold; full affine finite continuation
passes nominal30s/23.50s hold but0/12 raw noisy starts. These actual failures
are retained; tiny defects do not establish task success.
`conditioned_n12_001` uses this first finite design,
18s hanging feedback, online axial reference alignment and600 tracking commands
of its original1500-step design, including a9s swing. Nominal holds5.50s;
all45 development episodes pass, full30s/5.50s hold/no warning/exact replay.
Inclusive native cost540628. `qualify_n12_001` then passes all32 NEW reserved
starts20262316..20262347, every full30s with5.50s hold and exact replay,
no native warning.32 distinct initial states; maximum cart coordinate1.914864091m,
minimum substep margin1.085023719m, maximum force norm62.186687470N,
maximum launch tilt2.43815e-6rad. All384000 actual/proof native steps counted.
Independent saved-tape longest-streak and force/config checks pass. This
qualifies TWELVE. No search/state reset/extra episode time.
Full first-heldout video/normal-speed preview are rendered and inspected:
1501 frames/50fps,30.02s encoding,300-frame30s GIF. MP4 SHA256
2bc5c79efb073b417d3de42df7a7c9a38b14a680c5faa81c9e956863dd84c57f.
Packaged source96 tests passes; first held-out states/actions/tilts reproduce
byte-for-byte and its11 source module hashes match qualification.3708 staged/
archive hashes verify. Twelve is PUBLIC at tag2d-cartpole-12-link-v1, source
1ff88386cc1b563bc85b6938712f24894a36361d. MP4 is1142592 bytes with the hash
above. ZIP1753666834 bytes SHA256
b0774bac8ff0f3fc93ceb8747c3ee274f9ed4243a950f5edbec9b417afd49848.
Both public asset digests/sizes and exact tag/master commit verify. Do not rebuild
this published archive with later source. Current13-link freedom/rotation/gauge
extensions pass, bringing the lab to101 tests; twelve's frozen scope remains96.
`planar_lift_n13_001` completes strict import of the read-only9s physical-
relative donor: stored physical-tape parity0, hanging error0, native rod-direction
error2.165e-15. Native design304876 steps. Its direct baseline fails2.52s,
Y3.014954m/no hold, exact replay. The original file remains explicitly a
single-state/not_solution proposal. `n13_direct_repair_001` completes20 accepted
updates/12777076 native design steps, defect1.308216e-14/max6.583040e-8.
Its gap-free baseline fails8.00s/Y3.029320m with0.02s hold, exact replay.
`n13_hybrid_finite_001` passes full nominal23.50s hold but0/12 raw noisy starts.
`conditioned_n13_001` with18s online-gauge preparation passes30/45; no reserved
cohort is opened. All fifteen failed development starts/tapes are retained.

Optional exact-planar reference cleanup changes state/force only1.3644e-12/
2.5488e-12 normalized and reduces forbidden cross-plane gain5.1695e7->1.2006e5,
but its declared five-failure pilot passes only nominal (1/6). It has no measured
reliability gain. Optional physical-observable direction/transverse-rate tracking
also passes nominal23.50s/0 of12 raw starts and only1/6 prepared pilot. Its
kinematic factors independently match native observables, respect gauge-null
directions and leave genuine spin physical. Online alignment during hanging
preparation, using original arrays/18s clock, also passes only1/6; every original
payload array remains unchanged except the added flag. These are failed probes,
not qualified policies or evidence that all13 approaches are exhausted.

Current new trajectory: `n13_material_from14_001` transfers unwrapped cumulative
angle/rate fields at material centers from a historical14-link9s donor into
canonical13 mechanics, preserving cart state/cadence. Donor stored-tape parity0,
target hanging error0 and native direction error1.11e-15. This is a proposal
only. Donor's old nominal entry was earlier (24.16s hold versus23.50s), offering
a different arrival/time margin to test. Terminal `n13_direct_from14_001`
completes20 accepted full82D updates/12777076 design steps, with augmented
ridge and bounded LSMR fallback. Final defect6.129852e-16/max1.567505e-8;
its gap-free baseline fails7.36s with.26s hold. Finite-affine execution passes
nominal24.16s but0/12 raw noisy starts. The declared19s prepared gate reaches
44/45 with5.16s hold; only20261437 fails. A19.1s pilot and observed-cart-origin
alignment each still fail that start. Nine bounded observed-launch native
forecasts exactly reproduce the baseline failed suffix; only profile8,
reference offset(0,-.001)m, passes. Its actual pilot matches forecast exactly.
The fixed-offset full gate also reaches44/45; it fixes20261437 but fails20261444.
Neither fixed policy is qualified; all variants remain retained.

NEW causal launch admission forecasts BOTH fixed references from the current
observed state at command950, then selects the first full passing continuation.
No seed input, future actual state, physical recentering or resets. All forecast
states/actions/raw forces/tilts and query counts are saved. Pilot nominal plus
20261437/20261444 passes3/3, each full30s/5.16s hold/cold replay; every chosen
forecast equals actual suffix byte-for-byte. Inclusive pilot48380 native steps,
including11704 predictor steps. Unchanged `conditioned_n13_admission_001` is
terminal45/45, all5.16s holds/full30s/cold replay/chosen forecast suffix parity.
Inclusive737180 native steps include196504 predictor steps, within750000 budget.
Profile0 selected44 times/profile1 once. Payload35 arrays match pilot.
Frozen fresh qualification `qualify_n13_001` is terminal32/32, NEW noisy starts
20262416..47, all full30s/5.16s holds/no warning/exact cold replay. Every chosen
forecast states/actions/raw actions/tilts equals the actual suffix byte-for-byte.
All32 choose zero profile; the unchanged policy forecasts both candidates.
Inclusive524800 native steps include384000 actual+cold and140800 predictors.
All32 initial states differ. Maximum recorded-command cart coordinate1.883801165m,
minimum substep margin1.115844461m, maximum delivered force norm65.984695451N.
Thirteen QUALIFIES; publication/source/archive gates are being prepared.
Full first-heldout video1501frames50fps/30.02s encoding, normal-speed300frame
30s GIF and two contact sheets inspected. Video1144425 bytes SHA256
c87797b5a375c9bac212287e5e2817a5a7a4c727f6088cad2e0521539199976a.
Current suite131 tests passes. `n13_admission_n12_regression_001` preserves
twelve's first published states/actions/tilts byte-for-byte. Reserved13 seeds
20262416..47 are consumed successfully. Predictor cost is not hardware real-time proof.

Release packaging from13 includes complete current source/runtime, every new
or changed evidence file, and hash-bound immutable prior bundles for unchanged
historical evidence. All historical files remain in the Git checkout/full
inventory. Recursive ZIP verification is tested. Existing public ZIPs/tags are
create-only and unchanged. No13 archive exists before qualification/video gates.

Each new count needs a frozen controller, full nominal/development episode
tapes, fresh qualification, independent replay, source/runtime/config hashes,
and a full held-out video. Publish under the explicitly labeled `2d/` subtree,
using an isolated checkout (`publication/`), plus a `2d-cartpole-N-link-v1`
GitHub release with video and evidence. Never publish balance-only passes as
full hanging-start solves. Use a fast-forward push and preserve concurrent work.

The persistent goal remains active; no upper frontier or impossibility has been
established. Thirty-two reset-noise trials do not certify arbitrary disturbances.

## Resume checkpoint

One through TWELVE are qualified/public; full videos/evidence/asset hashes verify.
THIRTEEN is now qualified32/32 fresh starts; publication gates are pending.
Eleven direct transfer (`output/planar_lift_n11_001`)
fails nominal at 4.64s, X=3.261849m, no hold and exact replay. Repairs 001–004
and005 are terminal failures. Simultaneous direct repair001 closes proposal
defects but fails actual execution. The hybrid-document affine/finite design
`output/n11_hybrid_finite_001` passes full nominal23.50s hold, fails all12 raw
noisy starts. Its16s conditioned wrapper passes27/45 development episodes.
Static18s preparation improves to40/45;18.4s reaches43/45. Per-command axial
reference alignment at18s passes45/45 in conditioned004, then all32 fresh
qualification starts in qualify_n11_001. The full video is rendered/inspected.
Eleven publication is complete at84bd20a56dc15c7e9e470206a55f810c36c987eb,
with85 published-source tests, byte-identical first-heldout reproduction and
3335 staged/archive hashes. Invalid-input twelve001 is STOPPED and corrected;
corrected transfer002 has source physical-tape parity0, hanging error0, native
direction error1.776e-15, but direct policy fails1.14s. Direct repair002 is
terminal with20 accepted updates, tiny proposal defects and a failed4.24s-hold
baseline. Full affine1500-step finite001 passes nominal23.50s hold/0 of12 raw
noisy starts. Terminal output/conditioned_n12_001 uses
18s hanging/online-gauge feedback then600 commands from the same1500-step
design, passes45/45. Fresh32 qualification in qualify_n12_001 also passes all
full30s/5.50s holds/exact replays. Twelve publication is complete at
1ff88386cc1b563bc85b6938712f24894a36361d,3708 staged/archive hashes,
96 frozen-source tests, exact packaged-source first-tape parity, verified
public assets/tag/master. Thirteen strict transfer001 passes stored-tape parity0,
hanging error0/directions2.165e-15 but direct policy fails2.52s. Direct001 is
terminal; finite001 passes nominal23.50s but0/12 raw noisy starts, and its18s
online-gauge wrapper passes only30/45 development episodes. Cleanup/observable-
metric/online-parking pilots each fail all five reused failures (nominal1/6).
Earlier-entry material14->13 repair/finite design are terminal. Its original
and fixed-1mm-Y gates each reach44/45, with different failures. New observed-
launch two-profile native admission pilot passes3/3 and full gate45/45, with
exact chosen forecast suffix parity. Qualification output/qualify_n13_001 is
terminal32/32 NEW seeds20262416..47; all full30s/5.16s hold/no warning/cold
replay and forecast parity.524800 inclusive native steps. Publish13 with full
video/evidence after source/hash gates, then begin14. No live native job.
Twelve/thirteen reserved cohorts20262316..47/20262416..47 are consumed successfully.
No native search is used in the eleven qualified policy; goal remains active.
Historical proposal:
`runs/swingup11_uniform/eleven_link_controller.json`, 400 commands / 8s,
uniform eleven links, nominal cart maximum 2.071342m; source config
`configs/swingup11_uniform.yaml`. It is only a geometric proposal for unchanged
spatial mechanics. Current freedom/solver/affine/admission/archive checks:131 pass;
eleven/twelve frozen publications retain85/96-test source scopes respectively.
Keep physical freedom and near-planar-reference scope explicit. User authorizes
continued higher counts. Check live processes/terminal markers before resuming.
Historical planar route proposals are available in the original read-only
checkout; inspect their geometry, dynamics and timing before transfer.
The successful new tool is native fixed-target affine trajectory repair,
not force-only transfer. Preserve the strict 30s clock and all physical freedoms.
Record the chosen method, warm start, seeds and compute limit BEFORE execution.
Do not restart an experiment while its process is alive or its completed marker
exists. Checkpoints are proposals until full actual float32 gates pass. Eleven-link
reserved qualification seeds20262216..20262247 are now consumed successfully;
twelve's fresh cohort20262316..20262347 also passed; thirteen's NEW
20262416..20262447 frozen qualification passes32/32.
All future successful count releases require an explicitly 2D label, full video,
state/action evidence and hashes. No upper frontier has been established.

Source snapshots remain specific to each run. New experiments additionally
freeze their driver, input route, package metadata, dependency lock and thread
settings. Early auxiliary importer/gain-bank drivers were not copied before
execution; do not call those complete frozen synthesis reproductions. Native
state/action tapes and frozen final controllers remain authoritative evidence.
