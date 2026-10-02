# Fourteen-link canonical swing-up and hold

Fourteen uniform passive serial links pass the complete canonical repository
benchmark with one bounded cart force. The frozen controller passes noisy 20/20
(seed 214000), disjoint noisy 100/100 (214100), exact 20/20 (214200), and the
held-out video episode (214500). All 141 reserved episodes complete 1500 physical
steps / 30 seconds, hold every absolute link angle below 0.15 rad continuously
for 8.16 seconds, and remain in rail with valid simulation. Maximum reserved
cart excursion is 1.863168715 m.

The benchmark retains a 1 kg / 3 m chain, 1 kg cart, ±80 N force, ±3 m rail,
native MuJoCo binary64 RK4 with internal dt 0.005 s, 50 Hz held-force control,
float32 delivered actions and relative-angle/rate initial noise std 0.05.
Joint damping totals 0.015, cart damping is 0.02, per-DOF armature 0.0005,
and links retain capsule radius 0.025 m. No plant state is overwritten or reset
after initialization. This is an internal simulated benchmark result.

## Frozen evidence and reproduction

- Frozen source commit: `3277b4173a9333819bf35acb848e384f52a227ec`.
- Controller SHA256: `6bc6d9ea44c5fb7eba0b5182aeaaa7cd391721d52301fefc157e5ca99111a34d`.
- Video SHA256: `26f93fd2ff0b87db2bd6e7e429b8d1d4d3169fbbc328bb3831e22ea35214b7c3`.
- Policy/controller: `fourteen_link_policy.json`, `fourteen_link_controller.json`.
- Source: `fourteen_link_source.bundle`, `fourteen_link_source.json`.
- Reserved gates: `eval_swingup14_20.json`, `eval_swingup14_100.json`,
  `eval_swingup14_exact20.json`, `fourteen_link_video_episode.json`.
- Commands/manifest: `reserved_commands.json`, `fourteen_link_swingup_manifest.json`.
- Development: `experiment_ledger.json`, `synthesis_inputs.tar.gz`,
  `retiming_comparison.json`, `full_horizon_source.json`, and development chronology.
- Hashes: `SHA256SUMS`. Full video: `fourteen_link_swingup_success.mp4`.

All 419 repository/publication-source tests and 119 focused frozen-source tests
pass. The count-agnostic manifest verifier checks the complete release, including
all episodes and source/policy/video identities. The video is 1280×720 with
1500 recorded physical post-step frames at 50 fps, 30 seconds. Rendering neither
executes another controller nor interpolates the physical trajectory. The README
GIF shows the full episode at twice speed.

Clone the source bundle into a clean directory, use the runtime versions recorded
in metadata, set `PYTHONPATH=.:src:scripts`, `OMP_NUM_THREADS=1` and
`OPENBLAS_NUM_THREADS=1`, and run `reserved_commands.json` with local interpreter
and output paths. Preserve the controller/configuration bytes and every numerical
policy setting. Unpack `synthesis_inputs.tar.gz` at repository root to recover
development sources, exact commands, snapshots and failed candidates. These
commands include development orchestration; independent automatic rediscovery
is not an established result.

## Why simply repeating the thirteen-link recipe failed

The original continuation transfers thirteen's executed nine-second physical
curve in material coordinates, appends a five-second virtual tail, then performs
20 OSQP/legacy residual-SCvx iterations followed by 40 Clarabel/nonlinear-agreement
iterations. The final maximum native gap is 4.73368544e-8, yet physical replay
leaves the rail at 6.38 s with only 0.14 s upright. All declared 8/9/10/12 s handoffs
fail before their switch. Quiet virtual nodes and small residuals are inadequate
physical execution certificates for this unstable system.

Additional tests retain their negative outcomes: tighter gap weighting and QP
settings; native and square-root FDDP; pure QR and MP100 tracking designs;
smaller derivative increments; and a bounded native next-step value controller.
Pure QR and MP tracking produce bit-identical delivered physical prefixes despite
different designed gains. The native value probe extends the episode to 9.46 s
with 2.38 s hold but still leaves the rail. These experiments do not establish
that precision or nonlinear prediction can never help; they failed with the
specific sources, objectives and budgets tested.

The physical gap audit reproduces every recorded transition exactly. Thirteen
and the failed fourteen source have similarly small native gaps and comparable
one-step gap-to-action projections, while fourteen's feedback gains and physical
tracking errors are larger near failure. Fourth-order local linearization also
shows nonlinear remainder and saturation effects. This evidence supports a
conditioning and closed-loop trajectory problem, without assigning exact causal
shares to these coupled mechanisms.

## Timing recovery with the same physical benchmark

A bounded development bracket declares initializer swing durations 8,10 and 11 s,
in addition to the failed original 9 s. The reusable transfer helper uses cubic
Hermite position interpolation and the correctly scaled time derivatives for
velocities; control interpolation is only a warm start. It preserves winding,
geometry and endpoint identities. Retimed nodes are explicitly offline proposals,
and are not presumed dynamically feasible.

Each candidate receives the same 20+40 iteration repair and the same unchanged-
prefix 8/9/10/12 s handoff screen. The 8 s initializer ends at native gap
4.63267331e-8. Handoff 8 fails at 9.66 s; handoffs 9 and 10 complete 30 s with
24.16 s hold and cart excursion 1.91309675 m. Handoff 12 fails at 14.34 s despite
an earlier 6.50 s hold. The earliest full-pass rule selects 9 s. Ten- and eleven-
second initializer candidates fail at 5.40 and 7.06 s respectively, before any
declared switch. Their complete negative artifacts remain available.

The successful initializer is 8 seconds, while the released delivered tracking
route is 9 seconds. These durations must not be conflated. Timing changes several
trajectory properties together; this comparison establishes recovery under the
declared intervention, not a globally optimal duration or one isolated cause.

The successful route is rebuilt from actual delivered float32 actions and native
physical states. Initial and final checked gaps are exactly zero; independent
replay reproduces all 1500 states bit for bit. The unchanged launch policy parks
for 16 s with cart position/velocity costs 10/5, action cost 1000, cart target−0.05 m,
tracking scale 1 and no phase adaptation. Fresh development 20/20 (20264200) and
100/100 (20264300) pass before the policy/source freeze. Only then are reserved
cohorts evaluated. All four reserved gates pass without further tuning.

## Cadence, precision and the next boundary

Higher Hz remains a useful hypothesis to test, but this accepted result retains
50 Hz. Matched local mechanical capture studies at 50,100 and 200 Hz fail the
tested 14-link 1e-13 directions and pass 1e-15 directions; the tested 20-link local
controller fails both. A separate bounded linear 20-link transient study violates
force/rail limits despite stable mathematical poles, even with multiprecision arithmetic.
These are component tests of particular gains/directions, not canonical swing-up
tests or an impossibility argument for 20 links. Full resynthesis at another
cadence would constitute a separately labelled benchmark experiment.

The accepted capture gain uses 100-digit Riccati design. Execution, state storage
and nonlinear MuJoCo dynamics remain binary64 with float32 action delivery.
MP tracking/value variants are development probes and are not hidden additions
to the accepted static feedback policy.

Shared tools now cover physical transfer/retiming, residual shooting, handoff,
physical rebuilding, development gates, freezing, reserved evaluation and release
verification. The bounded runner initially failed on 14; successful retiming was
coordinated with declared development scripts. Count-specific controllers are
outputs of shared code, but this does not yet establish a universal policy,
fully automatic generalist synthesis, arbitrary morphology generalization,
hardware performance, sensor-noise robustness, delay or model mismatch tolerance.
Previous 7–13 results remain preserved. Fifteen through twenty remain unsolved.
