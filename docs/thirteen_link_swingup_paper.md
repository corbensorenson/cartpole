# Thirteen-link canonical swing-up and hold

Thirteen uniform passive serial links pass the complete internal canonical
release bundle from noisy hanging starts using one bounded cart force. The
frozen policy passes noisy 20/20(seed 213000), disjoint noisy 100/100(seed 213100),
exact 20/20(seed 213200), and the held-out video episode(seed 213500). All 141
reserved episodes run the full 1500 steps, hold upright 7.50 s, remain in rail,
and report valid simulation. Maximum reserved cart excursion is
1.943462883 m.

The benchmark retains total chain mass 1 kg, total length 3 m, cart mass 1 kg,
force limit 80 N, rail limit 3 m, internal RK 4 step 0.005 s,50 Hz control,
and thirty-second episodes. Canonical relative-angle and velocity noise has
standard deviation 0.05. Every absolute link angle must remain below 0.15 rad
for five continuous seconds. No plant state is reset or overwritten after
initialization. This is an internal repository benchmark, not an external
world-record claim or a hardware result.

## Frozen evidence

- Source commit: `aa16a383b1562696772b2c26cf8bf83bae22d824`.
- Controller SHA 256: `2eb6e2383ed96abb6fd041529c811bc1e21a05b401edf3e7043b6eaf85b65995`.
- Full video SHA 256: `f20e550d20c03b1d7bbe0274da3bc90c91dfd6c1147919c15f7dd1961160972d`.
- Clean source: `thirteen_link_source.bundle` and `thirteen_link_source.json`.
- Policy: `thirteen_link_policy.json`; route: `thirteen_link_controller.json`.
- Gates: `eval_swingup13_20.json`, `eval_swingup13_100.json`,
  `eval_swingup13_exact20.json`, and `thirteen_link_video_episode.json`.
- Manifest: `thirteen_link_swingup_manifest.json`.
- Exact commands: `reserved_commands.json`; hashes: `SHA256SUMS`.
- Video: `thirteen_link_swingup_success.mp4`,1280x 720,50fps,1500 frames,
  thirty seconds. Each frame depicts the held-out evaluator's recorded physical
  post-step state. Rendering performs no second controller run or interpolation.

All 401 working-tree tests and 73 focused tests from the frozen fresh clone
pass. The count-agnostic release verifier checks the final manifest. GitHub
publication runs the complete suite again in an isolated checkout containing
all published reference fixtures.

## Shared synthesis and why the naive continuation failed

This result inherits the released twelve-link nine-second physical curve in
material coordinates. Absolute angles and rates are interpolated at normalized
link centers; the resulting target nodes are not presumed dynamically feasible.
An explicit five-second virtual capture tail gives a fourteen-second offline
optimization horizon. These virtual nodes never replace the live plant.

Twenty residual-form SCvx iterations with the legacy OSQP recipe leave maximum
physical dynamics gap 0.00101238 and fail actual replay at 4.04 s despite an
upright-looking nominal endpoint. A further forty iterations on that same
saved source with Clarabel and legacy trust updates leave gap 0.000182364 and
fail at 6.06 s. Native QP convergence alone does not repair nonlinear feasibility.

The legacy trust rule keeps a large region and lowers regularization even after
very short accepted steps. With the same source, objective, native Clarabel
backend and forty outer iterations, nonlinear model agreement instead contracts
poorly predicted or tiny-step regions and expands accurately predicted full
steps. Maximum physical gap falls to 4.50616e-8. Actual replay now holds upright
7.64 s, but the late fourteen-second LQR handoff fails at 17.38 s.

A predeclared unchanged-prefix handoff screen tests 8,9,10 and 12 seconds. Eight
fails;9,10 and 12 pass all thirty seconds with 23.50 s hold and maximum cart
excursion 1.9934265 m. Nine is the earliest passing member and is selected.
At 9 s the actual capture value is 8.12798e-7, versus 4.75508 at 14 s. Even a
small-looking pose and a nominal capture threshold are insufficient certificates.
The complete physical episode is the deciding test.

The selected route is then rebuilt using serial physical feedback transitions
and actual delivered float 32 actions. Its initial and final checked dynamics
gaps are exactly zero, and its independent replay reproduces every physical
state of the successful source episode. Pure tracking feedback around the
separate inverse/restored initializer fails and is retained as a negative
control; it is not the successful controller.

The parked launch remains the eleven/twelve contract:16 seconds of hanging
contraction, cart-position/velocity costs 10/5, control cost 1000, target-0.05 m,
tracking scale 1 and deferred maintenance handoff. The rebuilt source passes
fresh noisy 20/20(seed 20263200) and disjoint 100/100(seed 20263300) before freezing.
Reserved seeds are used only after the policy and source are frozen; no reserved
result is used for tuning. No increase in control frequency is needed here.

The upright Riccati design uses 100-digit arithmetic. Nonlinear MuJoCo physics,
state storage and execution remain binary 64, with the existing float 32 action
interface. Arbitrary-precision design is not arbitrary-precision simulation.

## Reproduction and scope

Clone the source bundle to a fresh directory, set `PYTHONPATH=.:src:scripts`,
`OMP_NUM_THREADS=1` and`OPENBLAS_NUM_THREADS=1`, and run the exact commands in
`reserved_commands.json` from the frozen clone, adapting output paths only.
Use the pinned runtime recorded in source/evaluation metadata. The optional
interior-point synthesis backend is Clarabel 0.11.1, available through the
`interior-point` project extra. Its bounded-QP conversion follows the official
[Clarabel interface](https://clarabel.org/stable/python/getting_started_py/).
Native Clarabel normalized residuals and OSQP absolute residuals are reported
separately; inner iteration budgets 200 and 10000 are not equivalent work units.
Recorded search timing overlaps other independent experiments and is not a
serial speed benchmark.

The reusable procedure consists of material physical transfer, declared virtual
capture initialization, full-horizon residual shooting, unchanged-prefix handoff
screening, exact delivered-action rebuilding, development tests, frozen-source
reserved validation, trace rendering and manifest verification. Entry points are
`scripts/transfer_physical_capture_route.py`,
`scripts/synthesize_frontier_capture.py`, `scripts/replay_frontier_handoff.py`,
`scripts/rebuild_frontier_physical_route.py`, `scripts/freeze_frontier_release.py`
and `scripts/package_frontier_release.py`. The successful two-stage trajectory
repair is 20 OSQP/legacy iterations followed by 40 Clarabel/agreement iterations,
with initial QP tolerance caps 0.001 and 1e-5 respectively and floor 1e-8.
Exact commands and input/source snapshots accompany each run.

Count-specific controllers are outputs of shared code. This result does not
establish a universal policy, automatic independent rediscovery, arbitrary
morphology generalization, sensor-noise tolerance, delay robustness or model
mismatch tolerance. Fourteen through twenty remain unsolved. Full development
chronology, failed winding assumptions, inverse/FDDP failures and retained
negative controls are in `docs/thirteen_link_development.md` and the experiment
ledger. Previous 7–12 evidence remains preserved.


The additional forty-iteration OSQP continuation has also completed. Its
full result, native status history and compute are retained in the final
experiment ledger; it does not pass the physical capture gate. This negative
branch is independent of the accepted frozen policy and reserved validation.
