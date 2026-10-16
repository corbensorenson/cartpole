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
  442.54) did not yield a complete physical solve. No two-link release yet.
- Next steps: inspect final actual execution, qualify if successful, otherwise
  vary route duration/initialization with a declared budget. Advance to three
  only after passing two. Keep balance-only and full swing-up counts separate.

## Publication policy

Each new count needs a frozen controller, full nominal/development episode
tapes, fresh qualification, independent replay, source/runtime/config hashes,
and a full held-out video. Publish under the explicitly labeled `2d/` subtree,
using an isolated checkout (`publication/`), plus a `2d-cartpole-N-link-v1`
GitHub release with video and evidence. Never publish balance-only passes as
full hanging-start solves. Use a fast-forward push and preserve concurrent work.

The persistent goal remains active; no upper frontier or impossibility has been
established. Thirty-two reset-noise trials do not certify arbitrary disturbances.
