# Seven-to-Eight Transfer Review

The released seven-link architecture remains the reference: ten seconds of
hanging LQR conditioning, 228 Box-FDDP route actions at 50 Hz, trajectory
feedback scale 2, then upright LQR scale 1 after the complete route. Reusing
this architecture is justified. Copying its numerical route or gains without
target-plant optimization is only an initialization.

## Findings and Repairs

1. **P1: Incorrect feedback and state padding (fixed).**
   `scripts/pad_fddp_controller.py` appended two zero columns to a feedback
   matrix with position and velocity blocks. Adding a link shifts the velocity
   block; the old copy therefore applied the first velocity gain to the new
   angle channel. It also copied the terminal hinge velocity into the added
   relative joint, and duplicated a relative angle in physical states.
   Padding now preserves the two feedback blocks, duplicates only the absolute
   angle coordinate, and assigns zero relative angle/rate to collinear added
   links. A physical-coordinate and feedback-action invariance test covers it.
   Older padded-feedback failures cannot establish failure of correct transfer.

2. **P1: Control resampling altered the inherited waveform (fixed).**
   `scripts/force_proposal_to_fddp.py` treated N interval actions as N samples
   including the time T endpoint. Replaying at the original policy rate thus
   interpolated between the wrong samples, omitting the true final action.
   Saved actions now use zero-order hold on intervals `[i*T/N, (i+1)*T/N)`.
   A zero tail begins at T; feedback is indexed and padded consistently.
   Knotted proposals retain their source duration when a tail is appended.
   The adapter records the action actually applied by the environment, and
   disables both residual and switched auxiliary action controllers.

3. **P1: Warm starts inherited success evidence (fixed).**
   Padding carried the source's result and feasible/converged search status
   into the larger-chain artifact. It now creates an explicitly unverified
   search record and drops the source result. Separately, the release evaluator
   previously attached the default seven-link manifest and canonical label to
   arbitrary configs and execution settings. Canonical labeling now requires
   matching config/controller hashes and manifest execution settings;
   `--diagnostic` produces transfer evidence without a release manifest.

4. **P2: Nonlinear MPC used inconsistent time units and action precision (fixed).**
   `scripts/search_capture_online_mpc.py` counted physics substeps as policy
   steps when reporting the predicted arrival time and selecting its final
   scoring window. It also shifted a knot vector by the number of executed
   actions, and planned float64 forces while the live environment used float32
   actions. The planner now samples rollout states at policy boundaries,
   shifts knots with the existing elapsed-time helper, preserves its incumbent
   and zero-action baselines, and matches live action precision. Its fixed-state
   initializer now also clears inherited state scaling. Tests compare an entire
   batch trajectory against live MuJoCo steps, including a constrained plant.

5. **P1 research-design issue: the lock curriculum is discontinuous (open).**
   Only hinge 8 is constrained, not every hinge. At `p=0.999999`, the equality
   still has finite strength; it is deleted at `p=1`. For the same small last
   joint perturbation, its acceleration jumps from approximately `-0.2111` to
   `+0.5775 rad/s^2`. Thus `p=0.80` is neither 80% physical unlocking nor 80%
   completion. Even the endpoint retains split lengths/masses. Preserve this
   legacy condition for reproduction; do not treat it as a smooth path to the
   canonical eight-link plant. Any replacement continuation must first pass a
   measured dynamics-continuity check at its endpoint.

6. **P2: Conclusions outran the experiments (corrected in roadmap).**
   A few failed tail optimizers do not rule out linear capture or prove that a
   neural capture expert is required. The `p=0.80` custom-gain hold was reported
   from an inline run without a complete saved executable controller/result
   bundle. It remains provisional. The older claim that all `p=0.90` tails
   reached the rail is also inaccurate: the zero-action probe reached its time
   limit after losing upright, and MPC stayed below 3 m. Low momentum alone
   does not establish membership in a stabilizer's recoverable state region.

## Verification and Current Outcome

- Seven-link release replay: **20/20**, mean upright streak **15.48 s**, maximum
  cart excursion approximately **2.371 m**. Artifact:
  `runs/swingup7_review_regression20.json`. This repeats the declared cohort as
  a regression check; it is not an additional independent statistical cohort.
- Corrected padded transfer with the same full architecture: **0/3**, no upright
  interval, maximum cart excursion approximately **3.039 m**. Artifact:
  `runs/swingup8_review_corrected_transfer3.json`.
- Exact-clock eight-link warm start: all **228 actions** replayed, maximum cart
  excursion **2.124 m**. Subsequent Box-FDDP with a 100-iteration limit stopped
  at iteration **18**, unconverged; live replay held **0 s** and reached
  **3.081 m**. Artifacts: `runs/swingup8_review_exact_clock_warmstart.json` and
  `runs/swingup8_review_exact_clock_fddp100.json`.
- Local upright linearization: both seven and eight have closed-loop spectral
  radius below 1 (approximately **0.9951** and **0.9952**), but the gain norm
  rises from **1974** to **9183**. This suggests greater sensitivity to handoff
  errors and saturation; it is not a nonlinear robustness certificate.
  Reproduce with `scripts/audit_transfer_numerics.py`; output:
  `runs/transfer_review_numerics.json`.
- Corrected MPC on the same split `p=0.90` endpoint and seed: **0.20 s** upright
  streak versus the old **0.06 s**, with **0.16 s** low-momentum streak and
  time-limit termination. It still fails capture. Artifact:
  `runs/swingup8_review_p090_online_mpc_v2.json`.
- Regression suite: **89 tests** pass, including six new transfer tests, under
  both pytest and the existing unittest runner. Default pytest discovery is
  scoped to `tests/`; the imported research packets have separate environments.

## Next Experiment Contract

Keep the released architecture and run each transfer as an explicit ablation:
correct target-state reconstruction, Box-FDDP optimization, full route execution,
then capture from its actual terminal state. Record any difference in conditioning,
tracking scale, timing, objective, or plant. Retain the same controller to compare
open-loop and feedback replay. Require saved controls, coordinate transform,
feedback gains, terminal gain, resolved config, and a complete no-reset result
before accepting a new intermediate frontier. The next missing result is a
canonical eight-link route whose actual endpoint can be held by its capture
controller. A new controller family is not yet supported by this review.
