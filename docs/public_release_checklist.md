# Public Release Checklist

This checklist exists to keep the public result defensible. It separates the
passed canonical control gates from the remaining project and external-rule
audits.

## Current Release Status

The project now has a reproducible canonical noisy-start seven-link swing-up
and hold result:

- 7 uniform links, canonical MuJoCo plant, +/-3 m rail, +/-80 N force limit.
- A 10-second active hanging-equilibrium conditioning phase, followed by the
  swing route and terminal capture expert.
- 20/20 and 100/100 success on the canonical noisy hanging-start distribution.
- A 30.00 s held-out noisy-start video, zero resets, and all seven links in
  frame during the upright hold.
- Maximum cart excursion 2.3708 m across the 100-episode gate.

The evidence records the exact clean tracked source commit. A fresh-clone audit
on 2026-09-12 ran all 78 tests, the benchmark verifier, and the release
checksums. Public material may describe this as a released canonical seven-link
20/100-gate result, but should not call it a universal world record unless an
external competition's exact rules and submission requirements have also been
verified.

## Evidence Inventory

| Artifact | Purpose |
|---|---|
| `runs/swingup7_uniform/seven_link_release_controller.json` | Frozen route and feedback gains used by the hybrid controller |
| `runs/swingup7_uniform/seven_link_swingup_manifest.json` | Complete conditioning, swing, and capture policy manifest |
| `runs/swingup7_uniform/seven_link_swingup_success.mp4` | 30-second canonical noisy held-out video |
| `runs/swingup7_uniform/seven_link_swingup_success.video.json` | Video metadata, hashes, runtime, reset count, and final metrics |
| `runs/swingup7_uniform/eval_swingup7_20.json` | 20-episode canonical gate |
| `runs/swingup7_uniform/eval_swingup7_100.json` | 100-episode canonical gate |
| `runs/swingup7_uniform/robustness_sweep.json` | Non-canonical paired-seed stress characterization |
| `runs/swingup7_uniform/SHA256SUMS` | Integrity manifest for the public evidence bundle |
| `runs/eval_swingup7_fddp_two_expert_canonical20.json` | Negative control on the required noisy initial distribution |
| `docs/seven_link_swingup_paper.md` | Method, results, limitations, and reproduction commands |
| `docs/levers_and_pitfalls.md` | Full experiment ledger, including failed branches |
| `ROADMAP.md` | Frozen completion contract and unchecked final gates |

The exact controller, hanging-settle gain, video, configuration, generated XML,
and manifest hashes are recorded in the release metadata and cross-checked by
`SHA256SUMS` and `make verify-swingup7`.

## Allowed Public Language

Use language such as:

> We produced a reset-free canonical seven-link MuJoCo swing-up and hold.
> A settled-launch hybrid controller passes 20/20 and 100/100 noisy
> hanging-start episodes, then holds the uniform chain upright for the rest of
> the 30-second episode. The artifact, video, hashes, and negative controls
> are public. The project remains a hybrid model-based result pending
> external-rule comparison.

Avoid:

- “world record” or “competition winner”;
- “solved every seven-link benchmark” without naming the exact contract;
- implying that the video is a single neural policy rather than a hybrid chain;
- presenting a widened rail, altered morphology, curriculum stage, or
  near-upright run as canonical evidence;
- omitting the original no-settle 0/20 negative control when describing why
  the settled-launch phase matters.

## Before Calling The Project Complete

Do not check the final roadmap boxes until all of these are present:

1. `runs/swingup7_uniform/eval_swingup7_20.json` reports at least 0.80 success.
2. `runs/swingup7_uniform/eval_swingup7_100.json` reports at least 0.90 success.
3. The final video starts from a held-out noisy hanging state and reports
   `reset_count == 0`.
4. The final manifest publishes every expert, switch threshold, preprocessing
   rule, and hash.
5. A clean public commit and fresh-clone replay reproduce the metrics.
6. The post and paper are updated to identify the exact held-out seeds and
   benchmark configuration.

Items 1–6 have been satisfied for the repository's canonical result. A
universal world-record claim still requires matching the external
competition's exact plant, timing, scoring, and submission rules.
