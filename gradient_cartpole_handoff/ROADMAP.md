# Seven-Link Swing-Up Roadmap

## Purpose

This is the authoritative completion contract for the project.

The final research target is a reproducible uniform **7-link** MuJoCo cart-pole that starts hanging below the cart, swings up, captures, and remains upright. The existing 6-link work is a required calibration and debugging gate because a public 6-link result already exists; it is not the final claim.

The project goal can point directly at this file:

> Complete every required phase and the Final Completion Audit in `ROADMAP.md`. Produce reproducible public evidence that the canonical uniform 7-link MuJoCo cart-pole swings up from the hanging initial-state distribution and stabilizes upright, meeting the 20-episode and 100-episode success gates with published weights, hashes, metrics, and a reset-free 30-second video. Six-link end-to-end reproduction is a mandatory calibration gate, not the endpoint. Do not treat near-upright, curriculum-stage, widened-rail, altered-morphology, hand-picked-seed, or reset-containing runs as completion evidence.

## Final Definition Of Done

The project is complete only when all of these are true:

- The canonical 7-link benchmark below is implemented in `configs/swingup7_uniform.yaml` and documented without unresolved benchmark choices.
- A reset-free policy or explicit expert-chain manifest solves that exact benchmark from the hanging initial-state distribution.
- `runs/swingup7_uniform/eval_swingup7_20.json` reports `success_rate >= 0.80` over 20 deterministic held-out episodes.
- `runs/swingup7_uniform/eval_swingup7_100.json` reports `success_rate >= 0.90` over 100 deterministic held-out episodes.
- Both evaluation files contain per-episode seeds, returns, termination reasons, time to first upright, time to capture, maximum continuous upright streak, final upright streak, maximum cart excursion, and handoff data when two experts are used.
- Published weights and/or the expert-chain manifest have SHA-256 hashes recorded in the evidence JSON.
- `runs/swingup7_uniform/seven_link_swingup_success.mp4` is a 30-second held-out episode showing hanging start, swing-up, capture, and sustained stabilization.
- `runs/swingup7_uniform/seven_link_swingup_success.video.json` reports `reset_count == 0`, identifies the held-out evaluation seed, and contains no failure termination before successful episode completion.
- The evidence records the resolved config hash, generated MuJoCo XML hash, runtime/package versions, git commit, clean/dirty state, action frequency, wall-clock training time, and environment-step count.
- The final evaluation uses the uniform target plant, `+/-3 m` rail, target damping, zero hinge friction loss, and no curriculum-only training wheels.
- A fresh clone at the recorded commit can run the documented evaluation and rendering commands against the published weights.
- The public README states the exact scope and does not present near-upright or 6-link calibration evidence as the 7-link result.

Passing a curriculum stage, reaching upright briefly, solving a zero-noise seed, or producing a high shaped return does not satisfy this definition.

## Canonical Benchmark Contract

This contract is frozen for the final claim. Training may use curricula and easier plants, but final evaluation may not.

| Setting | Canonical value |
|---|---:|
| Links | `7` serial links |
| Total link length | `3.0 m` |
| Per-link length | uniform, `3/7 m` |
| Total link mass | `1.0 kg` |
| Per-link mass | uniform, `1/7 kg` |
| Cart mass | `1.0 kg` |
| Total joint damping | `0.015`, distributed uniformly |
| Joint friction loss | `0.0` |
| Cart damping | `0.02` |
| Joint armature | `0.0005` |
| Rail | `[-3.0, +3.0] m` |
| Action | one continuous normalized cart force in `[-1, 1]` |
| Force limit | `+/-80 N` |
| MuJoCo timestep | `0.005 s` |
| Frame skip | `4` |
| Policy/action frequency | `50 Hz` |
| Episode length | `30 s`, 1500 policy steps |
| Initial cart position | exactly `0.0 m` |
| Initial relative angles | `[pi, 0, 0, 0, 0, 0, 0] + Normal(0, 0.05)` |
| Initial generalized velocities | `Normal(0, 0.05)` |
| Angle failure termination | disabled |
| Failure termination | rail violation or non-finite simulation state |
| Upright condition | maximum absolute link angle `< 0.15 rad` |
| Minimum successful hold | at least `5.0 s` continuously upright |
| Evaluation | deterministic policy on held-out seeds |

The final evaluation config must use explicit `init_mode: hanging`. A training config may use `hanging_curriculum`, but evaluation must not depend on setting curriculum progress to obtain the real start.

With the current observation design and `obs_include_morphology: true`, the canonical 7-link observation has 51 values:

```text
cart position / rail                    1
cart velocity                           1
sin(absolute link angles)               7
cos(absolute link angles)               7
relative hinge angles                   7
hinge velocities                        7
length/mass/damping morphology vector  21
                                      ----
                                        51
```

If a different observation, action space, rail, force limit, or success definition is needed to match an external result, add a separately named matched-benchmark config. Do not silently alter this canonical contract.

## Permitted Solution Architecture

A single policy or a two-expert swing/capture policy is acceptable. A two-expert result must satisfy additional constraints:

- Both experts act through the same declared cart-force action.
- Switching occurs inside one uninterrupted MuJoCo episode.
- Handoff does not overwrite `qpos`, `qvel`, simulation time, or hidden simulator state.
- The switch rule is deterministic, versioned, and included in the policy manifest.
- Swing terminal states used to train capture are saved from actual swing-policy rollouts, split by episode seed, and hashed.
- Final evaluation starts from hanging; it may not start from a saved handoff state.
- All expert checkpoints, preprocessing, switch thresholds, and recurrent state rules are published.

An open-loop trajectory may be used for diagnostics or warm starts, but final evidence must use state feedback and pass the held-out initial-state distribution.

## Roadmap

### Phase 0: Lock The Benchmark And Verifier

Status: **Passed (2026-07-15)**. Run `make roadmap-p0` to regenerate the XML, exercise the rejection tests, and verify the runtime benchmark contract.

Deliverables:

- `configs/swingup7_uniform.yaml` implementing the canonical contract.
- A generated 7-link XML fixture or hash-producing export command.
- Smoke assertions for hanging geometry, 51-value observation, action bounds, force scaling, rail, timestep, and reset noise.
- A solution verifier that rejects missing weights, wrong link count, altered plant parameters, reset-containing video metadata, insufficient episodes, and success rates below the gates.
- Evidence fields for final upright streak and explicit termination reason if they are not already present.

Gate P0:

- The benchmark and verifier tests pass before any 7-link result is accepted.
- The config contains no unresolved placeholder values.

### Phase 1: Build A Reliable Six-Link Capture Basin

Purpose: solve the current bottleneck independently before spending compute on increasingly energetic swing policies.

Work:

- Freeze a seeded synthetic handoff-envelope generator with `|x| <= 1.25 m`, maximum absolute link angle `<= 0.15 rad`, `|cart_velocity| <= 0.50 m/s`, and hinge-velocity RMS `<= 0.75 rad/s`.
- Train capture from that synthetic envelope, then iterate with actual saved swing-policy states as Phase 2 produces them.
- Expand angle, hinge velocity, cart velocity, and cart-position distributions by mastery gates.
- Restore saved handoff velocities from zero to full scale while keeping the plant fixed.
- Compare nonlinear PPO, model-based MPC/iLQR warm starts, supervised policy distillation, and residual feedback using the same state distribution.
- Penalize rail consumption and measure final upright streak, not just first capture.

Gate P1:

- On 1,000 held-out seeded states from the frozen uniform 6-link synthetic handoff envelope, capture succeeds in at least 90% of episodes.
- Median continuous upright hold is at least `10 s` and no successful episode hits the rail.
- The synthetic envelope definition, generator version, seeds, and hashes are recorded.

### Phase 2: Produce Cold, Centered Six-Link Swing Handoffs

Purpose: train the swing expert against what the capture expert can actually recover.

Work:

- Optimize swing reward and checkpoint ranking for capture success or capture-value, not upright crossing alone.
- Require rail margin, low hinge velocity, low cart velocity, and centered cart position at handoff.
- Use longer rails, mass/length/damping gradients, friction, and easier starts only as training curricula.
- Anneal each training wheel independently and log the frontier where mastery fails.
- Save one best valid handoff per rollout episode for train/validation/test splits.

Gate P2:

- From the exact hanging-start uniform 6-link benchmark, at least 80 of 100 held-out swing episodes enter the proven P1 capture basin.
- Handoff acceptance is computed by replaying the capture expert, not solely by fixed angle/velocity thresholds.
- Saved handoffs come from the final uniform plant and real `+/-3 m` rail.
- Capture succeeds on at least 90% of accepted real handoffs and holds for a median of at least `10 s`.
- The real handoff dataset records source checkpoint, episode seed, `qpos`, `qvel`, morphology, rail, selection thresholds, split membership, and hashes.

Phases 1 and 2 are an intentional coupled loop: new real handoffs expand capture training, and the capture expert's measured basin supplies the swing objective. Their gates remain separate and both must pass.

### Phase 3: Integrate And Reproduce Six Links

Work:

- Run swing and capture/stabilize in one reset-free episode.
- Tune deterministic switching and hysteresis without changing simulator state.
- Evaluate failure categories: no swing-up, bad handoff, rail hit before capture, capture loss, and post-capture fall.

Deliverables:

- An explicit hanging-start uniform 6-link evaluation config using the canonical contract except `n_links: 6`.
- Publicly documented comparison against the known 6-link result, including any benchmark differences.
- Six-link end-to-end weights or expert manifest, 20-episode evidence, 100-episode evidence, and reset-free video.

Gate P3:

- `success_rate >= 0.80` over 20 held-out episodes.
- `success_rate >= 0.90` over 100 held-out episodes.
- At least one held-out 30-second video shows the entire sequence without reset.
- The integrated result can be reproduced from the published policy manifest.
- No evaluation reset and no curriculum plant is present in the final evidence.

P3 is the mandatory six-link calibration milestone that permits expensive seven-link sweeps. The existing near-upright LQR result does not count.

### Phase 4: Build The Seven-Link Capture Basin

Work:

- Port the frozen synthetic handoff envelope to the canonical uniform 7-link plant.
- Initialize from the proven 6-link capture method where architecture permits.
- Compare specialized 7-link policies with an N-conditioned graph/recurrent policy.
- Add real 7-link handoffs iteratively as Phase 5 produces them, without replacing the held-out synthetic gate.

Gate P4:

- On 1,000 held-out seeded states from the frozen uniform 7-link synthetic handoff envelope, capture succeeds in at least 90% of episodes.
- Median continuous upright hold is at least `10 s` and no successful episode hits the rail.
- The gate uses the canonical 7-link morphology, rail, force, damping, and observation.

### Phase 5: Scale Swing From Six To Seven

Work:

- Add link-count curricula such as `1 -> 2 -> ... -> 7` or `6 -> 7` while preserving total length and mass.
- Decide experimentally between per-link-count policies and a shared N-conditioned graph/recurrent policy.
- Carry forward the proven P4 capture basin and swing-to-capture objective.
- Apply length, mass, damping, friction, start-angle, and rail curricula one axis at a time so regressions are attributable.
- Keep a 7-link frontier ledger with plant progress, link count, handoff rate, capture rate, upright streak, rail use, and checkpoint hash.

Gate P5:

- Seven-link swing reaches the verified capture basin in at least 80% of 100 held-out episodes on the final plant.
- Seven-link capture succeeds from at least 90% of those held-out real handoffs.
- Median continuous upright hold after those real handoffs is at least `10 s`.
- Both component gates use the canonical rail, force, morphology, damping, and reset distribution.

### Phase 6: Solve The Integrated Seven-Link Benchmark

Deliverables:

```text
runs/swingup7_uniform/config.resolved.yaml
runs/swingup7_uniform/policy_manifest.json
runs/swingup7_uniform/checkpoints/best.safetensors             # single-policy architecture
runs/swingup7_uniform/checkpoints/swing_best.safetensors       # two-expert architecture
runs/swingup7_uniform/checkpoints/capture_best.safetensors     # two-expert architecture
runs/swingup7_uniform/eval_swingup7_20.json
runs/swingup7_uniform/eval_swingup7_100.json
runs/swingup7_uniform/seven_link_swingup_success.mp4
runs/swingup7_uniform/seven_link_swingup_success.video.json
runs/swingup7_uniform/SHA256SUMS
```

Gate P6:

- Every threshold and artifact in Final Definition Of Done is satisfied.
- Either the single-policy checkpoint exists or both two-expert checkpoints exist; the manifest identifies the chosen architecture and exact files.
- The 100-episode evaluation is run once against the frozen checkpoint and config after tuning ends.
- The video seed is present in the held-out evaluation set and is identified in its metadata.

### Phase 7: Publish And Independently Reproduce

Deliverables:

- Public repository commit containing code, frozen configs, commands, and result documentation.
- Weights published in the repository, Git LFS, or a versioned GitHub Release with matching hashes.
- One-command evaluation and rendering targets for the frozen result.
- A clean-environment reproduction run from a fresh clone.
- A comparison table separating the canonical benchmark, any Yacine-matched benchmark, the 6-link calibration, and the 7-link result.
- A limitations section covering continuous actions, simulation-only evidence, compute cost, and benchmark differences.

Gate P7:

- A fresh-clone reproduction regenerates matching evaluation metrics within deterministic expectations and verifies every SHA-256 hash.
- Public wording survives the Final Completion Audit without relying on local-only files.

### Stretch Phase: Eight Or More Links

After P7, evaluate 8+ links using the same contract and evidence process. This is a stretch result and is not required to complete the 7-link project.

## Experiment Discipline

- Every training run gets an immutable output directory and resolved config.
- Training seeds and evaluation seeds are disjoint and recorded.
- Hyperparameter selection uses development episodes; the frozen 100-episode test is not used for tuning.
- Checkpoints rank success, capture, final upright streak, and rail safety before shaped return.
- Failed runs remain documented with the specific gate they failed.
- Training-wheel ablations change one major axis at a time whenever practical.
- Runs on nonuniform morphology or rails wider than `+/-3 m` are labeled curriculum evidence only.
- Do not claim a record without a dated literature/repository search and an exact benchmark comparison.

## Decision Rules

- If P1 fails even from zero-velocity states, prioritize the stabilizer/controller and observation design before further swing search.
- If P2 reaches upright but misses the P1 basin, optimize handoff quality rather than raw upright crossings.
- If P3 cannot reproduce six links, investigate benchmark or algorithm mismatch; do not hide the failure by moving directly to seven links.
- If the canonical rail appears physically limiting, test wider-rail ablations, but do not redefine the canonical benchmark without a documented decision and a separately named claim.
- If a shared N-conditioned policy underperforms specialized policies, finishing seven links with specialized experts is acceptable; report the comparison honestly.

## Current Status

| Phase | Status | Current evidence |
|---|---|---|
| P0 benchmark/verifier | Passed | Canonical config, XML hash export, runtime assertions, rejection tests, and final artifact verifier pass via `make roadmap-p0`. |
| P1 six-link capture basin | In progress | The seeded 20k/2k/1k envelope and strict gate evaluator are frozen. Analytic LQR scores `0/1000`; the current component-wise curriculum passes fixed validation at `p=0.06` (`qpos=p^2`, `qvel=p^3`) and stalls at `p=0.0625`. |
| P2 six-link swing handoff | In progress | Best learned handoffs are from a progress-`0.3875` curriculum plant, not final uniform 6-link. |
| P3 integrated six | Not passed | Near-upright 6-link stabilization is solved; hanging-start end-to-end swing-up is not. |
| P4 seven-link capture basin | Not started | Blocked by the P3 calibration gate. |
| P5 seven-link swing | Not started | Blocked by the P3 calibration gate. |
| P6 integrated seven | Not started | No qualifying checkpoint or evidence. |
| P7 public reproduction | Not started | Public repo exists; final artifacts do not. |

## Final Completion Audit

- [x] P0 passed and canonical benchmark frozen.
- [ ] P1 passed on the frozen synthetic 6-link handoff envelope.
- [ ] P2 passed from uniform hanging-start 6-link rollouts and real handoffs.
- [ ] P3 passed with integrated reset-free 6-link reproduction evidence.
- [ ] P4 passed on the frozen synthetic 7-link handoff envelope.
- [ ] P5 component gates passed on the canonical 7-link plant.
- [ ] P6 20-episode evaluation passed.
- [ ] P6 100-episode evaluation passed.
- [ ] P6 reset-free 30-second video and metadata passed.
- [ ] All configs, XML, weights, manifests, and evidence hashes verified.
- [ ] Final evidence points to a clean public git commit.
- [ ] Fresh-clone reproduction completed.
- [ ] P7 public comparison and limitations documentation completed.

Only after every required checkbox is verified should the project goal be marked complete.
