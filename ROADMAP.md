# Escalating High-Link Swing-Up Roadmap

## Purpose

This is the authoritative completion contract for the project.

The project advances one link at a time. The current frontier is a reproducible uniform **8-link** MuJoCo cart-pole because the 7-link controller has passed the project's canonical 20/100 control gate. Every frontier starts hanging below the cart, swings up, captures, and remains upright. The existing 6-link work is a required calibration and debugging gate because a public 6-link result already exists; it is not the endpoint.

The project goal can point directly at this file:

> Complete every required phase and the Final Completion Audit in `ROADMAP.md`. At the active frontier `n`, produce reproducible public evidence that the canonical uniform `n`-link MuJoCo cart-pole swings up from the hanging initial-state distribution and stabilizes upright, meeting the 20-episode and 100-episode success gates with published weights, hashes, metrics, and a reset-free 30-second video. Once frontier `n` passes that gate, promote the active frontier to `n+1` and repeat until the user stops the escalation. Six-link end-to-end reproduction remains a mandatory calibration gate, not the endpoint. Do not treat near-upright, curriculum-stage, widened-rail, altered-morphology, hand-picked-seed, or reset-containing runs as completion evidence.

## Escalating Frontier Rule

The active frontier is the smallest link count greater than the latest passed
frontier result. A link count advances only after the previous count has all of
the following evidence under the same canonical MuJoCo contract:

- a reset-free hanging-start controller or explicit two-expert manifest;
- the held-out 20-episode gate at `>= 0.80` success;
- the held-out 100-episode gate at `>= 0.90` success;
- a reset-free 30-second video with the complete chain visible and metadata;
- published controller artifacts, hashes, resolved config, runtime, and exact replay commands.

Training wheels may be used for discovery, including longer rails, morphology
gradients, damping, friction, easier starts, and saved real handoff states.
They never advance the frontier. After a frontier advances, the next campaign
inherits the same evidence checklist and creates `n+1` config, manifest,
evaluation, video, and paper artifacts. The current ladder is:

### Method Inheritance Rule

Frontiers 8 and above inherit the released seven-link controller architecture
before any new discovery method is considered. The default campaign is:

1. Condition the noisy hanging state with the same hanging-equilibrium LQR.
2. Replay the released swing controls through the exact target-link MuJoCo
   plant and save the resulting target-chain states.
3. Re-optimize that state-consistent route with the same Box-FDDP terminal
   objective and time-varying feedback.
4. Hand the real terminal states to a capture/stabilize expert in the same
   uninterrupted episode; use terminal LQR where it is demonstrably stable,
   otherwise keep the capture controller inside the same exact FDDP family.
5. Evaluate the complete chain on the canonical target plant before promoting
   the next morphology or link count.

Only one continuation lever may change at a time. Rail widening, morphology
gradients, friction, damping, and longer horizons are temporary continuation
conditions and must be replayed back on the canonical target before they are
accepted. Broad PPO, global CEM, or unrelated controller-family searches are
not the default next step for a frontier that already has a working predecessor;
they require an explicit ledger entry explaining why the inherited chain was
falsified. Every inherited-route attempt must record its exact replay artifact,
terminal state, rail outcome, and capture outcome in `docs/levers_and_pitfalls.md`.

| Frontier | Status | Advancement target |
|---:|---|---|
| 6 | Calibration required | Complete the frozen six-link end-to-end gates |
| 7 | Canonical 20/100 control gate passed | Finish audit items, then retain as reference |
| 8 | Active | Pass the full canonical evidence bundle |
| 9+ | Queued | Start automatically after the preceding frontier passes |

## Final Definition Of Done

The project is complete only when all of these are true:

- The active-frontier benchmark below is implemented in `configs/swingup{n}_uniform.yaml` and documented without unresolved benchmark choices; the 7-link release remains the reference implementation.
- A reset-free policy or explicit expert-chain manifest solves that exact benchmark from the hanging initial-state distribution.
- `runs/swingup{n}_uniform/eval_swingup{n}_20.json` reports `success_rate >= 0.80` over 20 deterministic held-out episodes.
- `runs/swingup{n}_uniform/eval_swingup{n}_100.json` reports `success_rate >= 0.90` over 100 deterministic held-out episodes.
- Both evaluation files contain per-episode seeds, returns, termination reasons, time to first upright, time to capture, maximum continuous upright streak, final upright streak, maximum cart excursion, and handoff data when two experts are used.
- Published weights and/or the expert-chain manifest have SHA-256 hashes recorded in the evidence JSON.
- `runs/swingup{n}_uniform/{n}_link_swingup_success.mp4` is a 30-second held-out episode showing hanging start, swing-up, capture, and sustained stabilization.
- `runs/swingup{n}_uniform/{n}_link_swingup_success.video.json` reports `reset_count == 0`, identifies a held-out seed disjoint from both evaluation cohorts, and contains no failure termination before successful episode completion.
- The evidence records the resolved config hash, generated MuJoCo XML hash, runtime/package versions, git commit, clean/dirty state, action frequency, wall-clock training time, and environment-step count.
- The final evaluation uses the uniform `n`-link target plant, `+/-3 m` rail, target damping, zero hinge friction loss, and no curriculum-only training wheels.
- A fresh clone at the recorded commit can run the documented evaluation and rendering commands against the published weights.
- The public README states the exact active-frontier scope and does not present near-upright, lower-link calibration, or curriculum evidence as the `n`-link result.

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

A single policy or an explicit phased expert chain is acceptable. The released
seven-link runtime chain is the reference architecture for later frontiers and
has two experts: a swing-up expert and a
capture/stabilize expert. Maintenance is a separate training prerequisite for
the second expert and may be folded into its final checkpoint or switch logic.
Its gate and evidence remain separate. Any multi-expert result must satisfy
these constraints:

- Both experts act through the same declared cart-force action.
- Switching occurs inside one uninterrupted MuJoCo episode.
- Handoff does not overwrite `qpos`, `qvel`, simulation time, or hidden simulator state.
- The switch rule is deterministic, versioned, and included in the policy manifest.
- Swing terminal states used to train capture are saved from actual swing-policy rollouts, split by episode seed, and hashed.
- Final evaluation starts from hanging; it may not start from a saved handoff state.
- All expert checkpoints, preprocessing, switch thresholds, and recurrent state rules are published.
- The switch uses hysteresis. Enter and exit thresholds may use an LQR Lyapunov score, a learned capture value, or both, but final acceptance is established by uninterrupted nonlinear rollout under the canonical force and rail limits.
- Distillation into one policy is optional and may begin only after the explicit hybrid expert chain passes its component gates. The working expert chain remains the fallback and reference evaluator.

The training capabilities still have a strict dependency order. Maintenance is
trained from an upright start and then widened around that equilibrium.
Capture/stabilize is trained from real mastered-maintenance states and then
from actual swing-policy terminal states while its angle and velocity envelope
expands. Swing-up is trained against the measured capture value only after that
basin exists, and its final evaluation still starts from the canonical hanging
distribution. A brief upright crossing from a swing policy cannot bypass
either prerequisite.

An open-loop trajectory may be used for diagnostics or warm starts, but final evidence must use state feedback and pass the held-out initial-state distribution.

## External Packet Findings

The review of `frontier_swingup_study/`, `hybrid_training_study/`, `harmonic_codesign_study/`, and `cartpole_research_master_handoff/` is recorded in [`docs/packet_review.md`](docs/packet_review.md). The frontier study reinforces that global discovery must precede verification: it produced a genuine three-link model-based down-start swing-up, but no seven/eight-link solution under a different idealized benchmark. The hybrid study contributes evidence and resource-accounting discipline, not a swing-up controller. The harmonic study adds a bounded phase/energy and morphology-design diagnostic, with no canonical MuJoCo swing-up evidence. The master handoff adds immutable task identity, independent forward replay, positive/negative controls, energy-matching, and risk/accounting components, but its MuJoCo gate is blocked and its global high-link task remains unfinished.

The active experiment ledger is [`docs/levers_and_pitfalls.md`](docs/levers_and_pitfalls.md). It records exact levers, negative controls, hard-negative states, and the conditions under which a result is or is not admissible evidence.

None of these packets is canonical evidence. Their non-MuJoCo dynamics, nonuniform morphologies, force limits, rails, actuator contracts, and initial-state distributions differ from this roadmap's contract. In particular, the master handoff's legacy `+/-15 N`, `+/-2.4 m` rail, actuator lag, and 19 s horizon must not be silently substituted for the frozen uniform MuJoCo benchmark.

## Roadmap

### Phase 0: Lock The Benchmark And Verifier

Status: **Benchmark contract passed; full suite has a local runtime blocker**. The native MuJoCo/verifier subset passes (`41` tests). The full `make roadmap-p0` discovery is currently blocked when the MLX-dependent capture-envelope tests request a Metal device in this headless session; this is an environment issue, not accepted final evidence.

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

Status: **Active**. The exact-MuJoCo local-SCP teacher now solves all six fixed full-envelope development states, including reset-free recoveries for prior rail failures 4 and 5. This `6/6` model-based development audit validates teacher generation but does not satisfy the 1,000-state policy gate below.

Purpose: solve the current bottleneck independently before spending compute on increasingly energetic swing policies.

Work:

- Freeze a seeded synthetic handoff-envelope generator with `|x| <= 1.25 m`, maximum absolute link angle `<= 0.15 rad`, `|cart_velocity| <= 0.50 m/s`, and hinge-velocity RMS `<= 0.75 rad/s`.
- Train capture from that synthetic envelope, then iterate with actual saved swing-policy states as Phase 2 produces them.
- Expand angle, hinge velocity, cart velocity, and cart-position distributions by mastery gates.
- Restore saved handoff velocities from zero to full scale while keeping the plant fixed.
- Compare nonlinear PPO, model-based MPC/iLQR/Box-FDDP warm starts, supervised policy distillation, and residual feedback using the same state distribution.
- Penalize rail consumption and measure final upright streak, not just first capture.
- Fit a capture-value or funnel-membership model from exact capture-policy rollouts. Treat `x^T P x` from local LQR as a candidate feature, not proof of nonlinear constrained recoverability.
- Express LQR state costs and diagnostics in declared dimensionless coordinates. Record open/closed-loop eigenvalues and identify which modal directions dominate boundary failures before changing rewards or PPO settings.
- Label frontier states by modal composition and report recovery by mode. Introduce single-mode and small mode-group perturbations as a diagnostic curriculum only when the fixed all-state gate shows a repeatable modal failure pattern.

Gate P1:

- On 1,000 held-out seeded states from the frozen uniform 6-link synthetic handoff envelope, capture succeeds in at least 90% of episodes.
- Median continuous upright hold is at least `10 s` and no successful episode hits the rail.
- The synthetic envelope definition, generator version, seeds, and hashes are recorded.

### Phase 2: Produce Cold, Centered Six-Link Swing Handoffs

Purpose: train the swing expert against what the capture expert can actually recover.

Work:

- Optimize swing reward and checkpoint ranking for downstream capture success or capture-value, not upright crossing alone.
- Track rail margin, relative hinge velocity, cumulative absolute link-rate RMS/max, cart position, and cart velocity as shaping and ranking metrics. Treat a nominal low-momentum envelope as a useful curriculum target, not a universal pre-filter: a valid capture expert may need to arrest a state that still carries substantial momentum.
- Use longer rails, mass/length/damping gradients, friction, and easier starts only as training curricula.
- Use the harmonic co-design packet's bounded mass/length/damping profiles as predeclared discovery diagnostics. Record modal input participation, energy gap, force saturation, and capture outcomes, but label the graded mechanism separately and restore the canonical uniform plant before any gate claim.
- Anneal each training wheel independently and log the frontier where mastery fails.
- Save one best valid handoff per rollout episode for train/validation/test splits.
- Add direct collocation, DDP/iLQR, or nonlinear MPC trajectory generation whose terminal cost is entry into the measured P1 capture funnel, then train feedback tracking or imitation policies from successful trajectories.
- Use reverse-curriculum or return-to-promising-state training only for training; final swing evaluation must still begin from hanging. Every generated trajectory must be replayed in exact MuJoCo before it becomes a teacher label.
- Prefer a teacher/learner loop in which successful feedback trajectories create a split, hashed handoff dataset, followed by learner rollouts and planner labels on learner-visited states.

Gate P2:

- From the exact hanging-start uniform 6-link benchmark, at least 80 of 100 held-out swing episodes enter the proven P1 capture basin.
- Handoff acceptance is computed by replaying the capture expert, not solely by fixed angle/velocity thresholds.
- Saved handoffs come from the final uniform plant and real `+/-3 m` rail.
- Capture succeeds on at least 90% of accepted real handoffs and holds for a median of at least `10 s`.
- The real handoff dataset records source checkpoint, episode seed, `qpos`, `qvel`, morphology, rail, selection thresholds, split membership, and hashes.

Phases 1 and 2 are an intentional coupled loop: new real handoffs expand capture training, and the capture expert's measured basin supplies the swing objective. Their gates remain separate and both must pass.

### Global Discovery Feasibility Gate

Status: **Not passed**. This is an execution checkpoint between capture-basin work and full swing-policy scaling. It does not replace P1, P2, or P3.

Purpose: prove that the exact uniform six-link MuJoCo plant has at least one executable hanging-start route into sustained capture before investing in larger policy cohorts or validation acceleration.

Required evidence:

- At least one reset-free feedback-controlled rollout starts from the published hanging distribution, reaches the all-link upright condition, and holds for at least `5 s`.
- The same controller is replayed from at least three additional held-out hanging-start seeds.
- The canonical six-link rail, force, damping, action frequency, timestep, and failure rules are used.
- Full `qpos`, `qvel`, actions, seeds, config hash, controller hash, capture time, hold time, rail use, and termination reason are saved.
- Any coarse model, collocation plan, or optimizer endpoint is treated only as a warm start; exact MuJoCo replay is authoritative.

Decision rule: if this checkpoint fails after a declared search budget, change the global discovery method. Do not spend the next budget on verifier optimization, additional isolated capture teachers, or near-upright evidence.

- The first discovery pivot after the current CEM/iLQR attempts is phase/energy trajectory design with a bounded morphology screen, followed by exact MuJoCo replay. Do not promote a graded morphology or near-upright local recovery into canonical evidence.

### Global Discovery Method Branch: Phase/Energy Co-Design

Status: **Active diagnostic branch; no canonical solution yet**. The harmonic co-design packet motivates this branch, but does not supply canonical evidence.

Use a small, predeclared set of bounded morphology profiles and phase/energy trajectory parameterizations to search for a feedback-controlled route into the measured capture basin. The branch must:

- compare uniform, legacy, lighter/longer-tip, and all-increasing profiles under explicitly named resource budgets;
- add energy-gap-matched controls so lower lifting work is not mistaken for a resonance benefit;
- continue a candidate motion from an excitable hanging mode toward the upright capture region while penalizing transverse modal error, hinge momentum, cart excursion, and force saturation;
- replay every proposed route in exact MuJoCo at the canonical action cadence, with no reset into the capture set;
- treat the graded plant as a co-designed result or training curriculum only; the final seven-link claim remains the canonical uniform plant;
- preserve full trajectories, controller parameters, profile parameters, config hashes, and failure categories even when the route fails.

This branch is the next discovery method if the current exact six-link gate remains at zero sustained holds. It is not a relaxation of P1, P2, P3, or the final benchmark.

The first rail-length ablation supports making rail width an explicit homotopy
axis. With the same intermediate six-link energy-homotopy checkpoint, fixed
rail half-widths of `3`, `6`, and `9 m` produced no upright crossing, `12 m`
produced one crossing in two episodes, and `18 m` produced crossings in both
episodes. None produced a verified capture or hold. This is recorded in
[`docs/rail_length_homotopy.md`](docs/rail_length_homotopy.md). The next
discovery runs must therefore (1) learn the swing with enough rail to avoid
premature clipping, (2) save real crossing states, (3) train capture from those
states, and (4) anneal the rail back toward `+/-3 m` while ranking states by
downstream capture value. Low-momentum and centered states remain valuable
shaping targets and diagnostics, but missing a nominal rate threshold must not
discard a state before the real capture expert has tested it.

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

- Run the staged seven-link curriculum in [`docs/7_link_expert_curriculum.md`](docs/7_link_expert_curriculum.md).
- First pass the maintenance checkpoint: upright-start deterministic rollouts
  must hold for the required window while staying inside the rail. Advance the
  reset-angle and reset-velocity curriculum only after the current stage is
  mastered.
- Then pass the capture/recovery checkpoint from those real maintenance
  states. Keep the widest mastered initial-state envelope and its held-out
  seeds; do not describe this upright-start result as swing-up.
- Only after maintenance and capture pass may the swing expert be trained from
  `hanging_curriculum`. Preserve the checkpoint handoff and record which
  phase produced every final policy artifact.

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
- Prefer a link-growth homotopy that splits one existing link while preserving total length and mass, initially constrains or nearly removes the new joint's dynamic influence, and gradually releases its mass, inertia, and relative motion before annealing to the canonical uniform plant.
- During link release, isolate the newly introduced or materially changed unstable modes, replay previously mastered modes, and log per-mode recovery, force, rail use, and control energy.
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
runs/swingup7_uniform/seven_link_swingup_manifest.json
runs/swingup7_uniform/seven_link_release_controller.json
runs/swingup7_uniform/eval_swingup7_20.json
runs/swingup7_uniform/eval_swingup7_100.json
runs/swingup7_uniform/seven_link_swingup_success.mp4
runs/swingup7_uniform/seven_link_swingup_success.video.json
runs/swingup7_uniform/robustness_sweep.json
runs/swingup7_uniform/SHA256SUMS
```

Gate P6:

- Every threshold and artifact in Final Definition Of Done is satisfied.
- The release manifest identifies the settled-launch hybrid architecture, all three phases, the exact controller artifact, switching semantics, and hashes.
- The 100-episode evaluation is run once against the frozen checkpoint and config after tuning ends.
- The 20-episode cohort, 100-episode cohort, and video seed are mutually disjoint and identified in their metadata.

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

### Stretch Phase: Scalable Eight Or More Links

After P7, turn the seven-link result into an arbitrary-`n` scaling experiment. This work is downstream of the canonical seven-link swing-up claim and is not required to complete P0-P7.

- Implement a masked per-link graph, recurrent, or global-token policy whose input width and parameters do not need to be redesigned for each `n`.
- Grow `n -> n+1` through a link-splitting homotopy that preserves total length and mass, begins with a small strictly positive dynamic contribution, and gradually releases the new joint before annealing to the canonical uniform morphology.
- Recompute the model-based local oracle for every continuation plant and train the newly introduced unstable modes individually and compositionally while replaying previously mastered modes.
- Keep mixed-`n` replay so advancement at the frontier does not erase lower-link competence.
- Measure zero-shot transfer, fine-tuned success, environment steps, wall time, capture volume, weakest modal authority, force saturation, and rail use for each link count.
- Evaluate at least `n=8,9,10` under frozen per-link-count contracts and the same evidence discipline used for seven links; do not weaken force, rail, total-mass, or total-length constraints without naming a separate benchmark.

## Experiment Discipline

- Every training run gets an immutable output directory and resolved config.
- Training seeds and evaluation seeds are disjoint and recorded.
- Hyperparameter selection uses development episodes; the frozen 100-episode test is not used for tuning.
- Checkpoints rank success, capture, final upright streak, and rail safety before shaped return.
- Failed runs remain documented with the specific gate they failed.
- Training-wheel ablations change one major axis at a time whenever practical.
- Runs on nonuniform morphology or rails wider than `+/-3 m` are labeled curriculum evidence only.
- Keep canonical equal-link and co-designed morphology results as separate tracks. Optimized lengths, masses, damping, friction, rail, or force may initialize training and support ablations but never substitute for a canonical gate.
- Coarse or lower-fidelity simulation may rank proposals, but final labels and all gate decisions use exact MuJoCo at the declared action cadence. Record fidelity, action holds, numerical/domain failures, and replay discrepancies.
- Global discovery, capture, sustained hold, and statistical verification are separate metrics and separate evidence stages. Do not use a verifier result to imply that a controller was discovered.
- Handoff telemetry must include both relative hinge-rate RMS and cumulative absolute link angular-rate RMS/max. A quiet relative-joint state can still carry coordinated whole-chain motion. These quantities are shaping and reporting signals, not a universal hard gate; the authoritative handoff test is uninterrupted nonlinear capture from the actual saved state.
- Record CPU, GPU, RAM, worker layout, warmup, time overshoot, failed jobs, and numerical aborts in the experiment ledger.
- Do not claim a record without a dated literature/repository search and an exact benchmark comparison.

## Decision Rules

- If P1 fails even from zero-velocity states, prioritize the stabilizer/controller and observation design before further swing search.
- If the Global Discovery Feasibility Gate fails, pivot the discovery method before expanding cohorts or optimizing validation speed.
- If P2 reaches upright but misses the P1 basin, optimize handoff quality rather than raw upright crossings.
- If a trajectory optimizer reports convergence or a small defect but exact MuJoCo replay fails, classify the result as a failed proposal, not a swing-up candidate.
- If P3 cannot reproduce six links, investigate benchmark or algorithm mismatch; do not hide the failure by moving directly to seven links.
- If the canonical rail appears physically limiting, test wider-rail ablations, but do not redefine the canonical benchmark without a documented decision and a separately named claim.
- If a shared N-conditioned policy underperforms specialized policies, finishing seven links with specialized experts is acceptable; report the comparison honestly.
- Keep MuJoCo as the authoritative simulator. Build a separate batched planar backend only after profiling shows simulation throughput is a material bottleneck and budget its MuJoCo parity, Jacobian, conservation, and timestep-convergence tests as a distinct engineering phase.

## Current Status

| Phase | Status | Current evidence |
|---|---|---|
| P0 benchmark/verifier | Passed | Canonical config, XML hash export, runtime assertions, native MuJoCo tests, and the final artifact verifier are present; all 78 tests pass in the release environment. |
| Global discovery feasibility | Not passed | The CPU-safe exact-MuJoCo evaluator records `0/4` five-second holds for the current low-momentum swing plus LQR chain; the best baseline streak is `0.04 s`. Capture-ready CEM reduced nominal hinge RMS to about `0.835 rad/s` but still produced no sustained hold. The active phase/energy branch now has a measured rail-length diagnostic: the intermediate checkpoint crossed upright at `12 m` in `1/2` episodes and at `18 m` in `2/2`, but had `0/2` low-momentum handoffs and `0/2` captures. The completed 150-update real-handoff capture curriculum reached a best `0.28 s` upright streak and `0.7958` capture-quality score, but captured `0/2` episodes, succeeded `0/2`, and reached about `3.07 m` cart excursion. |
| P1 six-link capture basin | In progress | The seeded 20k/2k/1k envelope and strict gate evaluator are frozen. At `p=0.0700`, target planning reaches `217/256`, standard feedback MPC reaches `227/256`, and deterministic escalation reaches `233/256 = 91.02%` with a `13.90 s` median hold and no successful rail hits. The next `p=0.0725` cascade reaches only `220/256 = 85.94%`, so the accepted frontier remains `p=0.0700`. On representative `p=1.0` state 674, a frozen CEM-seeded DDP approach, settling tail, and LQR fallback succeeds in uninterrupted replay: funnel entry at `5.02 s`, minimum `V=0.12`, `9.30 s` upright hold, and maximum cart excursion `2.413 m`; its measured feedback tube is only `4/32` at normalized radius `0.005` and zero by `0.05`. Static action distillation, reward-only PPO, and raw-action DAgger are rejected. Predictive and receding-iLQR tails bottom out near `V=4,500`. Exact-MuJoCo Box-FDDP with stronger endpoint weights and policy-action precision parity now produces strict `10.56 s` replay successes through `alpha=0.7878125` toward held-out state 442. Full trajectory feedback recovers `32/32` perturbations through normalized radius `0.05`, `26/32` at `0.10`, and `13/32` at `0.20`, but `0/23` nearest distinct validation states. This is one useful local funnel rather than a reusable capture policy, and P1 is not passed. |
| P2 six-link swing handoff | In progress | Best learned handoffs are from a progress-`0.3875` curriculum plant, not final uniform 6-link. |
| P3 integrated six | Not passed | Near-upright 6-link stabilization is solved; hanging-start end-to-end swing-up is not. |
| P4 seven-link maintenance/capture | Superseded by integrated pass | The terminal upright LQR holds all 120 disjoint canonical release episodes after the Box-FDDP route; older curriculum failures remain preserved as research history. |
| P5 seven-link swing | Superseded by integrated pass | The settled-launch controller reaches capture in all 120 disjoint canonical release episodes; component-only gates were superseded by stronger end-to-end evidence. |
| P6 integrated seven | Passed | Frozen settled-launch hybrid passes disjoint 20/20 and 100/100 cohorts with a reset-free, independently seeded 30-second video. |
| P7 public reproduction | Passed for the canonical release | Public README, paper, commands, hashes, limitations, verifier, and fresh-clone audit are present. Independent third-party reproduction and external competition matching remain open. |

Latest P1 boundary update (2026-09-12): exact-state LQR sweeps across the
first 64 frozen full-envelope test states produced `0/64` success and `64/64`
rail exits across gain scales and control costs. A linear continuation from
the existing supervisor reached `87.5%` only at progress `0.05956`, then fell
to `28%` at `0.122` and `9%` at `0.185`. A gated continuation reproduced the
boundary: `78.125%` at `0.070`, followed by `59%` at `0.080` with no further
frontier advance. These runs are diagnostic only; P1 remains open and the
next inherited method is a reusable internal-mode recovery teacher evaluated
against the same frozen 1,000-state gate.

Latest P4/P5 update (2026-09-11): the CPU Torch finite-difference-LQR
residual curriculum advanced a narrow seven-link capture frontier through
`plant_progress=0.999`, but its checkpoint retained
`init_qpos_scale=init_qvel_scale=0.002`. The scaled-state audit appeared to
hold one state for `30 s`; the required exact unscaled replay with both scales
forced to `1.0` was `0/1`, held upright for `0.02 s`, and hit the `13 m` rail
after `1.32 s`. The artifact
`runs/swingup7_capture_lqr118_final` is therefore a rejected curriculum
checkpoint, not full-scale capture evidence. P4 and P5 remain open.

Latest P5 exact-plant probes (2026-09-11): a serial long-rail trajectory CEM
on the canonical uniform plant reached a late `0.130 rad` point with
`1.198 rad/s` relative hinge RMS and `0.04 s` upright streak, but exact replay
through the current capture expert hit the `20 m` rail. A chain-aware LQR
variant entered capture at `8.96 s` with `0.129 rad`, `1.733 rad/s` hinge RMS,
and `-0.947 m` cart position; it held only `0.06 s` before LQR saturation and
rail loss. Receding-horizon MPC briefly reached `0.039 rad` from that saved
state but did not hold. Unit-scale PPO from the same state remained `0/1`
after roughly `0.94M` environment steps. Final-plant modal and mass-matrix
energy searches reached no upright event; the best mass-matrix point had
energy fraction `1.000` but `1.058 rad` angle and `3.615 rad/s` hinge RMS.
These results strengthen the diagnosis: total energy is reachable, but the
seven internal modes are not arriving synchronized and quiet. P4/P5 remain
open, and no seven-link swing-up-and-hold claim is admissible.

Latest exact-handoff follow-up (2026-09-11): the hybrid replay artifact
`runs/swingup7_chain_lqr_replay_trace_seed20260987.json` now reproduces the
chain search's recorded seed and cart-noise semantics. It matches the
`8.96 s` capture switch and the saved `0.128716 rad / 1.732562 rad/s` post-step
state before the route reaches the `+/-20 m` rail at `10.48 s`. An exact-state
open-loop tail CEM, affine/rich/mixture/two-phase capture CEMs, and fixed-state
iLQR were then run from real handoffs; none produced absolute-low-momentum
hold, and the longest centered upright streak was `0.06 s`.

The swing objective now records cumulative absolute link-rate RMS. Its best
angle-constrained final-plant handoff is `0.149208 rad`, `1.064008 rad/s`
relative hinge RMS, `2.495875 rad/s` absolute-rate RMS, cart `0.148620 m`, and
cart velocity `0.043985 m/s`. A heavier absolute-rate objective found a
`1.4546 rad/s` absolute-rate point, but at `0.526 rad` angle, so it is not a
capture handoff. The immediate next P5 lever is Pareto-ranked trajectory
discovery against downstream capture value. Capture should be tested from
plausible real states even when they exceed a nominal low-momentum envelope;
the envelope is a curriculum and diagnostic tool, not a hard substitute for
the capture rollout. P4/P5 remain open.

Latest P5 optimizer follow-up (2026-09-11): terminal-rate-aware exact
multiple shooting from the real `0.149 rad` handoff reduced the best
cumulative absolute-rate RMS to about `1.84 rad/s` on the canonical-rail
probe, but the replay reached the `+/-3 m` rail. A `+/-13 m` development
diagnostic reached about `1.83 rad/s` and also rail-clamped; heavier terminal
absolute-rate weight did not improve it. A separate serial exact global CEM
with absolute-rate scoring, warm-started from a replayable force waveform,
finished with no late upright event; its best late point was about `1.46 rad`
angle, `2.28 rad/s` relative hinge RMS, and `4.98 rad/s` absolute-rate RMS.
These are negative diagnostics, not canonical evidence. The immediate P5
decision is to move to phase-aware feedback or a direct trajectory method
that can coordinate the seven internal modes; do not spend more time tuning
the same late handoff tail.

Latest P5 two-expert follow-up (2026-09-11):
`scripts/search_pfl_capture_controller.py` now tests the proposed two-expert
architecture directly: mass-matrix cart-acceleration energy shaping for the
swing phase, then exact seven-link LQR for gated capture. The `+/-12 m`
development probe reached no upright event and hit the rail. With a `+/-20 m`
diagnostic rail, CEM found a transient `0.592 rad` best angle but no upright
event, no capture-gate activation at that point, and `0.0 s` streak. Reusing
the quiet `absolute004` maintenance actor as the capture expert from the real
`0.149 rad / 2.496 rad/s` handoff produced only `0.02 s` upright streak and
`0.0 s` low-momentum hold. The two-expert decomposition is therefore still
the right interface, but its capture expert must be trained on high-rate
approach states; the quiet-state actor cannot serve as the arrest expert.
P4/P5 remain open.

Latest CPU curriculum follow-up (2026-09-11): the preserved run
`runs/swingup7_swing_power2_trust_region` advanced the teacher-backed
quadratic hanging curriculum through `progress=0.040` with two consecutive
eight-episode gates, briefly reached `progress=0.050`, and then oscillated
between passing and failing evaluations. It was stopped at update `849`
without a stage gate. At `progress=0.050`, the actual initial angle is only
`pi * 0.05^2 = 0.00785 rad`, so this is maintenance-curriculum evidence, not
meaningful swing-up evidence. The parallel linear-angle continuation stalled
at `progress=0.005`. The decision is to stop extending the low-exploration
maintenance branch and test an exploration-enabled swing curriculum with an
explicit energy/swing teacher, while keeping the final canonical evaluator
strictly uniform, hanging-start, rail `+/-3 m`, and reset-free.

Latest capture-interface follow-up (2026-09-11): measured 7-link approach
states were exported from the absolute-rate swing route into
`runs/swingup7_high_rate_capture_state_set/`. They are real MuJoCo states,
but are development data from a widened `+/-20 m` rail and span roughly
`0.14-0.65 rad` angle error with substantial cumulative link rates. A shared
linear/tanh capture actor searched over all 16 states reduced its aggregate
cost from about `61,653` to `14,045`, but produced `0/16` successful holds,
repeated rail violations, and `0.0 s` absolute-low-momentum streak. The actor
class and state set are therefore not yet a usable capture expert. The next
diagnostic narrows to the closest measured handoff and tests an open-loop
sequence/richer state-feedback law before expanding the held-out neighborhood.
The result remains component evidence only; canonical 7-link success is `0`.

Latest exploration follow-up (2026-09-11): initializing PPO with action
standard deviation `0.35` and linear-angle curriculum progress `0.005` did not
pass its first deterministic gate after about `0.67M` environment steps.
Deterministic success stayed `0/8` while reward improved, so increased noise
alone is not enough to discover the swing route from the maintenance policy.
Preserve this branch at `runs/swingup7_swing_exploration_linear`; the next
curriculum must inject a real swing/energy teacher or use a trajectory-level
interface rather than only widening PPO exploration.

Latest capture feasibility probes (2026-09-11): from the closest measured
high-rate handoff (`0.138 rad` angle error, `5.92 rad/s` relative hinge RMS,
`6.25 rad/s` cumulative absolute-rate RMS), an exact open-loop 8-second CEM,
a fixed two-phase affine controller, a richer angle-rate feedback CEM, and
50-step exact iLQR all failed to produce a low-momentum hold. The best richer
feedback replay stayed inside `0.45 m` cart excursion but reached only
`0.02 s` upright and `0.0 s` absolute-low-momentum hold; iLQR ended in a rail
violation. These are component diagnostics, not canonical evidence, and they
confirm that the swing stage must hand over a much quieter state.

The new whole-chain-rate-aware direct hanging-start action search
(`runs/swingup7_action_cem_absolute_capture_ready.json`) completed with no
candidate state: its best late angle was about `0.84 rad`, with `0.0 s`
upright streak. Its added objective is now part of the reusable search tool,
but this direct force-knot branch is rejected as a route to the capture basin.
The attempted 20-second serial global CEM also exposed and preserved a
configuration pitfall: the warm-start cart-target controller was not a
replayable force center under the current plant and hit the widened rail
before the first optimization population. No result from that run is counted.

Latest live-planner and rail-length diagnostic (2026-09-11): a new
low-dimensional receding-horizon exact-MuJoCo CEM is implemented in
`scripts/search_swingup_online_mpc.py`. On the canonical `+/-3 m` rail it left
the exact hanging equilibrium and reached only `0.379` normalized chain-height
proxy before a rail violation at `1.42 s`. On a `+/-12 m` discovery rail, the
same planner reached `0.995` height proxy, but the best geometric crossing
still had `0.495 rad` maximum angle, `22.16 rad/s` cumulative absolute-rate
RMS, and `6.875 m` cart excursion. Tightening the planner's cart and capture
weights kept the excursion below `3 m` and reached `0.984` height proxy with a
`0.465 rad` best angle, but still carried `21.30 rad/s` absolute-rate RMS and
never held. The result supports the rail-length hypothesis as a feasibility
lever, but not as a solution: additional travel currently produces a violent
high-energy pass, not a capturable seven-link handoff. P4/P5 remain open and
canonical seven-link success remains `0`.

Latest low-momentum interpretation and direct-force follow-up (2026-09-11):
low momentum is explicitly treated as a shaping objective and diagnostic, not
as an independent benchmark gate. A rate-weighted exact force refinement found
the strongest current canonical-rail swing crossing at about `0.0998 rad`
maximum angle, `2.084 rad/s` cumulative absolute-rate RMS, `1.309 rad/s`
relative hinge RMS, cart position `0.610 m`, cart velocity `-0.909 m/s`, and
maximum rail use `2.722 m`. Direct feedback CEM, open-loop sequence CEM, richer
interaction feedback, two-phase feedback, and fixed-state iLQR were all run from
that real state; none produced more than `0.02 s` of upright streak, and the
feedback/iLQR trials reached the rail. A `0.50 s` rolling-window force search
traded away the upright crossing to about `0.564 rad` while reducing cart speed,
so it did not produce a capture state either. These results support optimizing
the full downstream capture value and preserving the measured rates for
analysis, without declaring a route invalid solely because it is not yet low
momentum.

Latest captureability-frontier follow-up (2026-09-11): low momentum remains a
ranking signal, but the required acceptance test is now more concrete: replay
the exact saved MuJoCo state with a fixed capture controller and measure
uninterrupted hold. A batched full-state CEM appeared to find a quiet terminal
state (`0.083 rad` maximum angle, `0.096 rad` relative-angle RMS, `2.06 rad/s`
cumulative absolute-rate RMS), but exact serial replay diverged immediately
and produced no usable handoff. This is a rejected batched false positive.
Exact serial CEM warm-started from terminal iLQR improved the replayable
terminal point to about `0.091 rad` angle, `0.105 rad` relative-angle RMS,
`0.60 rad/s` relative hinge RMS, `0.60 rad/s` cumulative absolute-rate RMS,
and `0.65 m` cart position, but fixed-state iLQR capture, LQR variants, and
receding linear MPC still held for only about `0.08-0.10 s` before failure.
Centered iLQR refinement then produced progressively quieter real states; the
best ultraquiet state was about `0.0061 rad` maximum angle, `0.0127 rad/s`
relative hinge RMS, `0.0135 rad/s` cumulative absolute-rate RMS, `0.132 m`
cart position, and `0.015 m/s` cart speed. Even this state was not accepted:
zero control, scaled LQR/MPC, a focused PPO capture run, and an exact serial
8-second open-loop CEM all failed, with the open-loop search reaching only a
`0.24 s` upright streak. Therefore low momentum is neither sufficient nor a
standalone gate. The practical gate is empirical captureability under the
declared controller and exact plant; low-momentum metrics remain valuable for
curriculum shaping, Pareto ranking, diagnostics, and explaining why a handoff
failed.

Latest capture-basin scale probe (2026-09-11): the exact ultraquiet handoff
`runs/swingup7_ilqr_ultraquiet_a200_r500_state.json` was replayed with the
linear-MPC teacher while scaling its physical cart/angle/velocity displacement
toward upright and independently scaling the applied action. Exact upright
held for the full `4.02 s` probe. At only `2%` of the measured handoff and
`2%` action authority the best streak was `0.26 s`; most larger-action trials
hit the `+/-3 m` rail, while the remaining low-authority trials fell without a
hold. This is a narrow local basin and a controller-authority diagnostic, not
a new benchmark gate and not canonical capture evidence. The next capture
curriculum must expand a measured closed-loop basin with phase/rate-aware
feedback and hard rail penalties; it should not promote a state merely because
its low-momentum telemetry is below a fixed threshold.

Latest deliberately-absurd controller probe (2026-09-11): tested the hypothesis
that the seven-link chain only needs a stupid resonant rhythm. A chirped
square-wave "metronome" CEM on the exact `+/-3 m` canonical rail reached no
upright interval and hit the rail at `0.42 s` (best observed angle `2.253 rad`).
A blind sinusoidal "swing-set driver" with only cart PD centering found a more
organized trajectory: the search saw a `1.35 rad` angle crossing, while the
saved best-score replay reached `1.612 rad`, `1.825 rad/s` hinge RMS, and the
rail at `3.34 s`; it still produced `0.000 s` hold. Giving that same blind
family a deliberately absurd `+/-10 m` runway did not help: the best replay
walked to `9.15 m`, reached only `1.776 rad`, and hit the boundary without an
upright interval. Therefore a longer rail is a useful discovery diagnostic but
not a solution by itself, and a blind resonant driver is rejected as the
seven-link swing expert. A follow-up exact serial tail CEM aimed at the measured
quiet target from the three-second iLQR prefix ended at `1.068 rad`, `2.664
rad/s` absolute-rate RMS, and `1.104 m` cart position, so it did not repair the
handoff either. Artifacts are `runs/swingup7_resonant_bangbang_cem.json`,
`runs/swingup7_swing_set_driver_cem.json`,
`runs/swingup7_swing_set_driver_rail10_cem.json`, and
`runs/swingup7_tail_target_cem_start3_scale001.json`.

Latest explicit morphology-gradient screen (2026-09-11): the exact serial
energy/modal CEM was evaluated from the true hanging state at morphology
progress values `0.00`, `0.25`, `0.50`, `0.75`, and `1.00` on a fixed `+/-12 m`
development rail. Best angles were `0.572`, `0.902`, `0.775`, `0.778`, and
`0.931 rad`; all profiles scored `0/2` upright evaluation episodes and `0 s`
sustained hold. The p0 base-heavy, base-long, damped, and frictional profile
is the best small-screen discovery seed, but the result is non-monotonic and
does not transfer capture authority. Batched nonuniform proposals were also
replayed and rejected when their saved states disagreed with serial MuJoCo.
Use the profile only as a named training condition, regenerate labels at every
homotopy step, and return to the canonical uniform plant before any gate claim.
See [`docs/experiment_protocol.md`](docs/experiment_protocol.md) and
[`docs/levers_and_pitfalls.md`](docs/levers_and_pitfalls.md).

Latest isolated morphology and live-planner follow-up (2026-09-11): the
single-axis exact energy/modal screen reached best angles of `0.849` (length
only), `1.106` (mass only), `1.177` (damping only), and `0.959` (friction only)
on the same p0 `+/-12 m` development rail; all four had `0/2` upright episodes.
A receding-horizon planner on length-only reached `0.165 rad` at nearly full
height, but with `25.95 rad/s` relative hinge RMS and `25.92 m/s` cart speed.
The coupled p0 planner reached `0.373` and `0.423 rad` in two independent
seeds, with no upright interval. A tail CEM proposal from the first crossing
was replayed through exact serial MuJoCo and failed to create a quiet handoff.
The current interpretation is that gradients improve excitation and mode
alignment, but the remaining bottleneck is a robust nonlinear capture basin.
The direct p0 hanging PPO negative control was stopped at update 84 after
`688,128` steps and its first evaluation remained `0/8` upright. P4/P5 remain
open; none of these altered-morphology or widened-rail results is canonical
evidence.

The related fine PPO continuation from the p0.020 frontier passed repeated
eight-episode maintenance gates through p0.0275, then failed five consecutive
evaluations at p0.030 and was stopped at update `506`. Because its quadratic
hanging curriculum starts at `pi * progress^2`, p0.0275 is only `0.00238 rad`
from upright; because `swingup_slow` delays morphology changes, the plant is
still the full p0 gradient there. Preserve this checkpoint as maintenance
diagnostic evidence only, not as a seven-link swing-up result.

Latest measured two-expert handoff probe (2026-09-11): the best p0 online-MPC
trace was filtered into three exact visited states by
`scripts/extract_trajectory_states.py`. They were not interpolated or
synthetically quieted; their maximum angles were `0.373`, `0.537`, and
`0.600 rad`, with cumulative absolute-rate RMS values `11.69`, `12.42`, and
`16.32 rad/s`. A fixed p=0 Torch PPO capture probe trained for 250 gated
updates and `512,000` environment steps on those states, with no residual
teacher and a `+/-12 m` rail. It ended at `0/3` success, `0/3` ever upright,
and `0 s` maximum upright streak. A serial shared linear capture CEM over the
same state set also ended at `0/3`; state-specific actors reached transient
angles as low as `0.288 rad` but never entered the upright gate. This is
component evidence only. It rejects the current high-rate p0 arrival set as a
usable capture handoff and makes the next work item explicit: construct a
closed-loop capture-basin curriculum from lower-rate measured states, then
optimize the swing expert against capture value while annealing morphology and
rail back toward the canonical uniform plant.

The first two basin-bridge attempts are also recorded. A cold-start PPO policy
ramped the same real states from `2%` to `100%` qpos/qvel scale for 300 gated
updates (`614,400` steps), but never passed the initial gate and finished at
`0/3` success. A maintenance-checkpoint initialization retained its residual
teacher at the small stages and faded it toward zero; it still returned `0/3`
at the first exact-state evaluation and was stopped at update 223 after
repeated failed gates. These runs are
`runs/swingup7_p0000_capture_basin_curriculum_ppo` and
`runs/swingup7_p0000_capture_basin_teacher_bridge_ppo`. They are not
canonical evidence and reject simple state scaling or maintenance warm-start
as sufficient capture mechanisms.

## Final Completion Audit

### Current Escalation Target: Eight Links

The 7-link result has passed the canonical control evidence gate, so the active
research target is now uniform 8-link hanging-start swing-up and hold. The
8-link campaign must use `configs/swingup8_uniform.yaml` for final evaluation
and must publish the same 20/100 evaluation pair, reset-free video, manifest,
hashes, and paper before the roadmap promotes the target to 9 links. Gradient,
long-rail, damping, friction, and morphology variants are discovery branches
only; they belong in the experiment ledger and do not count as an 8-link pass.

The same rule applies recursively: after an `n`-link frontier passes, create
the `n+1`-link campaign and repeat the full evidence bundle. There is no fixed
upper stopping point in this roadmap; the user decides when to stop the ladder.

Latest inherited-method continuation update (2026-09-12): the exact released
seven-link force route was replayed through genuine uniform-eight dynamics,
retimed, repeated for longer windup, optimized open-loop and with feedback,
and handed to capture FDDP from measured upper-neighborhood states. The best
open-loop route crossed `0.243 rad` at `4.44 s` but reached the planned
handoff at `1.122 rad` and `x=2.807 m`; the best terminal-velocity homotopy
ended at about `1.50 rad` with `1.35 rad/s` hinge RMS and still failed capture.
The new terminal velocity factors are committed in
`scripts/search_fddp_capture.py` with `1.0` defaults. This remains an
inherited-method diagnostic boundary, not an eight-link result; the next
experiment must improve the same swing-to-low-momentum route before any
canonical 8-link evidence bundle is created.

Focused capture-tail continuation update (2026-09-12): the same inherited
seven-link route was replayed through a fixed 3.50-second eight-link prefix and
given a separate exact-MuJoCo arrest tail. The best canonical-rail-biased CEM
tail reached `0.534 rad` terminal angle at `1.396 m` cart position, with
`3.54 rad/s` hinge RMS and `2.16 m` maximum cart excursion; it is a warm start,
not a capture handoff. Rate-weight homotopy, a 10-second tail, constrained
shooting, iLQR refinement, full stitched Box-FDDP, and separate capture FDDP
all failed to produce a hold. The next inherited-method experiment must score
the tail by the actual post-tail capture/LQR replay while preserving the
canonical rail and the fixed swing prefix. It must not restart broad global or
morphology search merely because this tail objective is incomplete.

Capture-value continuation update (2026-09-12): the inherited eight-link tail
was then ranked by the actual seven-link-style terminal LQR from each emitted
state. Endpoint-only CEM reduced the selected angle to `0.481 rad`, but the
same state carried `7.27 rad/s` hinge RMS, `6.33 m/s` cart velocity, and
`2.66 m` cart position; the downstream LQR hold was `0.00 s`. The multi-state
window evaluator likewise found no upright streak before its short diagnostic
budget was stopped. A 10-second exact Box-FDDP capture search from that
measured endpoint reached live `min_v=388.15`, cart `3.006 m`, and `0.00 s`
hold. A switch-time sweep from `2.0` through `4.56 s`, LQR scales `0.25` through
`2.0`, and discovery rails through `+/-12 m` also produced no upright interval.
This closes the current terminal-LQR/timing branch without an eight-link
claim. The next permitted inherited-method step is a protected capture expert:
retain a known stabilizing teacher, train it on measured eight-link internal
modes, and only then rerank the same swing tail by that expert. Unconstrained
PPO is not an acceptable capture learner; the first eight-link maintenance
probe collapsed to `0/8` at progress `0` and was stopped at update `199`.

Exact-state and protected-teacher correction (2026-09-12): the shared
`fixed_state_cfg()` helper was found to clear only the base reset-noise fields,
not curriculum `*_start`/`*_end` fields or reset scales. Eight-link protected
capture probes that used a curriculum config were therefore not exact-state
probes. The helper now forces all reset noise to zero and both qpos/qvel reset
scales to one, with a regression test in `tests/test_fddp.py`. After the fix,
the same local two-expert method was rerun from exact `0.005 rad` internal-mode
states. Box-FDDP produced only `0.10-0.12 s` upright transients before the
canonical rail, short capture-sequence CEM produced `0.06 s`, and exact-model
feedback MPC produced `0.04 s`; none held. A release-seeded ghost-profile
continuation using the frozen seven-link route, exact replay-built states, and a
`+/-9 m` discovery rail reached `min_v=229.73` but still exited the rail with no
handoff. A full-authority residual PPO probe also failed to preserve even the
exact p0 maintenance gate after update 75 and was stopped at update 125. These
are corrected negative controls, not eight-link evidence. The next permitted
step remains the same architecture: produce a nonlinear capture teacher that
can recover the first measured internal-mode envelope, then optimize the
existing eight-link swing tail against that teacher.

The richer capture continuation was also run after fixing its hard-coded
seven-link seed dimension. A 66-feature angle/rate/cart-interaction actor seeded
from the bounded eight-link actor reached `0/1` success, a best `0.20 s` upright
streak, and a `3.079 m` rail exit. Feature interactions alone do not provide the
missing capture basin.

## Inherited Eight-Link Method Focus Boundary (2026-09-12)

The active eight-link campaign is now explicitly constrained to the method that
passed seven links: hanging-equilibrium conditioning, an exact target-chain
Box-FDDP swing expert, a measured-state capture expert, and terminal hold. The
recent experiments were continuation checks within that architecture, not a
new controller-family search.

| Attempt | Exact setup | Result | Decision |
|---|---|---|---|
| Shared internal-mode capture actor | One bounded nonlinear feedback actor, optimized over four exact `0.005 rad` single-link perturbations for 8 s | `0/4` holds; minimum upright streak `0.22-0.24 s`; every run exited the `+/-9 m` discovery rail | One static actor does not recover the eight-link internal-mode envelope |
| Padded seven-link release replay | Released seven-link route padded to eight and evaluated on exact uniform 8 links after the 10 s hanging LQR prelude | `0/1`; no upright event; rail at `3.031 m` after `13.42 s` | The predecessor force waveform does not reach a valid eight-link handoff |
| Inherited-route timing/amplitude screen | Seven-link force waveform only, retimed `0.60-1.80x` and amplitude-scaled `0.60-1.40x` on a `+/-12 m` diagnostic rail | No candidate reached the upright gate; best late composite still had about `3.04 rad` maximum angle | Simple retiming/amplitude changes are exhausted |
| Exact inherited-route terminal iLQR | Padded route, uniform 8 links, 6.56 s, upright terminal target, canonical rail penalty | Terminal angle `2.902 rad`, hinge RMS `22.70 rad/s`, cart `3.059 m`; no handoff | Direct local terminal refinement cannot repair the inherited route |

These results do not change the completion contract and do not justify an
eight-link claim. The next run remains narrow: first build a held-out
nonlinear capture teacher from real eight-link handoff states, then rerank or
refine the existing inherited swing tail against that teacher. Do not restart
the entire global policy or morphology search unless this inherited chain is
formally falsified by that capture-valued continuation.

### Current Canonical Seven-Link Result

The final canonical control evaluation now passes independently of the earlier
calibration phases. The settled-launch hybrid chain uses a 10-second
hanging-equilibrium LQR conditioning phase, the 228-step Box-FDDP swing route,
and terminal LQR capture. It passes `20/20` and `100/100` canonical noisy
hanging-start episodes. The evidence is:

- `runs/swingup7_uniform/eval_swingup7_20.json`
- `runs/swingup7_uniform/eval_swingup7_100.json`
- `runs/swingup7_uniform/seven_link_swingup_manifest.json`
- `runs/swingup7_uniform/seven_link_swingup_success.mp4`
- `runs/swingup7_uniform/seven_link_swingup_success.video.json`

This is a passed canonical seven-link control and release gate. The earlier
six-link calibration phases below remain incomplete research history and do
not weaken or substitute for the stronger integrated seven-link evidence.

- [x] P0 passed and canonical benchmark frozen.
- [ ] Global discovery feasibility gate passed on exact uniform six-link MuJoCo.
- [ ] P1 passed on the frozen synthetic 6-link handoff envelope.
- [ ] P2 passed from uniform hanging-start 6-link rollouts and real handoffs.
- [ ] P3 passed with integrated reset-free 6-link reproduction evidence.
- [x] P4 superseded by stronger integrated canonical evidence.
- [x] P5 superseded by stronger integrated canonical evidence.
- [x] P6 20-episode evaluation passed.
- [x] P6 100-episode evaluation passed.
- [x] P6 reset-free 30-second video and metadata passed.
- [x] All configs, XML, controller, manifests, and evidence hashes verified.
- [x] Final evidence points to a clean tracked git commit.
- [x] Fresh-clone verification completed.
- [x] P7 public comparison and limitations documentation completed.

Only after every required checkbox is verified should the project goal be marked complete.
