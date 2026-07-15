# Gradient n-Link Cart-Pole

## Active Project Goal

The final project target is now a reproducible **7-link hanging-start swing-up and stabilization result**. Six-link end-to-end reproduction is a mandatory calibration gate because six links has already been solved publicly; it is no longer the final research claim.

The authoritative phase gates, benchmark contract, artifacts, and completion audit are in [`ROADMAP.md`](ROADMAP.md). The project is not complete until every required roadmap phase through public fresh-clone reproduction passes.

## Status

This repository currently contains a reproducible **near-upright stabilization** result for a uniform 6-link MuJoCo cart-pole. It does **not** solve the harder swing-up problem from a collapsed or hanging-down initial state.

That distinction matters. If Yacine's benchmark starts with the links below the cart and requires swinging them upright, this repo should be treated as a baseline/tooling repo, not as a benchmark beat.

Roadmap Phase 0 is complete. The canonical seven-link config and verifier can be checked with:

```bash
make roadmap-p0
```

This regenerates the target MuJoCo XML, records its SHA-256, checks the hanging geometry and reset distribution in MuJoCo, and runs rejection tests for easier plants and incomplete evidence. It does not claim the seven-link control problem is solved.

Phase 1 now has a frozen six-link synthetic capture benchmark:

```bash
make generate-capture-envelope
make capture-envelope6
make eval-capture-envelope6
```

[`benchmarks/p1_capture_envelope.yaml`](benchmarks/p1_capture_envelope.yaml) fixes the uniform plant, control frequency, 15-second episode, success contract, gate thresholds, and seeded 20,000/2,000/1,000 train/validation/test state splits. The evaluator rejects altered rails, force limits, dynamics, or easier hold criteria and evaluates every held-out state exactly once. PPO curriculum gates replay a fixed seeded subset from the independent validation split and record the exact indices in checkpoint metadata. The current analytic LQR baseline fails the final gate with `0/1000` successes, `0.04 s` median maximum upright hold, and `1000` rail hits. A component-wise curriculum has passed the fixed validation gate at progress `0.06`, where reset positions are scaled by `p^2` and velocities by `p^3`; it stalls at `p=0.0625`. This is still a very narrow fraction of the physical envelope, so P1 is not passed.

The current boundary controller can be labelled and converted into a conservative empirical capture-funnel model with:

```bash
make fit-capture-funnel
make search-swingup-funnel
```

The first command deterministically regenerates the scale-`1.30` LQR checkpoint, evaluates 1,024 training states at `p=0.0625`, and fits a second-order logistic membership model. Its hash-determined training-only holdout reports `0.990` ROC AUC; the calibrated acceptance threshold has `0.951` precision and `1.000` recall. The model rejects states outside its measured coordinate domain and can replace the angle-only target in chain trajectory search. This is a policy-specific training objective at a very narrow curriculum stage, not held-out P1 evidence.

State-specific nonlinear planning can recover part of the boundary that fixed LQR misses:

```bash
make search-capture-target-teacher CAPTURE_TARGET_TEACHER_STATE=735
make build-capture-target-teachers
make eval-adaptive-capture-boundary
make analyze-capture-modal
make sweep-capture-modal-feedback
make refine-adaptive-capture-boundary
make analyze-refined-capture-modal
make eval-transfer-capture-boundary
make search-capture-hybrid
make eval-feedback-mpc-state
make refine-feedback-mpc-boundary
```

The planner searches a short cart-target schedule with deterministic CEM in exact MuJoCo, then executes the selected schedule in one uninterrupted rollout with LQR state feedback. On 256 fixed validation states at `p=0.065`, fixed LQR succeeds on `215/256`; the first planning budget recovers 18 failures and deterministic budget escalation recovers 3 more, producing `236/256 = 92.19%` success with a `13.94 s` median upright hold and no successful rail hits. This advances the accepted development frontier from `p=0.06` to `p=0.065`.

Dimensionless real-Schur analysis avoids the ill-conditioned raw eigenvector basis and identifies a slow weakly actuated block, dominated by absolute angles 4–6, as the strongest separator of planner recovery from failure (`0.9908 +/- 0.0080j`, refined-tail AUC `0.869`). Static feedback changes to that block destabilize the linear closed loop, and a seeded second search pass recovers `0/20`; the remaining tail needs a richer transient trajectory parameterization. These are compute-heavy online model-based diagnostics at a narrow partial envelope, not a learned policy and not P1: the formal gate still requires 1,000 frozen test states at `p=1.0`.

A dimensionless closed-loop Lyapunov diagnostic was then used to search a combined cart-target and bounded residual schedule. On representative unresolved state 475, the planned 3-second hybrid transient improved maximum upright hold from `0.50 s` to `0.90 s` but still hit the rail, while a forced 1-second handoff reached only `0.60 s`. The open-loop residual therefore did not provide the state-feedback correction needed by the slow mode. At the next curriculum increment, `p=0.0675`, fixed LQR transferred `210/256`; a standard target-planning pass reached `225/256`, and a 1,501-second high-budget escalation recovered only one more state for `226/256 = 88.28%`. Because that misses the 90% development threshold, `p=0.065` remains the accepted frontier and the scheduled-target controller family is considered saturated at `p=0.0675`.

Exact-model receding-horizon feedback MPC closes part of that gap. It searches short cart-target and bounded-residual schedules in cloned MuJoCo state while recomputing LQR feedback inside every candidate rollout, uses a dimensionless closed-loop Lyapunov terminal objective, and latches to plain LQR after entering a conservative measured funnel. On the 20 remaining `p=0.065` failures it recovers 6, raising the fixed cohort to `242/256 = 94.53%`. With the same frozen settings at `p=0.0675`, it recovers 8 of 30 failures and reaches `234/256 = 91.41%`, a `13.92 s` median upright hold, and zero successful rail hits. This advances the accepted development frontier to `p=0.0675`; it remains online model-based partial validation evidence, not P1.

At `p=0.0700`, transferred target controllers plus standard target replanning reach `217/256`, and the standard feedback-MPC stage reaches `227/256`. A deterministic escalation applied to all 29 remaining failures extends MPC authority to 3 seconds, tightens the handoff cart bound to `1.5 m`, and uses 4 CEM iterations with population 128. It recovers 6 more states for `233/256 = 91.02%`, a `13.90 s` median hold, and zero successful rail hits. The escalation batch takes 1,668 seconds on the recorded Apple Silicon runtime. This advances the accepted development frontier to `p=0.0700`, but remains partial validation evidence.

At the full `p=1.0` envelope, nominal LQR is heavily saturated: representative validation state 674 requests about `-78` normalized action before clipping. Feedback MPC now separates its planning feedback scale from the post-handoff LQR scale, so zero or low-gain planning can optimize direct bounded actions without changing the proven stabilizer. Neither direct-action nor low-gain CEM captured state 674; their best uninterrupted values remained above `1.7 million` and both eventually hit the rail.

The next roadmap methods are available through `make search-ilqr-capture`, `make refine-ilqr-capture-chain`, and `make search-multiple-shooting-capture`. Box-constrained finite-difference DDP operates in dimensionless absolute-angle coordinates, simulates every transition in exact MuJoCo, and tracks the resulting trajectory with time-varying feedback. Scalar active-set constraints improve the original rail-safe trajectory from `V=1.13M` to `V=0.431M`, but that basin remains folded and energetic. Initializing from the direct-action CEM rollout and explicitly penalizing terminal dimensionless state norm produces a qualitatively better state-674 approach: `V=30,286`, terminal cart `0.229 m`, and maximum relative-angle magnitude about `0.074 rad`.

Freezing that approach and optimizing a separate 1.5-second settling tail changes the outcome. A 24-knot direct-action CEM seed followed by box-constrained DDP reaches planned `V=2,961.94`; the uninterrupted approach, tail, and LQR fallback then enters the conservative `V <= 1,800` funnel at `5.02 s`, finishes the 15-second episode successfully with `9.30 s` maximum upright hold, reaches `V=0.12`, and stays within `2.413 m`. Replay from the saved controls reproduces the success and handoff step. `make eval-ilqr-chain-basin` measures its feedback tube: it recovers `0/23` nearest distinct validation states, `4/32` normalized-radius-`0.005` perturbations, and none by radius `0.05`. This establishes nonlinear recoverability of one representative full-envelope state, not a reusable capture policy or P1 evidence. Sparse exact-dynamics multiple shooting is also implemented with strict feasible-iterate retention, but SciPy's equality solvers have not improved the exact warm start without violating defects.

`configs/swingup6_capture_hybrid_policy.yaml` implements the explicit two-controller architecture for learning: deterministic hysteresis gives exact LQR control inside a conservative local region and full normalized-force authority to the neural policy outside it. The environment records every entry, exit, first-entry time, and per-controller action count without resetting state. A 50-update PPO probe at `p=0.0700` improves only from `11/256` zero-policy baseline captures to `12/256`, despite reducing maximum cart excursion to `1.172 m`; PPO learns survival rather than sustained capture, so this reward-only branch is rejected. The switch remains the integration target for planner-supervised on-policy learning.

The first training-only supervisor audit also rejects scheduled-target teachers as the sole DAgger expert. Frozen LQR fails `131/512` sampled training states at `p=0.0700`, and six-generation scheduled-target search solves only `4/32` selected failures. The next supervisor must escalate from feedback MPC to DDP and retain only successful, uninterrupted labels; validation planner artifacts are not training data.

An optional Stable-Baselines3 SAC residual branch is available through `make setup-sb3` and `make capture-sac-boundary`. Scaled capture coordinates fixed an observation-resolution defect, but conservative SAC only preserved the LQR baseline and weaker trust regularization eventually collapsed it. The branch is retained for reproducible ablation work; it did not advance the frontier.

Current generated split hashes:

```text
train       993d6563c4eccd23ae26aa3f5bd0064751790ad0bcf5376e2a60205d88462837
validation  aa32b8989d45223fca7e19e9d5eeb7f84da55c6dd5faa9a8508212e21a479de0
test        5d82bbd33b1cec6847fb26de836284be8623d00d47dd25c335130d68dd85c686
```

Current achieved artifact set:

```text
runs/uniform6_finetune/checkpoints/best.safetensors
runs/uniform6_finetune/eval_uniform6.json
runs/uniform6_finetune/eval_uniform6_100.json
runs/uniform6_finetune/six_link_uniform_success.mp4
runs/uniform6_finetune/six_link_uniform_success.video.json
```

The achieved config is:

```text
configs/uniform6_near_upright_lqr.yaml
```

It uses a finite-difference MuJoCo linearization plus discrete LQR to produce a deterministic MLX checkpoint for a narrow near-upright basin:

```text
init_angle_noise: 0.0003
init_vel_noise: 0.00009
```

Verified locally:

```text
20 deterministic episodes: success_rate = 1.0
100 deterministic episodes: success_rate = 1.0
30-second MP4: reset_count = 0, no termination event
```

## Research Direction

The original handoff hypothesis is still included:

> A heavily gradiented n-link cart-pole can be easier to learn than the uniform benchmark. Train on the easier morphology, then anneal length/mass/damping gradients back to the final uniform morphology.

This packet contains a **native Apple Silicon path** using:

- **MuJoCo** for physics.
- **MLX** for PPO policy/value training on Apple Silicon.
- A custom vectorized environment layer for throughput.
- Optional notes for **PufferLib 4.0** exploration.

## Important Scope Note

The current solved target is **upright stabilization from a very small near-upright perturbation**, not full swing-up from collapsed/hanging initial conditions. If Yacine's target starts below the cart, uses discrete actions, or has different rail/force/reward/termination details, those must be implemented and solved before claiming a direct comparison.

The code is written for `n` links. Six links is the calibration target; seven links is the final roadmap target.

## Six-Link Calibration Target

The required six-link end-to-end calibration is tracked separately from the near-upright baseline:

```text
configs/swingup6_uniform.yaml
runs/swingup6_uniform/checkpoints/best.safetensors
runs/swingup6_uniform/eval_swingup6.json
runs/swingup6_uniform/six_link_swingup_success.mp4
```

Environment spec:

| Setting | Value |
|---|---:|
| Links | 6 |
| Initial mode | `hanging_curriculum`, evaluated at `progress=1.0` |
| Initial relative angles at eval | `[pi, 0, 0, 0, 0, 0] + Normal(0, 0.05)` |
| Initial velocities | `Normal(0, 0.05)` |
| Action space | continuous normalized force in `[-1, 1]` |
| Force limit | `80 N` |
| Rail limit | `+/-3.0 m` |
| Episode length | `30 s` |
| Angle failure termination | disabled (`terminate_abs_angle: null`) |
| Failure termination | rail hit or numerical failure |
| Success | episode completes with at least `5 s` sustained upright |
| Upright threshold | max absolute link angle `< 0.15 rad` |
| Reward gate | no survival bonus; shaped return prioritizes upright angle, capture, and sustained upright |

During training, `best.safetensors` is selected by success/capture metrics before return: success rate, ever-upright rate, upright streak length, then shaped return. A long episode that never reaches upright should not become the best swing-up checkpoint. For gated swing pretraining, `frontier.safetensors` is also written when a curriculum stage passes; use that checkpoint for handoff export and uniform fine-tuning because it represents the farthest mastered swing stage, not merely the easiest high-scoring stage.

Run path:

```bash
make swingup-debug
make swingup6
make eval-swingup6
make render-swingup6
```

At the moment, the swing-up target is infrastructure-ready but not solved. The near-upright evidence in `runs/uniform6_finetune/` should not be used as a substitute.

## Current Two-Expert Curriculum

The active search direction is a two-expert setup:

1. a low-momentum swing expert that gets the hanging chain to the top with low hinge velocity, low cart velocity, and rail margin,
2. one capture/stabilize expert that takes over from those low-momentum top states and holds the uniform chain upright.

The gradient training-wheel path is:

```bash
make swingup6-gradient-low-momentum
make export-policy-handoff-states
make capture-policy-handoff-stage
make capture-policy-handoff-curriculum
make eval-capture-policy-handoff-stage
make eval-capture-policy-handoff-curriculum
make eval-mpc-policy-handoff-stage
make swingup6-uniform-low-momentum
make eval-swingup6-low-momentum
```

`configs/swingup6_gradient_low_momentum.yaml` starts with easier length, mass, damping, hinge friction-loss, and longer-rail gradients, then anneals them away. `configs/swingup6_uniform_low_momentum_finetune.yaml` removes those training wheels and evaluates at the final hanging-start task with the real `+/-3 m` rail.

`export-policy-handoff-states` is the boundary between the two experts: it replays the learned swing frontier policy and writes actual MuJoCo `qpos/qvel` low-momentum handoff states for the capture/stabilize expert. `capture-policy-handoff-stage` trains directly from that saved state file. `capture-policy-handoff-curriculum` is the stricter handoff curriculum: it pins the physical plant at the swing expert's `CAPTURE_STAGE_PROGRESS`, starts from the saved real positions with saved velocities scaled to zero, then restores the real saved velocities and reset noise through training progress. `eval-capture-policy-handoff-stage` and `eval-capture-policy-handoff-curriculum` run held-out deterministic capture eval, and `eval-mpc-policy-handoff-stage` tests whether the saved state is catchable by a receding-horizon diagnostic.

For an ad-hoc continuation run, point the handoff/export targets at that run without editing the Makefile:

```bash
make export-policy-handoff-states \
  SWING_HANDOFF_RUN=runs/swingup6_gradient_low_momentum_centered_gate025_from350_retry_240

make capture-policy-handoff-shaped6
```

The exporter uses `checkpoints/frontier.safetensors` by default and inherits the swing reward's centered low-momentum cart-position gate (`low_momentum_max_cart_abs`) unless `--max-cart-abs` is explicitly supplied.

Current best learned swing handoff artifact: `runs/swingup6_policy_handoff/swing_handoff_states_frontier03875_lmom_refine128.json`, exported from `runs/swingup6_gradient_low_momentum_lmom_refine_from3875_probe_120`. It contains only `2` low-momentum states, but the best one is near upright with `x = 0.171 m`, `cart_velocity = 0.396 m/s`, `max_abs_angle = 0.0849 rad`, and `hinge_velocity_rms = 0.100`. This is still curriculum-stage evidence at progress `0.3875`, not final uniform swing-up evidence.

For deterministic real-uniform trajectory probes, the search/export path can also target low-momentum handoff states before capture training:

```bash
make search-swingup-low-momentum
make search-swingup-sustain
make search-swingup-action-low-momentum
make search-swingup-action-cold
make search-linear-policy
make export-low-momentum-swingup-states
make search-capture-sequence
make eval-mpc-capture
make capture-wide6
make eval-capture-wide6
make capture-low-momentum-velocity-curriculum
make capture-state-curriculum
make capture-lqr-residual-velocity-curriculum
make capture-state-curriculum-lqr-init
```

`export-low-momentum-swingup-states` is the first-expert handoff file for the real-uniform path. It saves replayable MuJoCo `qpos/qvel` states from the swing trajectory, and the capture targets consume that same file through `SWING_TRAJECTORY_STATES_OUT`. `search-swingup-action-low-momentum` searches direct normalized-force knots as a richer first-expert handoff generator when fixed cart-position knots stall. `capture-wide6` trains a final-uniform robust near-upright capture basin from scheduled synthetic upright perturbations. `capture-low-momentum-velocity-curriculum` trains from real positions while annealing saved handoff velocity from zero back to full velocity. `capture-state-curriculum` additionally blends reset positions from exact upright to the saved handoff positions and expands from the easiest saved state to the full state list. `capture-lqr-residual-velocity-curriculum` uses the same state/velocity curriculum but applies a finite-difference LQR action bias around upright, so PPO learns a nonlinear residual catch policy instead of starting from an unaided random controller. `capture-state-curriculum-lqr-init` starts from the finite-difference LQR checkpoint and uses the same backward state-list curriculum with a linear policy. `eval-mpc-capture` is a deterministic nonlinear catchability diagnostic from one saved state; override `MPC_CAPTURE_QVEL_SCALE=0.0` to test whether the same handoff position would be easy if the swing expert arrived with no velocity.

If the exported frontier is still a gradiented curriculum stage, train the diagnostic capture expert on the same stage instead of forcing those states into the final uniform plant:

```bash
make capture-policy-handoff-stage \
  SWING_HANDOFF_RUN=runs/swingup6_gradient_low_momentum_centered_gate025_from3125_lowent_180 \
  CAPTURE_STAGE_PROGRESS=0.3375 \
  CAPTURE_STAGE_OUT=runs/swingup6_policy_handoff_capture_stage03375

make eval-capture-policy-handoff-stage \
  SWING_HANDOFF_RUN=runs/swingup6_gradient_low_momentum_centered_gate025_from3125_lowent_180 \
  SWING_HANDOFF_OUT=runs/swingup6_policy_handoff/swing_handoff_states.json \
  CAPTURE_STAGE_PROGRESS=0.3375 \
  CAPTURE_STAGE_OUT=runs/swingup6_policy_handoff_capture_stage03375
```

That target defaults to a forced `+/-3 m` capture rail and centered/low-momentum upright-streak rewards. Override `CAPTURE_STAGE_RAIL_LIMIT`, `CAPTURE_STAGE_CENTERED_MAX_CART_ABS`, `CAPTURE_STAGE_CENTERED_STREAK`, or `CAPTURE_STAGE_LOW_MOMENTUM_STREAK` to run ablations.

For a two-expert curriculum that keeps the same stage plant but restores the real handoff velocities over training:

```bash
make capture-policy-handoff-curriculum \
  SWING_HANDOFF_RUN=runs/swingup6_gradient_low_momentum_lmom_refine_from3875_probe_120 \
  SWING_HANDOFF_OUT=runs/swingup6_policy_handoff/swing_handoff_states_frontier03875_lmom_refine128.json \
  CAPTURE_STAGE_PROGRESS=0.3875 \
  CAPTURE_HANDOFF_CURRICULUM_OUT=runs/swingup6_policy_handoff_capture_curriculum03875

make eval-capture-policy-handoff-curriculum \
  SWING_HANDOFF_RUN=runs/swingup6_gradient_low_momentum_lmom_refine_from3875_probe_120 \
  SWING_HANDOFF_OUT=runs/swingup6_policy_handoff/swing_handoff_states_frontier03875_lmom_refine128.json \
  CAPTURE_STAGE_PROGRESS=0.3875 \
  CAPTURE_HANDOFF_CURRICULUM_OUT=runs/swingup6_policy_handoff_capture_curriculum03875
```

---

# 1. Mac setup

Use a native Apple Silicon shell, not Rosetta.

```bash
uname -m
# should print: arm64
```

Install Xcode command line tools if needed:

```bash
xcode-select --install
```

Then:

```bash
cd gradient_cartpole_handoff
bash scripts/setup_mac.sh
source .venv/bin/activate
```

The setup script installs:

```bash
pip install -r requirements-mac.txt
pip install -e .
```

MLX requires native Apple Silicon Python, Python >=3.10, and macOS >=14. If `pip install mlx` fails, the most common cause is accidentally running an x86/Rosetta Python.

---

# 2. Smoke test

```bash
python scripts/smoke_test.py
```

Expected output includes:

```text
obs_dim: ...
render frame: (180, 320, 3) uint8
MLX OK; default device: ...
```

Also export the generated MuJoCo XML to inspect it:

```bash
python scripts/export_xml.py \
  --config configs/gradient6_curriculum.yaml \
  --progress 0.0 \
  --out runs/gradient6_initial.xml

python scripts/export_xml.py \
  --config configs/gradient6_curriculum.yaml \
  --progress 1.0 \
  --out runs/gradient6_uniform.xml
```

Open the XML in MuJoCo viewer if desired.

---

# 3. Run the quick debug train

This checks the whole loop before committing a long run.

```bash
python scripts/train_mlx_ppo.py --config configs/debug6_fast.yaml
```

This is not expected to solve the problem. It should produce:

```text
runs/debug6_fast/train_log.csv
runs/debug6_fast/checkpoints/latest.safetensors
```

---

# 4. Generate the current near-upright LQR baseline

```bash
make lqr6
make eval6
make render6
```

This writes the required checkpoint/eval/video paths under:

```text
runs/uniform6_finetune/
```

For stronger evaluation:

```bash
python scripts/evaluate.py \
  --config configs/uniform6_near_upright_lqr.yaml \
  --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors \
  --episodes 100 \
  --progress 1.0 \
  --out runs/uniform6_finetune/eval_uniform6_100.json
```

Again: this is not a swing-up result.

# 5. Run the full 6-link gradient curriculum

```bash
python scripts/train_mlx_ppo.py --config configs/gradient6_curriculum.yaml
```

This trains with:

- 6 links
- strong mass gradient at the start
- mild length gradient
- moderate damping gradient
- gradual curriculum back to uniform
- continuous cart force action in `[-force_limit, +force_limit]`

Main output:

```text
runs/gradient6_curriculum/checkpoints/best.safetensors
runs/gradient6_curriculum/train_log.csv
```

Watch progress with:

```bash
tail -f runs/gradient6_curriculum/train_log.csv
```

---

# 6. Uniform 6-link fine-tune

After the gradient run produces a useful checkpoint, fine-tune on the fully uniform morphology:

```bash
python scripts/train_mlx_ppo.py \
  --config configs/uniform6_finetune.yaml \
  --init-checkpoint runs/gradient6_curriculum/checkpoints/best.safetensors
```

Output:

```text
runs/uniform6_finetune/checkpoints/best.safetensors
runs/uniform6_finetune/train_log.csv
```

---

# 7. Evaluate final uniform 6-link policy

```bash
python scripts/evaluate.py \
  --config configs/uniform6_finetune.yaml \
  --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors \
  --episodes 20 \
  --progress 1.0 \
  --out runs/uniform6_finetune/eval_uniform6.json
```

Suggested success bar for the first claim:

```text
success_rate >= 0.80 over 20 deterministic episodes
mean episode length close to configured max episode length
```

For a stronger claim, use 100 deterministic episodes and random seeds held out from training.

---

# 8. Render final video

```bash
python scripts/render_video.py \
  --config configs/uniform6_finetune.yaml \
  --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors \
  --out runs/uniform6_finetune/six_link_uniform_success.mp4 \
  --seconds 30 \
  --progress 1.0
```

The video script can keep rendering after resets for debugging, but evidence runs use `--fail-on-reset`. For a clean claim, inspect whether the agent actually started from the target distribution and stayed upright without a reset or failure termination.

---

# 9. Speed tuning on Mac

Before long runs:

```bash
export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
```

Start with these knobs:

| Knob | Default | Try if slow | Notes |
|---|---:|---:|---|
| `ppo.vec_backend` | `subprocess` | `serial` for debug | Subprocess has startup overhead but better CPU use. |
| `ppo.num_envs` | 48 | 16, 32, 64, 96 | Match Mac core count. More is not always faster. |
| `ppo.rollout_steps` | 256 | 128 or 512 | Larger amortizes PPO update overhead. |
| `env.frame_skip` | 4 | 2 or 5 | Higher means fewer policy calls but coarser control. |
| `env.curriculum_rebuild_every` | 25 | 50 or 100 | Rebuilding MuJoCo XML is expensive. |
| `ppo.hidden_sizes` | `[256,256]` | `[128,128]` | Use smaller net for speed tests. |

The biggest speed limit is MuJoCo CPU simulation. MLX accelerates neural net training/update work, not MuJoCo physics itself.

---

# 9. Scaling to n links

Seven links is governed by the frozen benchmark and phase gates in `ROADMAP.md`. Do not create the final task by casually overriding the six-link config: the canonical seven-link benchmark keeps total length at `3.0 m`, total link mass at `1.0 kg`, force at `+/-80 N`, and rail at `+/-3 m`.

Notes:

- Changing `n_links` changes observation dimension, so checkpoints are not directly reusable across `n`.
- If total length is fixed and `n` increases, links get shorter and the fastest unstable mode gets harder.
- Longer links, higher force, or wider rails may be used as labeled training curricula, but they are not final evidence.
- Complete the integrated six-link calibration gate before expensive seven-link sweeps.
- Run the linear sweep first to find plausible gradients.

---

# 10. Linear sweep before expensive RL

```bash
python scripts/linear_sweep.py --config configs/sweep_n.yaml
```

This writes:

```text
runs/linear_sweep_n/linear_sweep.csv
```

Use it to identify candidate `alpha_length` and `alpha_mass` before committing RL tokens/time.

The simplified linear model is not exact MuJoCo. Its purpose is fast ranking, not final proof.

---

# 11. What to do if it fails

## Falls immediately

Try:

```bash
--override env.init_angle_noise=0.015
--override env.init_vel_noise=0.005
--override morphology.start.alpha_mass=5.0
--override morphology.start.alpha_length=0.2
--override env.force_limit=120.0
```

## Learns gradiented plant but not uniform

Try keeping mass gradient longer:

- Edit `src/gcartpole/morphology.py`, `mass_last` schedule.
- Move mass anneal start from `0.45` to `0.65`.
- Or do a second curriculum where `alpha_mass` goes from `1.5` to `0.0` slowly.

## Too slow

Try:

```bash
--override ppo.num_envs=24
--override ppo.hidden_sizes='[128,128]'
--override env.curriculum_rebuild_every=100
```

## Policy saturates force constantly

Try:

```bash
--override env.reward.control=0.001
--override ppo.entropy_coef=0.004
```

## It balances but uses too much rail

Try:

```bash
--override env.reward.cart_pos=0.25
--override env.reward.cart_vel=0.02
```

---

# 12. Optional PufferLib 4 path

PufferLib 4.0 exists on the `4.0` GitHub branch and its `pyproject.toml` identifies version `4.0.0`. Install optionally with:

```bash
bash scripts/setup_mac.sh --with-puffer
```

Do not block the MVP on PufferLib. PufferLib's biggest advertised speed path is CUDA-first; the Mac path here uses MLX for neural nets and MuJoCo for CPU physics. The right future PufferLib integration is likely:

1. Keep this Gymnasium-compatible env.
2. Wrap it through PufferLib's current emulation/vectorization APIs.
3. Compare wall-clock steps/sec and success rate against the included MLX PPO.
4. If this project becomes serious, write a custom C/PufferLib env or a vectorized analytic simulator.

See `docs/pufferlib4_notes.md`.

---

# 13. Definition of done

`ROADMAP.md` is the only project-level definition of done. In summary, completion requires integrated hanging-start six-link calibration, the canonical seven-link 20- and 100-episode success gates, published weights and hashes, a reset-free 30-second video, and fresh-clone public reproduction. A smoke test, checkpoint, nonzero success rate, curriculum-stage result, or near-upright video is progress but not completion.
