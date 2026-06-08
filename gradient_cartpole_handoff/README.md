# Gradient n-Link Cart-Pole

## Status

This repository currently contains a reproducible **near-upright stabilization** result for a uniform 6-link MuJoCo cart-pole. It does **not** solve the harder swing-up problem from a collapsed or hanging-down initial state.

That distinction matters. If Yacine's benchmark starts with the links below the cart and requires swinging them upright, this repo should be treated as a baseline/tooling repo, not as a benchmark beat.

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

The code is written for `n` links. Six links is just the first target.

## Swing-Up Target

The real target for comparison is now tracked separately from the near-upright baseline:

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

After six works, try seven:

```bash
python scripts/train_mlx_ppo.py \
  --config configs/gradient6_curriculum.yaml \
  --override experiment.name=gradient7_curriculum \
  --override experiment.out_dir=runs/gradient7_curriculum \
  --override env.n_links=7 \
  --override env.total_length=3.5
```

Notes:

- Changing `n_links` changes observation dimension, so checkpoints are not directly reusable across `n`.
- If total length is fixed and `n` increases, links get shorter and the fastest unstable mode gets harder.
- If you increase `n`, consider increasing `total_length` or force limit.
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

Minimum:

- `scripts/smoke_test.py` passes.
- `gradient6_curriculum` produces a checkpoint.
- `uniform6_finetune` produces `best.safetensors`.
- Deterministic `progress=1.0` eval has nonzero success.
- Video is rendered.

Strong:

- `success_rate >= 0.80` over 20 deterministic uniform episodes.
- A 30-second MP4 shows all six links upright without reset.
- Same command works for `n=7` with adjusted config.

Stronger / public claim:

- Match Yacine's exact environment spec.
- Match action space, rail, force limit, reward, initial distribution, and termination.
- Report seeds, wall-clock time, total environment steps, and checkpoint hash.
