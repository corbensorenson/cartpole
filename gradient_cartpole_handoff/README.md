# Gradient n-Link Cart-Pole Handoff Packet

## Mission

Produce **weights** and a **video** of a working **6-link MuJoCo cart-pole**, using a gradient-to-uniform curriculum that also scales to arbitrary `n` links.

The first target is:

```text
runs/uniform6_finetune/checkpoints/best.safetensors
runs/uniform6_finetune/six_link_uniform_success.mp4
runs/uniform6_finetune/eval_uniform6.json
```

The experimental hypothesis is:

> A heavily gradiented n-link cart-pole can be easier to learn than the uniform benchmark. Train on the easier morphology, then anneal length/mass/damping gradients back to the final uniform morphology.

This packet contains a **native Apple Silicon path** using:

- **MuJoCo** for physics.
- **MLX** for PPO policy/value training on Apple Silicon.
- A custom vectorized environment layer for throughput.
- Optional notes for **PufferLib 4.0** exploration.

## Important scope note

The current default target is **upright stabilization**, not guaranteed full swing-up from hanging-down initial conditions. It initializes near the upright equilibrium with small noise and trains the agent to keep all 6 links upright. If Yacine's exact target includes swing-up, discrete actions, a different reward, different rail length, or different force limits, match those before claiming an exact reproduction.

The code is written for `n` links. Six links is just the first target.

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

# 4. Run the full 6-link gradient curriculum

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

# 5. Uniform 6-link fine-tune

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

# 6. Evaluate final uniform 6-link policy

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

# 7. Render final video

```bash
python scripts/render_video.py \
  --config configs/uniform6_finetune.yaml \
  --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors \
  --out runs/uniform6_finetune/six_link_uniform_success.mp4 \
  --seconds 30 \
  --progress 1.0
```

The video script resets if the episode terminates so it always produces a continuous MP4. For a clean claim, inspect whether the agent actually stayed upright for the full run rather than falling and resetting.

---

# 8. Speed tuning on Mac

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
