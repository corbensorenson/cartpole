# Troubleshooting

## `pip install mlx` fails

Check:

```bash
uname -m
python -c "import platform; print(platform.processor(), platform.machine())"
```

Expected native Apple Silicon output is `arm64` / `arm`-like. If it says `i386` or `x86_64`, you are probably in a Rosetta shell or using non-native Python.

## MuJoCo render fails

Try installing ffmpeg support:

```bash
pip install imageio-ffmpeg
```

For headless rendering issues, first verify:

```bash
python scripts/smoke_test.py
```

## Subprocess vector env hangs

Use serial to debug:

```bash
python scripts/train_mlx_ppo.py --config configs/debug6_fast.yaml --override ppo.vec_backend=serial
```

If serial works and subprocess hangs, reduce workers:

```bash
--override ppo.num_envs=8
```

## Training diverges / NaNs

Try:

```bash
--override ppo.learning_rate=0.0001
--override ppo.action_std_init=0.35
--override env.force_limit=60
--override ppo.entropy_coef=0.002
```

## It never survives long enough to learn

Start easier:

```bash
--override env.episode_seconds=8
--override env.init_angle_noise=0.015
--override morphology.start.alpha_mass=5.0
--override morphology.start.alpha_damping=3.0
--override morphology.start.total_damping=0.12
```

Then lengthen episodes and reduce crutches.

