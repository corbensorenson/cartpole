# Handoff Packet: Gradient-to-Uniform n-Link Cart-Pole

> Historical scope note: this packet records the original near-upright six-link starting point. The active project goal and authoritative completion gates are now in `ROADMAP.md`: reproduce hanging-start six links as calibration, then solve and publish the canonical seven-link benchmark.

## Objective

Use this Mac to produce near-upright six-link stabilization artifacts:

1. A trained 6-link uniform cart-pole checkpoint:

```text
runs/uniform6_finetune/checkpoints/best.safetensors
```

2. A video:

```text
runs/uniform6_finetune/six_link_uniform_success.mp4
```

The code supports arbitrary `n`; solve 6 first, then scale. This packet does not currently solve swing-up from a collapsed/hanging initial state.

## Current local baseline

The checked-in near-upright LQR path is:

```bash
make lqr6
make eval6
make render6
```

It is scoped to:

```text
configs/uniform6_near_upright_lqr.yaml
init_angle_noise: 0.0003
init_vel_noise: 0.00009
```

Do not describe this as a Yacine benchmark beat if Yacine starts below the cart and requires swing-up.

## Main idea

Train on an easier gradiented morphology, then anneal toward the uniform 6-link target:

```text
base-heavy / mildly base-long / damped  ->  uniform mass / uniform length / target damping
```

The default bet is that **mass gradient is the strongest curriculum knob**, with length gradient mild and damping treated as a temporary crutch.

## First commands

```bash
cd gradient_cartpole_handoff
bash scripts/setup_mac.sh
source .venv/bin/activate
python scripts/smoke_test.py
```

## Debug run

```bash
python scripts/train_mlx_ppo.py --config configs/debug6_fast.yaml
```

## Full 6-link gradient curriculum

```bash
python scripts/train_mlx_ppo.py --config configs/gradient6_curriculum.yaml
```

## Uniform 6-link fine-tune

```bash
python scripts/train_mlx_ppo.py \
  --config configs/uniform6_finetune.yaml \
  --init-checkpoint runs/gradient6_curriculum/checkpoints/best.safetensors
```

## Evaluate

```bash
python scripts/evaluate.py \
  --config configs/uniform6_finetune.yaml \
  --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors \
  --episodes 20 \
  --progress 1.0 \
  --out runs/uniform6_finetune/eval_uniform6.json
```

## Render video

```bash
python scripts/render_video.py \
  --config configs/uniform6_finetune.yaml \
  --checkpoint runs/uniform6_finetune/checkpoints/best.safetensors \
  --out runs/uniform6_finetune/six_link_uniform_success.mp4 \
  --seconds 30 \
  --progress 1.0
```

## Definition of done

Minimum:

- `smoke_test.py` passes.
- `best.safetensors` exists for uniform 6-link fine-tune.
- `eval_uniform6.json` exists.
- `six_link_uniform_success.mp4` exists.

Strong:

- `success_rate >= 0.80` over 20 deterministic uniform episodes.
- 30-second video shows no reset, no rail hit, all six links upright.

## Fairness note

This packet defaults to continuous-action near-upright stabilization. To claim an exact Yacine benchmark beat, match his precise action space, force bins, reward, initial-state distribution, rail, force limits, termination, MuJoCo version, and especially whether the initial state is collapsed/hanging and requires swing-up.
