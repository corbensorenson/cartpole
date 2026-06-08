# Six-Link Evidence Runbook

Use this after `make smoke` and the debug run pass.

There are two tracks:

- `make lqr6`: analytic finite-difference LQR checkpoint for a narrow near-upright basin.
- `make train6 && make uniform6`: PPO curriculum track from the handoff packet.

## Training path

```bash
make lqr6
make eval6
make render6
```

The target artifacts are:

```text
runs/uniform6_finetune/checkpoints/best.safetensors
runs/uniform6_finetune/eval_uniform6.json
runs/uniform6_finetune/six_link_uniform_success.mp4
runs/uniform6_finetune/six_link_uniform_success.video.json
```

## Minimum near-upright claim gate

Do not claim even near-upright six-link stabilization unless all of these are true:

- `eval_uniform6.json` reports `success_rate >= 0.80` over at least 20 deterministic episodes.
- `eval_uniform6.json` includes the checkpoint SHA-256, resolved config SHA-256, runtime versions, and git state.
- `six_link_uniform_success.video.json` reports `reset_count == 0`.
- `six_link_uniform_success.video.json` has no event with `"terminated": true`.
- The MP4 visually shows the cart staying within the rail and all six links staying upright for the requested duration.
- The claim text states the initialization scope. The analytic LQR checkpoint is only evidence for `init_angle_noise: 0.0003` and `init_vel_noise: 0.00009`, not the handoff packet's harder PPO default of `init_angle_noise: 0.040`.

For a stronger near-upright claim, rerun eval with `--episodes 100` and use a clean git commit so the evidence JSON points at an immutable code revision.

## Swing-up gap

If the target benchmark starts with links collapsed or hanging below the cart, use the swing-up path:

```bash
make swingup-debug
make swingup6
make eval-swingup6
make render-swingup6
```

The expected final artifacts are:

```text
runs/swingup6_uniform/checkpoints/best.safetensors
runs/swingup6_uniform/eval_swingup6.json
runs/swingup6_uniform/six_link_swingup_success.mp4
runs/swingup6_uniform/six_link_swingup_success.video.json
```

Do not claim swing-up success unless:

- `eval_swingup6.json` reports `success_rate >= 0.80` over at least 20 deterministic held-out episodes,
- the stronger 100-episode eval reports `success_rate >= 0.90`,
- `best.safetensors` was selected by success/capture metrics, not by raw shaped return alone,
- the video starts from the hanging/collapsed state,
- the video metadata reports `reset_count == 0`,
- the only done event is successful truncation at the episode limit,
- per-episode metrics include time-to-upright/capture and sustained-upright duration.

The current swing-up config is explicit, but the task is not solved yet. If matching an external benchmark, verify:

- initial joint angles near the downward/collapsed configuration,
- reward terms for energy injection and upright capture, with no survival-only shortcut,
- termination and rail limits matching the target,
- action space matching the target, especially if it is discrete,
- evaluation videos with no resets from that initial-state distribution.
