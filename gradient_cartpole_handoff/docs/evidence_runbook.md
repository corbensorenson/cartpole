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

## Minimum claim gate

Do not claim the six-link barrier is overcome unless all of these are true:

- `eval_uniform6.json` reports `success_rate >= 0.80` over at least 20 deterministic episodes.
- `eval_uniform6.json` includes the checkpoint SHA-256, resolved config SHA-256, runtime versions, and git state.
- `six_link_uniform_success.video.json` reports `reset_count == 0`.
- `six_link_uniform_success.video.json` has no event with `"terminated": true`.
- The MP4 visually shows the cart staying within the rail and all six links staying upright for the requested duration.
- The claim text states the initialization scope. The analytic LQR checkpoint is only evidence for `init_angle_noise: 0.0003` and `init_vel_noise: 0.00009`, not the handoff packet's harder PPO default of `init_angle_noise: 0.040`.

For a stronger claim, rerun eval with `--episodes 100` and use a clean git commit so the evidence JSON points at an immutable code revision.
