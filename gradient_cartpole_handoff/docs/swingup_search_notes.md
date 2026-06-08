# Swing-Up Search Notes

## Current Goal And Architecture

Working goal: solve the real six-link swing-up benchmark from the hanging/collapsed start, then stabilize the uniform chain upright. Starting from already-upright links is a separate near-upright control baseline and is not the target problem.

The current architecture is a two-expert system, not three independent experts:

1. low-momentum swing expert: swing the hanging six-link chain to the top while explicitly minimizing hinge velocity, cart velocity, and rail offset at the upright crossing,
2. capture/stabilize expert: take over from those low-momentum upright states and hold the chain inside the `0.15 rad` success threshold for `5 s`.

The earlier three-stage swing/capture/stabilize probes are still useful diagnostics, but the learned third-expert probe did not improve the reset-free chain. The real bottleneck is upstream: the handoff arrives upright too briefly and with too much momentum.

The handoff packet's original gradient idea is now represented in:

```text
configs/swingup6_gradient_low_momentum.yaml
configs/swingup6_uniform_low_momentum_finetune.yaml
```

The first config starts with a base-heavy, mildly base-long, high-damping morphology and anneals length, mass, and damping to the uniform target using `morphology.schedule_mode: swingup_slow`. It also uses `env.hanging_curriculum_power: 3.0`, so the start angle becomes hanging more slowly while morphology training wheels are being removed. Its reward includes `capture_quality`, low-velocity upright bonuses, and rail-margin penalties so checkpoint selection can prefer low-momentum top handoffs over high-speed upright flickers. This pretraining config uses `ppo.eval_progress: current`, so `best.safetensors` is selected at the active curriculum stage rather than only at the final hanging-start task. The second config removes the morphology training wheels and fine-tunes on the real uniform hanging-start task, where evaluation returns to final-task progress.

Run path:

```bash
make swingup6-gradient-low-momentum
make swingup6-uniform-low-momentum
make eval-swingup6-low-momentum
```

This is still not final evidence until the uniform checkpoint passes the held-out eval/video/hash gates.

A bounded `80` update probe before adding the slower hanging-start ramp showed why this matters:

```text
runs/swingup6_gradient_low_momentum_probe_80/checkpoints/best.meta.json
```

Training returns were positive on early easy morphology stages, then collapsed around progress `0.30` as the start angle became too hard. The best final-progress checkpoint was update `20` and still had `success_rate = 0.0`, `ever_upright_rate = 0.0`, and `low_momentum_upright_rate = 0.0`. That run is not solution evidence; it is a schedule diagnostic.

A bounded `40` update probe with `hanging_curriculum_power: 2.0` moved the training collapse later, around progress `0.62`, but final-progress eval still had no upright events. The checked-in config now uses a stronger `hanging_curriculum_power: 3.0` and the `swingup_slow` morphology schedule to keep damping, length gradient, and mass gradient available longer.

The current unsolved gap is capture, not only reachability. A direct cart-position trajectory probe can swing the exact hanging six-link chain into the upright angle threshold once:

```bash
make probe-swingup-trajectory
```

This writes:

```text
runs/swingup6_trajectory_probe/probe.json
```

Current zero-noise probe result:

- best max absolute angle is about `0.122 rad`, below the configured `0.15 rad` threshold,
- first upright event occurs around `5.84 s`,
- best-pass hinge velocity RMS is about `0.885 rad/s`,
- max upright streak is two simulation steps (`0.04 s`),
- maximum absolute cart position is about `2.64 m`, within the `3.0 m` rail,
- the run is not successful and must not be used as evidence for a solved policy.

The probe is useful because it proves the configured MuJoCo system can physically reach the upright basin from the hanging state within the rail. The remaining work is to produce a low-velocity capture state or a stronger catch controller that can hold the chain upright for at least `5 s`, then validate it over held-out noisy starts.

The probe and search outputs now include full MuJoCo `qpos` and `qvel` arrays for their best pass states. The current best pass state is also encoded in:

```text
configs/swingup6_capture_handoff.yaml
```

This config uses `env.init_mode: fixed_state` to start PPO directly from the real hanging-start crossing state, with small reset noise. It is a capture-training curriculum only:

```bash
make capture-handoff6
make eval-capture-handoff6
```

If this subproblem learns to hold the chain upright from the crossing state, the next step is to combine the hanging-start swing-up phase with the learned capture policy and evaluate the whole reset-free episode from `configs/swingup6_uniform.yaml`.

A bounded PPO probe from this handoff curriculum did not solve capture:

```text
runs/swingup6_capture_handoff_probe_120/eval_capture_handoff20.json
```

That `120` update run reported `success_rate = 0.0` over `20` deterministic handoff episodes, with `max_upright_streak_max = 0.04 s`. It confirms that simply starting PPO at the current crossing state is not enough; the crossing still needs either a lower-velocity/centered handoff or a stronger staged capture curriculum.

Legacy three-stage diagnostic path:

The old swing/capture/stabilize chain remains useful as a way to export states and measure handoff quality, but it is no longer the preferred architecture. Treat the capture and stabilize portions as one capture/stabilize expert unless future evidence shows a separate stabilizer materially improves the reset-free benchmark.

Generate replayable swing states for the capture expert with:

```bash
make export-swingup-states
```

Train/evaluate the capture expert over that state distribution with:

```bash
make capture-state-list6
make eval-capture-state-list6
```

Evaluate the reset-free expert chain with:

```bash
make eval-expert-chain
```

The chain evaluator records stage transitions, best pass `qpos/qvel`, and final success metrics in `runs/swingup6_expert_chain/eval_chain.json`. Until it reaches the held-out success gate and is rendered reset-free, it is diagnostic evidence only.

Current expert-chain baseline with no trained capture checkpoint:

- `make export-swingup-states` exports `35` replayable swing states from the zero-noise hanging trajectory after filtering out high-speed states (`hinge_velocity_rms <= 4.0` by default),
- `make eval-expert-chain` preserves the swing expert's upright crossing and then runs the capture stage for at least `0.50 s` before the stabilizer can take over,
- result remains `success = false`, `max_upright_streak_seconds = 0.04`, so the capture expert must improve this boundary before the chain can become solution evidence.

Bounded capture-expert probe:

```text
runs/swingup6_capture_state_list_probe_160/eval_capture_state_list20.json
runs/swingup6_expert_chain/eval_chain_capture_probe_160.json
```

The `160` update state-list PPO probe did not solve capture. Its held-out `20` episode capture eval reported `success_rate = 0.0`, `ever_upright_rate = 0.25`, and `max_upright_streak_max = 0.06 s`. In the reset-free expert chain, the checkpoint controlled the capture stage for `176` steps but still only reached `max_upright_streak_seconds = 0.04`. The next capture expert likely needs either a stronger objective around rail-centered recovery or a supervised/warm-start controller target instead of plain PPO from swing states.

A static LQR sweep over the best exported near-upright swing states also did not find a usable capture basin. Across broad control-cost, scale, and cart-target settings, the best near-upright states still topped out at `0.04 s` upright before rail failure. Relaxing the expert-chain stabilizer handoff gate likewise made LQR take over too early and rail out. This confirms the current missing expert is nonlinear capture, not just a threshold or LQR tuning issue.

Shaped capture reward:

```text
configs/swingup6_capture_state_list_shaped.yaml
runs/swingup6_capture_state_list_shaped_probe_160/eval_capture_state_list_shaped20.json
runs/swingup6_expert_chain/eval_chain_capture_shaped_probe_160.json
```

The shaped config adds optional capture-specific reward terms for low angle, low hinge/cart velocity, and centered rail position. A bounded `160` update probe still reported `success_rate = 0.0`, but it improved capture diagnostics: held-out `20` episode state-list eval reached upright in every episode (`ever_upright_rate = 1.0`) with `max_upright_streak_mean = 0.047 s` and `max_upright_streak_max = 0.06 s`. In the reset-free chain, that checkpoint controlled capture for the rest of the `30 s` episode without rail termination, improved best angle to `0.064 rad`, and increased max upright streak to `0.08 s`. It is still far below the `5 s` sustain requirement, so it is not solution evidence.

A short `80` update fine-tune from that shaped checkpoint with an added upright-streak reward regressed (`ever_upright_rate = 0.375`, `max_upright_streak_mean = 0.0225 s` in the best internal eval), so that streak term is available as an override but is not enabled in the shaped config by default.

Chain-level swing search:

```text
scripts/search_swingup_chain.py
runs/swingup6_chain_search/search_shaped_probe_4x6_30s.json
runs/swingup6_chain_search/search_shaped_probe_centered_8x12_20s.json
runs/swingup6_chain_search/centered_chain_states.json
```

This search optimizes the swing cart-position trajectory against the actual shaped capture checkpoint and optional stabilizer handoff, rather than scoring only an isolated upright crossing. A bounded full-horizon `4 x 6` probe found a slightly cleaner handoff (`x = 1.43`, best angle `0.102 rad`) and avoided rail termination for `30 s`, but max upright streak remained `0.08 s`. A tighter `rail-target-limit = 1.8` probe produced more centered near-upright states (`x` roughly `0.8-1.0`) and exported `202` chain-generated states with `max_abs_angle <= 0.60` and `hinge_velocity_rms <= 3.0`; those states are closer to the desired transition manifold but still high velocity.

Legacy learned stabilizer split probe:

```text
configs/swingup6_chain_state_list_shaped.yaml
runs/swingup6_chain_state_list_shaped_probe_120/eval_chain_state_list_shaped20.json
runs/swingup6_expert_chain/eval_chain_learned_probe_120.json
```

This older curriculum trains from chain-generated near-upright states, and `evaluate_expert_chain.py` can route the stabilize stage to a learned checkpoint instead of LQR. The bounded `120` update probe did not improve the full chain. Its held-out `20` episode state-list eval reported `success_rate = 0.0`, `ever_upright_rate = 0.5`, and `max_upright_streak_max = 0.06 s`. In reset-free chain eval with shaped capture plus the learned stabilizer, the stabilizer controlled `779` steps but still only reached `max_upright_streak_seconds = 0.08`. Treat this as evidence that the exported chain states were still too high-velocity, not as a reason to split the current architecture into three experts.

Trajectory search is now reproducible through:

```bash
make search-swingup-trajectory
```

This writes:

```text
runs/swingup6_trajectory_search/search.json
```

The search currently optimizes fixed cart-position PD knot trajectories for better reachability/capture candidates from the exact hanging state. It is a planning tool only; the benchmark is still unsolved until `runs/swingup6_uniform/checkpoints/best.safetensors`, held-out eval JSON, and reset-free video evidence satisfy the swing-up success gate.

Capture handoff search is reproducible through:

```bash
make search-swingup-capture
```

This writes:

```text
runs/swingup6_capture_search/search.json
```

That search uses the current hanging-start swing-up trajectory until a candidate switch time, then tries an LQR capture controller plus low-dimensional feedforward action knots. It is explicitly capture-focused: candidates are ranked primarily by sustained upright streak and post-switch angle violation. A bounded `8` iteration, `32` population run still topped out at a `0.04 s` upright streak with rail excursion near or slightly beyond `3.0 m`, so it is not a capture solution.

A bounded local residual-action prototype also did not improve beyond the current `0.04 s` upright streak, and a handoff-centered trajectory search reduced the best-pass hinge/cart velocity slightly while still crossing upright near `x = 1.47 m`. That cart position leaves little rail margin for capture, so centered low-velocity handoff remains a key subproblem.

Negative PPO probe with the corrected reward gate:

```text
runs/swingup6_probe_reward_gate_400/train_log.csv
```

Final full-hanging eval from that bounded run remained `success_rate = 0.0`, `ever_upright_rate = 0.0`, and `max_upright_streak_mean = 0.0`. Later curriculum stages collapsed into short rail-hit episodes, so that checkpoint is not a solution artifact.
