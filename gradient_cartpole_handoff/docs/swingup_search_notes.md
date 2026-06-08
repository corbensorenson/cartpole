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

The first config starts with a base-heavy, mildly base-long, high-damping morphology plus temporary hinge friction-loss and a longer `+/-9 m` rail. It anneals length, mass, damping, friction-loss, and rail length to the uniform target using `morphology.schedule_mode: swingup_slow` and environment rail scheduling. It also uses `env.hanging_curriculum_power: 3.0`, so the start angle becomes hanging more slowly while training wheels are being removed. Its reward includes `capture_quality`, low-velocity upright bonuses, and rail-margin penalties so checkpoint selection can prefer low-momentum top handoffs over high-speed upright flickers. This pretraining config uses `ppo.curriculum_mode: gated` and `ppo.eval_progress: current`, so training only advances after current-stage eval shows low-momentum upright handoffs. The second config removes the morphology/friction training wheels, anneals the rail from `+/-4.5 m` to the real `+/-3 m`, and fine-tunes on the real uniform hanging-start task, where evaluation returns to final-task progress.

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

Current-stage checkpointing exposed useful curriculum signal that final-progress eval hid:

```text
runs/swingup6_gradient_low_momentum_current_probe_120
runs/swingup6_gradient_low_momentum_friction_probe_120
runs/swingup6_gradient_low_momentum_gated_rail_probe_240
runs/swingup6_gradient_low_momentum_gated_longrail_continue_160
```

The current-stage `120` update probe reached low-momentum upright handoffs through roughly progress `0.33`, then lost upright events around progress `0.41`. Adding hinge friction-loss improved early returns but did not move that collapse boundary. A gated curriculum with friction-loss and rail scheduling advanced by mastery instead of update count: it reached progress `0.375`, with the `+/-9 m` rail schedule now configured to give more windup room, but still failed the gate there (`success_rate = 0.0`). A direct longer-rail continuation at progress `0.375` produced intermittent upright events but did not pass the low-momentum gate. The checked-in curriculum now uses a smaller `curriculum_step: 0.0125` and requires `curriculum_gate_mean_upright_streak: 0.25` so the next full pretrain removes training wheels more gradually through the `0.30-0.40` transition and does not advance on one lucky upright blip.

The swing pretrain reward config also sets `low_momentum_min_time_seconds: 0.50`, so checkpoint selection and gated curriculum advancement cannot count an immediate near-upright reset as a real swing handoff.

Post-control gate probe:

```text
runs/swingup6_gradient_low_momentum_postcontrol_probe_300
```

This run was stopped after update `174` because it remained stuck at curriculum progress `0.0`. The best checkpoint reached current-stage upright events (`ever_upright_rate = 1.0`, `max_upright_streak_mean = 0.513 s`, `max_upright_streak_max = 0.74 s` at update `100`) but only `2/6` eval episodes passed the stricter post-control low-momentum handoff gate. A `12` episode per-episode eval showed the policy was using the temporary `+/-9 m` rail and terminating at the rail, so the long rail was acting as an escape path instead of only a windup aid. The checked-in reward/eval path now supports `capture_cart_pos_abs_scale` and `low_momentum_max_cart_abs`; the active swing configs set a `1.5 m` absolute cart-position capture-quality scale and require low-momentum handoff events inside `2.5 m`.

The centered-handoff follow-up:

```text
runs/swingup6_gradient_low_momentum_centered_handoff_probe_180
runs/swingup6_gradient_low_momentum_centered_gate025_continue_80
runs/swingup6_gradient_low_momentum_centered_gate025_from0375_180
runs/swingup6_gradient_low_momentum_centered_gate025_from1125_lowent_180
runs/swingup6_gradient_low_momentum_centered_gate025_from225_lowent_180
runs/swingup6_gradient_low_momentum_centered_gate025_from3125_lowent_180
```

was stopped at update `100`. The best checkpoint still had `low_momentum_upright_rate = 0.333` over `6` eval episodes, but the centered reward changed the failure mode: `ever_upright_rate = 1.0`, `max_upright_streak_mean = 0.51 s`, `max_upright_streak_max = 0.86 s`, and `max_capture_quality_mean = 0.892`. The checked-in gate now uses `curriculum_gate_low_momentum_upright_rate: 0.25`, which requires at least two centered, post-control low-momentum handoffs in the default `8` episode eval while still allowing the curriculum to move off progress `0.0`. A short continuation from that checkpoint validated the change: it advanced at updates `25`, `30`, and `60`, reaching curriculum progress `0.0375`. A second continuation starting at `0.0375` advanced through `0.05`, `0.0625`, `0.075`, `0.0875`, and `0.10`; it ended at update `180` with `curriculum_progress_next = 0.1125`. A low-entropy continuation from there advanced through `0.125`, `0.1375`, `0.15`, `0.1625`, `0.175`, `0.1875`, `0.20`, and `0.2125`; it ended with `curriculum_progress_next = 0.225`. Continuing from `0.225` reached a frontier at progress `0.300`, with `low_momentum_upright_rate = 0.5`, `ever_upright_rate = 0.833`, `max_upright_streak_mean = 0.403 s`, and `max_capture_quality_mean = 0.688`. Continuing from `0.3125` reached a frontier at progress `0.3375`, with `low_momentum_upright_rate = 0.333`, `ever_upright_rate = 0.667`, `max_upright_streak_mean = 0.587 s`, `max_upright_streak_max = 1.34 s`, and `max_capture_quality_mean = 0.682`; the live policy reached progress `0.35` but failed that gate by update `180`. The strongest checkpoint in the earlier continuation was update `100` at progress `0.0625`, where held-out current-stage eval reported `low_momentum_upright_rate = 0.667`, `ever_upright_rate = 1.0`, `max_upright_streak_mean = 0.57 s`, and `max_capture_quality_mean = 0.902`. This is progress on the curriculum machinery, not solution evidence.

Gated swing pretraining now writes `checkpoints/frontier.safetensors` plus `frontier.meta.json` whenever a stage passes. Use the frontier checkpoint for policy handoff export and uniform fine-tuning, because `best.safetensors` can legitimately point to an easier lower-progress stage with a higher low-momentum rate.

Learned swing-policy handoff export:

```text
scripts/export_policy_handoff_states.py
configs/swingup6_policy_handoff_capture_shaped.yaml
runs/swingup6_policy_handoff/swing_handoff_states.json
runs/swingup6_policy_handoff_capture_shaped_probe_120/eval_capture_policy_handoff20.json
runs/swingup6_policy_handoff_capture_shaped_frontier03375_probe_160/eval_capture_policy_handoff_shaped20.json
```

This is now the explicit boundary between the two experts. The exporter replays a learned swing policy checkpoint and saves actual MuJoCo `qpos/qvel` states only when the policy reaches a low-momentum upright handoff after policy-controlled motion (`min_time` defaults to `0.5 s`). Handoff rows now include the morphology that produced the state (`lengths`, `masses`, `damping`, and `frictionloss`) so capture probes can be audited against the swing curriculum stage. A probe using the best available progress `0.375` swing checkpoint exported `11` valid handoff states from `256` deterministic episodes. A bounded `120` update capture/stabilize PPO run from those states did not solve capture: held-out `20` episode eval reported `success_rate = 0.0`, `max_upright_streak_max = 0.14 s`, and repeated rail termination. A broader export with relaxed angle/velocity thresholds produced `20` states from `256` episodes; an `80` update capture probe survived full `15 s` episodes but still failed capture (`success_rate = 0.0`, `max_upright_streak_max = 0.20 s`).

The corrected frontier-export path was validated with:

```bash
make export-policy-handoff-states \
  SWING_HANDOFF_RUN=runs/swingup6_gradient_low_momentum_centered_gate025_from3125_lowent_180
```

It exported `13` valid centered handoff states from `128` deterministic episodes at progress `0.3375`, using `max_cart_abs = 2.5` from the swing reward gate. The best exported handoff had `max_abs_angle = 0.055 rad`, `hinge_velocity_rms = 0.090 rad/s`, and `x = 0.167 m`. A bounded `160` update capture/stabilize PPO probe from those states did not solve capture: held-out `20` episode eval reported `success_rate = 0.0`, `ever_upright_rate = 0.9`, `low_momentum_upright_rate = 0.6`, `max_upright_streak_mean = 0.109 s`, and `max_upright_streak_max = 0.20 s`. This validates the two-expert data interface but shows the current capture expert still needs a stronger sustained-balance curriculum. These states are also from a progress `0.3375` gradiented plant, not the final uniform hanging task, so they are diagnostic rather than final evidence.

The existing near-upright LQR checkpoint was also evaluated against this same exported handoff state list. With the capture config's normal reset noise it reported `success_rate = 0.0`, `low_momentum_upright_rate = 0.1`, `max_upright_streak_max = 0.12 s`, and rapid rail termination. With all state-list reset noise disabled it still reported `success_rate = 0.0`, `low_momentum_upright_rate = 0.0`, and `max_upright_streak_max = 0.08 s`. This confirms the current progress-`0.3375` swing handoffs do not transfer directly into the final uniform LQR basin; capture training from early frontier states should either use the same curriculum-stage morphology or wait until the swing frontier is much closer to uniform.

A same-stage capture probe fixed the environment at progress `0.3375` and loaded the same exported state list, avoiding the gradient-to-uniform mismatch:

```text
runs/swingup6_policy_handoff_capture_stage03375_probe_120/eval_capture_stage03375_20.json
```

This improved the capture diagnostics but still did not solve sustained balance. The best internal eval at update `120` had `success_rate = 0.0`, `ever_upright_rate = 1.0`, `low_momentum_upright_rate = 0.125`, `max_upright_streak_mean = 0.43 s`, `max_upright_streak_max = 0.74 s`, and `max_capture_quality_mean = 0.725`. The held-out `20` episode eval reported `success_rate = 0.0`, `ever_upright_rate = 0.9`, `low_momentum_upright_rate = 0.05`, `max_upright_streak_mean = 0.278 s`, and `max_upright_streak_max = 0.66 s`. The evidence JSON now records both the scheduled rail at that progress (`6.975 m`) and the final configured rail (`3.0 m`). This makes the next concrete capture step a centered-rail sustain curriculum: same-stage capture can briefly catch, but it still rides out to the temporary long rail instead of holding near center.

Final-rail same-stage centered probe:

```text
runs/swingup6_policy_handoff_capture_stage03375_finalrail_centered_probe_160/eval_capture_stage03375_finalrail_centered20.json
```

This forced the same progress-`0.3375` morphology onto the final `+/-3 m` rail and increased centered cart penalties. It did not improve capture: best internal eval had `success_rate = 0.0`, `low_momentum_upright_rate = 0.125`, `max_upright_streak_mean = 0.313 s`, and `max_upright_streak_max = 0.58 s`; held-out `20` episode eval had `success_rate = 0.0`, `ever_upright_rate = 0.9`, `low_momentum_upright_rate = 0.05`, `max_upright_streak_mean = 0.224 s`, and `max_upright_streak_max = 0.54 s`. The eval episodes terminated at the real rail instead of the temporary long rail, which confirms that simply shrinking the rail and increasing cart-position penalties is not enough.

The environment now tracks `centered_upright_streak_seconds` and `low_momentum_upright_streak_seconds` separately from generic upright streaks. Capture reward configs can use `centered_upright_streak` and `low_momentum_upright_streak` terms, and checkpoint selection records centered/low-momentum streak metrics so the best capture checkpoint is not chosen only for a brief upright touch.

Centered/low-momentum streak capture probe:

```text
runs/swingup6_policy_handoff_capture_stage03375_centered_streak_probe_120/eval_capture_stage03375_centered_streak20.json
```

This used the same progress-`0.3375` morphology and exported swing-policy handoff states, forced the real `+/-3 m` rail, and added explicit centered/low-momentum upright-streak rewards. It improved the best internal capture checkpoint to `max_centered_upright_streak_mean = 0.428 s`, `max_centered_upright_streak_max = 0.66 s`, `max_low_momentum_upright_streak_mean = 0.163 s`, and `max_capture_quality_mean = 0.724`, but held-out `20` episode eval still reported `success_rate = 0.0`, `ever_upright_rate = 0.9`, `low_momentum_upright_rate = 0.05`, `max_centered_upright_streak_mean = 0.288 s`, and `max_centered_upright_streak_max = 0.66 s`. This is a better diagnostic reward and checkpoint selector, not a solved capture policy.

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

Low-momentum trajectory search:

```text
runs/swingup6_trajectory_search/low_momentum_probe_10x32.json
runs/swingup6_trajectory_search/low_momentum_continue_20x48.json
runs/swingup6_expert_chain/swing_states_low_momentum_probe_10x32.json
runs/swingup6_chain_search/eval_low_momentum_probe_10x32_lqr_30s.json
runs/swingup6_expert_chain/eval_lqr_on_low_momentum_probe_10x32_states20.json
runs/swingup6_capture_low_momentum_trajectory_probe_80/eval_capture_low_momentum_trajectory20.json
runs/swingup6_trajectory_search/sustain_probe_18x40_30s.json
runs/swingup6_capture_sequence/low_momentum_state_probe_24x64_8s.json
runs/swingup6_capture_sequence/low_momentum_state6_probe_14x48_8s.json
runs/swingup6_mpc_capture/low_momentum_state_probe_8s.json
runs/swingup6_mpc_capture/low_momentum_state_zero_velocity_probe_8s.json
runs/swingup6_capture_lqr_residual_probe_120/eval_capture_lqr_residual20.json
runs/swingup6_action_search/low_momentum_probe_10x64.json
runs/swingup6_action_search/longrail_low_momentum_continue_20x64.json
runs/swingup6_action_search/longrail_sustain_continue_10x64.json
runs/swingup6_capture_low_momentum_velocity_curriculum_probe_120/eval_capture_velocity_curriculum20.json
runs/swingup6_capture_low_momentum_velocity_curriculum_probe_120/eval_capture_velocity_curriculum_progress0_zeronoise20.json
```

`scripts/search_swingup_trajectory.py` now has `--score-mode low_momentum` and `--score-mode sustain` options and can warm-start from a prior controller JSON. `scripts/export_swingup_states.py` can export states from a searched controller JSON and no longer shadows the `--max-cart-abs` filter with observed max cart position. `scripts/search_capture_sequence.py` searches an open-loop action sequence from a replayed handoff state to test whether the state is physically catchable before spending more PPO time. `scripts/evaluate_mpc_capture.py` adds a receding-horizon random-shooting MPC diagnostic from the same saved state file, with an optional LQR baseline candidate. State-list and fixed-state resets support a progress-scheduled `env.init_qvel_scale_start` to `env.init_qvel_scale_end`, so capture can train on the real handoff positions while gradually restoring the saved handoff velocities. Reproduce the path with:

```bash
make search-swingup-low-momentum
make search-swingup-sustain
make search-swingup-action-low-momentum
make export-low-momentum-swingup-states
make search-capture-sequence
make eval-mpc-capture
make capture-low-momentum-velocity-curriculum
make capture-lqr-residual-velocity-curriculum
```

A bounded `10 x 32` exact-hanging uniform search found a more centered top crossing than the original fixed probe: best score handoff had `max_abs_angle = 0.079 rad`, `hinge_velocity_rms = 1.062 rad/s`, `x = 1.069 m`, and `cart_velocity = -0.030 m/s`; the best-streak candidate reached `0.10 s` upright with `max_abs_angle = 0.074 rad` and `hinge_velocity_rms = 1.091 rad/s`. A warm-started `20 x 48` continuation did not beat this; it found a lower-angle candidate (`0.059 rad`) but with higher hinge velocity (`1.302 rad/s`) and the same `0.08 s` streak. The exporter wrote `11` replayable low-momentum-search states from the best controller.

The better trajectory states still did not solve capture. Reset-free chain eval with LQR capture/stabilization reached capture but stayed at `max_upright_streak_seconds = 0.04`. Direct near-upright LQR eval from the exported states reported `success_rate = 0.0`, `ever_upright_rate = 0.45`, `low_momentum_upright_rate = 0.0`, and `max_upright_streak_max = 0.04 s`. A bounded `80` update PPO capture probe from those states reported held-out `20` episode `success_rate = 0.0`, `ever_upright_rate = 0.75`, `low_momentum_upright_rate = 0.0`, `max_upright_streak_mean = 0.053 s`, and `max_upright_streak_max = 0.10 s`. A `120` update velocity-curriculum PPO probe, training from real positions with saved velocities annealed from `0.0` to `1.0`, also failed: held-out final-velocity eval reported `success_rate = 0.0`, `ever_upright_rate = 0.75`, `low_momentum_upright_rate = 0.0`, `max_upright_streak_mean = 0.053 s`, and `max_upright_streak_max = 0.10 s`; even progress-`0.0` zero-noise eval only reached `max_upright_streak_max = 0.10 s`. A small LQR sweep over cart targets, gains, control costs, and velocity scales likewise topped out at `0.06 s`. A full `30 s` sustain-scored trajectory search also failed to improve the reset-free controller; its best candidate stayed at `0.08 s` upright. Open-loop action-sequence CEM from the best exported state improved the LQR baseline from `0.04 s` to only `0.06 s`, and the later state-index `6` probe reached only `0.02 s`.

A receding-horizon MPC diagnostic from the best saved handoff state confirms the same capture bottleneck with feedback replanning. With the real saved velocity, an `8 s`, `96` sample, `30` step horizon probe reached `success = false`, `max_upright_streak_seconds = 0.08`, best angle `0.082 rad`, hinge velocity RMS `1.728 rad/s`, and max cart excursion `2.885 m`. Replaying the same handoff position with `env.init_qvel_scale = 0.0` improved the streak to `0.12 s` and best angle to `0.043 rad`, but still failed the `5 s` sustain target. This improves the real-uniform first-expert state quality but confirms the capture expert still needs either a colder handoff, a more centered handoff, or a stronger nonlinear catch method.

The next capture variant is `configs/swingup6_capture_lqr_residual.yaml`. It keeps the real saved state-list and velocity curriculum, but the environment applies a finite-difference LQR action bias around upright plus the policy's learned residual action. This tests whether the capture learner benefits from starting inside the proven near-upright stabilizer's feedback structure while still being able to learn nonlinear corrections for swing handoff states. A bounded `120` update real-uniform probe was not better than the unaided velocity curriculum: held-out `20` episode eval reported `success_rate = 0.0`, `ever_upright_rate = 0.65`, `low_momentum_upright_rate = 0.0`, `max_upright_streak_mean = 0.038 s`, `max_upright_streak_max = 0.10 s`, and repeated rail termination. Treat this as evidence that a naive LQR residual bias is not enough for the current handoff states.

`scripts/search_swingup_action_sequence.py` adds a direct normalized-force-knot search to test whether fixed cart-position PD knots are hiding better swing handoffs. On the real `+/-3 m` rail, a bounded `10 x 64` probe improved only to `max_abs_angle = 0.802 rad` and exported no upright states. With a temporary `+/-9 m` rail, a warm-started `20 x 64` continuation reached the upright threshold (`min_best_pass_angle = 0.074 rad`, best score pass `0.138 rad`) near the center, but hinge velocity remained about `10-11 rad/s` and no low-momentum states passed export filters. A sustain-scored continuation did not improve the streak beyond `0.02 s`. This supports the longer-windup hypothesis for reachability, but not yet for catchable handoff quality.

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
