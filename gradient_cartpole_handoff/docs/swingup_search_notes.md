# Swing-Up Search Notes

## Current Goal And Architecture

Current roadmap: reproduce the real six-link swing-up benchmark from the hanging/collapsed start as a calibration gate, then solve and publish the canonical seven-link task. Starting from already-upright links remains a separate near-upright control baseline and is not completion evidence. See `ROADMAP.md` for the authoritative project gates.

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
runs/swingup6_gradient_low_momentum_centered_gate025_from350_retry_240
```

was stopped at update `100`. The best checkpoint still had `low_momentum_upright_rate = 0.333` over `6` eval episodes, but the centered reward changed the failure mode: `ever_upright_rate = 1.0`, `max_upright_streak_mean = 0.51 s`, `max_upright_streak_max = 0.86 s`, and `max_capture_quality_mean = 0.892`. The checked-in gate now uses `curriculum_gate_low_momentum_upright_rate: 0.25`, which requires at least two centered, post-control low-momentum handoffs in the default `8` episode eval while still allowing the curriculum to move off progress `0.0`. A short continuation from that checkpoint validated the change: it advanced at updates `25`, `30`, and `60`, reaching curriculum progress `0.0375`. A second continuation starting at `0.0375` advanced through `0.05`, `0.0625`, `0.075`, `0.0875`, and `0.10`; it ended at update `180` with `curriculum_progress_next = 0.1125`. A low-entropy continuation from there advanced through `0.125`, `0.1375`, `0.15`, `0.1625`, `0.175`, `0.1875`, `0.20`, and `0.2125`; it ended with `curriculum_progress_next = 0.225`. Continuing from `0.225` reached a frontier at progress `0.300`, with `low_momentum_upright_rate = 0.5`, `ever_upright_rate = 0.833`, `max_upright_streak_mean = 0.403 s`, and `max_capture_quality_mean = 0.688`. Continuing from `0.3125` reached a frontier at progress `0.3375`, with `low_momentum_upright_rate = 0.333`, `ever_upright_rate = 0.667`, `max_upright_streak_mean = 0.587 s`, `max_upright_streak_max = 1.34 s`, and `max_capture_quality_mean = 0.682`; the live policy reached progress `0.35` but failed that gate by update `180`. A retry from the best progress-`0.35` checkpoint with `8` eval episodes and a `0.50` ever-upright gate advanced again: update `120` passed progress `0.3625`, and update `130` passed progress `0.375`. The new frontier is progress `0.375` with `curriculum_progress_next = 0.3875`, `low_momentum_upright_rate = 0.375`, `ever_upright_rate = 0.5`, `max_upright_streak_mean = 0.353 s`, `max_upright_streak_max = 0.86 s`, and `max_capture_quality_mean = 0.518`. It then plateaued at progress `0.3875` through update `240`. The strongest checkpoint in the earlier continuation was update `100` at progress `0.0625`, where held-out current-stage eval reported `low_momentum_upright_rate = 0.667`, `ever_upright_rate = 1.0`, `max_upright_streak_mean = 0.57 s`, and `max_capture_quality_mean = 0.902`. This is progress on the curriculum machinery, not solution evidence.

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

The newer progress-`0.375` frontier was exported with:

```text
runs/swingup6_policy_handoff/swing_handoff_states_frontier0375_retry128.json
runs/swingup6_policy_handoff_capture_stage0375_retry_probe_120/eval_capture_stage0375_retry20.json
```

It produced `5` valid low-momentum handoff states from `128` deterministic episodes. The best exported state was high quality (`max_abs_angle = 0.055 rad`, `hinge_velocity_rms = 0.146 rad/s`, `x = 0.277 m`, `cart_velocity = 0.444 m/s`), but the same-stage final-rail capture probe still failed: held-out `20` episode eval reported `success_rate = 0.0`, `ever_upright_rate = 0.95`, `low_momentum_upright_rate = 0.0`, `max_upright_streak_mean = 0.227 s`, `max_upright_streak_max = 0.44 s`, `max_low_momentum_upright_streak_max = 0.28 s`, and repeated rail termination. This confirms the saved first-expert states are real and increasingly clean, but the current capture/stabilize expert still cannot hold them.

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
runs/swingup6_action_search/rail7_low_momentum_continue_30x64.json
runs/swingup6_action_search/rail7_cold_handoff_continue_20x64.json
runs/swingup6_action_search/rail7_sustain_continue_15x64.json
runs/swingup6_linear_policy_search_probe_20x64/search_linear_policy.json
runs/swingup6_uniform_time_linear_probe_20x64/search_linear_policy.json
runs/swingup6_capture_low_momentum_velocity_curriculum_probe_120/eval_capture_velocity_curriculum20.json
runs/swingup6_capture_low_momentum_velocity_curriculum_probe_120/eval_capture_velocity_curriculum_progress0_zeronoise20.json
runs/swingup6_capture_state_curriculum_probe_120/eval_capture_state_curriculum20.json
runs/swingup6_capture_state_curriculum_lqr_init_probe_120/eval_capture_state_curriculum_lqr_init20.json
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
make capture-state-curriculum
make capture-lqr-residual-velocity-curriculum
make capture-state-curriculum-lqr-init
```

A bounded `10 x 32` exact-hanging uniform search found a more centered top crossing than the original fixed probe: best score handoff had `max_abs_angle = 0.079 rad`, `hinge_velocity_rms = 1.062 rad/s`, `x = 1.069 m`, and `cart_velocity = -0.030 m/s`; the best-streak candidate reached `0.10 s` upright with `max_abs_angle = 0.074 rad` and `hinge_velocity_rms = 1.091 rad/s`. A warm-started `20 x 48` continuation did not beat this; it found a lower-angle candidate (`0.059 rad`) but with higher hinge velocity (`1.302 rad/s`) and the same `0.08 s` streak. The exporter wrote `11` replayable low-momentum-search states from the best controller.

The better trajectory states still did not solve capture. Reset-free chain eval with LQR capture/stabilization reached capture but stayed at `max_upright_streak_seconds = 0.04`. Direct near-upright LQR eval from the exported states reported `success_rate = 0.0`, `ever_upright_rate = 0.45`, `low_momentum_upright_rate = 0.0`, and `max_upright_streak_max = 0.04 s`. A bounded `80` update PPO capture probe from those states reported held-out `20` episode `success_rate = 0.0`, `ever_upright_rate = 0.75`, `low_momentum_upright_rate = 0.0`, `max_upright_streak_mean = 0.053 s`, and `max_upright_streak_max = 0.10 s`. A `120` update velocity-curriculum PPO probe, training from real positions with saved velocities annealed from `0.0` to `1.0`, also failed: held-out final-velocity eval reported `success_rate = 0.0`, `ever_upright_rate = 0.75`, `low_momentum_upright_rate = 0.0`, `max_upright_streak_mean = 0.053 s`, and `max_upright_streak_max = 0.10 s`; even progress-`0.0` zero-noise eval only reached `max_upright_streak_max = 0.10 s`. A small LQR sweep over cart targets, gains, control costs, and velocity scales likewise topped out at `0.06 s`. A full `30 s` sustain-scored trajectory search also failed to improve the reset-free controller; its best candidate stayed at `0.08 s` upright. Open-loop action-sequence CEM from the best exported state improved the LQR baseline from `0.04 s` to only `0.06 s`, and the later state-index `6` probe reached only `0.02 s`.

A receding-horizon MPC diagnostic from the best saved handoff state confirms the same capture bottleneck with feedback replanning. With the real saved velocity, an `8 s`, `96` sample, `30` step horizon probe reached `success = false`, `max_upright_streak_seconds = 0.08`, best angle `0.082 rad`, hinge velocity RMS `1.728 rad/s`, and max cart excursion `2.885 m`. Replaying the same handoff position with `env.init_qvel_scale = 0.0` improved the streak to `0.12 s` and best angle to `0.043 rad`, but still failed the `5 s` sustain target. This improves the real-uniform first-expert state quality but confirms the capture expert still needs either a colder handoff, a more centered handoff, or a stronger nonlinear catch method.

The next capture variant is `configs/swingup6_capture_lqr_residual.yaml`. It keeps the real saved state-list and velocity curriculum, but the environment applies a finite-difference LQR action bias around upright plus the policy's learned residual action. This tests whether the capture learner benefits from starting inside the proven near-upright stabilizer's feedback structure while still being able to learn nonlinear corrections for swing handoff states. A bounded `120` update real-uniform probe was not better than the unaided velocity curriculum: held-out `20` episode eval reported `success_rate = 0.0`, `ever_upright_rate = 0.65`, `low_momentum_upright_rate = 0.0`, `max_upright_streak_mean = 0.038 s`, `max_upright_streak_max = 0.10 s`, and repeated rail termination. Treat this as evidence that a naive LQR residual bias is not enough for the current handoff states.

The backward capture curriculum now starts the capture expert from exact upright states and gradually blends reset `qpos/qvel` back to the real saved first-expert handoff states while expanding from the easiest saved state to the full state list. The linear LQR-initialized variant (`make capture-state-curriculum-lqr-init`) was worse than the previous velocity-only and residual probes: held-out final-state eval reported `success_rate = 0.0`, `ever_upright_rate = 0.55`, `low_momentum_upright_rate = 0.0`, `max_upright_streak_mean = 0.021 s`, `max_upright_streak_max = 0.06 s`, and `max_capture_quality_max = 0.031`. The nonlinear PPO variant (`make capture-state-curriculum`) did reach upright states frequently, with held-out `ever_upright_rate = 0.75`, `max_capture_quality_max = 0.244`, and `max_upright_streak_max = 0.08 s`, but still had `success_rate = 0.0` and no low-momentum upright captures. This is useful negative evidence: the saved handoff positions are replayable and the curriculum is wired correctly, but the current capture learner still cannot hold the real uniform six-link chain after the swing expert hands it off.

`scripts/search_swingup_action_sequence.py` adds a direct normalized-force-knot search to test whether fixed cart-position PD knots are hiding better swing handoffs. On the real `+/-3 m` rail, a bounded `10 x 64` probe improved only to `max_abs_angle = 0.802 rad` and exported no upright states. With a temporary `+/-9 m` rail, a warm-started `20 x 64` continuation reached the upright threshold (`min_best_pass_angle = 0.074 rad`, best score pass `0.138 rad`) near the center, but hinge velocity remained about `10-11 rad/s` and no low-momentum states passed export filters. A sustain-scored continuation did not improve the streak beyond `0.02 s`. A rail-homotopy probe from that long-rail controller showed how much windup it was using: replay at `+/-7 m` degraded to `0.676 rad`, and replay at `+/-3 m` degraded to `1.846 rad`. Warm-started `+/-7 m` search recovered an upright flicker (`0.144 rad`) but with hinge RMS around `19 rad/s`; the new `cold_handoff` score found colder near-upright passes (`0.185 rad` at `11.6 rad/s`, and a `0.348 rad` pass at `5.37 rad/s` before the handoff time filter), but still exported no catchable states. This supports the longer-windup hypothesis for reachability, but not yet for catchable handoff quality.

`scripts/search_linear_policy.py` tests a different hypothesis: a small closed-loop policy search, saved in normal MLX `ActorCritic` checkpoint format. A stationary linear tanh policy on the default observation improved only to about `1.78 rad` best angle in a bounded `20 x 64` CEM probe and never reached upright. `configs/swingup6_uniform_time_linear.yaml` adds explicit time-basis observation features so a linear checkpoint can represent a windup schedule plus feedback; the matching `20 x 64` probe still never reached upright and topped out around `1.90 rad`. This is negative evidence that the quick low-dimensional closed-loop search is weaker than the existing PPO frontier and action-spline searches.

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

Policy handoff frontier and capture diagnostics:

```text
runs/swingup6_gradient_low_momentum_centered_gate025_from350_retry_240/checkpoints/frontier.safetensors
runs/swingup6_policy_handoff/swing_handoff_states_frontier0375_retry128.json
runs/swingup6_policy_handoff_capture_stage0375_retry_probe_120/eval_capture_stage0375_retry20.json
runs/swingup6_policy_handoff_capture_stage0375_retry_probe_120/eval_capture_stage0375_retry20_zeronoise.json
runs/swingup6_policy_handoff_capture_stage0375_retry_probe_120/eval_capture_stage0375_retry20_qvel0_zeronoise.json
runs/swingup6_mpc_capture/stage0375_frontier_best_8s.json
runs/swingup6_mpc_capture/stage0375_frontier_best_qvel0_8s.json
```

The best learned swing frontier passed the progress-`0.375` gate and exported `5` real MuJoCo low-momentum handoff states from `128` deterministic replay episodes. The best exported state is close to upright (`max_abs_angle = 0.0549 rad`, `hinge_velocity_rms = 0.146`) and centered enough for capture (`x = 0.277 m`, `cart_velocity = 0.444 m/s`), but it is still an intermediate curriculum state: progress `0.375`, scheduled rail `+/-6.75 m`, and a partially hanging start due `hanging_curriculum_power = 3.0`.

The same-stage capture PPO trained from those saved states improves the handoff but does not solve capture. Held-out `20` episode eval reports `success_rate = 0.0`, `ever_upright_rate = 0.95`, `max_upright_streak_mean = 0.227 s`, and `max_upright_streak_max = 0.44 s`. With reset noise removed it still fails (`success_rate = 0.0`, `max_upright_streak_max = 0.28 s`). Forcing saved handoff velocities to zero improves the diagnostic (`max_upright_streak_mean = 0.371 s`, max `0.50 s`) but remains far below the `5 s` sustain gate.

Finite-difference LQR is a valid near-upright stabilizer for the final uniform model, but it is not a reliable capture stabilizer for the progress-`0.375` gradient morphology: the stage linearization is highly ill-conditioned and the closed-loop eigenvalue remains slightly above `1.0`. Receding-horizon MPC from the best saved handoff state confirms the same gap. Full-velocity MPC reached only a `0.30 s` upright streak; zeroing saved qvel reached `0.50 s` and then used almost the whole `+/-3 m` rail. The next useful capture work is therefore a nonlinear learned catch policy from saved states, while the final proof still requires exporting states from a full-uniform, full-hanging swing expert.

Longer-rail continuation probe:

```text
runs/swingup6_gradient_low_momentum_longrail_from3875_probe_100/train_log.csv
runs/swingup6_gradient_low_momentum_longrail_from3875_probe_100/checkpoints/best.meta.json
```

This bounded `100` update run initialized from the previous latest checkpoint at progress `0.3875`, widened the scheduled rail at that stage from `+/-6.675 m` to `+/-8.5125 m`, relaxed the low-momentum cart-position allowance to `3.0 m`, and lowered entropy to `0.010`. It did not pass the gate or advance curriculum. The best checkpoint was update `20`: `ever_upright_rate = 0.875`, `low_momentum_upright_rate = 0.125`, `max_upright_streak_mean = 0.2325 s`, `max_upright_streak_max = 0.64 s`, `max_low_momentum_upright_streak_max = 0.50 s`, and `success_rate = 0.0`. Later evals regressed. This supports the user's longer-windup idea as a useful training-wheel probe, but the current policy still cannot reliably arrive with low momentum.

Low-momentum refinement and capture follow-up:

```text
runs/swingup6_gradient_low_momentum_lmom_refine_from3875_probe_120/train_log.csv
runs/swingup6_policy_handoff/swing_handoff_states_frontier03875_lmom_refine128.json
runs/swingup6_policy_handoff_capture_stage03875_lmom_refine_probe_160/eval_capture_policy_handoff_stage20.json
runs/swingup6_policy_handoff_capture_stage03875_lmom_refine_probe_160/eval_capture_policy_handoff_stage20_qvel0_zeronoise.json
```

The `120` update refinement restarted from the long-rail best checkpoint and added explicit centered/low-momentum streak rewards at progress `0.3875`. It still did not advance the gate, but it improved the best internal upright-streak diagnostic: update `120` reached `max_upright_streak_max = 2.04 s`, `max_upright_streak_mean = 0.4375 s`, and `max_low_momentum_upright_streak_mean = 0.10 s` with `success_rate = 0.0`. Exporting actual learned-policy handoff states from that checkpoint produced only `2` low-momentum states in `128` deterministic episodes. The best exported state was `x = 0.171 m`, `cart_velocity = 0.396 m/s`, `max_abs_angle = 0.0849 rad`, and `hinge_velocity_rms = 0.100`.

Training the same-stage capture expert from those two real states again failed the sustain gate. Held-out `20` episode eval reported `success_rate = 0.0`, `ever_upright_rate = 0.95`, `max_upright_streak_mean = 0.221 s`, and `max_upright_streak_max = 0.54 s`. Even with reset noise removed and saved qvel forced to zero, the capture checkpoint still terminated on the rail with `success_rate = 0.0`, `max_upright_streak_mean = 0.442 s`, and `max_upright_streak_max = 0.46 s`. The saved poses are therefore better first-expert artifacts, but the capture/stabilize basin remains too small.

`scripts/make_lqr_checkpoint.py` now accepts `--cart-target`, which lets the deterministic LQR checkpoint bias its linear policy around the saved handoff cart position. A scale/sign sweep around the best progress-`0.3875` handoff pose still did not stabilize even when saved qvel was forced to zero. The best tested static LQR variant reached only about `0.14 s` upright before rail termination. This is useful negative evidence: the failure is not just an omitted cart target or gain sign/scale issue.

The new `capture-policy-handoff-curriculum` target separates reset-schedule progress from plant progress. Set `env.plant_progress` through `CAPTURE_STAGE_PROGRESS` to keep the same MuJoCo plant as the swing expert, while training progress scales the saved `qvel` from `0.0` to `1.0` and ramps reset noise from zero to the real handoff-noise values. This implements the two-expert handoff contract directly: train capture from the swing expert's real saved positions, first with zeroed arrival momentum, then with the real saved arrival momentum.

A bounded `120` update probe of that target from `swing_handoff_states_frontier03875_lmom_refine128.json` improved the capture frontier but still failed the sustain gate. The best internal checkpoint was update `80` with `ever_upright_rate = 1.0`, `max_upright_streak_max = 0.86 s`, and `max_low_momentum_upright_streak_mean = 0.2075 s`. Held-out `20` episode full-velocity/noise eval reported `success_rate = 0.0`, `ever_upright_rate = 0.95`, `max_upright_streak_mean = 0.26 s`, `max_upright_streak_max = 0.76 s`, and `max_capture_quality_max = 0.913`. The qvel-zero/no-noise diagnostic from the same checkpoint was deterministic at `0.50 s` upright before rail termination. A follow-up `80` update full-velocity polish from that checkpoint regressed slightly (`max_upright_streak_max = 0.68 s` held-out). Net: the real-position/velocity-restoration curriculum is the best learned capture variant so far, but the capture expert still drives into the rail before a `5 s` hold.

Wide uniform capture/stabilizer probe:

```text
configs/uniform6_capture_wide.yaml
runs/uniform6_capture_wide_probe_80/eval_capture_wide20.json
```

The environment now supports scheduled reset noise (`*_start` / `*_end`) for initial angle, velocity, cart-position, and cart-velocity noise, and `evaluate.py` records both the effective noise at eval progress and the schedule fields in evidence JSON. `configs/uniform6_capture_wide.yaml` uses that support to train a robust final-uniform near-upright capture expert from progressively wider upright perturbations. A bounded `80` update probe did not learn the final wide-noise task: held-out `20` episode eval at progress `1.0` reported `success_rate = 0.0`, `ever_upright_rate = 0.15`, `max_upright_streak_max = 0.06 s`, and `max_capture_quality_mean = 0.0266`. This is not a solution, but it clarifies the next bottleneck: before final swing-up proof, the capture/stabilize expert must learn a much wider near-upright basin than the current analytic LQR baseline.

## Frozen P1 capture envelope (2026-07-15)

The synthetic six-link capture gate is now frozen in `benchmarks/p1_capture_envelope.yaml`. It uses the final uniform six-link morphology, `+/-3 m` rail, `+/-80 N` action contract, 50 Hz policy rate, and 15-second episodes. States independently cover `|x| <= 1.25 m`, maximum absolute link angle `<= 0.15 rad`, `|cart velocity| <= 0.50 m/s`, and a six-dimensional hinge-velocity RMS ball `<= 0.75 rad/s`. The deterministic splits are train seed `61001` (`20,000` states), validation seed `61002` (`2,000`), and test seed `61003` (`1,000`).

Regenerated artifact hashes after freezing the complete plant and gate contract:

```text
runs/p1_capture_envelope/train.json       993d6563c4eccd23ae26aa3f5bd0064751790ad0bcf5376e2a60205d88462837
runs/p1_capture_envelope/validation.json  aa32b8989d45223fca7e19e9d5eeb7f84da55c6dd5faa9a8508212e21a479de0
runs/p1_capture_envelope/test.json        5d82bbd33b1cec6847fb26de836284be8623d00d47dd25c335130d68dd85c686
```

`scripts/evaluate_capture_gate.py` now validates both the generated dataset and the resolved plant/gate config, runs deterministic batched policy inference across all 1,000 test states, and records per-state capture, hold, rail, termination, and excursion evidence. Diagnostic subsets or curriculum progress below `1.0` cannot pass. The exact final gate remains at least 90% success, median maximum continuous upright hold of at least 10 seconds, and no rail hit among successful episodes.

The existing narrow finite-difference LQR baseline was rerun after the benchmark freeze:

```text
runs/p1_capture_envelope/eval_lqr_test1000.json
success_rate = 0.0
max_upright_streak_median = 0.04 s
rail_hit_count = 1000
gate.passed = false
```

Previous wide PPO, state-list PPO, curriculum-stage PPO, and LQR-residual checkpoints also scored zero successes on the same 1,000 physical test states. Those earlier JSONs predate the expanded spec hash and are retained only as diagnostics, not authoritative gate evidence.

The first honest all-state backward curriculum uses the frozen 20,000-state train split at every stage, with position and velocity amplitudes scaled quadratically by curriculum progress. This avoids the optimistic `quality_prefix` ordering used in earlier probes. Its residual-PPO frontier is:

```text
runs/swingup6_capture_envelope_allstates_gated_probe_75/checkpoints/frontier.safetensors
mastered progress = 0.05
validation success at progress 0.05 = 30/32 internally; 124/128 for pure LQR diagnostic
validation success at progress 0.075 = 18/32 internally; 130/256 on an external evaluation
```

A fixed-progress continuation at `0.075` with lower learning rate and action variance remained near 50% success through update 50, so it did not advance the gate. This establishes a measured curriculum boundary, not a solution. The next PPO continuation should use materially smaller envelope increments than `0.025`; the parallel model-based branch should add a terminal stabilizer/value and longer low-dimensional horizon before being judged against the same frozen validation states.

Fine-step follow-up and validation-gate correction:

```text
runs/swingup6_capture_envelope_allstates_finegate_probe_300/
runs/p1_capture_envelope/eval_finegate_frontier0055_validation256.json
```

Reducing the curriculum increment from `0.025` to `0.005` allowed the policy to move beyond the previous jump failure. It requalified at progress `0.05` with `62/64` internal successes, then reached `58/64` at progress `0.055` after 60 updates at that stage. At progress `0.06`, three checks scored 72%, 80%, and 72%, so the run was stopped at update 156.

An independent exact-index check of the proposed `0.055` frontier on 256 frozen validation states scored only `84.8%` (`217/256`) with 39 rail hits. Audit showed that the trainer's old internal gate sampled the training state file rather than the independent validation split. The `0.055` advancement is therefore invalidated as a mastered frontier even though it is useful initialization. `ppo_mlx.evaluate_policy` now supports exact indexed resets, and gated training can set `ppo.curriculum_eval_states_path`, `ppo.curriculum_eval_seed`, and `ppo.eval_episodes`. The capture config uses a fixed 256-state subset of `runs/p1_capture_envelope/validation.json`; checkpoint metadata records the path, seed, and every selected index. The accepted held-out frontier remains progress `0.05` until the corrected validation gate passes.

The corrected trainer gate uses batched MLX inference over 64 independent MuJoCo environments. A four-state serial/batched comparison matched all discrete metrics exactly and floating-point metrics to approximately `1e-9`; a full 256-state check now takes about 35 seconds instead of several minutes. On the fixed seed-`61201` validation subset, the same candidate checkpoint scores `92.58%` at progress `0.05` and `82.03%` at progress `0.055`. This independently accepts the 5% frontier and rejects the 5.5% candidate before further training.

Hard-negative mining evaluated 2,048 deterministic states from the training split at progress `0.055` and found 373 failures (`18.2%`), closely matching validation. The generated mixture keeps all 2,048 anchors and repeats each training-only failure three times (`3,167` rows total). A 60-update focused PPO pass changed shaped return but left all 46 held-out failure identities unchanged, so it did not enlarge the basin.

A direct controller decomposition then identified the stalled homotopy. At progress `0.055`, the old schedule had already reduced the LQR bias from `1.0` to about `0.946` and increased learned-residual authority to about `0.060`. A fixed 64-state sweep showed 82% success near LQR scale `0.90`, at least 95% near `1.0`, and 98.4% with LQR scale `1.20`; increasing residual authority consumed more rail without improving success. On all 256 fixed validation states, pure scale-`1.20` LQR scores `93.75%` at progress `0.055`, `66.02%` at `0.075`, and `35.94%` at `0.10`. This passes the 5.5% component stage but remains far from P1.

The first revised config preserved LQR scale `1.20` early and annealed it to `0.02` with fourth-power progress, while learned-residual authority grew quadratically from zero to one. This established the intended training-wheel homotopy: retain the proven local stabilizer while expanding the reset envelope, then remove it as the nonlinear policy takes authority instead of weakening both controllers at the beginning.

Further held-out tuning found that LQR scale `1.30` is the narrow linear-stability optimum at the next boundary. On all 256 validation states at progress `0.06`, scale `1.30` scores `89.45%`; nearby scales `1.31` to `1.35` remain around 89%, while `1.40` is already too aggressive. Small additional cart-position or cart-velocity gains do not rescue failures, and modal gain variants from LQR control costs `30` through `30,000` are worse after angular-gain normalization.

An axis ablation at progress `0.06` explains the remaining two-to-three-episode gap: with the same scale-`1.30` LQR, the position-only envelope scores `92.19%`, velocity-only scores `100%`, and the combined envelope scores `89.45%`. The curriculum now expands reset positions quadratically and velocities cubically, so the controller masters displacement directions before the full arrival-momentum distribution catches up. Both schedules still equal exactly one at final progress; this changes training order, not the P1 gate.

The component-wise homotopy then passed the fixed 256-state validation gate at progress `0.06` with `91.80%` success at update 20. At `0.0625`, repeated checks remained at `87.50%`, so the accepted frontier is now `p=0.06`. Because the effective position and velocity scales are only `p^2` and `p^3`, this is a controller-development frontier rather than evidence that 6% of each physical bound has been mastered.

Direct-policy and distillation probes did not enlarge that basin. Initializing an exact linear MLX actor from scale-`1.30` LQR allowed ordinary PPO to transiently requalify at `p=0.06` (`90.23%`) but it collapsed to zero successes immediately after advancing to `p=0.06125`. A lower-variance, lower-rate continuation stayed at `86.33%` at the same boundary. Supervised distillation on 100,000 sampled states reached validation MSE `0.00233`, MAE `0.03448`, and maximum absolute action error `0.15874`, but the distilled nonlinear actor scored `0/256` at `p=0.06` and every episode hit the rail. For this unstable plant, a visually small action-approximation error is not a usable stabilizer.

A constrained linear-MPC diagnostic now uses a condensed action-only OSQP formulation, normalized Riccati terminal cost, exact frozen-state resets, physical action bounds, and rail constraints. On the first 64 frozen validation states at `p=0.06`, the 20-step (`0.4 s`) MPC scored `0/64` with median upright hold `0.44 s`; scale-`1.30` LQR succeeds on `54/64` of the same ordered states and has median hold `15.02 s`. Removing the MPC prediction rail bound does not rescue the controller, so the result is not caused by an over-tight rail constraint. This rules out the current linearized, short-horizon constrained MPC as a warm start; nonlinear trajectory optimization or a learned terminal value remains an open branch.

## Strategy review from the shared browser conversation (2026-07-15)

The strongest recommendations from the shared review are now explicit roadmap work: retain two composable problems (`reach the capture funnel` and `remain inside it`), use hysteresis at the expert switch, generate swing trajectories with direct collocation/DDP/nonlinear MPC toward the measured capture funnel, and use link splitting/release as the preferred link-count continuation. Distillation remains optional until the explicit hybrid chain works. Canonical uniform and co-designed morphology tracks remain separate, so a graded or optimized plant can teach the controller but cannot satisfy the final seven-link gate.

Two cautions prevent the proposal from becoming optimistic evidence. First, an LQR score such as `x^T P x` is only a candidate funnel coordinate; force saturation, nonlinear dynamics, and the finite rail mean membership must be checked by uninterrupted MuJoCo rollout. Second, direct collocation is a local nonconvex optimizer and a trajectory generator, not a guarantee that a six- or seven-link feedback controller has been solved. The frozen 1,000-state gates and hanging-start end-to-end evaluations remain authoritative.

Primary technical references used for this decision:

- Taeyoung Lee, Melvin Leok, and N. Harris McClamroch, `Dynamics and Control of a Chain Pendulum on a Cart`: https://arxiv.org/abs/1211.4604
- MIT Underactuated Robotics, `Trajectory Optimization`: https://underactuated.mit.edu/trajopt.html
- MIT Underactuated Robotics, `Acrobots, Cart-Poles, and Quadrotors`: https://underactuated.mit.edu/acrobot.html

The linear-policy CEM utility was also corrected during this review. Earlier generations changed development seeds, making cross-generation scores incomparable, and moved the search center away from the best known policy. It now keeps a fixed development cohort by default and searches around the global incumbent; optional cohort resampling is explicit. At `p=0.0625`, direct scale-`1.30` LQR failed on 154 of 1,024 deterministic training states (`15.04%`). Repeating those training-only failures five times produced a 1,794-state hard-tail mixture. CEM improved its fixed weighted cohort from `43.75%` to `51.56%`, but exact evaluation on 256 untouched validation states remained `87.89%` with 31 rail hits. The gain search did not advance the `p=0.06` accepted frontier.

Nonlinear recovery probes did not recover the failed boundary states. Random-shooting MuJoCo MPC avoided the rail for eight seconds on one validation failure but let the chain fall, and a 40-generation, 96-candidate short residual-prefix search improved one training failure's maximum upright streak from `0.26 s` to `0.76 s` while still hitting the rail. These are useful trajectory diagnostics, not reusable capture teachers.

An off-policy SAC residual branch was then tested at `p=0.0625`. Vanilla SAC and weakly regularized variants collapsed the 58/64 fixed validation baseline to near zero. Strong zero-residual trust regularization preserved the baseline but recovered no failures. The original observation was dominated by constant cosine and morphology values while boundary perturbations were only `p^2` and `p^3`; adding explicitly scaled capture coordinates made the actor state-dependent. Even with that fix, trust coefficient `10` remained at `58/64` after 10,000 steps, while coefficient `1` fell to `57/64` at 10,000 and `2/64` at 20,000. SAC therefore remains an ablation branch rather than the current route forward.

The exact 1,024 training rollouts now also produce `runs/p1_capture_envelope/funnel_lqr_scale130_p00625.json`. A second-order logistic model is fit on a deterministic stratified development subset and calibrated on a hash-determined 205-state holdout from the training split only. Holdout ROC AUC is `0.990`; at the conservative acceptance threshold, precision is `0.951` and recall is `1.000`. Explicit coordinate-domain limits prevent polynomial extrapolation from labeling energetic out-of-distribution swing states as capture-ready. `search_swingup_chain.py --capture-funnel-model ...` now uses this measured probability for trajectory ranking and requires accepted membership for the swing-to-capture switch. The frozen validation and test splits remain untouched by model fitting and remain authoritative for curriculum and P1 gates.

A hanging-start CEM search against this model confirmed that the present funnel is not yet a practical swing target. After correcting the objective to ignore the reset transient and rank only states after the declared minimum handoff time, 12 generations of 32 candidates reduced normalized distance to the measured domain from `74,224` to `15,812`, but the best arrival remained `0.82 rad` from upright and the search plateaued without entering the domain. This does not rule out a stronger trajectory optimizer; it does show that tuning upright crossings against the current `p=0.0625` funnel is the wrong priority. P1 must expand the recoverable position and especially velocity domain before the coupled P2 loop becomes productive.

## Long-form browser directive review (2026-07-15)

The original signed-in conversation contains a longer Codex directive than the public share snapshot. Its useful engineering additions are dimensionless LQR coordinates, modal failure diagnostics, mode-aware reset curricula, link-growth homotopy, frontier sampling, and explicit MuJoCo reference validation. Those additions are compatible with the current roadmap and the modal/continuation items are now recorded in P1 and P5.

The directive's immediate objective is not compatible with this project's active completion contract. It asks for seven-link near-upright stabilization first and treats full swing-up as a later tier. This repository already has a six-link near-upright baseline, and the user explicitly requires hanging-start swing-up. The authoritative order therefore remains P1/P2 capture and swing handoff, P3 integrated hanging-start six-link calibration, then P4-P6 seven-link work. A seven-link S0 balance result cannot replace any of those gates.

The larger proposed rewrites are deliberately deferred:

- A custom batched planar simulator creates a second dynamics implementation and a substantial parity/Jacobian/conservation test burden. It is justified only if profiling shows MuJoCo throughput is a material blocker.
- A universal variable-link graph/transformer policy is an ablation after a specialized controller-transfer baseline exists. Introducing it during P1 would add approximation error without addressing the measured capture boundary.
- A full `n=3..12` ablation matrix and extensive new documentation are publication work after P3/P6 evidence, not prerequisites for solving the current component gate.
- LQR `V(x)` and modal coordinates remain diagnostic features. Nonlinear constrained MuJoCo rollout is authoritative because force saturation and the finite rail invalidate purely linear membership claims.

The current adaptive scheduled-target planner is a concrete instance of the model-assisted direction recommended in the conversation: it uses exact nonlinear MuJoCo rollouts to choose a short cart-target schedule, executes that schedule with state-feedback LQR, and records planning cost and uninterrupted final-rollout evidence. It is still an online state-specific planner at a narrow curriculum boundary, not a learned capture policy and not P1 evidence.

## Adaptive scheduled-target capture boundary (2026-07-15)

A state-specific teacher search parameterizes eight cart-position targets over the first two seconds and one LQR gain scale. Deterministic CEM evaluates candidates in exact MuJoCo; after the target schedule expires, the cart target returns to zero. The final evidence rollout is separate from candidate scoring and runs without reset or simulator-state overwrite.

At `p=0.0625`, individual high-budget searches recovered five of the six fixed 64-state LQR failures. A training-only batch probe recovered 14 of 32 failed states with a smaller search budget, but controller transfer was strongly state-specific. This motivated online planning rather than a hidden state-to-teacher lookup table.

Strict adaptive evaluation first executes scale-`1.30` LQR. Only baseline failures invoke the fixed-budget planner, and the selected target schedule is then replayed in one authoritative nonlinear rollout. Results:

```text
runs/p1_capture_target_teachers/eval_adaptive_validation64_p00625.json
baseline 58/64; planner recovered 5; final 63/64 (98.44%)

runs/p1_capture_target_teachers/eval_adaptive_validation256_p00625.json
baseline 224/256; planner recovered 18; final 242/256 (94.53%)

runs/p1_capture_target_teachers/eval_adaptive_validation64_p0065.json
baseline 56/64; planner recovered 3; final 59/64 (92.19%)

runs/p1_capture_target_teachers/eval_adaptive_validation256_p0065.json
baseline 215/256; planner recovered 18; final 233/256 (91.02%)
median maximum upright hold 13.94 s; no successful rail hits
```

The last result advances the accepted development frontier to `p=0.065`. Its formal gate remains false by construction: P1 requires all 1,000 frozen test states and final progress `p=1.0`. The planner used 41 invocations and about 8.4 minutes wall time for the 256-state run, so compute cost and eventual amortization into a reusable recovery controller remain open problems.

### Dimensionless modal diagnosis and escalation

The browser directive's modal-curriculum proposal was tested against the exact `p=0.065` outcomes rather than adopted on intuition. `src/gcartpole/modal.py` defines an invertible transformation from MuJoCo's relative-hinge state to dimensionless cart position, absolute link angles, cart velocity, and hinge velocities using the frozen envelope bounds. The raw closed-loop eigenvector matrix has condition number about `2.4e4`, so raw eigenvector amplitudes are numerically misleading. The authoritative diagnostic therefore uses orthonormal real-Schur blocks while retaining the eigenvalue and conditioning report.

On the original 41 planner invocations, the strongest recovery separator is a slow weakly actuated pair at `0.9908 +/- 0.0080j`. Its basis energy is concentrated in absolute angles 4–6, and its state-energy share predicts planner failure with ROC AUC `0.824`. After high-budget refinement, the same block separates the remaining 20 failures from the 21 total planner recoveries with AUC `0.869`.

A targeted feedback sweep rules out the simplest response. Multipliers from `0.25` through `3.0` on that Schur block recover no unresolved state and make the linear closed loop unstable away from a very narrow neighborhood of `1.0`. Fine multipliers `0.975` and `0.99` each recover only state `265`, which the target planner had already recovered; all other variants recover zero baseline failures. Permanent mode-gain modification is therefore not the next controller.

Deterministic budget escalation is useful for the near tail. Re-running 12 generations of 256 candidates only on the 23 unresolved states recovered states `1252`, `1008`, and `463`. The complete replayed artifact reports:

```text
runs/p1_capture_target_teachers/eval_adaptive_refined_validation256_p0065.json
initial 233/256; refinement recovered 3/23; final 236/256 (92.19%)
median maximum upright hold 13.94 s; no successful rail hits
88,256 total candidate rollouts represented in the combined planning history
```

A subsequent six-generation, 128-candidate pass initialized from each best existing schedule recovered `0/20`, confirming that local search around the two-second schedule family is saturated. The next deep-tail experiment should add a longer horizon or direct transient action/residual knots and rank candidates with dense dimensionless Lyapunov reduction. None of these partial-progress validation artifacts pass P1.

### Full-envelope direct-action and iLQR probes

The `p=1.0` capture envelope exposes a controller-authority defect that is hidden near the curriculum frontier. On validation state 674, scale-`1.30` LQR requests approximately `-78` normalized action before clipping. Adding a bounded residual to that saturated signal cannot change the applied action over most of its range. Feedback MPC therefore now has a separate planning LQR scale while retaining scale `1.30` after verified handoff. Zero-feedback direct-action CEM lowers the uninterrupted minimum dimensionless Lyapunov value from the prior `2.14M` floor to `1.76M`; a scale-`0.005` guided variant reaches only `3.63M`. Both miss the funnel and hit the rail.

`src/gcartpole/ilqr.py` adds a bounded finite-difference iLQR trajectory optimizer. It operates in the frozen envelope's dimensionless absolute-angle coordinates, uses exact MuJoCo transitions, carries a steep soft barrier near the physical rail, and returns time-varying tracking gains. `scripts/search_ilqr_capture.py` executes the planned trajectory without reset and switches to the nominal LQR stabilizer only after `V <= 1,800` and the cart handoff bound both pass.

A feasible zero-action seed with terminal-weight continuation produced the strongest full-envelope state-674 trajectory:

```text
runs/p1_capture_ilqr/validation_674_p1_feasible_seed.json
terminal V = 1,599,033; planned max |x| = 1.614 m

runs/p1_capture_ilqr/validation_674_p1_continuation10.json
terminal V = 1,125,027; planned max |x| = 1.638 m
```

Live time-varying tracking matches the planned terminal value, but neither trajectory enters the conservative funnel; switching to saturated LQR then causes a rail failure. Raising terminal weight by another factor of ten makes no progress. Preserving the three-second control timing and appending a two-second horizon lowers transient `V` to `1.01M` but regresses terminal `V` to `66.4M`. The current local iLQR formulation is therefore retained as a trajectory diagnostic and warm-start tool, not treated as a P1 controller. The next distinct branch should use box-constrained DDP/direct collocation with an explicit terminal-state constraint or learn a broad recovery policy from a diverse set of optimized feasible trajectories; further scalar weight sweeps on this local optimum are not justified.

### Box-constrained DDP and multiple shooting

The DDP backward pass now solves the scalar box subproblem at each step. If the unconstrained feedforward update exceeds the remaining normalized action interval, the update is clamped and its local feedback gain is disabled. This avoids treating rollout clipping as though it were the optimizer. The final controller records active constraint steps and uses the time-varying feedback gains during uninterrupted MuJoCo evaluation.

The first box-constrained continuation improves the original folded state-674 trajectory from `V=1.13M` to `V=430,908`, but its normalized terminal state norm remains about `64`; extrapolated `x^T P x` had been rewarding a folded, high-velocity state outside the local funnel. Adding explicit terminal dimensionless-state cost to that incumbent barely changes the basin.

Initialization from the direct-action CEM rollout changes the result qualitatively. Successive active-set DDP continuation produces:

```text
runs/p1_capture_ilqr/validation_674_p1_boxddp_from_cem.json
terminal V = 60,588; terminal ||z|| = 2.761; planned max |x| = 2.404 m

runs/p1_capture_ilqr/validation_674_p1_boxddp_from_cem_refine10x.json
terminal V = 35,148; terminal ||z|| = 1.907; planned max |x| = 2.317 m

runs/p1_capture_ilqr/validation_674_p1_boxddp_from_cem_refine100x.json
terminal V = 30,286; planned max |x| = 2.413 m
```

The middle terminal state has cart position `0.265 m`, maximum relative-angle magnitude about `0.100 rad`, cart speed `0.083 m/s`, and maximum hinge speed `0.84 rad/s`. Planned and uninterrupted tracking values agree, but none reaches the conservative `V <= 1,800` handoff; nominal LQR after the transient hits the rail. Extending the best control timing from three to four seconds with a zero-action tail regresses terminal `V` to `316M`, so that extension is rejected.

`src/gcartpole/multiple_shooting.py` and `scripts/search_multiple_shooting_capture.py` add sparse state-node/action optimization with exact MuJoCo segment defects, hard action and node-rail bounds, and DDP tracking gains for final replay. Penalized defects as small as `5.8e-3` still diverge under exact replay. Equality-constrained trust-region and SLSQP probes reduce their node objectives only by violating dynamics constraints. The implementation therefore retains only objective improvements whose maximum defect is at most `1e-8`; otherwise it returns the exact feasible warm start. No equality-solver probe improved that warm start. The main actionable result is that trajectory initialization diversity, not further scalar weighting of one local basin, enabled the `37x` reduction from `1.13M` to `30,286`.
