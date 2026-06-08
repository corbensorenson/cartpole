# Swing-Up Search Notes

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
