# Fifteen-link development: 3.00 seconds upright, unsolved

Published October 15, 2026. **Fourteen remains the highest accepted canonical
link count.** This is a partial-result evidence package, not a successful
fifteen-link release.

The best recorded nominal fifteen-link hold is **150 consecutive post-command
states, or 3.00 seconds at 50 Hz**, with every absolute cumulative link angle
strictly below 0.15 rad. The acceptance gate requires 250 such states (5 seconds),
an uninterrupted 1,500-command (30-second) episode within the rail, and the
established frozen noisy/reserved validation and video gates. None of the
published fifteen-link cases passes the full nominal gate.

## What the saved episodes show

| Saved case | Best upright hold | Episode until rail failure | Maximum absolute cart position |
|---|---:|---:|---:|
| Original minimax continuation | 3.00 s | 9.52 s | 3.011613 m |
| Original minimum-change continuation | 3.00 s | 9.48 s | 3.064075 m |
| Later analytic-derivative study: baseline | 3.00 s | 10.88 s | 3.015751 m |
| Later analytic-derivative study: static protected center | 3.00 s | 10.12 s | 3.075214 m |

All four start from the exact nominal hanging state with zero velocities. They
reuse the development initialization associated with seed 20265100. They are
not independent noisy trials, a reserved cohort, or evidence of robustness.
The later cases do not improve the hold record. Complete actual state/action
tapes, including their terminal failures, are included.

The unchanged canonical plant has fifteen equal links totaling 3 m and 1 kg,
a 1 kg cart, 0.025 m link radius, 0.0005 joint armature, total link damping
0.015, and cart damping 0.02. Force is bounded by ±80 N and cart travel by ±3 m.
MuJoCo uses float64 state integration with RK4, a 0.005-second physics step
and four steps per command. Delivered normalized commands are float32 at
50 Hz; tapes retain those exact values as float64. Higher-rate and altered-plant
studies elsewhere in the research archive do not change these claims.

The observed limitation is loss of capture followed by rail failure. These
episodes do not establish physical impossibility or identify a unique cause.
Reaching a short upright interval is insufficient to establish a maintainable
bounded-force route. Further work must extend capture while preserving the
rail and delivered-command constraints, then pass the full validation gates.

## Video

[Watch the complete minimax attempt](../runs/fifteen_link_partial_20261015/fifteen_link_partial_full_attempt.mp4).
The video renders all 477 saved states in order at 50 frames per second, from
the hanging initial state through the out-of-rail terminal state at 9.52 s.
Its encoded duration is 9.54 s because the initial state occupies one frame.
It uses a schematic drawn from the recorded joint angles without integrating
new physics, interpolating states, or resetting the episode. The GIF is a
25-frame-per-second preview using every second state; use the MP4 for the
complete sequence. Both identify the result as unsolved.

## Inspecting the evidence

The [manifest](../runs/fifteen_link_partial_20261015/manifest.json) binds each
published original artifact to its retained research path and SHA-256 hash.
The package includes two selected 441-command forecast tapes, the original
native controller design, configuration, historical execution/accounting
records, and 812 original frozen source/configuration files in
`original_synthesis_source.tar.gz`. The two later complete tapes include their
historical verification and cold-replay reports. Their entire synthesis/search
archives remain preserved locally; this selected public package does not
pretend to contain every research artifact or provide independent rediscovery.

The original continuation used 396,615 inclusive native transition calls
(1,586,460 physics substeps), including 950 actual command ticks. The older
395,665 search count excludes those 950 ticks. These counts describe that
historical execution, not the entire fifteen-link research campaign.
Historical reports whose audit says `passed` mean that evidence checks passed;
they do not mean fifteen links passed the cart-pole task.

From a checkout with Python and NumPy:

```sh
python scripts/verify_fifteen_partial.py
```

This verifies artifact hashes, finite full tapes, nominal initialization,
float32 action delivery, upright/rail metrics, and the two selected forecast
prefixes. It reports `evidence_integrity_passed: true` and
`canonical15_solved: false`. It adds zero native integration calls and is not a
fresh simulator replay. The bundled historical replay reports state their own
scope. The saved current audit is
`runs/fifteen_link_partial_20261015/saved_tape_verification.json`.

To regenerate the schematic video, additionally install Pillow, imageio and
imageio-ffmpeg, then run `python scripts/render_fifteen_partial.py`. Fonts and
encoder versions can change rendered bytes; the delivered video is separately
hashed. `SHA256SUMS` covers the published package, documentation, preview and
audit/render scripts.
