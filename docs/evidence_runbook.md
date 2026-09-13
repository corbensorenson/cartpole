# Evidence Runbook

This runbook covers the released seven-link evidence pipeline and the earlier
six-link calibration work. The authoritative benchmark contract and frontier
rules remain in `ROADMAP.md`.

## Seven-link benchmark and final audit

Freeze and test the benchmark before accepting training results:

```bash
make roadmap-p0
```

The released settled-launch hybrid path uses:

```bash
make eval-swingup7-20
make eval-swingup7-100
make render-swingup7
make verify-swingup7
```

To regenerate the complete bundle from one clean tracked commit, use:

```bash
make release-swingup7
```

This writes disjoint 20- and 100-episode cohorts, an independently seeded
video, a non-canonical robustness sweep, `SHA256SUMS`, and a verifier report.
`runs/swingup7_uniform/seven_link_swingup_manifest.json` is the authoritative
policy manifest. Its top-level contract is:

```json
{
  "schema_version": 2,
  "claim_status": "released_canonical_noisy_swingup_and_hold",
  "architecture": "settled_launch_hybrid",
  "benchmark": {
    "config_sha256": "...",
    "generated_xml_sha256": "..."
  },
  "controller": {"path": "...", "sha256": "...", "route_steps": 228},
  "experts": {"conditioning": {}, "swing": {}, "capture": {}},
  "switch": {"state_reset_at_phase_boundaries": false},
  "evaluation": {
    "twenty_seed_start": 30732,
    "hundred_seed_start": 40732,
    "video_seed": 50732
  }
}
```

Paths are repository-relative. The verifier requires complete per-episode
metrics, non-overlapping cohorts, an independent video seed, clean tracked
source provenance, the controller and manifest hashes, runtime metadata, all
15 robustness scenarios, and a reset-free video ending at a successful time
limit. Historical diagnostics marked `not_claim` cannot satisfy this release
contract.

## Eight-link parked-route promotion

The eight-link result reuses the same evidence discipline with a saved
feedback route and an explicit hanging-cart park:

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
export OMP_NUM_THREADS=1
export PYTHONPATH=gradient_cartpole_handoff/.conda-aligator/lib/python3.12/site-packages:src:scripts
PY=gradient_cartpole_handoff/.conda-aligator/bin/python

$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup8_uniform.yaml \
  --controller runs/generalized_solver/n8_capture_fddp_feedback120.json \
  --episodes 20 --seed 81801 --park-seconds 14 --cart-target -0.15 \
  --tracking-gain-scale 0.75 \
  --out runs/generalized_solver/n8_fddp_parked_target015_14s_noisy20.json

$PY scripts/evaluate_fddp_parked_route.py \
  --config configs/swingup8_uniform.yaml \
  --controller runs/generalized_solver/n8_capture_fddp_feedback120.json \
  --episodes 100 --seed 80801 --park-seconds 14 --cart-target -0.15 \
  --tracking-gain-scale 0.75 \
  --out runs/generalized_solver/n8_fddp_parked_target015_14s_noisy100.json
```

The 20-episode seeds `81801--81820`, the 100-episode seeds `80801--80900`,
and the video seed `90901` are disjoint. The complete eight-link contract,
including the zoomed-out video command and hashes, is in
`docs/eight_link_swingup_paper.md` and
`runs/generalized_solver/eight_link_swingup_manifest.json`.

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

For current unsolved search notes, including a zero-noise trajectory probe that reaches the upright angle threshold once but does not capture, see `docs/swingup_search_notes.md`.

## External Packet Review And Discovery Gate

The review in `docs/packet_review.md` records four supporting packets. The frontier study is useful for global-discovery protocol and exact replay discipline, but its non-MuJoCo nonuniform seven/eight-link benchmark is not comparable evidence. The hybrid study is useful for candidate freezing, resource accounting, and risk budgeting, but its local 3/5-link results do not establish hanging-start swing-up. The harmonic study supplies phase/energy and morphology diagnostics, and the Cart-Pole Research Master Handoff supplies independent replay, task identity, and failure-classification patterns. Neither supplies canonical MuJoCo evidence; the master handoff's own MuJoCo gate is blocked.

Before extending verification or large policy cohorts, run the Global Discovery Feasibility Gate from `ROADMAP.md`: exact uniform six-link MuJoCo, hanging start, one five-second feedback-controlled hold, and three held-out hanging-start replays. Store full state/action trajectories and hashes. A collocation endpoint, near-upright result, or low-fidelity success is only a proposal until exact MuJoCo replay passes.
