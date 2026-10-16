# 1D cart-pole · results and reproduction

[← Project overview](../README.md) · [2D spatial benchmark](../2d/README.md) ·
[Development workflow](../DEVELOPMENT.md)

The 1D benchmark uses one bounded force on a rail cart and a passive planar
hinge chain. **Fourteen is the highest accepted published count.** Fifteen has
an explicitly unsolved partial result. This index separates those results from
the independently developed 2D cart / spatial ball-joint benchmark.

## Published results

| Links | Result / technical appendix | Frozen evidence | Full video |
| :--- | :--- | :--- | :--- |
| 7 | [Seven-link paper](seven_link_swingup_paper.md) | [Manifest](../runs/swingup7_uniform/seven_link_swingup_manifest.json) | [Video](../runs/swingup7_uniform/seven_link_swingup_success.mp4) |
| 8 | [Eight-link appendix](eight_link_swingup_paper.md) | [Manifest](../runs/generalized_solver/eight_link_swingup_manifest.json) | [Video](../runs/generalized_solver/eight_link_swingup_success.mp4) |
| 9 | [Nine-link appendix](nine_link_swingup_paper.md) | [Manifest](../runs/generalized_solver/nine_link_swingup_manifest.json) | [Video](../runs/generalized_solver/nine_link_swingup_success.mp4) |
| 10 | [Ten-link appendix](ten_link_swingup_paper.md) | [Manifest](../runs/generalized_solver/ten_link_swingup_manifest.json) | [Video](../runs/generalized_solver/n10_fddp_refined_route_feedback100_park16_targetm005.mp4) |
| 11 | [Eleven-link appendix](eleven_link_swingup_paper.md) | [Manifest](../runs/swingup11_uniform/eleven_link_swingup_manifest.json) | [Video](../runs/swingup11_uniform/eleven_link_swingup_success.mp4) |
| 12 | [Twelve-link appendix](twelve_link_swingup_paper.md) | [Manifest](../runs/swingup12_uniform/twelve_link_swingup_manifest.json) | [Video](../runs/swingup12_uniform/twelve_link_swingup_success.mp4) |
| 13 | [Thirteen-link appendix](thirteen_link_swingup_paper.md) | [Manifest](../runs/swingup13_uniform/thirteen_link_swingup_manifest.json) | [Video](../runs/swingup13_uniform/thirteen_link_swingup_success.mp4) |
| **14** | [Fourteen-link appendix](fourteen_link_swingup_paper.md) | [Manifest](../runs/swingup14_uniform/fourteen_link_swingup_manifest.json) | [Video](../runs/swingup14_uniform/fourteen_link_swingup_success.mp4) |
| 15 · partial | [Unsolved: 3.00 s nominal hold](fifteen_link_partial_result.md) | [Manifest](../runs/fifteen_link_partial_20261015/manifest.json) | [Full failed attempt](../runs/fifteen_link_partial_20261015/fifteen_link_partial_full_attempt.mp4) |

The [consolidated 7–10 paper](../output/pdf/seven_to_ten_link_swingup_paper.pdf)
and individual appendices retain their original methods, cohort definitions,
limitations and negative controls. Read each manifest for the precise plant,
source and validation conditions; earlier results are not retroactively
relabelled under a later protocol.

Tagged releases for the recent frontier:

- [Eleven](https://github.com/corbensorenson/cartpole/releases/tag/eleven-link-canonical-20261001)
- [Twelve](https://github.com/corbensorenson/cartpole/releases/tag/twelve-link-canonical-20261002)
- [Thirteen](https://github.com/corbensorenson/cartpole/releases/tag/thirteen-link-canonical-20261002)
- [Fourteen](https://github.com/corbensorenson/cartpole/releases/tag/fourteen-link-canonical-20261002)
- [Fifteen partial, explicitly unsolved](https://github.com/corbensorenson/cartpole/releases/tag/fifteen-link-partial-20261015)

## Current qualification contract

The canonical higher-count plant has a uniform 3 m / 1 kg chain, a 1 kg cart,
±80 N force and ±3 m rail. MuJoCo RK4 uses 0.005 s physics steps; a command is
held for four steps, giving 50 Hz control. Delivered actions are float32;
physical dynamics and stored states are binary64.

A successful episode starts from the declared hanging-state distribution,
uses no mid-episode state reset, finishes all 30 seconds with valid simulation
and no rail exit, and holds **all absolute cumulative world link angles strictly below
0.15 rad for at least 250 consecutive post-command states** (five seconds).
A temporary hold followed by a rail failure does not pass.

Fourteen's frozen controller passes reserved noisy 20/20, a disjoint noisy
100/100, exact 20/20, and one held-out video episode. Every episode holds for
8.16 s, with maximum cart excursion 1.863168715 m. The qualification follows
source/controller freeze and does not tune against reserved cohorts.

See [ROADMAP.md](../ROADMAP.md) for the full 1D campaign contract and the
[release checklist](public_release_checklist.md) for packaging context. The
frozen release manifest and appendix are authoritative for a specific result.

## Reproduce the fourteen-link release

1. Read the [appendix](fourteen_link_swingup_paper.md#frozen-evidence-and-reproduction)
   and [manifest](../runs/swingup14_uniform/fourteen_link_swingup_manifest.json).
2. Obtain the [release](https://github.com/corbensorenson/cartpole/releases/tag/fourteen-link-canonical-20261002),
   including the [source bundle](../runs/swingup14_uniform/fourteen_link_source.bundle),
   source metadata, controller/configuration files, hashes and command file.
3. Clone the bundle into a new directory. Its frozen source commit is
   `3277b4173a9333819bf35acb848e384f52a227ec`. Recreate the runtime versions
   recorded with the release. Historical synthesis uses the Crocoddyl/BoxFDDP
   environment; other root tools also use separate Aligator and learning stacks.
4. Follow [reserved_commands.json](../runs/swingup14_uniform/reserved_commands.json),
   adapting local interpreter and output paths. Preserve controller/config bytes
   and numerical settings. Use one BLAS/OpenMP thread and the recorded Python
   import paths (`PYTHONPATH=.:src:scripts` on the recorded Unix-style runtime).
5. Compare complete episode metrics, replay and hashes against the manifest.
   Keep new outputs separate from the frozen evidence.

The original environment files are [requirements-mac.txt](../requirements-mac.txt),
[requirements-fddp.txt](../requirements-fddp.txt), and
[environment-aligator.yml](../environment-aligator.yml). Match the chosen
release's metadata before using them. Reproducing a saved controller is a
different claim from independently regenerating its trajectory and gains.

## Fifteen-link partial evidence

The published best nominal hold is 150 commands / 3.00 s, below the required
250 commands. Every published attempt later leaves the rail. Development used
reused nominal starts; no final noisy or reserved cohort passes.

The [partial-result verifier](../scripts/verify_fifteen_partial.py) checks saved
artifact integrity, tapes and reported arithmetic without running a new native
simulation. From a Python environment with NumPy, at the repository root:

```powershell
python scripts/verify_fifteen_partial.py --checkout .
```

A passing integrity audit confirms the partial evidence's consistency. It does
not qualify fifteen links. See the [partial-result appendix](fifteen_link_partial_result.md)
for the complete claim boundary and failed-attempt video.

## Method and research navigation

- [Shared generalized solver](generalized_solver.md): architecture and synthesis tools.
- [Eleven-link frontier review](eleven_link_frontier_review.md): diagnostics and qualification gaps at that stage.
- [Twelve](twelve_link_development.md), [thirteen](thirteen_link_development.md), and
  [fourteen](fourteen_link_development.md): development chronology and failures.
- [Experiment protocol](experiment_protocol.md) and [evidence runbook](evidence_runbook.md): research context.
- [Campaign roadmap](../ROADMAP.md): the sequential 1D target through twenty.

Historical method notes describe the experiments and frontiers at the time they
were written. Use the [project overview](../README.md) and this result index for
current published status.
