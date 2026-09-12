# Seven-Link Cart-Pole Swing-Up & Hold

> **7 links · noisy hanging start · 100/100 successful episodes · zero resets · ±3 m rail**

<p align="center">
  <a href="runs/swingup7_uniform/seven_link_swingup_success.mp4">
    <img src="assets/seven-link-swingup.gif" alt="Seven-link cart-pole swinging from a noisy hanging start and holding upright" width="760">
  </a>
</p>

<p align="center">
  <strong><a href="runs/swingup7_uniform/seven_link_swingup_success.mp4">Watch the full 30-second run</a></strong>
  · <a href="docs/seven_link_swingup_paper.md">Read the method paper</a>
  · <a href="runs/swingup7_uniform/seven_link_swingup_manifest.json">Inspect the controller manifest</a>
</p>

This repository demonstrates a reset-free swing-up and sustained hold of a
uniform seven-link cart-pole in MuJoCo. One frozen hybrid controller passed
both held-out gates—`20/20` and `100/100`—from the declared noisy hanging-start
distribution. The full evidence bundle, controller hashes, negative controls,
and reproduction commands are public.

## The seven-link record

| Benchmark property | Result |
|---|---:|
| Links | **7** |
| 20-episode gate | **20/20 (100%)** |
| 100-episode gate | **100/100 (100%)** |
| Resets inside episodes | **0** |
| Rail failures in the 100-episode gate | **0** |
| First upright | **14.54 s** |
| Upright hold after first upright | **15.48 s** |
| Maximum cart excursion across 100 episodes | **2.3709 m** |

The frozen contract uses a uniform seven-link plant, continuous `±80 N`
force, a `±3 m` rail, 50 Hz control, a noisy hanging start, and a five-second
upright requirement. This is the repository's canonical benchmark record. It
has not yet been certified against an external competition's exact plant and
submission rules, so we do not present it as a universal world-record claim.
See the [fairness notes](docs/yacine_fairness_notes.md) and
[public release checklist](docs/public_release_checklist.md) for that boundary.

## How it works

```text
noisy hanging start
        │
        ▼
10 s hanging-equilibrium LQR conditioning
        │
        ▼
4.56 s Box-FDDP swing route with time-varying feedback
        │
        ▼
terminal LQR capture and upright hold
```

The conditioning phase damps the initial perturbation and recenters the cart
without changing simulator state. A finite-horizon Box-FDDP controller then
executes the swing route, and a terminal LQR takes over only after the chain
arrives quiet enough to hold. The complete explanation—including the state
coordinate bug, the failed approaches, and the decisive robustness changes—is
in **[Seven-Link Cart-Pole Swing-Up: A Settled-Launch Two-Expert Controller](docs/seven_link_swingup_paper.md)**.

## Evidence

| Artifact | What it establishes |
|---|---|
| [Held-out video](runs/swingup7_uniform/seven_link_swingup_success.mp4) | 30-second noisy-start replay, reset-free, all seven links visible |
| [Video metadata](runs/swingup7_uniform/seven_link_swingup_success.video.json) | Seed, runtime, frame count, reset count, final metrics, and hashes |
| [20-episode evaluation](runs/swingup7_uniform/eval_swingup7_20.json) | First canonical acceptance gate: 20/20 |
| [100-episode evaluation](runs/swingup7_uniform/eval_swingup7_100.json) | Final canonical acceptance gate: 100/100 |
| [No-settle negative control](runs/eval_swingup7_fddp_two_expert_canonical20.json) | Same noisy-start gate without conditioning: 0/20 |
| [Controller manifest](runs/swingup7_uniform/seven_link_swingup_manifest.json) | Conditioning, swing, capture, switching rules, preprocessing, and hashes |
| [FDDP route](runs/swingup7_fddp_full_hanging_ilqr_terminal100k_deferred_lqr1.json) | Nominal states, controls, and feedback gains |
| [Method paper](docs/seven_link_swingup_paper.md) | Benchmark, method, results, limitations, and reproduction |

## Current frontier: eight links

Seven links is the established result; **eight links is active research and is
not solved yet**. The current campaign explores direct uniform optimization,
gradient morphology continuation, and ghost-link homotopy. Its configs and
diagnostic artifacts are included so progress and failures remain auditable:

- [`configs/swingup8_uniform.yaml`](configs/swingup8_uniform.yaml)
- [`configs/swingup8_gradient_discovery.yaml`](configs/swingup8_gradient_discovery.yaml)
- [`configs/swingup8_ghost_continuation.yaml`](configs/swingup8_ghost_continuation.yaml)
- [`ROADMAP.md`](ROADMAP.md)

The project advances one link only after the same evidence bundle passes.

## Reproduce the result

Requirements: Python 3.10+, MuJoCo, and a local environment capable of running
Crocoddyl/Box-FDDP. The recorded setup targets Apple Silicon; see
[`requirements-fddp.txt`](requirements-fddp.txt) and
[`scripts/setup_aligator.sh`](scripts/setup_aligator.sh) for the planner stack.

```bash
git clone https://github.com/corbensorenson/cartpole.git
cd cartpole

make setup
make roadmap-p0
make eval-swingup7-20
make eval-swingup7-100
```

For the exact settled-launch replay command, seeds, and output contract, follow
the [paper's reproduction section](docs/seven_link_swingup_paper.md#5-reproduction).

## Repository map

| Path | Purpose |
|---|---|
| [`src/gcartpole`](src/gcartpole) | MuJoCo environment, controllers, optimizers, and evidence code |
| [`configs`](configs) | Frozen benchmark and curriculum configurations |
| [`scripts`](scripts) | Training, search, evaluation, replay, and rendering entry points |
| [`tests`](tests) | Dynamics, optimizer, morphology, and evidence-contract tests |
| [`docs`](docs) | Paper, roadmap support, experiment ledger, and reproduction notes |
| [`runs/swingup7_uniform`](runs/swingup7_uniform) | Curated public seven-link evidence bundle |

Research history is intentionally preserved, including negative results. Start
with the [method paper](docs/seven_link_swingup_paper.md) for the clean narrative
and [`docs/levers_and_pitfalls.md`](docs/levers_and_pitfalls.md) for the full
experiment ledger.
