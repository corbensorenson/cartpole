# Cart-Pole · 1D & 2D Control

**Swinging up passive chains with bounded cart forces in MuJoCo.**

Two control benchmarks share this repository: a cart on a **1D rail** carrying a
planar hinge chain, and a cart on a **2D plane** carrying a spatial ball-joint
chain. Both start hanging, use full-state feedback, and must finish an
uninterrupted episode within the cart boundary.

[![1D: 14 links qualified](https://img.shields.io/badge/1D-14_links_qualified-2563eb)](#1d-cart-pole)
[![2D: 13 links qualified](https://img.shields.io/badge/2D-13_links_qualified-7c3aed)](#2d-cart-pole)
[![MuJoCo 3.3.5](https://img.shields.io/badge/MuJoCo-3.3.5-059669)](https://mujoco.readthedocs.io/en/3.3.5/)

[1D results](#1d-cart-pole) · [2D results](#2d-cart-pole) ·
[Get started](#get-started) · [Repository map](#repository-map) ·
[Development](DEVELOPMENT.md)

<table>
  <tr>
    <th width="50%">1D cart · planar hinge chain</th>
    <th width="50%">2D cart · spatial ball-joint chain</th>
  </tr>
  <tr>
    <td><a href="https://github.com/corbensorenson/cartpole/releases/tag/fourteen-link-canonical-20261002"><img src="assets/fourteen-link-swingup.gif" alt="Fourteen passive planar links swing up and hold; preview runs at twice speed" width="360"></a></td>
    <td><a href="https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-13-link-v1"><img src="2d/videos/2d-cartpole-13-link-preview.gif" alt="Thirteen passive spatial links swing up and hold on an unrestricted XY cart" width="360"></a></td>
  </tr>
  <tr>
    <td><strong>14 links qualified.</strong> Full 30-second episodes; 8.16 seconds upright. Preview at 2× speed.</td>
    <td><strong>13 links qualified.</strong> Full 30-second episodes; 5.16 seconds upright after 19 seconds hanging preparation. Preview at normal speed.</td>
  </tr>
  <tr>
    <td><a href="runs/swingup14_uniform/fourteen_link_swingup_success.mp4">Full video</a> · <a href="docs/fourteen_link_swingup_paper.md">Technical appendix</a> · <a href="https://github.com/corbensorenson/cartpole/releases/tag/fourteen-link-canonical-20261002">Release</a></td>
    <td><a href="https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-13-link-v1/2d-cartpole-13-link-swingup.mp4">Full video</a> · <a href="2d/README.md">Method & evidence</a> · <a href="https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-13-link-v1">Release</a></td>
  </tr>
</table>

*Published status: October 16, 2026. Link counts belong to their respective
benchmarks; their plants and qualification cohorts differ.*

## 1D cart-pole

One force moves a cart along a finite rail. The links are passive revolute
joints, so the chain moves in a vertical plane. The canonical plant has a
3 m / 1 kg uniform chain, a 1 kg cart, **±80 N** force and **±3 m** rail.
Control runs at **50 Hz**, with native MuJoCo RK4 integration at 200 Hz.

**Seven through fourteen links have published accepted results.** The latest,
fourteen, passes reserved noisy **20/20**, disjoint noisy **100/100**, exact
**20/20**, and a held-out video episode. All 141 episodes finish 30 seconds and
hold every link's absolute world angle below 0.15 rad for 8.16 consecutive
seconds. Maximum cart excursion is 1.863169 m.

The controller combines a settled launch, a trajectory with time-varying
feedback, and upright capture/maintenance. Higher-count synthesis transfers
predecessor routes, repairs them on the exact target plant, and rebuilds
feedback from physical execution. Delivered actions are float32; simulation
and stored physical states are binary64. The fourteen-link capture gain uses
100-digit Riccati design. Saved-controller validation and automatic
re-synthesis are separate claims.

| Milestone | Status | Evidence |
| :--- | :--- | :--- |
| 7–10 links | Accepted; historical regression references | [Consolidated paper](output/pdf/seven_to_ten_link_swingup_paper.pdf) · [Result index](docs/one-dimensional-guide.md#published-results) |
| 11–13 links | Accepted; full release bundles | [11](docs/eleven_link_swingup_paper.md) · [12](docs/twelve_link_swingup_paper.md) · [13](docs/thirteen_link_swingup_paper.md) |
| **14 links** | **Highest accepted 1D count in this repository** | [Appendix](docs/fourteen_link_swingup_paper.md) · [Manifest](runs/swingup14_uniform/fourteen_link_swingup_manifest.json) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/fourteen-link-canonical-20261002) |
| 15 links | **Unsolved:** published nominal partial hold of 3.00 s | [Failure analysis](docs/fifteen_link_partial_result.md) · [Full attempt](runs/fifteen_link_partial_20261015/fifteen_link_partial_full_attempt.mp4) |

The fifteen-link partial result is below the required five-second hold. Every
published attempt later exits the rail before the full episode ends; no noisy
or reserved fifteen-link cohort has passed. The [1D campaign](ROADMAP.md)
targets sequential qualification through twenty links.

**Explore 1D:** [Results & reproduction guide](docs/one-dimensional-guide.md) ·
[Shared solver](docs/generalized_solver.md) · [Campaign contract](ROADMAP.md)

## 2D cart-pole

Two forces move a cart across an XY plane. Each link has an **unrestricted
passive ball joint**: it can tilt in both horizontal directions and twist.
This is spatial motion, with quaternion orientations and tangent-rotation
feedback differences.

The uniform chain is 3 m / 1 kg and the cart is 1 kg. The **total XY force norm
is capped at 80 N**, and the cart must stay inside a **6 m × 6 m square**,
including physics substeps. Commands run at 50 Hz; RK4 integrates at 200 Hz.
Success requires a complete 30-second episode and every rod's world tilt below
0.15 rad for five consecutive seconds. Free twist is not scored as tilt.

| Milestone | Fresh noisy qualification | Upright hold | Evidence |
| :--- | :--- | :--- | :--- |
| 1 link | **32/32** complete episodes | 21.74 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-1-link-v1/2d-cartpole-1-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-1-link-v1) |
| 2 links | **32/32** complete episodes | 21.68 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-2-link-v1/2d-cartpole-2-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-2-link-v1) |
| 3 links | **32/32** fresh complete episodes | 6.78 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-3-link-v1/2d-cartpole-3-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-3-link-v1) |
| 4 links | **32/32** fresh complete episodes | 7.22 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-4-link-v1/2d-cartpole-4-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-4-link-v1) |
| 5 links | **32/32** fresh complete episodes | 21.70 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-5-link-v1/2d-cartpole-5-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-5-link-v1) |
| 6 links | **32/32** fresh complete episodes | 14.06 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-6-link-v1/2d-cartpole-6-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-6-link-v1) |
| 7 links | **32/32** fresh complete episodes | 13.48 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-7-link-v1/2d-cartpole-7-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-7-link-v1) |
| 8 links | **32/32** fresh complete episodes | 14.20 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-8-link-v1/2d-cartpole-8-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-8-link-v1) |
| 9 links | **32/32** fresh complete episodes | 8.18 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-9-link-v1/2d-cartpole-9-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-9-link-v1) |
| 10 links | **32/32** fresh complete episodes | 6.12 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-10-link-v1/2d-cartpole-10-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-10-link-v1) |
| 11 links | **32/32** fresh complete episodes | 5.50 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-11-link-v1/2d-cartpole-11-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-11-link-v1) |
| 12 links | **32/32** fresh complete episodes | 5.50 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-12-link-v1/2d-cartpole-12-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-12-link-v1) |
| **13 links** | **32/32** fresh complete episodes | 5.16 s | [Full video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-13-link-v1/2d-cartpole-13-link-swingup.mp4) · [Release](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-13-link-v1) |
| 14 links | In development | — | [Progress log](2d/PROGRESS.md) |

Native MuJoCo finite differences feed trajectory optimization and sparse shooting
repair. One through ten track the swing and hand off to upright discrete LQR.
Eleven through thirteen retain affine offsets and design a finite upright continuation.
The two-link result uses world tilt/rate coordinates for stationary feedback;
the failed local-coordinate handoff is preserved alongside it.

Three links uses 12 seconds of hanging feedback, measured axial-frame alignment,
an 11.28-second swing, and upright feedback. Every physical state evolves under
legal cart forces; no joint lock or state reset is introduced. The unconditioned
route failed its first qualification at 25/32. Those failures remain preserved;
the frozen conditioned policy passes a new, disjoint 32-start cohort.

Four links uses native affine tracking repair to adapt a historical motion to
the exact spatial benchmark mechanics. It starts its 23.04-second swing
immediately from the noisy hanging state, then switches to world-coordinate
upright feedback. The frozen first repaired checkpoint passes nominal plus
12 development starts and a fresh 32-start cohort. Altered-mechanics components
and failed alternatives remain separate in the evidence.

Five links uses the same native tracking repair on an 8.42-second historical
motion proposal, correcting its mechanical mismatch and bringing the inherited
3.33m cart excursion inside the unchanged 3m boundary. It passes nominal plus
12 development starts and a fresh 32-start cohort without preliminary conditioning.

Six links uses 12 seconds of hanging feedback and measured axial-frame alignment
before a 4.18-second repaired swing. All phases fit inside the same 30-second
episode. The raw controller failed four of 12 noisy starts; the conditioned
policy passes all 44 development starts and a new 32-start qualification cohort.

**Why some videos look planar:** the plant has independent XY cart motion and
unrestricted ball joints, but the four- through thirteen-link nominal swing references
are essentially planar. Y feedback corrects spatial disturbances. The
[independent physics audit](2d/audits/spatial_freedom_001) checks every joint's
transverse freedom, passive spin and persistent native two-axis dynamics.
Its [controlled six-link XY-loop video](2d/audits/spatial_freedom_001/controlled_xy_orbit.mp4)
shows deliberate motion across both axes while keeping all six rods upright.

Seven links passes nominal plus 76 noisy development starts, then a NEW 32-start
qualification cohort. Its raw predecessor failed at 31/32; those seeds were
reused only as development. The final policy combines 12 seconds hanging
preparation, a 4.56-second repaired swing and upright feedback within the same
30-second clock. All world tilts stay below 0.15rad for 13.48 consecutive seconds.

Eight links uses 12 seconds hanging preparation followed by a 3.94-second native
repaired swing and upright feedback. The raw policy failed all 12 noisy starts;
the conditioned policy passes nominal plus 44 development starts and 32 fresh
qualification starts. All complete 30 seconds with 14.20 seconds upright.

Nine links uses 18 seconds hanging preparation followed by a 3.94-second native
repaired swing and upright feedback. The raw controller again failed all 12
noisy starts; the frozen prepared controller passes nominal plus 44 development
starts and 32 fresh qualification starts. Every final episode holds upright
for 8.18 seconds. Maximum cart coordinate is 2.716248m. Native physical-freedom
regression checks now include nine links; 52 tests pass.

Ten links uses 16 seconds hanging preparation, an 8-second native repaired
swing and upright feedback. Its raw policy failed all 12 noisy starts; the
prepared frozen controller passes nominal plus 44 development starts and a
new 32-start qualification cohort. Every final episode holds for 6.12 seconds,
with maximum cart coordinate 2.022729m and a minimum substep boundary margin
of 0.977271m. Native ten-link freedom checks bring the suite to 56 tests.

Eleven links uses full affine QR feedback with a finite upright continuation and
18 seconds of hanging preparation. Axial reference frames are aligned at each
observed feedback step while physical twist and spin remain free. Nominal plus
44 development starts and32 fresh reserved starts pass complete30-second
episodes with5.50 seconds upright. It executes a prefix of its full30-second
feedback design; design time adds no episode time. Maximum cart coordinate is
2.072389m and the minimum substep boundary margin is0.927587m. The85-test
suite includes eleven-link freedoms and axial-frame feedback invariance.

Twelve links uses the same18s preparation and per-command axial reference
alignment, with a9s repaired swing and3s upright tracking. It passes nominal
plus44 development starts then32 fresh reserved starts, all full30s with5.50s
hold and exact replay. Maximum cart coordinate is1.914864m and minimum substep
margin1.085024m. The96-test source includes12-link freedoms, factored QR design
and explicit donor-coordinate regression. Earlier incorrectly decoded donor
experiments are retained with correction notes; the accepted controller uses
the corrected source matrix and independent stored physical-tape checks.

Thirteen transfers a material angle/rate field, repairs it on the native spatial
plant and prepares for19s. Two fixed reference offsets each pass44/45 development
episodes with different failures. The selected controller forecasts both complete
continuations from the observed launch state and takes the first passing one,
without a seed lookup or live-state reset. It passes45/45 development and32/32
fresh qualification starts, all30s/5.16s hold/exact cold replay. Every selected
forecast matches actual suffix states and forces byte-for-byte. Maximum recorded
cart coordinate1.883801m; minimum substep margin1.115844m. The131-test suite
includes predictor isolation and archive integrity. Prediction cost is retained;
no hardware real-time claim is made. The current ZIP contains complete current
source and new evidence; unchanged historical tapes are hash-bound to immutable
prior bundles and remain in this Git checkout. Fourteen is the next count.

The final qualification uses fresh 3D hanging-start perturbations and checks
actual float32 forces, full state/action tapes, and independent replay.
Near-upright balance studies have their own results and do not promote a
hanging-start swing-up count.

**Explore 2D:** [Setup, physics & method](2d/README.md) ·
[Progress](2d/PROGRESS.md) · [Research plan](2d/research/plan.md) ·
[Frozen evidence](2d/evidence)

## Get started

### Watch or inspect a result

Start with a release above for the full video and downloadable evidence.
The [1D guide](docs/one-dimensional-guide.md) indexes historical results and
explains frozen-source reproduction. The [2D guide](2d/README.md) includes
installation, controller evaluation, and video commands.

### Run the 2D environment

From the repository root on Windows, create its separate Python 3.11 environment:

```powershell
cd 2d
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install -e .
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
.\.venv\Scripts\python.exe -m pytest -q
```

Then follow the [frozen-controller evaluation commands](2d/README.md#run-on-windows).
Use a new output directory for each run.

### Reproduce a 1D release

Use that release's **frozen source bundle, recorded runtime, and command file**.
The latest repository combines ongoing work and several optimizer environments;
installing one generic environment is not a substitute for a release's recorded
setup. Begin with the [fourteen-link reproduction notes](docs/one-dimensional-guide.md#reproduce-the-fourteen-link-release).

## Repository map

| Path | Purpose |
| :--- | :--- |
| [`src/gcartpole/`](src/gcartpole) | 1D environment and controllers |
| [`configs/`](configs) · [`scripts/`](scripts) | 1D plants, synthesis, evaluation and verification |
| [`docs/`](docs) | 1D papers, method notes and evidence guides |
| [`runs/`](runs) | Reviewed public 1D controllers, manifests, tapes and videos |
| [`2d/src/spatialpole/`](2d/src/spatialpole) | 2D spatial environment and controller tools |
| [`2d/evidence/`](2d/evidence) · [`2d/videos/`](2d/videos) | Frozen 2D evidence and full videos |
| [`2d/research/`](2d/research) | 2D experiment contracts and research records |
| [`tests/`](tests) · [`2d/tests/`](2d/tests) | Each track's tests |
| [`DEVELOPMENT.md`](DEVELOPMENT.md) | Contribution, worktree and publication workflow |

## Evidence and citation

Accepted results include frozen controllers and source identities, declared
cohorts, complete episodes, and full videos. Failed attempts and partial results
keep their original status. A passing reset-noise cohort establishes performance
on that distribution; broader disturbance robustness and indefinite balance
require separate experiments.

Use [CITATION.cff](CITATION.cff) to cite the project, and include the relevant
release tag or source commit when reporting a numerical result.
