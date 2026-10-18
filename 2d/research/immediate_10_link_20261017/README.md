# Ten-link 2D cartpole: immediate swing-up

**32/32 new reserved starts**, after 46/46 development starts. Every episode runs
30 seconds and holds every link within .15 radians of world upright for at least
22.12 seconds. Hanging preparation is zero.
[Full first-reserved video and complete evidence](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-10-link-immediate-v1).

The cart moves freely in XY; every passive joint is a spatial ball joint. The
original 3m/1kg chain, 1kg cart, 80N actuator disk, +/-3m boundary, 50Hz controls
and 200Hz RK4 remain unchanged. All 32 physical velocity coordinates, including
spin, are retained. Control uses full-state finite affine feedback with online axial-frame alignment, selected by complete owned forecasts: affine first, native-corrected second. The motion reference lasts
8.00s and starts at reset. There is no learned policy or hardware
real-time claim. Additional force-gust robustness is unqualified for this variant.

The full native model and physical factors are audited. Every qualification
state and substep rail margin cold-replays exactly; reset states, raw/delivered
commands, world tilts, gates and native accounting reconstruct. One qualification
episode repeats from frozen source with every array and metadata byte equal,
including its complete query and forecast ledgers. Qualification seeds
20263416..47 were reserved before testing and
used once. Preserve them as reproduction data, not fresh selection evidence.

The evidence retains the unsuccessful corrected-feedback development family,
lookahead and effort variants, and the affine family's failed31/32 qualification
on20263216..47. Its failed start becomes explicit development data; the new bank
uses a fresh range. The runtime passes 294 regression tests. The prepared fourteen-link
record and original 1D source remain unchanged.

Extract all evidence ZIPs together and run these commands from the extracted
`2d` directory. Use Python 3.11 and the dependency lock on the recorded platform,
then reproduce the development pilot:

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install -e .
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python scripts/reproduce_ten_link_immediate_bank.py --route evidence/qualification32_001/frozen_route.npz --out output/reproduce_immediate_10
```

Qualification charges 576,000 native integrations;
its independent audit adds 210,000. The full frozen
runtime, reference/models/controllers, positive and negative tapes, drivers,
registered plans, audits and SHA256 inventories are included.

The nominal reference is mostly planar, while the physical simulation remains
fully spatial. Across the new qualification cohort, the largest recorded
absolute Y cart displacement is0.136529m and the minimum native substep boundary
margin is0.927397m. The two force axes and all ball-joint motion remain free.
