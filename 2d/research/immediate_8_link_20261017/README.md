# Eight-link 2D cartpole — immediate swing-up

**32/32 new reserved starts**, after 46/46 development starts. Every episode
completes 30s with at least 26.20s upright and zero hanging preparation.
[Full first-reserved video and complete evidence](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-8-link-immediate-v1).

The cart has unrestricted XY slides and every passive joint is a ball joint.
The original 3m/1kg chain, 1kg cart, 80N force disk, +/-3m boundary, 50Hz controls,
200Hz RK4 and world tilt <.15rad for 5s remain unchanged. All 26 physical
velocity coordinates, including spin, are retained. Seven's immediate release
and the prepared fourteen-link record remain immutable.

Two frozen full-motion feedback/value profiles use effort penalties .001 and .0001.
At reset, owned complete native forecasts select the first full passing profile
from observed state. Selection takes computation; this is not a 20ms hardware
deadline claim. The live cart never waits, resets or receives an episode seed.
Selected feedback executes again against observations, and every actual state,
raw/delivered force and tilt matches its forecast through the final command.
Both profiles use one-step native action correction with a fixed four-step
lookahead window at commands 170..269. Full forecast physics and nested action
queries, execution queries and independent cold replay are all counted.

The original single-profile family passed 45/45 development but failed its old
fresh cohort 31/32. That failure, 20263035, became explicit development; a new,
untouched cohort 20263316..47 was used once for this bank. Stronger feedback
alone and a clock-spliced alternative also failed development. All positive
and negative eight-link tapes, models, controllers, frozen source/drivers,
registered plans, runtime locks, integrity audits and SHA256 inventories are
included across the primary ZIP and its hash-bound supplemental ZIPs. Extract
all ZIPs together. This release does not qualify additional force-gust rejection
or a learned controller.

All 32 new reserved starts selected the first profile. The second profile is
exercised by the former failed start in the targeted and 46-start development
gates. Qualification uses 7,530,616 inclusive native
integrations; its independent audit adds 427,016.

From the extracted 2d directory, use Python 3.11 and the recorded dependency lock:

```powershell
python -m pip install -r requirements-lock.txt
python -m pip install -e .
$env:OPENBLAS_NUM_THREADS="1"
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python scripts/evaluate_immediate_bank.py --route evidence/n8_immediate_bank_design_001/route.npz --cohort development46 --previous-qualification evidence/qualify_n8_immediate_001 --out output/reproduce_immediate_8
```

Qualification source is authoritative. Stored reserved seeds support reproduction;
they must not be reused as fresh evidence to select another controller.
