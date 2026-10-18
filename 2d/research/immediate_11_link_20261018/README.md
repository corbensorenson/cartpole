# Eleven-link 2D cartpole: immediate noisy swing-up

**32/32 fresh reserved starts pass after 45/45 development cases.** Every
episode completes 30 seconds and holds every rod within 0.15 radians of world
upright for at least 23.50 seconds. Hanging
preparation is zero. [Full first-reserved video](https://github.com/corbensorenson/cartpole/releases/download/2d-cartpole-11-link-immediate-v1/2d-cartpole-11-link-immediate.mp4) · [Release and artifacts](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-11-link-immediate-v1).

The original XY cart and unrestricted passive ball joints remain unchanged:
3 m / 1 kg uniform chain, 1 kg cart, 80 N radial force limit, +/-3 m square,
50 Hz controls and 200 Hz binary64 MuJoCo RK4. All 35 physical velocities,
including joint spin, remain in the model and feedback. Across qualification,
minimum native substep boundary margin is 0.913636 m;
maximum delivered force is 71.120819 N.

The fixed classical controller synthesizes a full trajectory from the observed
initial state, then applies state feedback during the swing and upright hold.
It reuses one fixed, audited reference trajectory and native model. Native prediction,
explicit physical defects, bounded Newton corrections and independently
checked constrained solves support each accepted trajectory. No physical state
reset, joint lock or hanging wait occurs during the episode.

Planning is **offline**: 4.30–4.84
minutes per reserved start on the recorded runtime. The simulation does not
advance during planning. Hardware real-time operation, a learned controller,
and robustness to additional mid-swing force gusts are unqualified. The nominal
motion is mostly planar; the simulator remains fully spatial and feedback
handles the original spatial start perturbations.

Qualification seeds 20263516–20263547 were reserved before testing and consumed
once by the frozen candidate. All cases pass separate reset, native model,
physical-rate, certificate, feedback, full-episode and cold-replay audits. The
first reserved case also repeats its complete planning, actual episode and cold
replay from the extracted runtime: all planning arrays and the physical tape
match exactly. Reproduction adds no fresh sample. Development inherits its
13 already audited pilot cases instead of rerunning them.

The release includes the full 1,501-frame, 50 fps video of the first reserved
case, complete current case histories and audits, and the selected synthesis
family's negative forecasts, continuation trials and corrections. Earlier
closed research families remain in their original stores; their reports and
sealed inventories are included. This does not claim to upload every derivative
cache from the entire laboratory. The manifest defines exact archive coverage
and native-step accounting.

## Source and reproduction

[Manifest](manifest.json) · [Frozen algorithm](source/helpers/eleven_observed_synthesis.py) ·
[Frozen physical package](source/spatialpole) · [Source hashes](source/SHA256SUMS.json).
These files expose the exact candidate for review. For execution, download the
runtime ZIP and every evidence ZIP from the release. Extract evidence parts
together and extract the runtime ZIP into its own runtime directory. Its README
and bootstrap preserve the required directory layout and pinned Windows /
Python 3.11 numerical runtime. Use seed 20263516 with `--expected-case` pointing
to `evidence/observed_synthesis_qualification32_n11_001/seed20263516` for the recorded first case. This is repeated
evidence, not a new qualification cohort. The archives contain binary data;
source snapshots in this study retain exact bytes through local Git attributes.

The original 1D work, [14-link conditioned 2D result](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-14-link-v1),
and [10-link immediate result](../immediate_10_link_20261017/README.md) remain
preserved. Eleven is the new immediate-swing-up count; fourteen remains the
highest qualified conditioned comparison count.
