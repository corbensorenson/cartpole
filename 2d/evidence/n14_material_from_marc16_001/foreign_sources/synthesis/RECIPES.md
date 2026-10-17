# Original stages and runnable public commands

Start with the installation and thread settings in [README.md](README.md).
Run each command in a fresh Python process from the repository root. Every
`--out` path must be new. Original source is under `original/{15,16}/`;
`stage.py` supplies portable paths and records the exact child argv in
`adapter.json`, with complete original stdout/stderr in `stage.log`.

The sequence is deliberately staged. A stage uses its specified retained
checkpoint unless you pass the optional `--reference` or `--episode` shown
below. This permits matched reproduction of a selected stage even if a fresh
earlier nonlinear solve diverges. No stage silently substitutes a checkpoint
for its own output. Inspect its original `product/receipt.json` and its array
comparison before treating the new output as feasible or as the selected one.

## Inputs and derivative providers

```sh
python3 synthesis/regenerate.py fetch-inputs --out .artifacts/synthesis/upstream
.venv-synthesis/bin/python synthesis/regenerate.py model \
  --count 15 --out .artifacts/synthesis/model15
.venv-synthesis/bin/python synthesis/regenerate.py model \
  --count 16 --out .artifacts/synthesis/model16
```

The downloader fetches Corben's exact fourteen controller JSON and partial
fifteen `actual_minimax.npz` at the pinned upstream commit. It checks both
SHA256 values. The partial fifteen warm start is an initializer, not a passing
fifteen result. No upstream Python package is imported.

The new model command calls the original `planning_functions`: compiled
capsule mass/COM/inertia, relative-joint damping/armature, held normalized cart
force, four RK4 steps, and implicit-equation derivatives. JIT flags are `-O2`
and `-ffp-contract=off`. It recompiles locally rather than shipping native
libraries. It checks sampled native transitions and compares all 700 selected
reference Jacobians with the new public derivative checkpoint. Model build
time is recorded separately from controller time.

## Fifteen reference and selected feedback

```sh
.venv-synthesis/bin/python synthesis/stage.py 15-reference \
  --model .artifacts/synthesis/model15 --inputs .artifacts/synthesis/upstream \
  --out .artifacts/synthesis/reference15

.venv-synthesis/bin/python synthesis/stage.py 15-first-native-pass \
  --model .artifacts/synthesis/model15 \
  --out .artifacts/synthesis/first-native15

.venv-synthesis/bin/python synthesis/stage.py 15-physical-reference \
  --out .artifacts/synthesis/physical-reference15

.venv-synthesis/bin/python synthesis/stage.py 15-prototype \
  --model .artifacts/synthesis/model15 \
  --out .artifacts/synthesis/prototype15
```

The selected historical sequence was:

1. Reflect the public partial-fifteen route into the `+pi` hanging branch.
   Construct a 650-command reference guess, retain its first 400-command
   warm prefix and taper toward upright. Repair it with the original
   `repair_canonical_reference.py` worker. Its source exposes all boundary,
   rail/force/state and final-hold constraints, objective and IPOPT options.
   The historical selected repaired reference is `data/15/repaired-reference.npz`.
2. Native-value control from exact hanging, without parking, produced the
   first actual full successful tape (`data/15/first-native-pass.npz`). This
   is different from declaring the optimizer's nominal reference successful.
   The worker uses native float32 forecasts after route tick 180. The selected
   design-stage receipt records tracking cost 1000 and native affine defects.
3. The original manual stage extracted the first **700 delivered commands and
   701 actual states** of that tape. `15-physical-reference` makes that stage
   explicit and checks against `data/15/physical-reference.npz`.
   Add `--replay-native` to regenerate those states from exact hanging by
   replaying the retained original delivered controls. This is an explicitly
   recorded-command replay, not a claim of newly reproducing the original
   online action selection.
4. The selected prototype used that physical reference, binary64 QR feedback
   and value factors, 800 parking commands, cart target zero, parking decay
   1.2, one-command native value forecasts and joint lookahead two from route
   tick 390. `15-prototype` runs the original worker with those literal flags,
   then the original packager with `--bounded-planning`. That packages the
   frozen ordered 32-profile list used by the released live planner.

To propagate your own regenerated artifacts, add:

- `--reference .artifacts/synthesis/reference15/product/reference.npz` to
  `15-first-native-pass`;
- `--episode .artifacts/synthesis/first-native15/product/episode-3156000.npz`
  to `15-physical-reference`;
- `--reference .artifacts/synthesis/physical-reference15/product/reference.npz`
  to `15-prototype`.

These are new design/diagnostic attempts. The adapters use a published exact
diagnostic recipe, never a fresh-qualification claim. The final original source
retains later options that were added during experimentation; the selected
arguments and historical outcome checkpoints distinguish the used path.
The original fifteen repair command's complete argv was not retained with
its receipt. The public invocation reconstructs its recorded inputs and
650-command settings using the preserved final worker; its measured agreement
is reported, rather than asserted to recover an unrecorded earlier script revision.

## Sixteen moving orbit and hanging-start bridge

```sh
.venv-synthesis/bin/python synthesis/stage.py 16-lift \
  --inputs .artifacts/synthesis/upstream --out .artifacts/synthesis/lift16

.venv-synthesis/bin/python synthesis/stage.py 16-base-reference \
  --model .artifacts/synthesis/model16 --out .artifacts/synthesis/base-reference16

.venv-synthesis/bin/python synthesis/stage.py 16-orbit \
  --model .artifacts/synthesis/model16 --out .artifacts/synthesis/orbit16

.venv-synthesis/bin/python synthesis/stage.py 16-launch \
  --out .artifacts/synthesis/launch16

.venv-synthesis/bin/python synthesis/stage.py 16-bridge \
  --model .artifacts/synthesis/model16 --out .artifacts/synthesis/bridge16

.venv-synthesis/bin/python synthesis/stage.py 16-probe07 \
  --model .artifacts/synthesis/model16 --out .artifacts/synthesis/probe07
```

`16-lift` uses the original material-coordinate interpolation of the attributed
fourteen donor, preserving cart state and interpolating cumulative angles/rates
before converting back to relative coordinates. It checks the numeric lift.

`16-base-reference` reruns the original 700-command repair with a 450-command
swing guess. This is a warm reference, not a successful sixteen controller.
`16-orbit` reruns all three registered frequencies and their fixed selection
rule. `16-launch` starts at exact hanging, applies the selected parking gain
for 800 actual native commands and compares the complete parked prefix with
the original launch recording.

`16-bridge` runs the original early-bend worker against the retained base warm
reference, selected refined orbit and original actual postparking launch state.
Its public relocation views contain the real historical component outcomes
and exact hash-bound inputs. They are explicitly not fresh owner receipts.
The new launch and orbit commands let you independently regenerate and compare
those inputs before rerunning the bridge. The original optimizer solves the
450-command bridge with a delivered-float32 value map and a disclosed
straight-through input Jacobian. It then appends 250 fixed orbit commands,
substitutes literal offline boundary values and rechecks all native rows.
Those boundary operations never reset a live episode.

To test your freshly repaired base as a new bridge input, add
`--reference .artifacts/synthesis/base-reference16/product/reference.npz` to
`16-bridge`. To run the controller built from your new bridge, add
`--reference .artifacts/synthesis/bridge16/product/reference.npz` to `16-probe07`.
Any numerical differences and physical failures are retained.

The original selected probe07 flags are 16 seconds parking, cart target `-0.05`,
tracking cost 1000, native affine bias, float64 dot arithmetic, zero terminal
root and zero post-route force. Its original launch argv and resource caps are
in [the historical command record](evidence/16/original-commands.json).

## Regenerate the upright and parking designs

```sh
.venv-synthesis/bin/python synthesis/regenerate.py design \
  --count 15 --out .artifacts/synthesis/design15
.venv-synthesis/bin/python synthesis/regenerate.py design \
  --count 16 --out .artifacts/synthesis/design16
.venv-synthesis/bin/python synthesis/stage.py 16-parking \
  --out .artifacts/synthesis/parking16
```

Both upright recipes promote compiled physical constants into 100-digit
Riccati design and round their returned arrays to float64. This is separate
from fifteen's **double-precision trajectory feedback recursion**. The finite
feedback recipe in `regenerate.py` regenerates the selected fifteen parking
gain from its original hanging derivative, costs and decay, and compares it
with the actual released payload.

The sixteen parking stage reruns native centered held-input derivatives,
then the five-profile angular-cost screen on the same ten diagnostic starts.
It retains every design/tape and checks `angles10-design.npz` against the
selected parking checkpoint. Its twenty-second parking tapes are component
tests, not swing-up qualification.

## Exact diagnostic with rebuilt feedback

Use the short commands in [README.md](README.md) after inspecting the stages.
With a different reference, use `regenerate.py feedback --reference PATH --model
PATH`; a saved derivative checkpoint is refused for a different reference.
Inspect the comparison and diagnostic result even when an optimizer exits zero:
solver termination, native reference defect and complete physical task success
are different fields. Validation outcomes and known reconstruction gaps are
published in [validation/README.md](validation/README.md).
