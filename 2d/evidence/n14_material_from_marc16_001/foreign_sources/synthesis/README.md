# Fifteen and sixteen: synthesis source, recipes and checkpoints

This export answers the design-reproduction questions behind the public
[fifteen](../benchmark/README.md) and [sixteen](../benchmark/canonical16/README.md)
controllers. It includes the reference-generation workers, moving-orbit
construction, actual feedback/parking recipes, selected intermediate arrays,
and retained comparisons. Read [the six detailed answers](ANSWERS.md) and
[the staged command sequence](RECIPES.md).

The original benchmark-specific Python source and numeric artifacts are
preserved byte for byte under `original/` and `data/`; hashes and source
provenance are in [original-manifest.json](original-manifest.json).
`regenerate.py`, `stage.py` and the portable metadata are new public adapters.
Neither this export nor its diagnostics replace a historical qualification,
controller freeze, recording or consumed evaluation label.

## Small runnable reconstruction

Run from the repository root. Python 3.12.3 and the recorded package versions
are recommended. The rebuild commands additionally require a C compiler for
CasADi JIT and use IPOPT/MUMPS from the CasADi wheel. Everything here runs on
CPU; no robot or GPU service is involved.

```sh
git clone https://github.com/marctheshark3/pendulum-arena.git
cd pendulum-arena
uv venv --python 3.12.3 .venv-synthesis
uv pip install --python .venv-synthesis/bin/python -r synthesis/requirements.txt
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 synthesis/regenerate.py verify

.venv-synthesis/bin/python synthesis/regenerate.py feedback \
  --count 15 --out .artifacts/synthesis/feedback15
.venv-synthesis/bin/python synthesis/regenerate.py diagnostic \
  --count 15 --controller .artifacts/synthesis/feedback15 \
  --out .artifacts/synthesis/diagnostic15

.venv-synthesis/bin/python synthesis/regenerate.py feedback \
  --count 16 --out .artifacts/synthesis/feedback16
.venv-synthesis/bin/python synthesis/regenerate.py diagnostic \
  --count 16 --controller .artifacts/synthesis/feedback16 \
  --out .artifacts/synthesis/diagnostic16
```

Every output directory must be new. This reconstructs the selected controller
payloads from the selected **reference checkpoints**, recomputed native defects,
the original feedback code and the explicit parking design. It compares every
array's shape, dtype and bytes with the released payload. The sixteen command
also regenerates the original 700-command `probe07-controller.npz`, then retains
the actual selected 650-entry prefix without recomputing its backward recursion.

The default derivative input is a new numeric checkpoint evaluated from the
original selected CasADi provider after qualification. Its provenance is in
`data/{15,16}/derivatives.json`. It avoids requiring a compiler for this short
path. It is not a learned model or a historical training artifact. The original
provider's source is included, and the following command rebuilds it and checks
all reference Jacobians against that checkpoint:

```sh
.venv-synthesis/bin/python synthesis/regenerate.py model \
  --count 16 --out .artifacts/synthesis/model16
.venv-synthesis/bin/python synthesis/regenerate.py feedback \
  --count 16 --model .artifacts/synthesis/model16 \
  --out .artifacts/synthesis/feedback16-rebuilt-model
```

Use `--count 15` with fresh paths for the fifteen provider. To regenerate the
references themselves, follow [RECIPES.md](RECIPES.md), starting with the pinned
upstream donor downloads. Those commands expose the original manual stages and
bounded optimizations instead of silently replacing a divergent solve with a
saved success.

## What the validation establishes

The release's [validation record](validation/README.md) distinguishes array
reconstruction, new exact-hanging diagnostics, rebuilt derivatives and attempted
optimization stages. On the recorded Linux/aarch64 stack, both selected
feedback payloads reconstruct exactly, including fifteen's value factors and
sixteen's original probe07 and shortened table. The new exact-hanging
diagnostics complete 1,500 commands and hold upright for 8.18 s and 7.34 s,
respectively; cold native replay has zero transition defect.

The optimizer reference is not the actual physical episode. In particular,
sixteen's selected 700-command optimizer reference has a native cold-map gap
around `3.06e-6`; its affine feedback incorporates those measured defects.
The actual released source calibration and actual diagnostic recordings are
checked separately. Do not substitute optimizer feasibility, low gain norm or
a partial local hold for a complete passing physical episode.

The historical runtime is Linux/aarch64, Python 3.12.3, NumPy 2.5.2, SciPy 1.18.1,
CasADi 3.8.1, MuJoCo 3.3.5 and mpmath 1.3.0, with one BLAS/OMP thread. Package
versions do not identify all native binaries. Other architectures or solvers
can produce different arrays or optimization paths; reports retain differences
and failures. No general numerical-portability claim follows. Fifty hertz is
simulated action cadence, not a hardware or wall-clock deadline claim.

## Scope and attribution

The task and initial public donors come from
[Corben Sorenson's cartpole work](https://github.com/corbensorenson/cartpole/tree/b84c6413974d5c0831cdf261ce15e58ba65cd3f5).
The two original donor files are downloaded from that pinned source and verified
against [upstream-inputs.json](upstream-inputs.json); upstream code, Git history
and the original donor files are not vendored here. Our own derived references,
source, authored simulation data and documentation carry their included MIT
license. Third-party packages retain their licenses.

Original private Git history, robot/legacy-body assets, credentials, native
runtime binaries and reserved activation commands are excluded. The scripts
use either exact hanging or explicitly published diagnostic recipes, without
touching a permanent qualification ledger. Historical fifteen CLI labels in
`stage.py` are reproduction of known diagnostics, not fresh held-out trials.

The sixteen result uses native-model forecasts and classical feedback. Earlier
[JEPA/classical experiments](https://github.com/marctheshark3/rage-jepa) and the
bounded seventeen follow-up are discussed under [learning](ANSWERS.md#learning-and-seventeen).
No JEPA advantage is attributed to the sixteen qualification.
