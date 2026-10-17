# 7-link 2D cartpole — immediate swing-up

**32/32 fresh noisy starts**, plus 45/45 development starts, complete 30s and hold
upright for at least 25.48s. **Zero hanging preparation.**

[Full video, source and evidence](https://github.com/corbensorenson/cartpole/releases/tag/2d-cartpole-7-link-immediate-v1).

The cart has two forces and all passive ball joints retain unrestricted tilt
and twist. Original 3m/1kg chain, 1kg cart, 80N force disk, 3m square boundary,
50Hz controls/200Hz RK4, world tilt <.15rad for 5s; no plant or gate changes.
This qualifies the immediate-start variant. Fourteen remains the highest
qualified overall count. No added-force-gust qualification or learned model
is claimed. All previous count releases remain immutable.

A complete native 1500-command motion supplies the reference. Full native
finite-difference models and finite affine QR values retain every physical
velocity; axial quotient coordinates affect planning/reference frames only.
Online frame-aligned feedback corrects the noisy reset during motion. The chosen
controller performs no online native action search and never receives a seed.
Independent replay verifies all actual tapes; frozen-runtime reexecution of the
first fresh start matches every state/raw/delivered force/tilt byte. Native
single-transition stencil, physical-factor and finite-value audits pass.

The evidence ZIP includes all this variant's development/qualification tapes,
frozen source/drivers/runtime/controller, native model cache, tests and per-file
SHA256 inventory. Six-link negative terminal-value/frame-only/gust variants
are included in the six-link bundle; the seven-link audit accounting correction
is retained in its bundle. Earlier failures remain in their original releases.

From an extracted ZIP, install requirements-lock.txt and pip install -e .;
set OPENBLAS_NUM_THREADS=1,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1. Reproduce with:

```powershell
python scripts/evaluate_immediate_finite.py --route output/n7_immediate_finite_design_001/affine_no_search.npz --cohort development45 --out output/reproduce_immediate_7
```

Frozen runtime from qualification/source is authoritative. Qualification seeds
are recorded for reproduction and must not be reused to select another family.
