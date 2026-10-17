# Corben–Marc hybrid ideas transferred to spatial cartpole

Review date: 2026-10-16. The user requested this review while eleven-link
development was underway. Source: revision3 of
`D:/ChatGPT/cartpole/corben-marc-hybrid.md`, SHA256
`9d45c767cd826011753e6d9cad1771f2119ddf1e52a8b1dff04fa1a664edd919`.
Its read-only source copy and identity are retained in
`sources/corben_marc_hybrid_2d_20261016/`. The original1D checkout was not changed.

**Useful result already measured:** preserving affine native defects and
designing a finite upright continuation produces an eleven-link nominal full
30s pass with23.50s hold. Raw noisy starts fail, and static preparation improves
only to43/45. Adding causal per-command axial reference alignment then passes
45/45 development and32/32 NEW reserved episodes, every full30s with5.50s
hold and exact cold replay. Eleven is qualified; its source/video release is
being prepared. The document review supplied useful ideas; the actual spatial
experiments and qualification establish the new count.

## What transfers and what needs adaptation

| Document idea | Existing spatial implementation / new action | Evidence or next test |
|---|---|---|
| Full affine QR values, including defects and residuals (11.1,12.2) | New `affine_value.py` retains additive force offsets and every value root/residual | Independent condensed optimum test and matched actual11-link comparison |
| Finite continuation and explicit design/execution clocks (11.2–11.3) | New1500-step upright continuation, finite terminal value;1550-step guard comparison | Both affine designs pass nominal23.50s; neither passes raw noisy starts |
| Native delivered-action correction (14.3) | New bounded2D candidate policy with full affine value and owned native predictor | Same1.92s hold as plain affine on8s design; no demonstrated benefit in this test |
| Reachable moving upright targets (9,10.6) | High-priority alternate destination if fixed-target reliability stalls | Still proposed; no spatial orbit-bridge solve from this review |
| Sparse residual shooting / no normal-equation conditioning (4.1,10) | Existing full-spatial shooting; new augmented ridge diagnostic | Stronger measured local defect correction; standalone actual controller still fails |
| Physical references and separate incumbents (12.1) | Keep raw proposal, rebuilt route and actual tapes separately | Failed rebuild retained; nominal finite-affine pass preserved independently |
| Measured launch correction demand (8,12) | Existing hanging wrapper, full affine values and causal per-command axial reference alignment | Static16s:27/45;18s:40/45;18.4s:43/45; online18s:45/45 development then32/32 reserved |
| Actual admission, fresh qualification, cold replay (14–16) | Existing gates retained | No eleven reserved seeds consumed by these development tests |

The scalar B17 controller is not copied. Spatial errors have quaternion tangent
coordinates; there are two force components with a joint80N Euclidean norm
limit, two cart coordinates, passive twist and two physical tilt directions at
every joint. World rod tilt is the gate, not independent planar angle sums.
The document's17-link1D release is described there as external evidence, not an
independent reproduction in this spatial lab. Its gains, states, frequencies and
hold times are not spatial controller certificates.

## Matched experiments and limits

Every experiment has a pre-execution amendment in `plan.md`, frozen driver,
source, route, runtime, native accounting and actual/cold-replay tapes.
The common donor is the full-spatial70-dimensional shooting proposal from
`output/n11_direct_repair_001`. Intermediate nodes are never physical resets.

| Experiment / arm | Nominal outcome | Noisy development |
|---|---|---|
| `n11_direct_repair_001`, gap-free rebuilt baseline | Fails4.58s, no hold | Not opened |
| `n11_hybrid_affine_001`, same raw reference/gains, zero offset | Fails5.08s, no hold | Not opened |
| Same, full affine offsets |1.92s hold; fails10.88s | Not opened |
| Same, affine plus native action search |1.92s hold; fails10.66s | Not opened |
| `n11_hybrid_finite_001`,1500-step design, zero offset | Fails4.94s, no hold | Not opened |
| Same, affine | Full30s,23.50s hold |0/12 |
| `n11_hybrid_guard_001`,1550 design /1500 execution, affine | Full30s,23.50s hold |0/12 |
| `conditioned_n11_001`,16s preparation +700-command prefix | Full30s,7.50s hold |26/44;27/45 including nominal |
| `conditioned_n11_002`,18s preparation +600-command prefix | Full30s,5.50s hold |39/44;40/45 including nominal |
| `conditioned_n11_003`,18.4s preparation +580-command prefix | Full30s,5.10s hold |42/44;43/45 including nominal |
| `n11_online_gauge_pilot_001`,18s +per-command axial reference alignment | Full30s,5.50s hold |5/5 reused failures +nominal |
| `conditioned_n11_004`,same frozen pilot arrays | Full30s,5.50s hold |44/44 +nominal |
| `qualify_n11_001`,same frozen policy | Full30s,5.50s hold |32/32 fresh RESERVED; distinct from development |

All terminal actual episodes replay exactly. Failures remain failures even if
they briefly hold upright. The finite comparison changes both the reference
continuation and terminal treatment relative to the8s design; it does not isolate
which change caused the improvement. Within each design, zero-offset and affine
arms share gains/reference/Q/R/initialization. Their difference directly tests
preserving the affine force offset in that finite local controller family.

The8s affine trajectory reaches tilt0.0003054rad at8s, but grows to0.09035rad
at8.40s after switching to stationary balance. Finite design avoids that switch.
This is an observed seam and useful intervention, not proof of an exclusive cause
or a guarantee of noisy capture. The guard adds no demonstrated reliability gain
on the tested raw cohort. Native search spends25600 additional physics steps
without extending the measured hold; its local value improvement is insufficient
to admit it as a better full-task controller.

The1500-step finite controller was selected as the first nominal passer before
guard outcomes were inspected. Conditioning executes a prefix of its original
full design; regenerating a shorter value recursion would be another controller.
Axial alignment rotates state references, gain columns and value-root columns;
force offsets and value residuals remain unchanged. It never edits a live state.

Current verification:85 tests pass, including independent dense affine optimum,
nonzero-residual action selection, native candidate replay, live/predictor
isolation, cap/deduplication, gauge transport and online force invariance at1/3/11
links under equivalent observed axial frame coordinates. The published ten-link first
held-out episode is byte-identical in states/actions/tilts under the current
source (`output/n11_hybrid_ten_regression_001/parity.json`).

## Next decisions

Publish eleven's full source,32 complete held-out tapes and full first-heldout
video after staged/archive hashes and independent packaged-source reproduction.
Transfer the procedure to twelve with fresh mechanics/derivatives, a bounded
synthesis budget and new qualification seeds. Moving-tail/earlier bridge
proposals remain alternatives. Periodic closure alone cannot establish hanging-start reachability.
Do not treat2.5/5/6.25Hz1D seeds as universal spatial prescriptions.

The online variant aligns the current reference and gain columns with the
observed state each command using a closest axial polar rotation. It retains
measured spin/angular velocity and all physical joints, uses no future actual
state and never resets the plant. It is a changed nonlinear feedback law,
not a claim that raw fixed-gauge gains are arbitrarily robust. Qualification
covers32 finite noisy starts, not arbitrary disturbances or indefinite stability.

Continue publishing every newly qualified count explicitly as2D with its full
held-out video. No new count is promoted by document claims alone.
