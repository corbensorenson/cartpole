# Twelve-link development evidence

Twelve now has successful full exact-start development episodes at the
canonical 50 Hz control rate. Reserved release validation is still pending.
The eleven-link release remains the accepted frontier until twelve completes
the same frozen-source, noisy-cohort, exact-cohort and video gates.

## Procedure and the recovered failure

The architecture remains hanging contraction, an optimized feedback swing,
then upright maintenance. Twelve required a residual-form sparse sequential
convexification step on the full horizon, followed by an earlier maintenance
handoff. This is an extension of the shared synthesis procedure, with
count-specific trajectories and gains as outputs. Automatic re-synthesis
across counts and generalization to new morphologies remain separate claims.

The initializer combines an eight-second physical hanging-to-near-upright
route with a five-second virtual upright tail. Offline multiple-shooting
nodes are explicit optimizer variables. They are never imposed on the live
plant. Simultaneous state/control increments, L1 virtual-dynamics penalties,
hard cart bounds, hard virtual capture-angle bounds, and explicit quadratic
residual variables avoid forming the ill-conditioned cost Gram matrix.
Every accepted candidate must decrease the actual nonlinear merit and retain
the hard virtual-node bounds. The full physical episode is measured
independently. Native QP status does not certify physical capture.

The matched suffix-only variants freeze six seconds of the source. Strict
1e-8 native QP tolerances accept no step within the prescribed budgets.
Adaptive tolerances accept full updates; a separately marked inexact-QP
candidate decreases actual merit further. The largest physical defect falls
from 0.001757417 to 0.00000596958, but the actual hold is only 0.28 seconds.
Smaller gaps alone therefore do not establish a usable trajectory.

Allowing the entire thirteen-second horizon to change accepts thirteen
updates in twenty outer iterations. The final maximum local physical defect
is 9.93866e-8, and the total scaled L1 defect is 1.64592e-5. Search takes
983.09 seconds on the recorded runtime. Its QPs have 69,603 variables and
89,515 constraint rows, with at most 10,000 native iterations per outer step.
The actual hanging-start episode holds upright for 6.52 seconds, then hits
the rail at 13.46 seconds. It is a negative full-episode result.

The failure occurs as the route ends. At thirteen seconds the final tracking
gain norm has fallen to 1.655; the physical state looks geometrically close
to upright but its directional capture value is about 7.18e7. Switching to
the upright regulator immediately saturates the force. Earlier handoffs at
8, 9, 10, 11 and 12 seconds all pass the full exact-start episode, with
23.50 seconds upright and maximum cart excursion 1.913864 m. The saved
prefixes, feedback and physical states are unchanged through each handoff.
This identifies a concrete endpoint problem: waiting for the last planned
node can let a weakly controlled direction leave viable capture.

The nine-second handoff was selected before the corrected eight-second
replay completed. The first eight-second attempt was an argument-validation
failure because its unused capture objective began at the route endpoint.
The corrected replay removes that unused objective; it retains the same
physical prefix and gains. All attempts and the selection order are saved.

The successful nine-second route is rebuilt offline by serial feedback
execution, storing delivered float32 actions and actual physical nodes.
The rebuilt route has exactly zero checked dynamics gaps and independently
reproduces every physical state of the successful thirty-second episode.
This turns the optimizer output into a physically consistent controller
artifact suitable for the existing evaluator.

## Development validation and negative controls

The unreconstructed nine-second candidate passes a fresh twenty-seed noisy
development cohort after sixteen seconds of hanging regulation. The rebuilt
candidate is tested separately because its references and delivered controls
have changed. Hanging cart-position/velocity weights remain 10/5, control
penalty 1000, translated target -0.05 m, and tracking scale one. Every
episode runs continuously from its noisy hanging reset without state
replacement at the phase transitions. These are development cohorts;
reserved seeds remain separate until policy/source freezing.

Pure tracking feedback designed independently of the optimizer's clipped
descent step does not rescue the suffix-only route: both regularization zero
and 625 hold for 0.28 seconds. Applying regularization-zero pure tracking to
the full-horizon route worsens its hold to 1.34 seconds. The successful
candidate retains the optimizer's saved feedback; independently recomputing
gains is not an interchangeable operation on this sensitive plant.

A separate small-periodic-motion component hypothesis also fails the tested
robustness probes. With control penalty 1000, stationary upright feedback
passes four 1e-11 perturbations; three tested sinusoidal near-upright orbits
pass none. All three track their unperturbed initializers for thirty seconds.
Their orbit defects are explicitly nonzero at about 1e-14. These component
episodes begin near upright, not hanging, and cannot advance the count.
They use binary64 nonlinear MuJoCo and binary64 sparse/QR calculations.
The first component report inherited an environment counter that includes
the initial upright state, giving 30.02 seconds for a thirty-second episode;
subsequent reports count post-step physical samples independently and retain
the raw environment metric alongside it.

## Evidence locations

- Full-horizon synthesis: `runs/frontier_campaign_20261001/n12_precision80_residual_scvx_capture5s_full_horizon_penalty1e5_inexact_qp_iter20/`.
- Nine-second handoff: `runs/frontier_campaign_20261001/n12_precision80_residual_scvx_full_horizon_handoff9_replay/`.
- Exact physical rebuild: `runs/frontier_campaign_20261001/n12_precision80_residual_scvx_handoff9_physical_rebuild_delivered_controls/`.
- Matched suffix comparison: `runs/frontier_campaign_20261001/n12_precision80_residual_scvx_fixed_adaptive_inexact_comparison/`.
- Rebuilt development cohorts: `runs/frontier_campaign_20261001/n12_precision80_scvx_rebuilt_handoff9_park16_cartQ10_vQ5_noisy20_development/` and `...noisy100_development/`.

Commands, budgets, source/input snapshots and hashes accompany these
artifacts. All 374 current tests pass. This development note does not replace
the promotion manifest or constitute an external record claim.
